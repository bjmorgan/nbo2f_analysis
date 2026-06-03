"""Parallel per-window starting-configuration search.

Uses ``multiprocessing`` with the ``spawn`` context because icet's
C++ extensions crash under ``fork`` on macOS. Each worker loads the
cluster expansion independently from disk.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path

import numpy as np
from ase import Atoms

from nbo2f_analysis.ce_tools import (
    atoms_from_f_mask_stable,
    build_tiled_groundstate_atoms,
)
from nbo2f_analysis.rewl.config import MovesCfg

# Upper bound for per-anneal child seeds; comfortably within the range
# numpy's random generator accepts.
_SEED_RANGE = 2**31
# Grace period (seconds) for search workers to observe the stop signal and
# exit cleanly before any straggler is terminated.
_SHUTDOWN_TIMEOUT_S = 15.0
# Sentinel a worker puts on the result queue to report an uncaught
# exception, so the parent re-raises it rather than misreporting the fault
# as an unfillable window.
_WORKER_ERROR = "__worker_error__"


@dataclass(frozen=True)
class SearchParams:
    """Knobs for the annealing starting-config search."""
    temperature_high: float
    temperature_low: float
    n_temperature_levels: int
    sweeps_per_level: int
    harvest_interval_sweeps: int
    max_anneals_per_worker: int
    backstop_temperature: float
    backstop_sweeps: int


def _geometric_schedule(
    t_high: float, t_low: float, n_levels: int
) -> list[float]:
    """Geometric temperatures from ``t_high`` down to ``t_low`` inclusive.

    ``n_levels == 1`` is a degenerate single-temperature schedule: it
    returns ``[t_high]`` only, with no cooling.
    """
    if n_levels == 1:
        return [t_high]
    ratio = (t_low / t_high) ** (1.0 / (n_levels - 1))
    return [t_high * ratio**k for k in range(n_levels)]


def _windows_containing(
    e: float, windows: list[tuple[float, float]]
) -> list[int]:
    """Indices of every window whose band contains energy ``e``."""
    return [i for i, (lo, hi) in enumerate(windows) if lo <= e <= hi]


def _record_config(
    found: list[list[np.ndarray]],
    seen: list[set[bytes]],
    windows: list[tuple[float, float]],
    counts: list[int],
    e: float,
    numbers: np.ndarray,
) -> bool:
    """Record ``numbers`` into each still-short window containing ``e``.

    Mutates ``found`` and ``seen`` in place: dedups per window by
    occupation-vector bytes and never exceeds a window's target count.
    Returns ``True`` once every window has reached its target.
    """
    key = numbers.tobytes()
    for i in _windows_containing(e, windows):
        if len(found[i]) < counts[i] and key not in seen[i]:
            seen[i].add(key)
            found[i].append(numbers.copy())
    return all(len(found[i]) >= counts[i] for i in range(len(counts)))


def _inject_ground_state(
    found: list[list[np.ndarray]],
    seen: list[set[bytes]],
    windows: list[tuple[float, float]],
    counts: list[int],
    e_gs: float,
    gs_numbers: np.ndarray,
) -> None:
    """Seed the exact ground state into the lowest window that contains it.

    Mutates ``found`` and ``seen`` in place. No-op if ``e_gs`` lies outside
    every window: ground-state seeding is a best-effort guarantee, and such
    a window is left for the anneal/backstop to fill (or the caller's hard
    error).
    """
    for i, (lo, hi) in enumerate(windows):
        if lo <= e_gs <= hi:
            key = gs_numbers.tobytes()
            if len(found[i]) < counts[i] and key not in seen[i]:
                seen[i].add(key)
                found[i].append(gs_numbers.copy())
            return


def _anneal_worker(
    seed: int,
    ce_path: str,
    n_sc: int,
    windows: list[tuple[float, float]],
    moves_cfg: MovesCfg,
    params: SearchParams,
    stop_event: MpEvent,
    result_queue: MpQueue,
) -> None:
    """Run independent anneals, emitting distinct in-window configs.

    A spawn-process target. Builds the calculator and production move set
    once, then repeatedly anneals from a fresh random fill down the
    temperature schedule, putting each novel in-window occupation vector on
    ``result_queue`` as ``(energy, numbers)``. Returns early whenever
    ``stop_event`` is set.

    Any exception is reported to the parent as a ``(_WORKER_ERROR,
    traceback)`` sentinel rather than dying quietly and being misread as an
    unfillable window. A hard C++ crash that kills the process outright
    cannot be caught here; the parent still detects that via its
    dead-worker check.
    """
    try:
        from icet import ClusterExpansion
        from mchammer.calculators import ClusterExpansionCalculator
        from mchammer_moves import CustomCanonicalEnsemble

        from nbo2f_analysis.rewl.nbo2f import (
            build_moves,
            resolve_anion_sublattice_index,
        )

        ce = ClusterExpansion.read(ce_path)
        atoms_gs = build_tiled_groundstate_atoms(n_sc=n_sc)
        n_atoms = len(atoms_gs)
        calc = ClusterExpansionCalculator(atoms_gs.copy(), ce)
        sublattice_index = resolve_anion_sublattice_index(calc)
        moves = build_moves(n_sc, sublattice_index, moves_cfg)

        rng = np.random.default_rng(seed)
        temps = _geometric_schedule(
            params.temperature_high,
            params.temperature_low,
            params.n_temperature_levels,
        )
        n_anion = 3 * n_sc**3
        steps_per_level = params.sweeps_per_level * n_atoms
        harvest_batch = params.harvest_interval_sweeps * n_atoms
        emitted: set[bytes] = set()

        def _maybe_emit(numbers: np.ndarray) -> None:
            e = float(calc.calculate_total(occupations=numbers))
            if not _windows_containing(e, windows):
                return
            key = numbers.tobytes()
            if key in emitted:
                return
            emitted.add(key)
            result_queue.put((e, numbers.copy()))

        for _ in range(params.max_anneals_per_worker):
            if stop_event.is_set():
                return
            mask = np.zeros(n_anion, dtype=bool)
            mask[rng.choice(n_anion, size=n_sc**3, replace=False)] = True
            current = atoms_from_f_mask_stable(n_sc, mask)
            for t in temps:
                if stop_event.is_set():
                    return
                ens = CustomCanonicalEnsemble(
                    structure=current,
                    calculator=calc,
                    temperature=t,
                    moves=moves,
                    random_seed=int(rng.integers(_SEED_RANGE)),
                )
                done = 0
                while done < steps_per_level:
                    if stop_event.is_set():
                        return
                    batch = min(harvest_batch, steps_per_level - done)
                    ens.run(batch)
                    done += batch
                    _maybe_emit(ens.structure.numbers)
                current = ens.structure.copy()
    except Exception:  # report any worker fault to the parent
        import traceback

        result_queue.put((_WORKER_ERROR, traceback.format_exc()))


def _lingering_backstop(
    found: list[list[np.ndarray]],
    seen: list[set[bytes]],
    windows: list[tuple[float, float]],
    counts: list[int],
    atoms_gs: Atoms,
    calc,
    moves,
    params: SearchParams,
) -> None:
    """Top up still-short windows by lingering near their band.

    For each window with an anchor but fewer than ``counts[i]`` configs,
    runs a fixed-temperature canonical walk (production moves, Metropolis
    on the real CE energy) seeded from one of its found configs, harvesting
    further distinct in-window configs. Windows with no anchor are left for
    the caller's hard error. Mutates ``found`` and ``seen`` in place.
    """
    from mchammer_moves import CustomCanonicalEnsemble

    n_atoms = len(atoms_gs)
    total_steps = params.backstop_sweeps * n_atoms
    harvest_batch = params.harvest_interval_sweeps * n_atoms

    for i, (lo, hi) in enumerate(windows):
        if len(found[i]) >= counts[i] or not found[i]:
            continue
        anchor = atoms_gs.copy()
        anchor.numbers = found[i][0]
        ens = CustomCanonicalEnsemble(
            structure=anchor,
            calculator=calc,
            temperature=params.backstop_temperature,
            moves=moves,
            random_seed=i + 1,  # fixed, reproducible per-window backstop seed
        )
        done = 0
        while done < total_steps and len(found[i]) < counts[i]:
            batch = min(harvest_batch, total_steps - done)
            ens.run(batch)
            done += batch
            numbers = ens.structure.numbers
            e = float(calc.calculate_total(occupations=numbers))
            key = numbers.tobytes()
            if lo <= e <= hi and key not in seen[i]:
                seen[i].add(key)
                found[i].append(numbers.copy())


def find_all_window_configs(
    ce_path: str | Path,
    n_sc: int,
    windows: list[tuple[float, float]],
    counts: list[int],
    moves_cfg: MovesCfg,
    n_workers: int,
    params: SearchParams,
) -> list[list[Atoms]]:
    """Find ``counts[i]`` distinct in-window configs per window by annealing.

    Runs ``n_workers`` independent simulated anneals (spawn processes) that
    propose from the production move set ``moves_cfg`` and harvest distinct
    in-window configurations on the way down a geometric temperature
    schedule. The exact ground state is injected once into the lowest
    window that contains it. Windows the anneals leave short are topped up
    by a fixed-temperature lingering backstop (when ``backstop_sweeps >
    0``); any window still short after that raises ``RuntimeError``.

    Returns one list of ``Atoms`` per window, in window order, each inner
    list of length ``counts[i]`` and all configs in that window distinct.

    Raises:
        RuntimeError: if any window cannot be filled to its target count,
            or if a search worker terminates with an exception.
    """
    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator

    from nbo2f_analysis.rewl.nbo2f import (
        build_moves,
        resolve_anion_sublattice_index,
    )

    n_windows = len(windows)
    found: list[list[np.ndarray]] = [[] for _ in range(n_windows)]
    seen: list[set[bytes]] = [set() for _ in range(n_windows)]

    atoms_gs = build_tiled_groundstate_atoms(n_sc=n_sc)
    ce = ClusterExpansion.read(str(ce_path))
    calc = ClusterExpansionCalculator(atoms_gs.copy(), ce)
    e_gs = float(calc.calculate_total(occupations=atoms_gs.numbers))
    _inject_ground_state(found, seen, windows, counts, e_gs, atoms_gs.numbers)

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_anneal_worker,
            args=(
                seed,
                str(ce_path),
                n_sc,
                list(windows),
                moves_cfg,
                params,
                stop_event,
                result_queue,
            ),
        )
        for seed in range(n_workers)
    ]
    print(f"  launching {n_workers} annealing search processes (spawn)...")
    for p in procs:
        p.start()

    def _satisfied() -> bool:
        return all(len(found[i]) >= counts[i] for i in range(n_windows))

    worker_error: list[str] = []

    def _handle(item: tuple) -> bool:
        """Record a queue item, or capture a worker-error sentinel.

        Returns ``True`` once every window has reached its target.
        """
        tag, payload = item
        if tag == _WORKER_ERROR:
            worker_error.append(payload)
            return False
        return _record_config(found, seen, windows, counts, tag, payload)

    def _drain() -> None:
        while not worker_error:
            try:
                item = result_queue.get_nowait()
            except queue.Empty:
                return
            _handle(item)

    satisfied = _satisfied()
    while not satisfied and not worker_error:
        try:
            item = result_queue.get(timeout=0.5)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                break
            continue
        satisfied = _handle(item)

    # Wind the workers down. Keep draining the queue while they exit: a
    # worker blocked on a full queue cannot observe ``stop_event`` and would
    # have to be killed mid-``put``, which can corrupt the very queue we
    # then need to drain. Continuous draining lets each worker reach its
    # next ``stop_event`` check and exit cleanly; ``terminate`` is only a
    # last resort for a straggler that overran the deadline.
    stop_event.set()
    deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_S
    while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
        _drain()
        for p in procs:
            p.join(timeout=0.1)
    _drain()
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    _drain()

    if worker_error:
        raise RuntimeError(
            "A search worker terminated with an exception; its traceback "
            f"follows:\n{worker_error[0]}"
        )

    if not _satisfied() and params.backstop_sweeps > 0:
        sublattice_index = resolve_anion_sublattice_index(calc)
        moves = build_moves(n_sc, sublattice_index, moves_cfg)
        _lingering_backstop(
            found, seen, windows, counts, atoms_gs, calc, moves, params
        )

    missing = [
        (i, len(found[i]), counts[i])
        for i in range(n_windows)
        if len(found[i]) < counts[i]
    ]
    if missing:
        raise RuntimeError(
            f"Could not fill windows (index, found, target): {missing}"
        )

    out: list[list[Atoms]] = []
    for i in range(n_windows):
        per_window: list[Atoms] = []
        for numbers in found[i][:counts[i]]:
            atoms = atoms_gs.copy()
            atoms.numbers = numbers
            per_window.append(atoms)
        out.append(per_window)
    return out
