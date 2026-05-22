"""Parallel per-window starting-configuration search.

Uses ``multiprocessing`` with the ``spawn`` context because icet's
C++ extensions crash under ``fork`` on macOS. Each worker loads the
cluster expansion independently from disk.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

from nbo2f_analysis.ce_tools import (
    atoms_from_f_mask_stable,
    build_tiled_groundstate_atoms,
)


@dataclass(frozen=True)
class SearchParams:
    """Knobs for the per-window starting-config search."""
    max_swaps: tuple[int, ...]
    attempts_per_swap_count: int
    random_attempts: int


def _worker(
    seed: int,
    ce_path: str,
    n_sc: int,
    windows: list[tuple[float, float]],
    params: SearchParams,
    stop_event,
    result_queue,
) -> None:
    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator

    ce = ClusterExpansion.read(ce_path)
    atoms_gs = build_tiled_groundstate_atoms(n_sc=n_sc)
    calc = ClusterExpansionCalculator(atoms_gs.copy(), ce)
    rng = np.random.default_rng(seed)
    cation_offset = n_sc ** 3
    n_anion = 3 * n_sc ** 3

    local_found: set[int] = set()

    def _check(e, numbers):
        for i, (lo, hi) in enumerate(windows):
            if i not in local_found and lo <= e <= hi:
                local_found.add(i)
                result_queue.put((i, numbers.copy()))
                print(
                    f"    win {i}: E={e:.2f} eV (seed {seed})",
                    flush=True,
                )

    e_gs = float(calc.calculate_total(occupations=atoms_gs.numbers))
    _check(e_gs, atoms_gs.numbers)

    for n_swaps in params.max_swaps:
        if stop_event.is_set():
            return
        for _ in range(params.attempts_per_swap_count):
            if stop_event.is_set():
                return
            atoms = atoms_gs.copy()
            numbers = atoms.numbers.copy()
            anion_nums = numbers[cation_offset:]
            o_idx = np.where(anion_nums == 8)[0]
            f_idx = np.where(anion_nums == 9)[0]
            n = min(n_swaps, len(o_idx), len(f_idx))
            if n == 0:
                continue
            so = rng.choice(o_idx, size=n, replace=False)
            sf = rng.choice(f_idx, size=n, replace=False)
            for oi, fi in zip(so, sf):
                anion_nums[oi], anion_nums[fi] = (
                    anion_nums[fi],
                    anion_nums[oi],
                )
            numbers[cation_offset:] = anion_nums
            atoms.numbers = numbers
            e = float(calc.calculate_total(occupations=numbers))
            _check(e, numbers)

    for _ in range(params.random_attempts):
        if stop_event.is_set():
            return
        mask = np.zeros(n_anion, dtype=bool)
        mask[rng.choice(n_anion, size=n_sc ** 3, replace=False)] = True
        atoms = atoms_from_f_mask_stable(n_sc, mask)
        e = float(calc.calculate_total(occupations=atoms.numbers))
        _check(e, atoms.numbers)


def find_all_window_configs(
    ce_path: str | Path,
    n_sc: int,
    windows: list[tuple[float, float]],
    n_workers: int,
    params: SearchParams,
) -> list[Atoms]:
    """Find a starting configuration per window via parallel search.

    Returns one ``Atoms`` per window, in the same order as ``windows``.
    Each returned configuration has an energy in its window.

    Raises:
        RuntimeError: if any window could not be populated after
            exhausting all workers' search budgets.
    """
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker,
            args=(seed, str(ce_path), n_sc, list(windows), params, stop_event, result_queue),
        )
        for seed in range(n_workers)
    ]
    print(f"  launching {n_workers} search processes (spawn)...")
    for p in procs:
        p.start()

    found: dict[int, np.ndarray] = {}
    n_windows = len(windows)
    while len(found) < n_windows:
        try:
            idx, nums = result_queue.get(timeout=0.5)
            if idx not in found:
                found[idx] = nums
                print(f"  ({len(found)}/{n_windows} windows found)")
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                break

    stop_event.set()
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    missing = [i for i in range(n_windows) if i not in found]
    if missing:
        bounds = [(i, windows[i]) for i in missing]
        raise RuntimeError(
            f"Could not find configs for windows: {bounds}"
        )

    atoms_gs = build_tiled_groundstate_atoms(n_sc=n_sc)
    out: list[Atoms] = []
    for i in range(n_windows):
        atoms = atoms_gs.copy()
        atoms.numbers = found[i]
        out.append(atoms)
    return out
