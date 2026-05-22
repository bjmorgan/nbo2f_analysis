"""REWL execution entry points: fresh-start `run` and `resume`.

Both functions assume the current working directory is the run's
output directory (the CLI does the chdir before calling in). All
output paths used here are relative to CWD.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYMBAR_DISABLE_JAX", "1")

from icet import ClusterExpansion
from mchammer.calculators import ClusterExpansionCalculator
from mchammer_pt import WangLandauProgressPrinter
from mchammer_pt.contrib import CoordinatedCustomWangLandauEnsemble
from mchammer_pt.wl import WangLandauParallelTempering

from nbo2f_analysis.ce_tools import build_tiled_groundstate_atoms
from nbo2f_analysis.rewl.config import RewlConfig, resolve_ce_path
from nbo2f_analysis.rewl.nbo2f import (
    build_moves,
    resolve_anion_sublattice_index,
)
from nbo2f_analysis.rewl.postprocess import write_all
from nbo2f_analysis.rewl.search import (
    SearchParams,
    find_all_window_configs,
)


def _build_ensemble_kwargs(cfg: RewlConfig, moves, n_atoms: int) -> dict:
    block_size = n_atoms * cfg.wl.block_size_sweeps
    # icet's BaseEnsemble does `step % trajectory_write_interval` unguarded
    # and asserts the value is an int (the `is np.inf` identity check icet
    # uses to skip writes does not survive pickling across spawn workers,
    # so we cannot pass np.inf either). Use sys.maxsize as the "disabled"
    # sentinel: it is an int, modulo is cheap, and no realistic step count
    # will reach it so no trajectory rows are written.
    traj_iv = (
        n_atoms * cfg.wl.trajectory_write_interval_sweeps
        if cfg.wl.trajectory_write_interval_sweeps > 0
        else sys.maxsize
    )
    return {
        "moves": moves,
        "schedule": cfg.wl.schedule,
        "fill_factor_limit": cfg.wl.fill_factor_limit,
        "flatness_limit": cfg.wl.flatness_limit,
        "flatness_check_interval": block_size,
        "ensemble_data_write_interval": block_size,
        "trajectory_write_interval": traj_iv,
    }


def _build_moves_and_kwargs(cfg: RewlConfig, ce):
    """Replicate the move + kwargs construction used by both run and resume."""
    atoms_gs = build_tiled_groundstate_atoms(n_sc=cfg.system.n_sc)
    n_atoms = len(atoms_gs)
    calc = ClusterExpansionCalculator(atoms_gs.copy(), ce)
    sublattice_index = resolve_anion_sublattice_index(calc)
    moves = build_moves(cfg.system.n_sc, sublattice_index)
    kwargs = _build_ensemble_kwargs(cfg, moves, n_atoms)
    return atoms_gs, n_atoms, moves, kwargs


def run(cfg: RewlConfig, *, force: bool = False) -> None:
    """Fresh REWL run from a `RewlConfig`."""
    cwd = Path.cwd()
    checkpoint_path = cwd / cfg.checkpoint.filename
    if checkpoint_path.exists() and not force:
        raise RuntimeError(
            f"Checkpoint already exists at {checkpoint_path}. "
            f"Use `rewl resume` to continue, or pass --force to overwrite."
        )

    print(f"Loading CE from {resolve_ce_path(cfg)}")
    ce = ClusterExpansion.read(str(resolve_ce_path(cfg)))

    atoms_gs, n_atoms, _, ensemble_kwargs = _build_moves_and_kwargs(cfg, ce)
    block_size = n_atoms * cfg.wl.block_size_sweeps
    n_cycles_target = cfg.wl.n_trials_per_walker // block_size
    n_windows = len(cfg.windows.list)
    print(
        f"L={cfg.system.n_sc}: {n_atoms} atoms; "
        f"block={block_size} steps ({cfg.wl.block_size_sweeps} sweeps); "
        f"cycles target={n_cycles_target}; "
        f"windows={n_windows}; "
        f"walkers={cfg.windows.walkers_per_window}"
    )

    n_workers = (
        cfg.config_search.n_workers
        if cfg.config_search.n_workers is not None
        else min(os.cpu_count() or 1, n_windows)
    )
    search_params = SearchParams(
        max_swaps=cfg.config_search.max_swaps,
        attempts_per_swap_count=cfg.config_search.attempts_per_swap_count,
        random_attempts=cfg.config_search.random_attempts,
    )
    print("Finding starting configurations (parallel search)...")
    atoms_per_window = find_all_window_configs(
        ce_path=str(resolve_ce_path(cfg)),
        n_sc=cfg.system.n_sc,
        windows=cfg.windows.bounds,
        n_workers=n_workers,
        params=search_params,
    )

    print("Constructing WangLandauParallelTempering (process pool)...")
    pt = WangLandauParallelTempering.process_pool(
        cluster_expansion=ce,
        atoms=atoms_per_window,
        windows=cfg.windows.bounds,
        energy_spacing=cfg.windows.energy_spacing,
        block_size=block_size,
        random_seed=cfg.random_seed,
        data_container_file=None,
        ensemble_cls=CoordinatedCustomWangLandauEnsemble,
        ensemble_kwargs=ensemble_kwargs,
        n_walkers_per_window=cfg.windows.walkers_per_window,
        flatness_mode=cfg.wl.flatness_mode,
        merge_cadence=cfg.wl.merge_cadence,
    )
    pt.attach_cycle_callback(WangLandauProgressPrinter(pt.pool, interval=10))
    if cfg.checkpoint.interval_cycles > 0:
        pt.attach_checkpoint_writer(
            str(checkpoint_path),
            interval=cfg.checkpoint.interval_cycles,
        )

    print(
        f"Running REWL: {n_cycles_target} cycles, "
        f"{block_size} steps/block, {n_windows} windows..."
    )
    t0 = time.perf_counter()
    pt.run(n_cycles=n_cycles_target)
    dt = time.perf_counter() - t0
    print(f"REWL complete in {dt:.0f} s")

    # Final checkpoint guarantee (even if periodic writes disabled).
    pt.save_checkpoint(str(checkpoint_path))

    print("Writing analysis artefacts...")
    write_all(pt, cfg)
    print("Done.")


def resume(cfg: RewlConfig, *, extra_cycles: int | None = None) -> None:
    """Resume a previously-checkpointed REWL run."""
    from mchammer_pt.history import ExchangeHistory, read_hdf5

    cwd = Path.cwd()
    checkpoint_path = cwd / cfg.checkpoint.filename
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"No checkpoint at {checkpoint_path}; use `rewl run` to start."
        )

    print(f"Loading CE from {resolve_ce_path(cfg)}")
    ce = ClusterExpansion.read(str(resolve_ce_path(cfg)))
    _, n_atoms, _, ensemble_kwargs = _build_moves_and_kwargs(cfg, ce)
    block_size = n_atoms * cfg.wl.block_size_sweeps

    print(f"Resuming from {checkpoint_path}")
    pt = WangLandauParallelTempering.resume_process_pool(
        str(checkpoint_path),
        cluster_expansion=ce,
        ensemble_cls=CoordinatedCustomWangLandauEnsemble,
        ensemble_kwargs=ensemble_kwargs,
    )

    # `resume_process_pool` rebuilds replicas and orchestrator state but
    # does NOT repopulate `pt._history` from the checkpoint, so
    # `pt.history` is None right after resume. Read it back directly
    # from the HDF5 file so we can concatenate it with the next segment.
    prior_history, _containers, _meta = read_hdf5(str(checkpoint_path))
    n_done = prior_history.energies_per_cycle.shape[0] - 1
    target_total_cycles = cfg.wl.n_trials_per_walker // block_size
    if extra_cycles is not None:
        n_extra = int(extra_cycles)
    else:
        n_extra = max(0, target_total_cycles - n_done)
    print(
        f"Cycles done={n_done}, target={target_total_cycles}, "
        f"running {n_extra} more."
    )

    if n_extra > 0:
        pt.attach_cycle_callback(
            WangLandauProgressPrinter(pt.pool, interval=10)
        )
        if cfg.checkpoint.interval_cycles > 0:
            pt.attach_checkpoint_writer(
                str(checkpoint_path),
                interval=cfg.checkpoint.interval_cycles,
            )
        t0 = time.perf_counter()
        pt.run(n_cycles=n_extra)
        dt = time.perf_counter() - t0
        print(f"REWL complete in {dt:.0f} s")
        pt.save_checkpoint(str(checkpoint_path))
        history = ExchangeHistory.concatenate(prior_history, pt.history)
    else:
        print("Already at target; no additional cycles run.")
        history = prior_history

    print("Writing analysis artefacts...")
    write_all(pt, cfg, history=history)
    print("Done.")
