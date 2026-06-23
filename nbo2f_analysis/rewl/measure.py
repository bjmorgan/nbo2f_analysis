"""Frozen-g measurement pass for the REWL driver.

Loads a converged REWL checkpoint in mchammer-pt's frozen-g measurement
mode (g(E) held fixed, coordinator disabled), attaches the NbO2F
:class:`~nbo2f_analysis.chain_order_observer.ChainOrderObserver`, and runs
a configured number of measurement cycles. The pass writes its own
checkpoint (distinct from the DOS checkpoint) and resumes/chains off it to
build up per-bin observable statistics; the canonical reduction (DOS
stitch, observable stitch, reweight) is a separate manual step.

Assumes the current working directory is the run's output directory (the
CLI does the chdir before calling in), as ``run``/``resume`` do.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYMBAR_DISABLE_JAX", "1")

from icet import ClusterExpansion
from mchammer_pt import WangLandauProgressPrinter, completed_cycles
from mchammer_pt.contrib import CoordinatedCustomWangLandauEnsemble
from mchammer_pt.history import read_hdf5
from mchammer_pt.wl import WangLandauParallelTempering

from nbo2f_analysis.chain_order_observer import build_chain_order_observer
from nbo2f_analysis.rewl.config import RewlConfig, resolve_ce_path
from nbo2f_analysis.rewl.run import (
    _status,
    _validate_extra_cycles,
    build_moves_and_kwargs,
)


def measure(
    cfg: RewlConfig,
    *,
    extra_cycles: int | None = None,
    allow_kwargs_mismatch: bool = False,
) -> None:
    """Run or resume a frozen-g measurement pass from a converged checkpoint.

    Reads the converged DOS checkpoint (``cfg.checkpoint.filename``) on the
    first invocation and the measurement checkpoint
    (``cfg.measurement.checkpoint_filename``) on resumes, attaches the
    configured :class:`ChainOrderObserver`, and advances the frozen-g walk
    until the measurement cycle budget (``cfg.measurement.n_measure_cycles``)
    is reached. ``extra_cycles`` overrides the automatic remainder.

    Args:
        cfg: The REWL configuration; must include a ``measurement`` section.
        extra_cycles: Run this many measurement cycles instead of the
            automatic remainder to the budget.
        allow_kwargs_mismatch: Forwarded to
            ``measure_from_checkpoint_process_pool``. When True, an
            ensemble-kwargs hash mismatch against the checkpoint is
            downgraded from an error to a warning (the CE-identity check
            stays strict). Needed to measure a checkpoint written in a
            different software environment -- e.g. a cluster-written DOS
            checkpoint measured locally -- where the move objects pickle to
            different bytes despite identical physics.

    Raises:
        ValueError: if ``extra_cycles`` is negative.
        RuntimeError: if ``cfg`` has no ``measurement`` section, or if the
            DOS checkpoint does not exist.
    """
    if cfg.measurement is None:
        raise RuntimeError(
            "config has no 'measurement' section; `rewl measure` requires one."
        )
    _validate_extra_cycles(extra_cycles)
    meas = cfg.measurement  # local binding: keeps the None-narrowing across calls
    cwd = Path.cwd()
    dos_ckpt = cwd / cfg.checkpoint.filename
    meas_ckpt = cwd / meas.checkpoint_filename
    if not dos_ckpt.exists():
        raise RuntimeError(
            f"No converged DOS checkpoint at {dos_ckpt}; run `rewl run` first."
        )

    _status(f"Loading CE from {resolve_ce_path(cfg)}")
    ce = ClusterExpansion.read(str(resolve_ce_path(cfg)))
    _, _, _, ensemble_kwargs = build_moves_and_kwargs(cfg, ce)

    # Baseline: completed DOS cycles. In frozen mode every walker advances
    # block_size per cycle in lockstep, so measurement cycles done equals
    # completed_cycles(measurement) - completed_cycles(DOS), exactly.
    _, dos_containers, dos_meta = read_hdf5(str(dos_ckpt))
    block_size = int(dos_meta["block_size"])
    dos_baseline = completed_cycles(dos_containers, block_size)

    if meas_ckpt.exists():
        source = meas_ckpt
        _, meas_containers, meas_meta = read_hdf5(str(meas_ckpt))
        # The measurement inherits the DOS block_size; a mismatch means the
        # two files are not a matching pair, which would make the cycle
        # count meaningless.
        meas_block_size = int(meas_meta["block_size"])
        if meas_block_size != block_size:
            raise RuntimeError(
                f"Measurement checkpoint block_size ({meas_block_size}) does "
                f"not match the DOS checkpoint ({block_size}); the two "
                f"checkpoints are not a matching pair."
            )
        # A measurement inherits its DOS run's windows verbatim, so they are
        # bit-identical for a genuine pair. Differing windows mean the
        # measurement descends from a different run -- e.g. a stray
        # checkpoint from a sibling same-system run directory.
        # equal_nan=True: open window edges can encode as NaN, and NaN != NaN
        # would otherwise falsely reject a genuine matching pair. A no-op for
        # today's always-finite REWL windows.
        if not np.array_equal(
            np.asarray(dos_meta["windows"]),
            np.asarray(meas_meta["windows"]),
            equal_nan=True,
        ):
            raise RuntimeError(
                f"Measurement checkpoint {meas_ckpt} has different energy "
                f"windows from the DOS checkpoint; they are not a matching "
                f"pair. Delete {meas_ckpt} to remeasure on the current DOS."
            )
        meas_total = completed_cycles(meas_containers, block_size)
        meas_done = meas_total - dos_baseline
        # A valid measurement checkpoint only ever adds cycles on top of the
        # DOS baseline, so meas_total >= dos_baseline always. A negative
        # difference means this checkpoint does not descend from the current
        # DOS baseline -- fail loudly rather than run a wrong cycle count.
        if meas_done < 0:
            raise RuntimeError(
                f"Measurement checkpoint {meas_ckpt} has {meas_total} "
                f"completed cycles, fewer than the DOS baseline "
                f"{dos_baseline}: it does not descend from the current DOS "
                f"checkpoint (the DOS run was extended after measurement, or "
                f"this is a stale/foreign measurement checkpoint). Delete "
                f"{meas_ckpt} to remeasure on the current DOS."
            )
    else:
        source = dos_ckpt
        meas_done = 0

    target = meas.n_measure_cycles
    if extra_cycles is not None:
        n_extra = int(extra_cycles)
    else:
        n_extra = max(0, target - meas_done)
    _status(
        f"Measurement cycles done={meas_done}, target={target}, "
        f"running {n_extra} more."
    )
    if n_extra <= 0:
        _status("Already at or beyond measurement budget; nothing to run.")
        return

    if allow_kwargs_mismatch:
        _status(
            "allow_kwargs_mismatch=True: an ensemble-kwargs hash mismatch "
            "will be downgraded to a warning (CE identity stays strict)."
        )
    _status(f"Loading checkpoint (frozen-g measurement) from {source}")
    pt = WangLandauParallelTempering.measure_from_checkpoint_process_pool(
        str(source),
        cluster_expansion=ce,
        ensemble_cls=CoordinatedCustomWangLandauEnsemble,
        ensemble_kwargs=ensemble_kwargs,
        allow_kwargs_mismatch=allow_kwargs_mismatch,
    )
    observer = build_chain_order_observer(
        cfg.system.n_sc,
        meas.observer_interval,
        meas.ops,
    )
    pt.record_observable(observer)
    pt.attach_cycle_callback(WangLandauProgressPrinter(pt.pool, interval=10))
    if meas.checkpoint_interval_cycles > 0:
        pt.attach_checkpoint_writer(
            str(meas_ckpt),
            interval=meas.checkpoint_interval_cycles,
        )

    _status(f"Running frozen-g measurement: {n_extra} cycles...")
    t0 = time.perf_counter()
    pt.run(n_cycles=n_extra)
    _status(f"Measurement complete in {time.perf_counter() - t0:.0f} s")
    pt.save_checkpoint(str(meas_ckpt))
    _status("Done.")
