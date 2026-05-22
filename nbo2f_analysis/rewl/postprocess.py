"""Post-processing: derive CSVs and the diagnostic figure from a live
`WangLandauParallelTempering` orchestrator.

The HDF5 checkpoint is the source of truth; the artefacts written here
are derived and re-written on every subcommand invocation. Stitching
(``mchammer-pt-stitch rewl_state.h5``) and canonical reweighting
(``mchammer-pt-reweight stitched_dos.csv``) are separate
post-processing steps and are not part of this writer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import pandas as pd

from mchammer_pt import swap_acceptance_rates
from mchammer_pt.analysis.dos import stitch_entropy

from nbo2f_analysis.rewl.plots import diagnostic_figure


def write_all(pt: Any, cfg: Any, *, history: Any | None = None) -> None:
    """Write all derived artefacts to the current working directory.

    Produces: convergence.csv, exchange_rates.csv,
    per_move_rejection_rates.csv, rewl_diagnostics.png. Stitching
    (``mchammer-pt-stitch rewl_state.h5``) and canonical reweighting
    (``mchammer-pt-reweight stitched_dos.csv``) are separate
    post-processing steps run by job scripts; the checkpoint .h5 is
    the single durable artefact the stitcher consumes.

    The diagnostic figure's stitched-DOS panel is computed
    transiently inside this function (via ``stitch_entropy`` applied
    to ``pt.results()``'s walker-merged per-window curves) and is
    never persisted to disk.

    ``history`` may be supplied explicitly to override ``pt.history`` —
    needed by ``resume``, which must pass the concatenated pre-run +
    new-segment history because ``pt.run(...)`` replaces
    ``pt._history`` with a fresh per-segment one.
    """
    out = Path.cwd()
    results = pt.results()
    h = history if history is not None else pt.history

    # Per-window walker-merged entropies (kept in memory for the
    # transient stitched-DOS figure panel; not written to disk).
    per_window_entropy: list[pd.DataFrame] = [
        wr.get_entropy() for wr in results
    ]

    # Convergence.
    stats = pt.pool.per_window_stats()
    walkers = cfg.windows.walkers_per_window
    conv_df = pd.DataFrame([
        {
            "win": i,
            "n_walkers": walkers[i],
            "halvings": s["halvings"],
            "fill_factor": s["fill_factor"],
            "converged": s["converged"],
        }
        for i, s in enumerate(stats)
    ])
    conv_df.to_csv(out / "convergence.csv", index=False)

    # Exchange rates (cumulative — uses caller-supplied history).
    rates = swap_acceptance_rates(h)
    n_windows = len(cfg.windows.list)
    bounds = cfg.windows.bounds
    exchange_rows = []
    for i in range(n_windows - 1):
        rate = rates[i] if i < len(rates) else float("nan")
        overlap_eV = bounds[i][1] - bounds[i + 1][0]
        exchange_rows.append({
            "pair": f"{i}-{i + 1}",
            "acceptance_rate": float(rate),
            "overlap_eV": float(overlap_eV),
        })
    exchange_df = pd.DataFrame(exchange_rows)
    exchange_df.to_csv(out / "exchange_rates.csv", index=False)

    # Per-move rejection.
    reject_rows = []
    for i, wr in enumerate(results):
        dc = wr.containers[0]
        obs = dc.observables
        move_names = sorted({
            o.replace("_window_rejection_rate", "")
            for o in obs if o.endswith("_window_rejection_rate")
        })
        for mname in move_names:
            row = {"win": i, "move": mname}
            for suffix, key in [
                ("_window_rejection_rate", "window_reject_rate"),
                ("_wl_rejection_rate", "wl_reject_rate"),
                ("_acceptance_rate", "acceptance_rate"),
            ]:
                tag = f"{mname}{suffix}"
                if tag in obs:
                    row[key] = float(dc.get(tag).mean())
            reject_rows.append(row)
    if reject_rows:
        pd.DataFrame(reject_rows).to_csv(
            out / "per_move_rejection_rates.csv", index=False,
        )

    # Transient stitched DOS for the figure panel only (no CSV
    # written; mchammer-pt-stitch produces the persistent artefact).
    stitched, _overlap_errors = stitch_entropy(
        per_window_entropy, cfg.windows.energy_spacing,
    )

    # Figure (WL-health only; canonical reweighting is post-processing).
    rates_list = [
        rates[i] if i < len(rates) else 0.0
        for i in range(n_windows - 1)
    ]
    fig = diagnostic_figure(
        per_window_entropy=per_window_entropy,
        convergence_df=conv_df,
        exchange_rates=rates_list,
        stitched_dos=stitched,
        cfg=cfg,
    )
    fig.savefig(out / "rewl_diagnostics.png", dpi=120)
    import matplotlib.pyplot as plt
    plt.close(fig)


def postprocess(cfg) -> None:
    """Re-derive CSVs and the figure from the checkpoint, no WL run."""
    from pathlib import Path

    from icet import ClusterExpansion
    from mchammer_pt.contrib import CoordinatedCustomWangLandauEnsemble
    from mchammer_pt.wl import WangLandauParallelTempering

    from nbo2f_analysis.rewl.config import resolve_ce_path
    from nbo2f_analysis.rewl.run import _build_moves_and_kwargs

    cwd = Path.cwd()
    checkpoint_path = cwd / cfg.checkpoint.filename
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"No checkpoint at {checkpoint_path}; nothing to postprocess."
        )

    ce = ClusterExpansion.read(str(resolve_ce_path(cfg)))
    _, _, _, ensemble_kwargs = _build_moves_and_kwargs(cfg, ce)
    pt = WangLandauParallelTempering.resume_process_pool(
        str(checkpoint_path),
        cluster_expansion=ce,
        ensemble_cls=CoordinatedCustomWangLandauEnsemble,
        ensemble_kwargs=ensemble_kwargs,
    )
    write_all(pt, cfg)
    print("Done.")
