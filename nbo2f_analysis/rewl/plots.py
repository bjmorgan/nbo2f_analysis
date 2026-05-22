"""Four-panel REWL diagnostic figure.

The figure is a WL-run health snapshot. Canonical thermodynamics
(Cv, <E>(T)) is a post-processing concern handled by the separate
`mchammer-pt-reweight` tool and is not produced by the driver.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def diagnostic_figure(
    per_window_entropy: list[pd.DataFrame],
    convergence_df: pd.DataFrame,
    exchange_rates: list[float],
    stitched_dos: pd.DataFrame,
    cfg: Any,
) -> plt.Figure:
    """Build the four-panel REWL diagnostic figure.

    Panels (row-major):
        (0,0) per-window raw entropy
        (0,1) stitched ln g(E)
        (1,0) exchange acceptance per window pair
        (1,1) halvings completed per window
    """
    n_windows = len(per_window_entropy)
    walkers = cfg.windows.walkers_per_window
    block_size_sweeps = cfg.wl.block_size_sweeps
    flatness_limit = cfg.wl.flatness_limit

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for i, ent in enumerate(per_window_entropy):
        if ent is not None and len(ent) > 0:
            ax.plot(ent["energy"], ent["entropy"], lw=1.0, label=f"win {i}")
    ax.set_xlabel("E (eV)")
    ax.set_ylabel("ln g(E) per window (raw)")
    ax.set_title(f"Per-window entropy (W={walkers})")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ln_g = stitched_dos["entropy"].to_numpy() - stitched_dos["entropy"].max()
    ax.plot(stitched_dos["energy"], ln_g, lw=1.2, color="black")
    ax.set_xlabel("E (eV)")
    ax.set_ylabel("ln g(E) (stitched)")
    ax.set_title("Stitched density of states")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    pair_labels = [f"{i}-{i + 1}" for i in range(n_windows - 1)]
    rate_values = [
        exchange_rates[i] if i < len(exchange_rates) else 0.0
        for i in range(n_windows - 1)
    ]
    ax.bar(pair_labels, rate_values, color="steelblue")
    ax.set_xlabel("Window pair")
    ax.set_ylabel("Swap acceptance rate")
    ax.set_title("REWL exchange rates")
    ax.set_ylim(0, max(rate_values + [0.01]) * 1.3)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    halvings = convergence_df["halvings"].to_numpy()
    ax.bar(range(n_windows), halvings, color="steelblue", alpha=0.7)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Halvings completed")
    ax.set_title("Per-window WL halvings")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"REWL L={cfg.system.n_sc}: walkers={walkers}, "
        f"block={block_size_sweeps} sweeps, "
        f"flatness={flatness_limit}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig
