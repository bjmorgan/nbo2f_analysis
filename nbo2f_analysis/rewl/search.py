"""Per-window starting-configuration search.

A thin adapter over ``mchammer_pt.seed_window_configs`` (the generic,
material-agnostic window-seeding search). This module supplies the three
NbO2F-specific contracts the generic search needs -- the ground-state
anchor, a correct-composition random fill, and the production move set --
and forwards everything else unchanged.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from icet import ClusterExpansion
from mchammer_pt import SeedSearchParams, seed_window_configs

from nbo2f_analysis.ce_tools import (
    atoms_from_f_mask_stable,
    build_tiled_groundstate_atoms,
)
from nbo2f_analysis.rewl.config import MovesCfg


def find_all_window_configs(
    ce: ClusterExpansion,
    n_sc: int,
    windows: list[tuple[float, float]],
    counts: list[int],
    energy_spacing: float,
    moves_cfg: MovesCfg,
    params: SeedSearchParams,
    seed: int,
) -> list[list[Atoms]]:
    """Find ``counts[i]`` distinct in-window configs per window.

    Builds the three NbO2F-specific contracts and delegates the search to
    ``mchammer_pt.seed_window_configs``:

    - ``bottom_anchor`` -- the tiled ground state, from which lower windows
      climb up into their band;
    - ``random_fill`` -- a fresh, correct-composition, disordered structure
      on the *same* stable lattice as the anchor (mandatory: the library
      binds one calculator to the anchor and reuses it for every walk), from
      which upper windows settle down into their band;
    - ``moves`` -- the production move set described by ``moves_cfg``.

    Determinism flows from ``seed``. Ground-state injection, per-window
    dedup, and the unfillable-window hard error are handled upstream.

    Args:
        ce: the trained cluster expansion (object, not a path).
        n_sc: supercell side in primitive cells.
        windows: per-window ``(lo, hi)`` energy bands, in window order.
        counts: number of distinct configs wanted per window (walker count).
        energy_spacing: Wang-Landau energy-grid bin size.
        moves_cfg: the production move-set specification.
        params: seed-search tuning knobs.
        seed: master random seed; all per-walk seeds derive from it.

    Returns:
        One list of ``Atoms`` per window, in window order, each inner list
        of length ``counts[i]`` and all configs in that window distinct.

    Raises:
        RuntimeError: (from the library) if a window cannot be filled.
    """
    from mchammer.calculators import ClusterExpansionCalculator

    from nbo2f_analysis.rewl.nbo2f import (
        build_moves,
        resolve_anion_sublattice_index,
    )

    bottom_anchor = build_tiled_groundstate_atoms(n_sc=n_sc)
    calc = ClusterExpansionCalculator(bottom_anchor.copy(), ce)
    sublattice_index = resolve_anion_sublattice_index(calc)
    moves = build_moves(n_sc, sublattice_index, moves_cfg)

    n_anion = 3 * n_sc**3
    n_f = n_sc**3

    def random_fill(fill_seed: int) -> Atoms:
        """Fresh disordered NbO2F config on the anchor's stable lattice."""
        rng = np.random.default_rng(fill_seed)
        mask = np.zeros(n_anion, dtype=bool)
        mask[rng.choice(n_anion, size=n_f, replace=False)] = True
        return atoms_from_f_mask_stable(n_sc, mask)

    return seed_window_configs(
        ce,
        moves,
        list(windows),
        list(counts),
        energy_spacing,
        bottom_anchor,
        random_fill,
        random_seed=seed,
        params=params,
    )
