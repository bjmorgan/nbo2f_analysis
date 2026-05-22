"""NbO2F-specific helpers for the REWL driver: move set and sublattice
resolution.

These functions encapsulate the only material-specific composition in
the driver: the fixed five-move set used in all production REWL runs,
and the convention that the O/F anion sublattice is the one with
chemical symbols ``{"O", "F"}``.
"""
from __future__ import annotations

from typing import Any

from mchammer_moves import IndexSetSwap, PairSwap

from nbo2f_analysis.chain_geometry import (
    MotifShift,
    RowReflect,
    RowShift,
    build_anion_chains,
)


def resolve_anion_sublattice_index(calculator: Any) -> int:
    """Return the index of the O/F sublattice on the calculator."""
    for i, sl in enumerate(calculator.sublattices):
        if set(sl.chemical_symbols) == {"O", "F"}:
            return i
    raise RuntimeError("No O/F sublattice found on calculator")


def build_moves(n_sc: int, sublattice_index: int) -> list[tuple]:
    """Return the standard NbO2F REWL move set as ``(move, weight)`` pairs.

    The set is fixed across the production runs: a pairwise anion swap,
    a row shift, a motif shift, an anion-chain index-set swap, and a
    row reflection. Weights mirror the existing scripts.
    """
    chains = build_anion_chains(n_sc)
    return [
        (PairSwap(sublattice_index=sublattice_index), 0.1),
        (RowShift(n_sc), 0.2),
        (MotifShift(n_sc), 0.5),
        (
            IndexSetSwap(
                index_sets=chains,
                name="chain_swap",
                require_matching_composition=False,
            ),
            0.5,
        ),
        (RowReflect(n_sc), 0.5),
    ]
