"""NbO2F-specific helpers for the REWL driver: move-set registry and
sublattice resolution.

These functions encapsulate the only material-specific composition in
the driver: the registry of move types that the YAML config can refer
to by name, and the convention that the O/F anion sublattice is the
one with chemical symbols ``{"O", "F"}``.
"""
from __future__ import annotations

from collections.abc import Callable
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


# Registry of move builders. Each entry is a string name (the value
# users write in YAML) mapped to a callable that constructs the move
# from ``(n_sc, sublattice_index)``. Adding a new move type means
# adding an entry here; users cannot construct arbitrary move classes
# from YAML alone.
_MOVE_BUILDERS: dict[str, Callable[[int, int], Any]] = {
    "pair_swap": lambda n_sc, sublattice_index: PairSwap(
        sublattice_index=sublattice_index,
    ),
    "row_shift": lambda n_sc, sublattice_index: RowShift(n_sc),
    "motif_shift": lambda n_sc, sublattice_index: MotifShift(n_sc),
    "chain_swap": lambda n_sc, sublattice_index: IndexSetSwap(
        index_sets=build_anion_chains(n_sc),
        name="chain_swap",
        require_matching_composition=False,
    ),
    "row_reflect": lambda n_sc, sublattice_index: RowReflect(n_sc),
}

ALLOWED_MOVE_TYPES: frozenset[str] = frozenset(_MOVE_BUILDERS)


def build_moves(
    n_sc: int,
    sublattice_index: int,
    moves_cfg: Any,
) -> list[tuple]:
    """Return the move set described by ``moves_cfg`` as ``(move, weight)`` pairs.

    ``moves_cfg`` is a ``MovesCfg`` from the YAML config, whose ``.list``
    is a tuple of ``MoveSpec(type, weight)`` entries. Each entry's
    ``type`` is looked up in the registry to construct the actual move.
    """
    out: list[tuple] = []
    for spec in moves_cfg.list:
        builder = _MOVE_BUILDERS[spec.type]
        out.append((builder(n_sc, sublattice_index), spec.weight))
    return out
