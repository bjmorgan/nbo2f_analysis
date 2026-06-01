"""Tests for the REWL move registry wiring."""
from __future__ import annotations

from nbo2f_analysis.cell_symmetry import CellReflect
from nbo2f_analysis.rewl.config import MoveSpec, MovesCfg
from nbo2f_analysis.rewl.nbo2f import ALLOWED_MOVE_TYPES, build_moves


def test_cell_reflect_is_allowed():
    assert "cell_reflect" in ALLOWED_MOVE_TYPES


def test_build_moves_constructs_cell_reflect():
    cfg = MovesCfg(list=(MoveSpec(type="cell_reflect", weight=0.1),))
    moves = build_moves(n_sc=4, sublattice_index=0, moves_cfg=cfg)
    assert len(moves) == 1
    move, weight = moves[0]
    assert isinstance(move, CellReflect)
    assert weight == 0.1
