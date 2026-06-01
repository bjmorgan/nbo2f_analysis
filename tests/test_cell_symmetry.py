"""Tests for nbo2f_analysis.cell_symmetry."""
from __future__ import annotations

import pytest

from nbo2f_analysis.cell_symmetry import (
    CellReflect,
    build_reflection_permutation,
)
from nbo2f_analysis.ce_tools import anion_index, index_to_anion


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reflection_is_involution(axis):
    n_sc = 4
    perm = build_reflection_permutation(n_sc, axis)
    for src, dst in perm.items():
        assert perm[dst] == src


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reflection_support_is_closed_bijection(axis):
    n_sc = 4
    perm = build_reflection_permutation(n_sc, axis)
    assert set(perm.keys()) == set(perm.values())
    assert len(set(perm.values())) == len(perm)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reflection_preserves_sublattice(axis):
    n_sc = 4
    cation_offset = n_sc**3
    perm = build_reflection_permutation(n_sc, axis)
    for src, dst in perm.items():
        s_src = index_to_anion(n_sc, src - cation_offset)[0]
        s_dst = index_to_anion(n_sc, dst - cation_offset)[0]
        assert s_src == s_dst


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reflection_omits_fixed_points(axis):
    perm = build_reflection_permutation(4, axis)
    assert all(src != dst for src, dst in perm.items())


def test_reflection_geometry_half_integer_sublattice():
    # axis=0, s=0 carries the half-integer x offset: i -> n_sc-1-i.
    n_sc = 4
    cation_offset = n_sc**3
    perm = build_reflection_permutation(n_sc, 0)
    src = cation_offset + anion_index(n_sc, 0, 1, 2, 3)
    dst = cation_offset + anion_index(n_sc, 0, n_sc - 1 - 1, 2, 3)
    assert perm[src] == dst


def test_reflection_geometry_integer_sublattice():
    # axis=0, s=1 has integer x: i -> (n_sc-i) % n_sc.
    n_sc = 4
    cation_offset = n_sc**3
    perm = build_reflection_permutation(n_sc, 0)
    src = cation_offset + anion_index(n_sc, 1, 1, 2, 3)
    dst = cation_offset + anion_index(n_sc, 1, (n_sc - 1) % n_sc, 2, 3)
    assert perm[src] == dst


def test_build_reflection_permutation_rejects_bad_axis():
    with pytest.raises(ValueError, match="axis must be 0, 1, or 2"):
        build_reflection_permutation(4, 3)


def test_cell_reflect_name_and_operation_count():
    move = CellReflect(n_sc=4)
    assert move.name == "cell_reflect"
    assert move.n_operations == 3


def test_cell_reflect_operations_match_builder():
    n_sc = 4
    move = CellReflect(n_sc=n_sc)
    expected = [build_reflection_permutation(n_sc, axis) for axis in (0, 1, 2)]
    assert move.operations == expected


@pytest.mark.slow
def test_reflecting_groundstate_is_isoenergetic_enantiomer():
    from pathlib import Path

    import numpy as np
    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator

    from nbo2f_analysis.ce_tools import build_tiled_groundstate_atoms

    n_sc = 3
    ce_path = (
        Path(__file__).resolve().parent.parent
        / "nbo2f_analysis" / "data" / "ces" / "paircut9_5_5_ardr_n96.ce"
    )
    ce = ClusterExpansion.read(str(ce_path))
    gs = build_tiled_groundstate_atoms(n_sc=n_sc)
    occ = gs.numbers.copy()

    calc = ClusterExpansionCalculator(gs.copy(), ce)
    e_before = float(calc.calculate_total(occupations=occ))

    # Apply the x-reflection the way SitePermutation does: new[i] = old[j].
    perm = build_reflection_permutation(n_sc, axis=0)
    reflected = occ.copy()
    for src, dst in perm.items():
        reflected[src] = occ[dst]

    # The chiral GS is genuinely altered (it lands in the enantiomer)...
    assert not np.array_equal(reflected, occ)
    # ...but the cluster-expansion energy is unchanged.
    e_after = float(calc.calculate_total(occupations=reflected))
    assert e_after == pytest.approx(e_before, abs=1e-6)
