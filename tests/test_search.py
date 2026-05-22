"""Tests for nbo2f_analysis.rewl.search."""
from __future__ import annotations

from pathlib import Path

import pytest

from nbo2f_analysis.ce_tools import build_tiled_groundstate_atoms
from nbo2f_analysis.rewl.search import find_all_window_configs, SearchParams


def _ce_path() -> str:
    return str(
        Path(__file__).resolve().parent.parent
        / "nbo2f_analysis" / "data" / "ces" / "paircut9_5_5_ardr_n96.ce"
    )


@pytest.mark.slow
def test_find_all_window_configs_returns_in_window_atoms():
    n_sc = 3
    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator

    ce = ClusterExpansion.read(_ce_path())
    gs = build_tiled_groundstate_atoms(n_sc=n_sc)
    calc = ClusterExpansionCalculator(gs.copy(), ce)
    e_gs = float(calc.calculate_total(occupations=gs.numbers))
    windows = [
        (e_gs - 0.5, e_gs + 1.5),
        (e_gs + 0.5, e_gs + 3.0),
    ]
    params = SearchParams(
        max_swaps=(1, 2, 3, 5),
        attempts_per_swap_count=50,
        random_attempts=100,
    )
    atoms_list = find_all_window_configs(
        ce_path=_ce_path(),
        n_sc=n_sc,
        windows=windows,
        n_workers=2,
        params=params,
    )
    assert len(atoms_list) == 2
    for atoms, (lo, hi) in zip(atoms_list, windows):
        e = float(calc.calculate_total(occupations=atoms.numbers))
        assert lo <= e <= hi, f"energy {e} outside [{lo}, {hi}]"
