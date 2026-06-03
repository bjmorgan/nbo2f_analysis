"""Tests for nbo2f_analysis.rewl.search."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nbo2f_analysis.ce_tools import build_tiled_groundstate_atoms
from nbo2f_analysis.rewl.search import (
    find_all_window_configs,
    SearchParams,
    _geometric_schedule,
    _windows_containing,
    _record_config,
    _inject_ground_state,
)


def test_geometric_schedule_endpoints_and_monotone():
    sched = _geometric_schedule(2000.0, 100.0, 4)
    assert len(sched) == 4
    assert sched[0] == 2000.0
    assert abs(sched[-1] - 100.0) < 1e-9
    assert all(sched[i] > sched[i + 1] for i in range(len(sched) - 1))


def test_geometric_schedule_single_level():
    assert _geometric_schedule(2000.0, 100.0, 1) == [2000.0]


def test_windows_containing():
    windows = [(-10.0, -8.0), (-9.0, -7.0)]
    assert _windows_containing(-8.5, windows) == [0, 1]
    assert _windows_containing(-7.5, windows) == [1]
    assert _windows_containing(-11.0, windows) == []


def test_record_config_dedups_and_caps_per_window():
    windows = [(-10.0, -8.0), (-9.0, -7.0)]
    counts = [2, 1]
    found = [[], []]
    seen = [set(), set()]
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([4, 5, 6], dtype=np.int64)
    # -8.5 is in both windows; fills window 1 (cap 1) and one slot of window 0.
    assert _record_config(found, seen, windows, counts, -8.5, a) is False
    # Identical vector is deduped everywhere.
    assert _record_config(found, seen, windows, counts, -8.5, a) is False
    # A distinct vector fills window 0's second slot; window 1 already capped.
    assert _record_config(found, seen, windows, counts, -8.5, b) is True
    assert len(found[0]) == 2
    assert len(found[1]) == 1


def test_inject_ground_state_uses_only_lowest_containing_window():
    windows = [(-10.0, -8.0), (-9.0, -7.0)]
    counts = [2, 2]
    found = [[], []]
    seen = [set(), set()]
    gs = np.array([7, 8, 9], dtype=np.int64)
    _inject_ground_state(found, seen, windows, counts, -8.5, gs)
    assert len(found[0]) == 1
    assert len(found[1]) == 0


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
