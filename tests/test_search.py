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
    _lingering_backstop,
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


def test_inject_ground_state_is_noop_when_outside_all_windows():
    windows = [(-10.0, -8.0), (-9.0, -7.0)]
    counts = [2, 2]
    found = [[], []]
    seen = [set(), set()]
    gs = np.array([7, 8, 9], dtype=np.int64)
    # e_gs below every window: best-effort seeding records nothing.
    _inject_ground_state(found, seen, windows, counts, -50.0, gs)
    assert found == [[], []]
    assert seen == [set(), set()]


def _ce_path() -> str:
    return str(
        Path(__file__).resolve().parent.parent
        / "nbo2f_analysis" / "data" / "ces" / "paircut9_5_5_ardr_n96.ce"
    )


@pytest.mark.slow
def test_find_all_window_configs_returns_distinct_in_window_atoms():
    n_sc = 3
    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator

    from nbo2f_analysis.rewl.config import MovesCfg, MoveSpec

    ce = ClusterExpansion.read(_ce_path())
    gs = build_tiled_groundstate_atoms(n_sc=n_sc)
    calc = ClusterExpansionCalculator(gs.copy(), ce)
    e_gs = float(calc.calculate_total(occupations=gs.numbers))
    windows = [
        (e_gs - 0.5, e_gs + 1.5),
        (e_gs + 0.5, e_gs + 3.0),
    ]
    counts = [2, 1]
    moves_cfg = MovesCfg(
        list=(
            MoveSpec(type="pair_swap", weight=0.2),
            MoveSpec(type="chain_swap", weight=0.5),
            MoveSpec(type="row_reflect", weight=0.5),
        )
    )
    params = SearchParams(
        temperature_high=3000.0,
        temperature_low=100.0,
        n_temperature_levels=6,
        sweeps_per_level=4,
        harvest_interval_sweeps=1,
        max_anneals_per_worker=40,
        backstop_temperature=200.0,
        backstop_sweeps=20,
    )
    per_window = find_all_window_configs(
        ce_path=_ce_path(),
        n_sc=n_sc,
        windows=windows,
        counts=counts,
        moves_cfg=moves_cfg,
        n_workers=2,
        params=params,
    )
    assert [len(w) for w in per_window] == counts
    for atoms_list, (lo, hi) in zip(per_window, windows):
        keys = set()
        for atoms in atoms_list:
            e = float(calc.calculate_total(occupations=atoms.numbers))
            assert lo <= e <= hi, f"energy {e} outside [{lo}, {hi}]"
            keys.add(atoms.numbers.tobytes())
        assert len(keys) == len(atoms_list), "configs within a window not distinct"


@pytest.mark.slow
def test_lingering_backstop_tops_up_a_short_window():
    n_sc = 3
    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator

    from nbo2f_analysis.rewl.config import MovesCfg, MoveSpec
    from nbo2f_analysis.rewl.nbo2f import (
        build_moves,
        resolve_anion_sublattice_index,
    )

    ce = ClusterExpansion.read(_ce_path())
    gs = build_tiled_groundstate_atoms(n_sc=n_sc)
    calc = ClusterExpansionCalculator(gs.copy(), ce)
    e_gs = float(calc.calculate_total(occupations=gs.numbers))
    # One wide window around the GS so the lingering walk stays in-band.
    windows = [(e_gs - 1.0, e_gs + 5.0)]
    counts = [3]
    # Seed the window with the GS as its single anchor.
    found = [[gs.numbers.copy()]]
    seen = [{gs.numbers.tobytes()}]
    moves_cfg = MovesCfg(
        list=(
            MoveSpec(type="pair_swap", weight=0.3),
            MoveSpec(type="chain_swap", weight=0.5),
        )
    )
    sub = resolve_anion_sublattice_index(calc)
    moves = build_moves(n_sc, sub, moves_cfg)
    params = SearchParams(
        temperature_high=3000.0,
        temperature_low=100.0,
        n_temperature_levels=4,
        sweeps_per_level=2,
        harvest_interval_sweeps=1,
        max_anneals_per_worker=1,
        backstop_temperature=400.0,
        backstop_sweeps=200,
    )
    _lingering_backstop(found, seen, windows, counts, gs, calc, moves, params)
    assert len(found[0]) == 3
    keys = {n.tobytes() for n in found[0]}
    assert len(keys) == 3
    for numbers in found[0]:
        e = float(calc.calculate_total(occupations=numbers))
        assert windows[0][0] <= e <= windows[0][1]


@pytest.mark.slow
def test_find_all_window_configs_raises_when_window_unfillable():
    n_sc = 3
    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator

    from nbo2f_analysis.rewl.config import MovesCfg, MoveSpec

    ce = ClusterExpansion.read(_ce_path())
    gs = build_tiled_groundstate_atoms(n_sc=n_sc)
    calc = ClusterExpansionCalculator(gs.copy(), ce)
    e_gs = float(calc.calculate_total(occupations=gs.numbers))
    # An empty band far below any reachable energy can never be filled.
    windows = [(e_gs - 100.0, e_gs - 50.0)]
    counts = [1]
    moves_cfg = MovesCfg(list=(MoveSpec(type="pair_swap", weight=1.0),))
    params = SearchParams(
        temperature_high=2000.0,
        temperature_low=100.0,
        n_temperature_levels=2,
        sweeps_per_level=1,
        harvest_interval_sweeps=1,
        max_anneals_per_worker=1,
        backstop_temperature=200.0,
        backstop_sweeps=0,  # disable the backstop: isolate the hard-error path
    )
    with pytest.raises(RuntimeError, match="Could not fill windows"):
        find_all_window_configs(
            ce_path=_ce_path(),
            n_sc=n_sc,
            windows=windows,
            counts=counts,
            moves_cfg=moves_cfg,
            n_workers=1,
            params=params,
        )
