"""Tests for nbo2f_analysis.rewl.search (adapter over seed_window_configs)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from nbo2f_analysis.ce_tools import build_tiled_groundstate_atoms
from nbo2f_analysis.rewl import search
from nbo2f_analysis.rewl.config import MovesCfg, MoveSpec


def _ce_path() -> str:
    return str(
        Path(__file__).resolve().parent.parent
        / "nbo2f_analysis" / "data" / "ces" / "paircut9_5_5_ardr_n96.ce"
    )


def test_find_all_window_configs_forwards_contracts(monkeypatch):
    from icet import ClusterExpansion
    from mchammer_pt import SeedSearchParams

    n_sc = 3
    ce = ClusterExpansion.read(_ce_path())
    windows = [(-1.0, 1.0), (0.5, 2.0)]
    counts = [2, 1]
    moves_cfg = MovesCfg(list=(MoveSpec(type="pair_swap", weight=1.0),))
    params = SeedSearchParams(
        window_search_penalty=3.0,
        walk_sweeps=7,
        max_walks_per_window=5,
        n_workers=2,
    )

    captured: dict = {}
    sentinel = [[object(), object()], [object()]]

    def spy(
        cluster_expansion,
        moves,
        win,
        cnt,
        energy_spacing,
        bottom_anchor,
        random_fill,
        *,
        random_seed,
        params,
        anchors=None,
    ):
        captured.update(
            cluster_expansion=cluster_expansion,
            moves=moves,
            windows=win,
            counts=cnt,
            energy_spacing=energy_spacing,
            bottom_anchor=bottom_anchor,
            random_fill=random_fill,
            random_seed=random_seed,
            params=params,
            anchors=anchors,
        )
        return sentinel

    monkeypatch.setattr(search, "seed_window_configs", spy)

    result = search.find_all_window_configs(
        ce=ce,
        n_sc=n_sc,
        windows=windows,
        counts=counts,
        energy_spacing=0.5,
        moves_cfg=moves_cfg,
        params=params,
        seed=99,
    )

    assert result is sentinel
    assert captured["cluster_expansion"] is ce
    assert captured["windows"] == windows
    assert captured["counts"] == counts
    assert captured["energy_spacing"] == 0.5
    assert captured["random_seed"] == 99
    assert captured["params"] is params
    assert len(captured["moves"]) == 1

    gs = build_tiled_groundstate_atoms(n_sc=n_sc)
    anchor = captured["bottom_anchor"]
    assert isinstance(anchor, Atoms)
    assert np.allclose(anchor.get_positions(), gs.get_positions())
    assert anchor.get_chemical_symbols() == gs.get_chemical_symbols()

    # random_fill yields a stable-ordered Atoms on the GS lattice (same
    # positions index-for-index), correct F count, differing only in species.
    fill = captured["random_fill"](0)
    assert isinstance(fill, Atoms)
    assert np.allclose(fill.get_positions(), gs.get_positions())
    symbols = fill.get_chemical_symbols()
    assert symbols.count("F") == n_sc**3
    assert symbols.count("Nb") == gs.get_chemical_symbols().count("Nb")


@pytest.mark.slow
def test_find_all_window_configs_fills_windows_end_to_end():
    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator
    from mchammer_pt import SeedSearchParams

    n_sc = 3
    ce = ClusterExpansion.read(_ce_path())
    gs = build_tiled_groundstate_atoms(n_sc=n_sc)
    calc = ClusterExpansionCalculator(gs.copy(), ce)
    e_gs = float(calc.calculate_total(occupations=gs.numbers))
    # A narrow low window (K>1, seeded from the GS) and a higher window.
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
    params = SeedSearchParams(
        window_search_penalty=2.0,
        walk_sweeps=40,
        max_walks_per_window=40,
        n_workers=2,
    )
    per_window = search.find_all_window_configs(
        ce=ce,
        n_sc=n_sc,
        windows=windows,
        counts=counts,
        energy_spacing=0.5,
        moves_cfg=moves_cfg,
        params=params,
        seed=0,
    )
    assert [len(w) for w in per_window] == counts
    for atoms_list, (lo, hi) in zip(per_window, windows):
        keys = set()
        for atoms in atoms_list:
            e = float(calc.calculate_total(occupations=atoms.numbers))
            assert lo <= e <= hi, f"energy {e} outside [{lo}, {hi}]"
            keys.add(atoms.numbers.tobytes())
        assert len(keys) == len(atoms_list), "configs within a window not distinct"
