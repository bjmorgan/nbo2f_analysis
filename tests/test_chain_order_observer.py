"""Tests for the lifted NbO2F chain-ordering observer."""
from __future__ import annotations

import numpy as np
import pytest

from nbo2f_analysis.ce_tools import (
    atoms_from_f_mask_stable,
    build_tiled_groundstate_atoms,
)
from nbo2f_analysis.chain_order_observer import (
    ALLOWED_OPS,
    ChainOrderObserver,
    _generate_orbit_references,
    build_chain_order_observer,
)

_ALL_OPS = (
    "chi_11", "closest_chi", "closest_sim", "icoh_global",
    "icoh_nn", "oof_amp", "cis_frac", "nbo4f2_frac",
)

# Golden values captured from the original PT-script observer
# (data_NbO2F/analysis/pt/run_pt_with_custom_moves.py) on fixed N=3
# structures. closest_orbit (categorical) is deliberately excluded.
_GOLDEN_TILED_GS_N3 = {
    "chi_11": 0.4444444444444444,
    "closest_chi": 0.4444444444444444,
    "closest_sim": 1.0,
    "icoh_global": 0.9999999999999997,
    "icoh_nn": 1.0,
    "oof_amp": 0.3333333333333333,
    "cis_frac": 1.0,
    "nbo4f2_frac": 1.0,
}
_GOLDEN_RANDOM_N3 = {
    "chi_11": 0.0,
    "closest_chi": 0.024691358024691357,
    "closest_sim": 0.7283950617283951,
    "icoh_global": 0.29229929851825565,
    "icoh_nn": 0.33639188948512944,
    "oof_amp": 0.2222222222222222,
    "cis_frac": 0.18518518518518517,
    "nbo4f2_frac": 0.25925925925925924,
}


@pytest.fixture(scope="module")
def refs():
    return _generate_orbit_references(n_sc_orbit=3)


@pytest.fixture(scope="module")
def observer(refs):
    return ChainOrderObserver(n_sc=3, interval=1, orbit_refs=refs, ops=_ALL_OPS)


def _random_stable_atoms_n3(seed: int = 12345):
    n_anion = 3 * 3**3
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_anion, dtype=bool)
    mask[rng.choice(n_anion, size=3**3, replace=False)] = True
    return atoms_from_f_mask_stable(3, mask)


def test_reproduces_golden_on_tiled_gs(observer):
    out = observer.get_observable(build_tiled_groundstate_atoms(n_sc=3))
    assert set(out) == set(_ALL_OPS)
    for k, v in _GOLDEN_TILED_GS_N3.items():
        assert out[k] == pytest.approx(v, abs=1e-9), k


def test_reproduces_golden_on_random_config(observer):
    out = observer.get_observable(_random_stable_atoms_n3())
    for k, v in _GOLDEN_RANDOM_N3.items():
        assert out[k] == pytest.approx(v, abs=1e-9), k


def test_tiled_gs_invariants(observer):
    out = observer.get_observable(build_tiled_groundstate_atoms(n_sc=3))
    assert out["closest_sim"] == pytest.approx(1.0)          # exact match to orbit 11
    assert out["chi_11"] == pytest.approx(out["closest_chi"])  # orbit 11 is closest
    assert out["chi_11"] > 0                                  # one signed enantiomer
    assert out["icoh_global"] == pytest.approx(1.0, abs=1e-9)


def test_ops_selects_subset(refs):
    obs = ChainOrderObserver(3, 1, refs, ops=("oof_amp", "cis_frac"))
    out = obs.get_observable(build_tiled_groundstate_atoms(n_sc=3))
    assert set(out) == {"oof_amp", "cis_frac"}


def test_closest_orbit_not_recordable(refs):
    assert "closest_orbit" not in ALLOWED_OPS
    with pytest.raises(ValueError):
        ChainOrderObserver(3, 1, refs, ops=("closest_orbit",))


def test_empty_ops_rejected(refs):
    with pytest.raises(ValueError):
        ChainOrderObserver(3, 1, refs, ops=())


def test_n_sc_must_be_multiple_of_orbit():
    with pytest.raises(ValueError):
        ChainOrderObserver(n_sc=4, interval=1, orbit_refs={}, ops=("oof_amp",))


def test_generate_orbit_references_from_package_data(refs):
    assert set(refs) == {f"{i:02d}" for i in range(12)}
    for label, (proper, improper) in refs.items():
        assert proper, label
        assert improper, label


def test_build_chain_order_observer_constructs_working_observer():
    obs = build_chain_order_observer(n_sc=3, interval=5, ops=("oof_amp",))
    assert obs.interval == 5
    out = obs.get_observable(build_tiled_groundstate_atoms(n_sc=3))
    assert out == pytest.approx({"oof_amp": 0.3333333333333333}, abs=1e-9)
