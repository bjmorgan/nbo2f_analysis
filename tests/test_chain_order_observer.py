"""Tests for the lifted NbO2F chain-ordering observer."""
from __future__ import annotations

import numpy as np
import pytest

from nbo2f_analysis.ce_tools import (
    atoms_from_f_mask_stable,
    build_tiled_groundstate_atoms,
    nb_anion_neighbours,
)
from nbo2f_analysis.chain_order_observer import (
    ALLOWED_OPS,
    ChainOrderObserver,
    _apply_spacegroup_op,
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


def _direct_collinear_ff_density(f_mask, n_sc):
    """Independent geometric oracle: mean collinear (trans) F-Nb-F pairs per
    cation, divided by the three axes -- the same per-(cation, axis) density
    the FF-along-chain motif route should yield.
    """
    fn = f_mask[nb_anion_neighbours(n_sc)]  # (n_nb, 6) in [+x,-x,+y,-y,+z,-z]
    pairs = ((fn[:, 0] & fn[:, 1]).astype(int)
             + (fn[:, 2] & fn[:, 3]) + (fn[:, 4] & fn[:, 5]))
    return float(pairs.mean()) / 3.0


def test_collinear_ff_zero_in_ground_state(refs):
    obs = ChainOrderObserver(3, 1, refs, ops=("collinear_ff",))
    out = obs.get_observable(build_tiled_groundstate_atoms(n_sc=3))
    assert out["collinear_ff"] == 0.0


@pytest.mark.parametrize("n_sc", [3, 6])
@pytest.mark.parametrize("seed", [0, 7, 42, 2024])
def test_collinear_ff_matches_direct_geometric_count(refs, n_sc, seed):
    # Equality to an independent geometric oracle on random fixed-composition
    # configs. The observer route (atoms -> SublatticeOccupation -> FF motif)
    # and the oracle (neighbour table -> same-axis pair count) share no code,
    # so agreement cross-validates the FF-along-chain identity. n_sc=6
    # exercises longer chains/wrap than the n_sc=3 base case.
    n_anion = 3 * n_sc**3
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_anion, dtype=bool)
    mask[rng.choice(n_anion, n_sc**3, replace=False)] = True
    obs = ChainOrderObserver(n_sc, 1, refs, ops=("collinear_ff",))
    out = obs.get_observable(atoms_from_f_mask_stable(n_sc, mask))
    assert out["collinear_ff"] == pytest.approx(
        _direct_collinear_ff_density(mask, n_sc), abs=1e-12)


def test_collinear_ff_saturated_all_f(refs):
    # Every length-2 window is FF, so the per-(cation, axis) density is its
    # maximum, 1.0. This pins the chosen normalisation: a per-Nb count would
    # give 3.0 here.
    n_sc = 3
    mask = np.ones(3 * n_sc**3, dtype=bool)
    obs = ChainOrderObserver(n_sc, 1, refs, ops=("collinear_ff",))
    out = obs.get_observable(atoms_from_f_mask_stable(n_sc, mask))
    assert out["collinear_ff"] == 1.0


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
    for label, ref in refs.items():
        assert ref.proper, label
        assert ref.improper, label


def test_apply_spacegroup_op_identity():
    # Identity permutation with all-positive signs returns the input unchanged.
    occ = np.random.default_rng(0).integers(0, 2, size=(3, 3, 3, 3))
    out = _apply_spacegroup_op(occ, perm=(0, 1, 2), signs=(1, 1, 1))
    assert np.array_equal(out, occ)


def test_apply_spacegroup_op_permutes_cells():
    # Any space-group op permutes occupation cells, so the value multiset
    # (and hence the F count) is preserved.
    occ = np.random.default_rng(1).integers(0, 2, size=(3, 3, 3, 3))
    out = _apply_spacegroup_op(occ, perm=(1, 2, 0), signs=(1, -1, 1))
    assert out.shape == occ.shape
    assert sorted(out.ravel().tolist()) == sorted(occ.ravel().tolist())


def test_apply_spacegroup_op_reflection_is_involution():
    # A pure single-axis reflection is its own inverse; applying it twice
    # must recover the input. This exercises the half-integer vs integer
    # sign-dependent negation, where an off-by-one would not self-cancel.
    occ = np.random.default_rng(2).integers(0, 2, size=(3, 3, 3, 3))
    refl = _apply_spacegroup_op(occ, perm=(0, 1, 2), signs=(-1, 1, 1))
    twice = _apply_spacegroup_op(refl, perm=(0, 1, 2), signs=(-1, 1, 1))
    assert np.array_equal(twice, occ)
    assert not np.array_equal(refl, occ)  # the reflection actually moves cells


def test_chi_11_flips_sign_under_reflection(observer):
    # chi_11 is signed (Z2): the enantiomer must give the opposite sign,
    # which the Binder even-moment analysis relies on.
    gs = build_tiled_groundstate_atoms(n_sc=3)
    chi0 = observer.get_observable(gs)["chi_11"]
    mirror = gs.copy()
    scaled = mirror.get_scaled_positions(wrap=True)
    scaled[:, 0] = (-scaled[:, 0]) % 1.0
    mirror.set_scaled_positions(scaled)
    chi_mirror = observer.get_observable(mirror)["chi_11"]
    assert chi0 > 0
    assert chi_mirror == pytest.approx(-chi0, abs=1e-9)


def test_build_chain_order_observer_constructs_working_observer():
    obs = build_chain_order_observer(n_sc=3, interval=5, ops=("oof_amp",))
    assert obs.interval == 5
    out = obs.get_observable(build_tiled_groundstate_atoms(n_sc=3))
    assert out == pytest.approx({"oof_amp": 0.3333333333333333}, abs=1e-9)
