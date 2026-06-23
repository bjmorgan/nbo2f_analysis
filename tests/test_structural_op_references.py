"""Tests for the structural OP reference values."""
from __future__ import annotations

import math
from fractions import Fraction

import pytest

from nbo2f_analysis import structural_op_references as sor


def test_random_local_limits_at_one_third():
    # NbO4F2 = C(6,2) f^2 (1-f)^4 = 80/243; cis = 12 f^2 (1-f)^4 = 64/243;
    # collinear_ff = f^2 = 1/9, all exact at f = 1/3.
    assert sor.random_local_limits(Fraction(1, 3)) == {
        "nbo4f2_frac": Fraction(80, 243),
        "cis_frac": Fraction(64, 243),
        "collinear_ff": Fraction(1, 9),
    }


def test_random_local_limits_general_f():
    # f = 1/2 pins the f-general forms, not just the physical case:
    # nbo4f2 = 15/64, cis = 12/64 = 3/16, collinear = 1/4.
    assert sor.random_local_limits(Fraction(1, 2)) == {
        "nbo4f2_frac": Fraction(15, 64),
        "cis_frac": Fraction(3, 16),
        "collinear_ff": Fraction(1, 4),
    }


def test_oof_amp_random_decays_as_inverse_sqrt_size():
    # The floor decays as N^{-1/2}, so oof_amp_random(N) * sqrt(N) is
    # size-independent -- an independent check of the scaling, distinct from
    # the MC-anchored value test below.
    assert sor.oof_amp_random(4) * math.sqrt(4) == pytest.approx(
        sor.oof_amp_random(9) * math.sqrt(9)
    )


def test_oof_amp_random_known_values():
    # The MC-verified floor: 0.171 at L=6, 0.121 at L=12.
    assert sor.oof_amp_random(6) == pytest.approx(0.1706, abs=1e-4)
    assert sor.oof_amp_random(12) == pytest.approx(0.1206, abs=1e-4)


@pytest.mark.parametrize("n_sc", [3, 6])
def test_ground_state_anchors_exact(n_sc):
    # L-independent P3_121 anchors, verified against the production observer.
    # chi_11 = 4/9 (not 1): an orbit and its enantiomer share 5/9 of sites.
    gs = sor.ground_state_reference(n_sc)
    assert gs["chi_11"] == pytest.approx(4 / 9, abs=1e-9)
    assert gs["icoh_global"] == pytest.approx(1.0, abs=1e-9)
    assert gs["oof_amp"] == pytest.approx(1 / 3, abs=1e-9)
    assert gs["cis_frac"] == pytest.approx(1.0, abs=1e-9)
    assert gs["nbo4f2_frac"] == pytest.approx(1.0, abs=1e-9)
    assert gs["collinear_ff"] == pytest.approx(0.0, abs=1e-9)
