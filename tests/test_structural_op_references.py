"""Tests for the structural OP reference values."""
from __future__ import annotations

import importlib.metadata
import io
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


def test_oof_amp_random_matches_formula():
    # Equals 0.5 * sqrt(pi f (1-f) / L) at f = 1/3 for several sizes.
    for n_sc in (6, 9, 12):
        assert sor.oof_amp_random(n_sc) == pytest.approx(
            0.5 * math.sqrt(math.pi * (1 / 3) * (2 / 3) / n_sc)
        )


def test_oof_amp_random_known_values():
    # The MC-verified floor: 0.171 at L=6, 0.121 at L=12.
    assert sor.oof_amp_random(6) == pytest.approx(0.1706, abs=1e-4)
    assert sor.oof_amp_random(12) == pytest.approx(0.1206, abs=1e-4)
