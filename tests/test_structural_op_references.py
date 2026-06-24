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


@pytest.mark.parametrize("n_sc", [3, 6, 9])
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
    assert gs["chirality"] == pytest.approx(1 / 4, abs=1e-9)


def test_monte_carlo_requires_two_samples():
    # A standard error of the mean is undefined below two samples.
    with pytest.raises(ValueError, match="n_samples"):
        sor.monte_carlo_random_reference(3, 1, seed=0)


def test_monte_carlo_converges_to_local_limits():
    # At L = 6 the MC means of the local-coordination OPs sit within a few
    # SEM of their exact independent-site limits. Restrict ops to the local
    # ones so the expensive chi_11 similarity loop is not run.
    limits = sor.random_local_limits()
    ref = sor.monte_carlo_random_reference(
        6, 200, seed=0, ops=("cis_frac", "nbo4f2_frac", "collinear_ff"),
    )
    for op, exact in limits.items():
        mean, sem = ref[op]
        assert abs(mean - float(exact)) <= 5 * sem, (op, mean, float(exact), sem)


@pytest.mark.slow
def test_monte_carlo_oof_matches_rayleigh_floor():
    # The Rayleigh floor is the leading-order (large-L, independent-site)
    # prediction. At L = 6 the fixed-composition MC mean sits a few
    # thousandths below it (canonical sampling plus the O(1/L) correction),
    # and that systematic gap does not shrink with sample count, so compare
    # to the floor within an absolute tolerance that brackets it. The bound
    # still excludes a wrong formula -- a factor-of-2 error would land near
    # 0.085 or 0.34, far outside it.
    ref = sor.monte_carlo_random_reference(6, 400, seed=0, ops=("oof_amp",))
    mean, _ = ref["oof_amp"]
    assert mean == pytest.approx(sor.oof_amp_random(6), abs=0.01)


@pytest.mark.slow
def test_random_oof_and_icoh_decrease_with_size():
    # Both vanishing OPs decay towards 0 with size. Asserting a monotone
    # decrease (not a fitted exponent) is robust to sampling noise; the L
    # gaps are far larger than the SEM at this sample count.
    oof, icoh = [], []
    for n_sc in (3, 6, 9):
        ref = sor.monte_carlo_random_reference(
            n_sc, 100, seed=0, ops=("oof_amp", "icoh_global"),
        )
        oof.append(ref["oof_amp"][0])
        icoh.append(ref["icoh_global"][0])
    assert oof[0] > oof[1] > oof[2]
    assert icoh[0] > icoh[1] > icoh[2]


@pytest.mark.slow
def test_chi_11_mean_near_zero_and_rms_decreases():
    # <chi_11> ~ 0 by Z2 symmetry, and its fluctuation scale shrinks with
    # size. At fixed n_samples the returned SEM is proportional to the RMS
    # (the mean is ~ 0), so a decreasing SEM witnesses the decreasing RMS.
    ref3 = sor.monte_carlo_random_reference(3, 150, seed=0, ops=("chi_11",))
    ref6 = sor.monte_carlo_random_reference(6, 150, seed=0, ops=("chi_11",))
    mean3, sem3 = ref3["chi_11"]
    mean6, sem6 = ref6["chi_11"]
    assert abs(mean3) <= 4 * sem3
    assert abs(mean6) <= 4 * sem6
    assert sem3 > sem6


def test_reference_table_smoke():
    rows = sor.reference_table((3,), 8, seed=0)
    # One row per referenced OP, in REFERENCE_OPS order.
    assert [r["op"] for r in rows] == list(sor.REFERENCE_OPS)
    assert set(rows[0]) == {
        "op", "n_sc", "analytic_random", "mc_mean", "mc_sem", "ground_state",
    }
    by_op = {r["op"]: r for r in rows}
    # analytic_random: exact local limit, Rayleigh floor, and the 0 limit.
    assert by_op["collinear_ff"]["analytic_random"] == pytest.approx(1 / 9)
    assert by_op["oof_amp"]["analytic_random"] == pytest.approx(
        sor.oof_amp_random(3)
    )
    assert by_op["chi_11"]["analytic_random"] == 0.0
    # ground_state column carries the exact anchors.
    assert by_op["icoh_global"]["ground_state"] == pytest.approx(1.0, abs=1e-9)
    assert by_op["collinear_ff"]["ground_state"] == pytest.approx(0.0, abs=1e-9)


def test_provenance_lines_record_versions():
    lines = sor.provenance_lines((6,), 10, 0)
    text = "\n".join(lines)
    assert all(line.startswith("#") for line in lines)
    assert importlib.metadata.version("nbo2f-analysis") in text
    assert importlib.metadata.version("chainorder") in text
    assert "n_samples=10" in text
    assert "seed=0" in text


def test_write_csv_has_provenance_header_and_rows():
    rows = [
        {"op": "oof_amp", "n_sc": 3, "analytic_random": 0.24,
         "mc_mean": 0.2, "mc_sem": 0.01, "ground_state": 0.3333},
    ]
    buf = io.StringIO()
    sor.write_csv(rows, buf, provenance=["# nbo2f-analysis 0.11.0"])
    out = buf.getvalue()
    assert out.startswith("# nbo2f-analysis 0.11.0\n")
    body = [ln for ln in out.splitlines() if not ln.startswith("#")]
    assert body[0] == "op,n_sc,analytic_random,mc_mean,mc_sem,ground_state"
    assert body[1].startswith("oof_amp,3,")


def test_monte_carlo_raises_on_non_finite_op(monkeypatch):
    # A fixed-composition config can leave a whole chain direction without
    # period-3 structure, making icoh_global undefined (NaN). The reference
    # must surface that loudly, not average a NaN into the citable table.
    class _NaNObserver:
        def get_observable(self, atoms):
            return {"icoh_global": float("nan")}

    monkeypatch.setattr(
        sor, "build_chain_order_observer", lambda *a, **k: _NaNObserver()
    )
    with pytest.raises(ValueError, match="non-finite"):
        sor.monte_carlo_random_reference(3, 2, seed=0, ops=("icoh_global",))


def test_monte_carlo_accepts_two_samples():
    # Two samples is the first count with a defined SEM; a <=2 vs <2
    # off-by-one in the guard would wrongly reject it.
    mean, sem = sor.monte_carlo_random_reference(
        3, 2, seed=0, ops=("collinear_ff",)
    )["collinear_ff"]
    assert math.isfinite(mean)
    assert math.isfinite(sem)


def test_analytic_random_rejects_unknown_op():
    # _analytic_random routes every REFERENCE_OP explicitly; an unrouted op
    # raises rather than silently taking the wrong default.
    with pytest.raises(KeyError):
        sor._analytic_random("not_an_op", 6, sor.random_local_limits())


def test_reference_table_orders_by_size_and_repeats_ground_state():
    rows = sor.reference_table((3, 6), 2, seed=0)
    n_ops = len(sor.REFERENCE_OPS)
    # Size-major order: all of size 3 (in REFERENCE_OPS order), then size 6.
    assert [r["n_sc"] for r in rows] == [3] * n_ops + [6] * n_ops
    assert [r["op"] for r in rows[:n_ops]] == list(sor.REFERENCE_OPS)
    # The ground-state column is L-independent, so it repeats across blocks.
    gs3 = {r["op"]: r["ground_state"] for r in rows[:n_ops]}
    gs6 = {r["op"]: r["ground_state"] for r in rows[n_ops:]}
    for op in sor.REFERENCE_OPS:
        assert gs6[op] == pytest.approx(gs3[op], abs=1e-9)


def test_analytic_random_chirality_is_zero():
    # The projected chirality is a signed pseudoscalar: its independent-site
    # random limit is exactly 0, with the finite-size floor in the MC column.
    assert sor._analytic_random("chirality", 6, sor.random_local_limits()) == 0.0


def test_reference_table_includes_chirality():
    rows = sor.reference_table((3,), 8, seed=0)
    by_op = {r["op"]: r for r in rows}
    assert "chirality" in by_op
    assert by_op["chirality"]["analytic_random"] == 0.0
    assert by_op["chirality"]["ground_state"] == pytest.approx(1 / 4, abs=1e-9)


def test_emit_csv_wires_table_and_provenance():
    # Smoke the main() wiring on a tiny grid: the defaults route through
    # reference_table and write_csv, and the run parameters are recorded
    # faithfully (a swapped seed/n_samples would show here).
    buf = io.StringIO()
    sor._emit_csv((3,), 2, 0, buf)
    out = buf.getvalue()
    assert out.startswith("# nbo2f-analysis")
    assert "n_samples=2" in out
    assert "seed=0" in out
    lines = out.splitlines()
    assert "op,n_sc,analytic_random,mc_mean,mc_sem,ground_state" in lines
    assert any(line.startswith("oof_amp,3,") for line in lines)
