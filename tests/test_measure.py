"""Tests for nbo2f_analysis.rewl.measure wiring and cycle accounting."""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from nbo2f_analysis.rewl import measure as measure_mod
from nbo2f_analysis.rewl.config import load_yaml

_BASE_CFG = """
random_seed: 0
system: {n_sc: 3, ce: paircut9_5_5_ardr_n96}
windows:
  energy_spacing: 0.1
  list:
    - [-10.0, -9.0, 1]
    - [-9.5, -8.5, 1]
wl:
  flatness_limit: 0.8
  fill_factor_limit: 1.0e-12
  schedule: "1_over_t"
  flatness_mode: "pooled"
  merge_cadence: "at_halve"
  n_trials_per_walker: 1000
  block_size_sweeps: 10
  trajectory_write_interval_sweeps: 0
moves:
  - {type: pair_swap, weight: 0.1}
  - {type: row_shift, weight: 0.2}
config_search: {n_workers: 1, window_search_penalty: 2.0, walk_sweeps: 10, max_walks_per_window: 4}
checkpoint: {filename: c.h5, interval_cycles: 0}
"""

_MEAS_BLOCK = """
measurement:
  checkpoint_filename: rewl_measure.h5
  observer_interval: 81
  n_measure_cycles: 100
  checkpoint_interval_cycles: 10
  ops: [chi_11, oof_amp]
"""


def _cfg(tmp_path: Path, with_measurement: bool = True):
    p = tmp_path / "cfg.yaml"
    p.write_text(_BASE_CFG + (_MEAS_BLOCK if with_measurement else ""))
    return load_yaml(p)


class _SpyPT:
    """Stands in for the frozen-g orchestrator: records calls."""

    def __init__(self):
        self.pool = object()
        self.run_calls: list[int] = []
        self.save_calls = 0
        self.recorded: list[object] = []

    def record_observable(self, observer):
        self.recorded.append(observer)

    def attach_cycle_callback(self, cb):
        pass

    def attach_checkpoint_writer(self, *a, **k):
        pass

    def run(self, n_cycles):
        self.run_calls.append(n_cycles)

    def save_checkpoint(self, path):
        self.save_calls += 1


def _container(last_step):
    return types.SimpleNamespace(_last_state={"last_step": last_step})


def _install(monkeypatch, tmp_path, *, dos_step, meas_step=None, block_size=10):
    """Stub everything except the real cycle-accounting logic.

    With block_size=10 the config's n_measure_cycles=100 is the target.
    dos_step sets the DOS baseline (dos_step // block_size); meas_step,
    when given, creates the measurement checkpoint and sets its total
    walker step (so meas_done = meas_step // block_size - baseline).
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c.h5").write_bytes(b"")            # DOS checkpoint exists
    if meas_step is not None:
        (tmp_path / "rewl_measure.h5").write_bytes(b"")  # measurement ckpt exists

    spy = _SpyPT()

    def fake_read_hdf5(path, *a, **k):
        if str(path).endswith("rewl_measure.h5"):
            return (None, [_container(meas_step)], {"block_size": block_size})
        return (None, [_container(dos_step)], {"block_size": block_size})

    monkeypatch.setattr(
        measure_mod.ClusterExpansion, "read", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        measure_mod, "build_moves_and_kwargs",
        lambda cfg, ce: (None, None, None, {}),
    )
    monkeypatch.setattr(measure_mod, "read_hdf5", fake_read_hdf5)
    monkeypatch.setattr(
        measure_mod.WangLandauParallelTempering,
        "measure_from_checkpoint_process_pool",
        lambda *a, **k: spy,
    )
    monkeypatch.setattr(
        measure_mod, "build_chain_order_observer",
        lambda *a, **k: types.SimpleNamespace(args=a, kwargs=k),
    )
    monkeypatch.setattr(
        measure_mod, "WangLandauProgressPrinter", lambda *a, **k: object()
    )
    return spy


def test_first_run_loads_dos_and_runs_full_budget(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    spy = _install(monkeypatch, tmp_path, dos_step=500)  # baseline 50, no meas ckpt
    measure_mod.measure(cfg)
    assert spy.run_calls == [100]
    assert spy.save_calls == 1
    assert len(spy.recorded) == 1


def test_resume_subtracts_dos_baseline(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    # baseline 50; meas total 80 -> meas_done 30 -> 70 remaining of 100.
    spy = _install(monkeypatch, tmp_path, dos_step=500, meas_step=800)
    measure_mod.measure(cfg)
    assert spy.run_calls == [70]


def test_no_op_when_budget_met(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    # baseline 50; meas total 150 -> meas_done 100 == target -> 0 remaining.
    spy = _install(monkeypatch, tmp_path, dos_step=500, meas_step=1500)
    measure_mod.measure(cfg)
    assert spy.run_calls == []
    assert spy.save_calls == 0


def test_extra_cycles_overrides_auto_count(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    spy = _install(monkeypatch, tmp_path, dos_step=500)
    measure_mod.measure(cfg, extra_cycles=7)
    assert spy.run_calls == [7]


def test_observer_built_with_configured_interval_and_ops(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    captured: dict = {}
    _install(monkeypatch, tmp_path, dos_step=500)

    def fake_build(n_sc, interval, ops, **k):
        captured.update(n_sc=n_sc, interval=interval, ops=ops)
        return object()

    monkeypatch.setattr(measure_mod, "build_chain_order_observer", fake_build)
    measure_mod.measure(cfg)
    assert captured == {"n_sc": 3, "interval": 81, "ops": ("chi_11", "oof_amp")}


def test_missing_dos_checkpoint_raises(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.chdir(tmp_path)  # no c.h5 created
    with pytest.raises(RuntimeError, match="No converged DOS checkpoint"):
        measure_mod.measure(cfg)


def test_missing_measurement_section_raises(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, with_measurement=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="measurement"):
        measure_mod.measure(cfg)


def test_status_reports_counts(tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    _install(monkeypatch, tmp_path, dos_step=500, meas_step=800)  # done 30
    measure_mod.measure(cfg)
    assert (
        "Measurement cycles done=30, target=100, running 70 more."
        in capsys.readouterr().out
    )
