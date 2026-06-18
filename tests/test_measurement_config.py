"""Tests for the optional `measurement` config section."""
from __future__ import annotations

from pathlib import Path

import pytest

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

_VALID_MEAS = """
measurement:
  checkpoint_filename: rewl_measure.h5
  observer_interval: 81
  n_measure_cycles: 100
  checkpoint_interval_cycles: 10
  ops: [chi_11, oof_amp]
"""


def _write(tmp_path: Path, measurement_block: str = "") -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(_BASE_CFG + measurement_block)
    return p


def test_measurement_parses(tmp_path):
    cfg = load_yaml(_write(tmp_path, _VALID_MEAS))
    assert cfg.measurement is not None
    m = cfg.measurement
    assert m.checkpoint_filename == "rewl_measure.h5"
    assert m.observer_interval == 81
    assert m.n_measure_cycles == 100
    assert m.checkpoint_interval_cycles == 10
    assert m.ops == ("chi_11", "oof_amp")


def test_measurement_absent_is_none(tmp_path):
    cfg = load_yaml(_write(tmp_path))
    assert cfg.measurement is None


def test_measurement_ops_required(tmp_path):
    block = """
measurement:
  checkpoint_filename: rewl_measure.h5
  observer_interval: 81
  n_measure_cycles: 100
  checkpoint_interval_cycles: 10
"""
    with pytest.raises(ValueError, match="ops"):
        load_yaml(_write(tmp_path, block))


def test_measurement_ops_unknown_rejected(tmp_path):
    block = _VALID_MEAS.replace("[chi_11, oof_amp]", "[chi_11, bogus]")
    with pytest.raises(ValueError, match="ALLOWED_OPS|unknown|not in"):
        load_yaml(_write(tmp_path, block))


def test_measurement_ops_closest_orbit_rejected(tmp_path):
    block = _VALID_MEAS.replace("[chi_11, oof_amp]", "[closest_orbit]")
    with pytest.raises(ValueError):
        load_yaml(_write(tmp_path, block))


def test_measurement_ops_duplicates_rejected(tmp_path):
    block = _VALID_MEAS.replace("[chi_11, oof_amp]", "[oof_amp, oof_amp]")
    with pytest.raises(ValueError, match="duplicate"):
        load_yaml(_write(tmp_path, block))


def test_measurement_checkpoint_must_differ_from_dos(tmp_path):
    block = _VALID_MEAS.replace("rewl_measure.h5", "c.h5")
    with pytest.raises(ValueError, match="differ"):
        load_yaml(_write(tmp_path, block))


def test_measurement_observer_interval_must_be_positive(tmp_path):
    block = _VALID_MEAS.replace("observer_interval: 81", "observer_interval: 0")
    with pytest.raises(ValueError, match="observer_interval"):
        load_yaml(_write(tmp_path, block))


def test_measurement_budget_must_be_positive(tmp_path):
    block = _VALID_MEAS.replace("n_measure_cycles: 100", "n_measure_cycles: 0")
    with pytest.raises(ValueError, match="n_measure_cycles"):
        load_yaml(_write(tmp_path, block))
