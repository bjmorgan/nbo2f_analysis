"""Tests for nbo2f_analysis.rewl.config."""
from __future__ import annotations

from pathlib import Path

import pytest

from nbo2f_analysis.rewl.config import RewlConfig, load_yaml

DATA = Path(__file__).parent / "data"


def test_load_yaml_parses_minimal_config():
    cfg = load_yaml(DATA / "L9_minimal.yaml")
    assert isinstance(cfg, RewlConfig)
    assert cfg.random_seed == 42
    assert cfg.system.n_sc == 9
    assert cfg.system.ce == "paircut9_5_5_ardr_n96"
    assert cfg.system.ce_path is None
    assert cfg.windows.energy_spacing == 0.1
    assert len(cfg.windows.list) == 3
    assert cfg.windows.list[0].lo == -25474.0
    assert cfg.windows.list[0].hi == -25472.5
    assert cfg.windows.list[0].walkers == 2
    assert cfg.windows.bounds == [
        (-25474.0, -25472.5),
        (-25473.5, -25471.5),
        (-25472.5, -25471.0),
    ]
    assert cfg.windows.walkers_per_window == [2, 2, 1]
    assert cfg.wl.flatness_limit == 0.8
    assert cfg.checkpoint.filename == "rewl_state.h5"


def test_load_yaml_rejects_non_overlapping_windows(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("""
random_seed: 0
system: {n_sc: 3, ce: paircut9_5_5_ardr_n96}
windows:
  energy_spacing: 0.1
  list:
    - [-10.0, -9.0, 1]
    - [-8.5, -7.5, 1]
wl: {flatness_limit: 0.8, fill_factor_limit: 1.0e-12, schedule: "1_over_t",
     flatness_mode: "pooled", merge_cadence: "at_halve",
     n_trials_per_walker: 1000, block_size_sweeps: 10,
     trajectory_write_interval_sweeps: 0}
config_search: {n_workers: 1, max_swaps: [1], attempts_per_swap_count: 1, random_attempts: 1}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="overlap"):
        load_yaml(bad)


def test_load_yaml_rejects_both_ce_and_ce_path(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("""
random_seed: 0
system: {n_sc: 3, ce: paircut9_5_5_ardr_n96, ce_path: /tmp/foo.ce}
windows:
  energy_spacing: 0.1
  list:
    - [-10.0, -9.0, 1]
    - [-9.5, -8.5, 1]
wl: {flatness_limit: 0.8, fill_factor_limit: 1.0e-12, schedule: "1_over_t",
     flatness_mode: "pooled", merge_cadence: "at_halve",
     n_trials_per_walker: 1000, block_size_sweeps: 10,
     trajectory_write_interval_sweeps: 0}
config_search: {n_workers: 1, max_swaps: [1], attempts_per_swap_count: 1, random_attempts: 1}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="ce.*ce_path"):
        load_yaml(bad)


def test_load_yaml_rejects_lo_ge_hi(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("""
random_seed: 0
system: {n_sc: 3, ce: paircut9_5_5_ardr_n96}
windows:
  energy_spacing: 0.1
  list:
    - [-9.0, -10.0, 1]
    - [-9.5, -8.5, 1]
wl: {flatness_limit: 0.8, fill_factor_limit: 1.0e-12, schedule: "1_over_t",
     flatness_mode: "pooled", merge_cadence: "at_halve",
     n_trials_per_walker: 1000, block_size_sweeps: 10,
     trajectory_write_interval_sweeps: 0}
config_search: {n_workers: 1, max_swaps: [1], attempts_per_swap_count: 1, random_attempts: 1}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="lo < hi"):
        load_yaml(bad)
