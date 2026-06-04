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
    assert len(cfg.moves.list) == 5
    assert cfg.moves.list[0].type == "pair_swap"
    assert cfg.moves.list[0].weight == 0.1
    assert {m.type for m in cfg.moves.list} == {
        "pair_swap", "row_shift", "motif_shift", "chain_swap", "row_reflect",
    }


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
moves:
  - {type: pair_swap, weight: 0.1}
  - {type: row_shift, weight: 0.2}
config_search: {n_workers: 1, temperature_high: 2000.0, temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, harvest_interval_sweeps: 1, max_anneals_per_worker: 4, backstop_temperature: 150.0, backstop_sweeps: 2}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="overlap"):
        load_yaml(bad)


def test_load_yaml_rejects_single_window(tmp_path):
    # A lone window passes the schema but cannot drive replica-exchange PT,
    # which needs at least two windows to swap between.
    bad = tmp_path / "bad.yaml"
    bad.write_text("""
random_seed: 0
system: {n_sc: 3, ce: paircut9_5_5_ardr_n96}
windows:
  energy_spacing: 0.1
  list:
    - [-10.0, -9.0, 1]
wl: {flatness_limit: 0.8, fill_factor_limit: 1.0e-12, schedule: "1_over_t",
     flatness_mode: "pooled", merge_cadence: "at_halve",
     n_trials_per_walker: 1000, block_size_sweeps: 10,
     trajectory_write_interval_sweeps: 0}
moves:
  - {type: pair_swap, weight: 0.1}
  - {type: row_shift, weight: 0.2}
config_search: {n_workers: 1, temperature_high: 2000.0, temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, harvest_interval_sweeps: 1, max_anneals_per_worker: 4, backstop_temperature: 150.0, backstop_sweeps: 2}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="at least two windows"):
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
moves:
  - {type: pair_swap, weight: 0.1}
  - {type: row_shift, weight: 0.2}
config_search: {n_workers: 1, temperature_high: 2000.0, temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, harvest_interval_sweeps: 1, max_anneals_per_worker: 4, backstop_temperature: 150.0, backstop_sweeps: 2}
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
moves:
  - {type: pair_swap, weight: 0.1}
  - {type: row_shift, weight: 0.2}
config_search: {n_workers: 1, temperature_high: 2000.0, temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, harvest_interval_sweeps: 1, max_anneals_per_worker: 4, backstop_temperature: 150.0, backstop_sweeps: 2}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="lo < hi"):
        load_yaml(bad)


def _cfg_with(tmp_path, config_search_line: str):
    p = tmp_path / "cfg.yaml"
    p.write_text(f"""
random_seed: 0
system: {{n_sc: 3, ce: paircut9_5_5_ardr_n96}}
windows:
  energy_spacing: 0.1
  list:
    - [-10.0, -9.0, 1]
    - [-9.5, -8.5, 1]
wl: {{flatness_limit: 0.8, fill_factor_limit: 1.0e-12, schedule: "1_over_t",
     flatness_mode: "pooled", merge_cadence: "at_halve",
     n_trials_per_walker: 1000, block_size_sweeps: 10,
     trajectory_write_interval_sweeps: 0}}
moves:
  - {{type: pair_swap, weight: 0.1}}
  - {{type: row_shift, weight: 0.2}}
{config_search_line}
checkpoint: {{filename: c.h5, interval_cycles: 0}}
""")
    return p


def test_load_yaml_parses_config_search_annealing_knobs(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 3, temperature_high: 2500.0, "
        "temperature_low: 120.0, n_temperature_levels: 6, sweeps_per_level: 4, "
        "harvest_interval_sweeps: 2, max_anneals_per_worker: 8, "
        "backstop_temperature: 200.0, backstop_sweeps: 5}",
    )
    cfg = load_yaml(p)
    cs = cfg.config_search
    assert cs.n_workers == 3
    assert cs.temperature_high == 2500.0
    assert cs.temperature_low == 120.0
    assert cs.n_temperature_levels == 6
    assert cs.sweeps_per_level == 4
    assert cs.harvest_interval_sweeps == 2
    assert cs.max_anneals_per_worker == 8
    assert cs.backstop_temperature == 200.0
    assert cs.backstop_sweeps == 5


def test_load_yaml_rejects_temperature_high_not_above_low(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 1, temperature_high: 100.0, "
        "temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, "
        "harvest_interval_sweeps: 1, max_anneals_per_worker: 4, "
        "backstop_temperature: 150.0, backstop_sweeps: 2}",
    )
    with pytest.raises(ValueError, match="temperature_high > temperature_low > 0"):
        load_yaml(p)


def test_load_yaml_rejects_zero_temperature_levels(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 1, temperature_high: 2000.0, "
        "temperature_low: 100.0, n_temperature_levels: 0, sweeps_per_level: 2, "
        "harvest_interval_sweeps: 1, max_anneals_per_worker: 4, "
        "backstop_temperature: 150.0, backstop_sweeps: 2}",
    )
    with pytest.raises(ValueError, match="n_temperature_levels must be >= 1"):
        load_yaml(p)


def test_load_yaml_rejects_zero_sweeps_per_level(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 1, temperature_high: 2000.0, "
        "temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 0, "
        "harvest_interval_sweeps: 1, max_anneals_per_worker: 4, "
        "backstop_temperature: 150.0, backstop_sweeps: 2}",
    )
    with pytest.raises(ValueError, match="sweeps_per_level must be >= 1"):
        load_yaml(p)


def test_load_yaml_rejects_zero_harvest_interval(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 1, temperature_high: 2000.0, "
        "temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, "
        "harvest_interval_sweeps: 0, max_anneals_per_worker: 4, "
        "backstop_temperature: 150.0, backstop_sweeps: 2}",
    )
    with pytest.raises(ValueError, match="harvest_interval_sweeps must be >= 1"):
        load_yaml(p)


def test_load_yaml_rejects_zero_max_anneals_per_worker(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 1, temperature_high: 2000.0, "
        "temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, "
        "harvest_interval_sweeps: 1, max_anneals_per_worker: 0, "
        "backstop_temperature: 150.0, backstop_sweeps: 2}",
    )
    with pytest.raises(ValueError, match="max_anneals_per_worker must be >= 1"):
        load_yaml(p)


def test_load_yaml_rejects_zero_backstop_temperature(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 1, temperature_high: 2000.0, "
        "temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, "
        "harvest_interval_sweeps: 1, max_anneals_per_worker: 4, "
        "backstop_temperature: 0.0, backstop_sweeps: 2}",
    )
    with pytest.raises(ValueError, match="backstop_temperature must be > 0"):
        load_yaml(p)


def test_load_yaml_rejects_negative_backstop_sweeps(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 1, temperature_high: 2000.0, "
        "temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, "
        "harvest_interval_sweeps: 1, max_anneals_per_worker: 4, "
        "backstop_temperature: 150.0, backstop_sweeps: -1}",
    )
    with pytest.raises(ValueError, match="backstop_sweeps must be >= 0"):
        load_yaml(p)


def test_load_yaml_rejects_zero_n_workers(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 0, temperature_high: 2000.0, "
        "temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, "
        "harvest_interval_sweeps: 1, max_anneals_per_worker: 4, "
        "backstop_temperature: 150.0, backstop_sweeps: 2}",
    )
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        load_yaml(p)


def test_load_yaml_accepts_zero_backstop_sweeps(tmp_path):
    # 0 disables the backstop and must remain a valid value (the guard is
    # `< 0`, asymmetric with its `>= 1` neighbours by design).
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 1, temperature_high: 2000.0, "
        "temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, "
        "harvest_interval_sweeps: 1, max_anneals_per_worker: 4, "
        "backstop_temperature: 150.0, backstop_sweeps: 0}",
    )
    cfg = load_yaml(p)
    assert cfg.config_search.backstop_sweeps == 0


def test_resolve_ce_path_raises_when_both_none():
    from dataclasses import replace
    from nbo2f_analysis.rewl.config import resolve_ce_path

    cfg = load_yaml(DATA / "L9_minimal.yaml")
    cfg_no_ce = replace(cfg, system=replace(cfg.system, ce=None, ce_path=None))
    with pytest.raises(ValueError, match="neither 'ce' nor 'ce_path'"):
        resolve_ce_path(cfg_no_ce)


def test_load_yaml_rejects_missing_moves_section(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("""
random_seed: 0
system: {n_sc: 3, ce: paircut9_5_5_ardr_n96}
windows:
  energy_spacing: 0.1
  list:
    - [-10.0, -9.0, 1]
    - [-9.5, -8.5, 1]
wl: {flatness_limit: 0.8, fill_factor_limit: 1.0e-12, schedule: "1_over_t",
     flatness_mode: "pooled", merge_cadence: "at_halve",
     n_trials_per_walker: 1000, block_size_sweeps: 10,
     trajectory_write_interval_sweeps: 0}
config_search: {n_workers: 1, temperature_high: 2000.0, temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, harvest_interval_sweeps: 1, max_anneals_per_worker: 4, backstop_temperature: 150.0, backstop_sweeps: 2}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="moves: section is required"):
        load_yaml(bad)


def test_load_yaml_rejects_unrecognised_move_type(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("""
random_seed: 0
system: {n_sc: 3, ce: paircut9_5_5_ardr_n96}
windows:
  energy_spacing: 0.1
  list:
    - [-10.0, -9.0, 1]
    - [-9.5, -8.5, 1]
wl: {flatness_limit: 0.8, fill_factor_limit: 1.0e-12, schedule: "1_over_t",
     flatness_mode: "pooled", merge_cadence: "at_halve",
     n_trials_per_walker: 1000, block_size_sweeps: 10,
     trajectory_write_interval_sweeps: 0}
moves:
  - {type: pair_swap, weight: 0.1}
  - {type: not_a_real_move, weight: 0.5}
config_search: {n_workers: 1, temperature_high: 2000.0, temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, harvest_interval_sweeps: 1, max_anneals_per_worker: 4, backstop_temperature: 150.0, backstop_sweeps: 2}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="not recognised"):
        load_yaml(bad)


def test_load_yaml_rejects_duplicate_move_type(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("""
random_seed: 0
system: {n_sc: 3, ce: paircut9_5_5_ardr_n96}
windows:
  energy_spacing: 0.1
  list:
    - [-10.0, -9.0, 1]
    - [-9.5, -8.5, 1]
wl: {flatness_limit: 0.8, fill_factor_limit: 1.0e-12, schedule: "1_over_t",
     flatness_mode: "pooled", merge_cadence: "at_halve",
     n_trials_per_walker: 1000, block_size_sweeps: 10,
     trajectory_write_interval_sweeps: 0}
moves:
  - {type: pair_swap, weight: 0.1}
  - {type: pair_swap, weight: 0.2}
config_search: {n_workers: 1, temperature_high: 2000.0, temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, harvest_interval_sweeps: 1, max_anneals_per_worker: 4, backstop_temperature: 150.0, backstop_sweeps: 2}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="more than once"):
        load_yaml(bad)


def test_load_yaml_rejects_zero_weight(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("""
random_seed: 0
system: {n_sc: 3, ce: paircut9_5_5_ardr_n96}
windows:
  energy_spacing: 0.1
  list:
    - [-10.0, -9.0, 1]
    - [-9.5, -8.5, 1]
wl: {flatness_limit: 0.8, fill_factor_limit: 1.0e-12, schedule: "1_over_t",
     flatness_mode: "pooled", merge_cadence: "at_halve",
     n_trials_per_walker: 1000, block_size_sweeps: 10,
     trajectory_write_interval_sweeps: 0}
moves:
  - {type: pair_swap, weight: 0.0}
config_search: {n_workers: 1, temperature_high: 2000.0, temperature_low: 100.0, n_temperature_levels: 4, sweeps_per_level: 2, harvest_interval_sweeps: 1, max_anneals_per_worker: 4, backstop_temperature: 150.0, backstop_sweeps: 2}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="weight must be > 0"):
        load_yaml(bad)
