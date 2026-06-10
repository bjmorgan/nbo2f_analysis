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
config_search: {n_workers: 1, window_search_penalty: 2.0, walk_sweeps: 10, max_walks_per_window: 4}
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
moves:
  - {type: pair_swap, weight: 0.1}
  - {type: row_shift, weight: 0.2}
config_search: {n_workers: 1, window_search_penalty: 2.0, walk_sweeps: 10, max_walks_per_window: 4}
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
config_search: {n_workers: 1, window_search_penalty: 2.0, walk_sweeps: 10, max_walks_per_window: 4}
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


def test_load_yaml_defaults_gate_knobs_when_absent():
    # A config without the new keys reproduces the visit_once schedule.
    cfg = load_yaml(DATA / "L9_minimal.yaml")
    assert cfg.wl.one_over_t_gate == "visit_once"
    assert cfg.wl.bp_stall_multiple == 4.0


def test_load_yaml_parses_flatness_gate(write_min_cfg):
    p = write_min_cfg('  one_over_t_gate: "flatness"\n  bp_stall_multiple: 2.0')
    cfg = load_yaml(p)
    assert cfg.wl.one_over_t_gate == "flatness"
    assert cfg.wl.bp_stall_multiple == 2.0


def test_load_yaml_rejects_unknown_one_over_t_gate(write_min_cfg):
    p = write_min_cfg('  one_over_t_gate: "nonsense"')
    with pytest.raises(ValueError, match="one_over_t_gate"):
        load_yaml(p)


def test_load_yaml_rejects_non_positive_bp_stall_multiple(write_min_cfg):
    p = write_min_cfg("  bp_stall_multiple: 0.0")
    with pytest.raises(ValueError, match="bp_stall_multiple must be a finite"):
        load_yaml(p)


@pytest.mark.parametrize("value", [".nan", ".inf"])
def test_load_yaml_rejects_non_finite_bp_stall_multiple(write_min_cfg, value):
    # nan/inf slip past a bare ``<= 0`` check; reject them at config load
    # rather than after the expensive starting-configuration search.
    p = write_min_cfg(f"  bp_stall_multiple: {value}")
    with pytest.raises(ValueError, match="bp_stall_multiple must be a finite"):
        load_yaml(p)


def test_load_yaml_parses_config_search_knobs(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 3, window_search_penalty: 1.5, "
        "walk_sweeps: 40, max_walks_per_window: 8}",
    )
    cfg = load_yaml(p)
    cs = cfg.config_search
    assert cs.n_workers == 3
    assert cs.window_search_penalty == 1.5
    assert cs.walk_sweeps == 40
    assert cs.max_walks_per_window == 8


def test_load_yaml_accepts_null_n_workers(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: null, window_search_penalty: 2.0, "
        "walk_sweeps: 10, max_walks_per_window: 4}",
    )
    cfg = load_yaml(p)
    assert cfg.config_search.n_workers is None


def test_load_yaml_rejects_zero_n_workers(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 0, window_search_penalty: 2.0, "
        "walk_sweeps: 10, max_walks_per_window: 4}",
    )
    with pytest.raises(ValueError, match="n_workers must be >= 1"):
        load_yaml(p)


def test_load_yaml_rejects_non_positive_window_search_penalty(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 1, window_search_penalty: 0.0, "
        "walk_sweeps: 10, max_walks_per_window: 4}",
    )
    with pytest.raises(ValueError, match="window_search_penalty must be > 0"):
        load_yaml(p)


def test_load_yaml_rejects_zero_walk_sweeps(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 1, window_search_penalty: 2.0, "
        "walk_sweeps: 0, max_walks_per_window: 4}",
    )
    with pytest.raises(ValueError, match="walk_sweeps must be >= 1"):
        load_yaml(p)


def test_load_yaml_rejects_zero_max_walks_per_window(tmp_path):
    p = _cfg_with(
        tmp_path,
        "config_search: {n_workers: 1, window_search_penalty: 2.0, "
        "walk_sweeps: 10, max_walks_per_window: 0}",
    )
    with pytest.raises(ValueError, match="max_walks_per_window must be >= 1"):
        load_yaml(p)


def test_load_yaml_rejects_max_walks_below_walker_count(tmp_path):
    # Two windows with 3 walkers each, but only 2 walks allowed per window:
    # a window can never collect its 3rd config (one config per round).
    bad = tmp_path / "bad.yaml"
    bad.write_text("""
random_seed: 0
system: {n_sc: 3, ce: paircut9_5_5_ardr_n96}
windows:
  energy_spacing: 0.1
  list:
    - [-10.0, -9.0, 3]
    - [-9.5, -8.5, 3]
wl: {flatness_limit: 0.8, fill_factor_limit: 1.0e-12, schedule: "1_over_t",
     flatness_mode: "pooled", merge_cadence: "at_halve",
     n_trials_per_walker: 1000, block_size_sweeps: 10,
     trajectory_write_interval_sweeps: 0}
moves:
  - {type: pair_swap, weight: 0.1}
  - {type: row_shift, weight: 0.2}
config_search: {n_workers: 1, window_search_penalty: 2.0, walk_sweeps: 10, max_walks_per_window: 2}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="max_walks_per_window"):
        load_yaml(bad)


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
config_search: {n_workers: 1, window_search_penalty: 2.0, walk_sweeps: 10, max_walks_per_window: 4}
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
config_search: {n_workers: 1, window_search_penalty: 2.0, walk_sweeps: 10, max_walks_per_window: 4}
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
config_search: {n_workers: 1, window_search_penalty: 2.0, walk_sweeps: 10, max_walks_per_window: 4}
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
config_search: {n_workers: 1, window_search_penalty: 2.0, walk_sweeps: 10, max_walks_per_window: 4}
checkpoint: {filename: c.h5, interval_cycles: 0}
""")
    with pytest.raises(ValueError, match="weight must be > 0"):
        load_yaml(bad)
