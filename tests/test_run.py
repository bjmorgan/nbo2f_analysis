"""Tests for nbo2f_analysis.rewl.run wiring (forwarding to process_pool)."""
from __future__ import annotations

from pathlib import Path

import pytest

from nbo2f_analysis.rewl import run as run_mod
from nbo2f_analysis.rewl.config import load_yaml


def _cfg(tmp_path: Path, gate_lines: str) -> Path:
    """Write a minimal n_sc=3 config, appending ``gate_lines`` to the wl block.

    ``gate_lines`` must already be indented to sit under ``wl:`` (or be empty).
    """
    p = tmp_path / "cfg.yaml"
    p.write_text(f"""
random_seed: 0
system: {{n_sc: 3, ce: paircut9_5_5_ardr_n96}}
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
{gate_lines}
moves:
  - {{type: pair_swap, weight: 0.1}}
  - {{type: row_shift, weight: 0.2}}
config_search: {{n_workers: 1, window_search_penalty: 2.0, walk_sweeps: 10, max_walks_per_window: 4}}
checkpoint: {{filename: rewl_state.h5, interval_cycles: 0}}
""")
    return p


class _StopAfterPool(Exception):
    """Stands in for the process_pool failure path: the spy captures the
    forwarded kwargs and raises this, stopping run() before a real pool
    launches."""


@pytest.mark.parametrize(
    "gate_lines, expected_gate, expected_multiple",
    [
        # Explicit non-default knobs reach process_pool.
        ('  one_over_t_gate: "flatness"\n  bp_stall_multiple: 2.0', "flatness", 2.0),
        # Keys omitted: the defaults are forwarded, so old configs behave
        # unchanged (the PR's backward-compatibility promise).
        ("", "visit_once", 4.0),
    ],
)
def test_run_forwards_gate_knobs_to_process_pool(
    tmp_path, monkeypatch, gate_lines, expected_gate, expected_multiple
):
    cfg = load_yaml(_cfg(tmp_path, gate_lines))
    monkeypatch.chdir(tmp_path)

    captured: dict = {}

    def spy(**kwargs):
        captured.update(kwargs)
        raise _StopAfterPool

    # The parallel starting-configuration search is expensive and unrelated
    # to the wiring under test; return one config-per-window placeholder.
    monkeypatch.setattr(
        run_mod, "find_all_window_configs", lambda **kw: [object(), object()]
    )
    monkeypatch.setattr(
        run_mod.WangLandauParallelTempering, "process_pool", spy
    )

    with pytest.raises(_StopAfterPool):
        run_mod.run(cfg)

    assert captured.get("one_over_t_gate") == expected_gate
    assert captured.get("bp_stall_multiple") == expected_multiple
