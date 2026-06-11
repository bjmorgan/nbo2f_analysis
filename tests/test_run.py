"""Tests for nbo2f_analysis.rewl.run wiring (forwarding to process_pool)."""
from __future__ import annotations

import pytest

from nbo2f_analysis.rewl import run as run_mod
from nbo2f_analysis.rewl.config import load_yaml


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
    tmp_path, monkeypatch, write_min_cfg, gate_lines, expected_gate, expected_multiple
):
    cfg = load_yaml(write_min_cfg(gate_lines))
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


@pytest.mark.parametrize(
    "entry_lines, expected_entry",
    [
        # An explicit non-default policy reaches process_pool.
        ('  one_over_t_entry: "f_continuous"', "f_continuous"),
        # Key omitted: the default is forwarded, so old configs behave
        # unchanged (the PR's backward-compatibility promise).
        ("", "window_clock"),
    ],
)
def test_run_forwards_one_over_t_entry_to_process_pool(
    tmp_path, monkeypatch, write_min_cfg, entry_lines, expected_entry
):
    cfg = load_yaml(write_min_cfg(entry_lines))
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

    assert captured.get("one_over_t_entry") == expected_entry
