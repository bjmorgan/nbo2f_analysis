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
    "wl_lines, expected",
    [
        # Explicit non-default knobs reach process_pool.
        (
            '  one_over_t_gate: "flatness"\n'
            '  bp_stall_multiple: 2.0\n'
            '  one_over_t_entry: "f_continuous"',
            {
                "one_over_t_gate": "flatness",
                "bp_stall_multiple": 2.0,
                "one_over_t_entry": "f_continuous",
            },
        ),
        # Keys omitted: the defaults are forwarded, so old configs behave
        # unchanged.
        (
            "",
            {
                "one_over_t_gate": "visit_once",
                "bp_stall_multiple": 4.0,
                "one_over_t_entry": "window_clock",
            },
        ),
    ],
)
def test_run_forwards_wl_knobs_to_process_pool(
    tmp_path, monkeypatch, write_min_cfg, wl_lines, expected
):
    cfg = load_yaml(write_min_cfg(wl_lines))
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

    for key, value in expected.items():
        assert captured.get(key) == value
