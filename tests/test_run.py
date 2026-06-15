"""Tests for nbo2f_analysis.rewl.run wiring (forwarding to process_pool)."""
from __future__ import annotations

import types

import pytest

from mchammer_pt.history import ExchangeHistory
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


class _SpyPT:
    """Stands in for the resumed orchestrator: records run() calls."""

    def __init__(self, n_replicas: int):
        self.pool = object()
        self.run_calls: list[int] = []
        self.save_calls = 0
        self.history = ExchangeHistory.empty(n_cycles=1, n_replicas=n_replicas)

    def attach_cycle_callback(self, cb):
        pass

    def attach_checkpoint_writer(self, *a, **k):
        pass

    def run(self, n_cycles):
        self.run_calls.append(n_cycles)

    def save_checkpoint(self, path):
        self.save_calls += 1


def _install_resume_stubs(monkeypatch, tmp_path, *, last_step, block_size=10):
    """Stub out everything in resume() except the real cycle-count logic.

    The config from `write_min_cfg` has n_trials_per_walker=1000, so the
    target is ``1000 // block_size`` cycles. `prior` is padded to that
    target length, reproducing the bug state: a walltime-killed checkpoint
    carries a per-cycle history pre-allocated to the full target, so the old
    shape-based count (`shape[0] - 1`) read it as "complete" and no-opped.
    The under-target tests therefore fail against that logic and pass only
    with the walker-step count.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c.h5").write_bytes(b"")  # resume() only checks existence

    spy = _SpyPT(n_replicas=2)
    containers = [
        types.SimpleNamespace(_last_state={"last_step": last_step}),
        types.SimpleNamespace(_last_state={"last_step": last_step}),
    ]
    target_cycles = 1000 // block_size
    prior = ExchangeHistory.empty(n_cycles=target_cycles, n_replicas=2)

    monkeypatch.setattr(
        run_mod.ClusterExpansion, "read", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(
        run_mod, "build_moves_and_kwargs", lambda cfg, ce: (None, None, None, {})
    )
    monkeypatch.setattr(
        run_mod.WangLandauParallelTempering,
        "resume_process_pool",
        lambda *a, **k: spy,
    )
    # Patch read_hdf5 at its definition site, not on run_mod: resume()
    # imports it locally (`from mchammer_pt.history import read_hdf5`), so
    # the local binding resolves to this attribute at call time.
    monkeypatch.setattr(
        "mchammer_pt.history.read_hdf5",
        lambda *a, **k: (prior, containers, {"block_size": block_size}),
    )
    monkeypatch.setattr(
        run_mod, "WangLandauProgressPrinter", lambda *a, **k: object()
    )
    monkeypatch.setattr(run_mod, "write_all", lambda *a, **k: None)
    return spy


def test_resume_runs_remaining_cycles_from_walker_step(
    tmp_path, monkeypatch, write_min_cfg
):
    cfg = load_yaml(write_min_cfg(""))
    spy = _install_resume_stubs(monkeypatch, tmp_path, last_step=50)  # 5 done
    run_mod.resume(cfg)
    assert spy.run_calls == [95]  # target 100 - 5 done


def test_resume_no_ops_when_complete(tmp_path, monkeypatch, write_min_cfg):
    cfg = load_yaml(write_min_cfg(""))
    spy = _install_resume_stubs(monkeypatch, tmp_path, last_step=1000)  # 100 done
    run_mod.resume(cfg)
    assert spy.run_calls == []


def test_resume_extra_cycles_overrides_auto_count(
    tmp_path, monkeypatch, write_min_cfg
):
    cfg = load_yaml(write_min_cfg(""))
    spy = _install_resume_stubs(monkeypatch, tmp_path, last_step=50)
    run_mod.resume(cfg, extra_cycles=7)
    assert spy.run_calls == [7]


def test_resume_tolerates_off_block_walker_step(
    tmp_path, monkeypatch, write_min_cfg
):
    # A walker that converged mid-block freezes off a block boundary; resume
    # must not abort. With last_step=55 and block_size=10, floor division
    # gives 5 completed cycles, so it runs the remaining 95 of the 100 target.
    cfg = load_yaml(write_min_cfg(""))
    spy = _install_resume_stubs(monkeypatch, tmp_path, last_step=55)
    run_mod.resume(cfg)
    assert spy.run_calls == [95]


def test_resume_count_tracks_cumulative_walker_step(
    tmp_path, monkeypatch, write_min_cfg
):
    # The count comes from the walker step, not the history length: a larger
    # restored step (a later resume segment) leaves fewer cycles remaining.
    cfg = load_yaml(write_min_cfg(""))
    spy = _install_resume_stubs(monkeypatch, tmp_path, last_step=500)  # 50 done
    run_mod.resume(cfg)
    assert spy.run_calls == [50]  # target 100 - 50 done


def test_resume_divides_by_checkpoint_block_size_not_config(
    tmp_path, monkeypatch, write_min_cfg
):
    # n_done and the target both use the checkpoint's own block_size: a
    # block_size of 20 gives target 1000 // 20 = 50 and 600 // 20 = 30 done,
    # so 20 remaining -- independent of the run config's block size.
    cfg = load_yaml(write_min_cfg(""))
    spy = _install_resume_stubs(
        monkeypatch, tmp_path, last_step=600, block_size=20
    )
    run_mod.resume(cfg)
    assert spy.run_calls == [20]
