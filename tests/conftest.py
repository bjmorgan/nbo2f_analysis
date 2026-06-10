"""Shared fixtures for the nbo2f_analysis test suite."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def write_min_cfg(tmp_path: Path):
    """Return a builder for a minimal valid REWL config in ``tmp_path``.

    ``wl_extra`` is appended verbatim to the ``wl`` block, so callers must
    indent it to sit under ``wl:`` (or leave it empty); the builder returns
    the path of the written file. Shared by tests that need a parseable
    n_sc=3 config differing only in the ``wl`` block — passing raw lines
    keeps control of the literal YAML, which the rejection tests rely on to
    inject edge values such as ``.nan`` or an unknown gate name.
    """
    def _write(wl_extra: str = "") -> Path:
        path = tmp_path / "cfg.yaml"
        path.write_text(f"""
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
{wl_extra}
moves:
  - {{type: pair_swap, weight: 0.1}}
  - {{type: row_shift, weight: 0.2}}
config_search: {{n_workers: 1, window_search_penalty: 2.0, walk_sweeps: 10, max_walks_per_window: 4}}
checkpoint: {{filename: c.h5, interval_cycles: 0}}
""")
        return path

    return _write
