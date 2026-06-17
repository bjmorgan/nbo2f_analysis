"""End-to-end REWL driver smoke test on a tiny N=3 system."""
from __future__ import annotations

from pathlib import Path

import pytest

from nbo2f_analysis.ce_tools import build_tiled_groundstate_atoms


def _tiny_yaml(tmp_path: Path, e_gs: float) -> Path:
    """Write a minimal YAML config for an n_sc=3 smoke run."""
    p = tmp_path / "tiny.yaml"
    lo0, hi0 = e_gs - 0.5, e_gs + 1.5
    lo1, hi1 = e_gs + 0.5, e_gs + 3.0
    p.write_text(f"""
random_seed: 42
system:
  n_sc: 3
  ce: paircut9_5_5_ardr_n96
windows:
  energy_spacing: 0.5
  list:
    - [{lo0:.3f}, {hi0:.3f}, 1]
    - [{lo1:.3f}, {hi1:.3f}, 1]
wl:
  flatness_limit: 0.5
  fill_factor_limit: 1.0e-2
  schedule: "1_over_t"
  flatness_mode: "pooled"
  merge_cadence: "at_halve"
  n_trials_per_walker: 5400
  block_size_sweeps: 10
  trajectory_write_interval_sweeps: 0
moves:
  - {{type: pair_swap, weight: 0.1}}
  - {{type: row_shift, weight: 0.2}}
  - {{type: motif_shift, weight: 0.5}}
  - {{type: chain_swap, weight: 0.5}}
  - {{type: row_reflect, weight: 0.5}}
config_search:
  n_workers: 2
  window_search_penalty: 2.0
  walk_sweeps: 10
  max_walks_per_window: 4
checkpoint:
  filename: rewl_state.h5
  interval_cycles: 5
""")
    return p


@pytest.mark.slow
def test_run_then_resume_then_postprocess(tmp_path, monkeypatch):
    # Compute the GS energy so we can place windows correctly.
    from pathlib import Path as _Path

    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator

    ce_path = str(
        _Path(__file__).resolve().parent.parent
        / "nbo2f_analysis" / "data" / "ces" / "paircut9_5_5_ardr_n96.ce"
    )
    ce = ClusterExpansion.read(ce_path)
    gs = build_tiled_groundstate_atoms(n_sc=3)
    calc = ClusterExpansionCalculator(gs.copy(), ce)
    e_gs = float(calc.calculate_total(occupations=gs.numbers))

    yaml_path = _tiny_yaml(tmp_path, e_gs)
    monkeypatch.chdir(tmp_path)

    # --- run ---
    from nbo2f_analysis.rewl.cli import main
    rc = main(["run", str(yaml_path)])
    assert rc == 0
    for fname in [
        "rewl_state.h5",
        "convergence.csv",
        "exchange_rates.csv",
        "rewl_diagnostics.png",
    ]:
        assert (tmp_path / fname).is_file(), f"missing artefact: {fname}"

    # --- resume with extra cycles ---
    rc = main(["resume", str(yaml_path), "--extra-cycles", "3"])
    assert rc == 0

    # --- postprocess ---
    # Remove the figure so we can confirm postprocess rebuilds it.
    (tmp_path / "rewl_diagnostics.png").unlink()
    rc = main(["postprocess", str(yaml_path)])
    assert rc == 0
    assert (tmp_path / "rewl_diagnostics.png").is_file()

    # --- upstream mchammer-pt-stitch on the checkpoint ---
    import subprocess
    rc = subprocess.run(
        [
            "mchammer-pt-stitch",
            str(tmp_path / "rewl_state.h5"),
            "-o", str(tmp_path / "stitched_dos.csv"),
        ],
        check=False,
    ).returncode
    assert rc == 0
    assert (tmp_path / "stitched_dos.csv").is_file()

    # --- upstream mchammer-pt-reweight on the stitched DOS ---
    rc = subprocess.run(
        [
            "mchammer-pt-reweight",
            str(tmp_path / "stitched_dos.csv"),
            "--T-min", "200", "--T-max", "800", "--T-step", "50",
            "-o", str(tmp_path / "canonical.csv"),
        ],
        check=False,
    ).returncode
    assert rc == 0
    assert (tmp_path / "canonical.csv").is_file()


def _tiny_yaml_with_measurement(tmp_path: Path, e_gs: float) -> Path:
    """Write a tiny n_sc=3 config with a measurement section."""
    p = tmp_path / "tiny_measure.yaml"
    lo0, hi0 = e_gs - 0.5, e_gs + 1.5
    lo1, hi1 = e_gs + 0.5, e_gs + 3.0
    p.write_text(f"""
random_seed: 42
system:
  n_sc: 3
  ce: paircut9_5_5_ardr_n96
windows:
  energy_spacing: 0.5
  list:
    - [{lo0:.3f}, {hi0:.3f}, 1]
    - [{lo1:.3f}, {hi1:.3f}, 1]
wl:
  flatness_limit: 0.5
  fill_factor_limit: 1.0e-2
  schedule: "1_over_t"
  flatness_mode: "pooled"
  merge_cadence: "at_halve"
  n_trials_per_walker: 5400
  block_size_sweeps: 10
  trajectory_write_interval_sweeps: 0
moves:
  - {{type: pair_swap, weight: 0.1}}
  - {{type: row_shift, weight: 0.2}}
  - {{type: motif_shift, weight: 0.5}}
config_search:
  n_workers: 2
  window_search_penalty: 2.0
  walk_sweeps: 10
  max_walks_per_window: 4
checkpoint:
  filename: rewl_state.h5
  interval_cycles: 5
measurement:
  checkpoint_filename: rewl_measure.h5
  observer_interval: 108
  n_measure_cycles: 30
  checkpoint_interval_cycles: 5
  ops: [chi_11, icoh_global, oof_amp, cis_frac]
""")
    return p


@pytest.mark.slow
def test_measure_end_to_end(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path as _Path

    from icet import ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator

    ce_path = str(
        _Path(__file__).resolve().parent.parent
        / "nbo2f_analysis" / "data" / "ces" / "paircut9_5_5_ardr_n96.ce"
    )
    ce = ClusterExpansion.read(ce_path)
    gs = build_tiled_groundstate_atoms(n_sc=3)
    calc = ClusterExpansionCalculator(gs.copy(), ce)
    e_gs = float(calc.calculate_total(occupations=gs.numbers))

    yaml_path = _tiny_yaml_with_measurement(tmp_path, e_gs)
    monkeypatch.chdir(tmp_path)

    from nbo2f_analysis.rewl.cli import main

    # --- DOS run, then frozen-g measurement ---
    assert main(["run", str(yaml_path)]) == 0
    assert main(["measure", str(yaml_path)]) == 0
    assert (tmp_path / "rewl_measure.h5").is_file()

    # --- manual canonical reduction via the mchammer-pt console scripts ---
    assert subprocess.run(
        ["mchammer-pt-stitch", str(tmp_path / "rewl_state.h5"),
         "-o", str(tmp_path / "dos.csv")],
        check=False,
    ).returncode == 0

    assert subprocess.run(
        ["mchammer-pt-stitch-observables", str(tmp_path / "rewl_measure.h5"),
         "-o", str(tmp_path / "observables")],
        check=False,
    ).returncode == 0
    assert (tmp_path / "observables" / "ChainOrder.csv").is_file()

    assert subprocess.run(
        ["mchammer-pt-reweight-observables",
         str(tmp_path / "observables"), str(tmp_path / "dos.csv"),
         "--T-min", "200", "--T-max", "800", "--T-step", "100",
         "-o", str(tmp_path / "canonical")],
        check=False,
    ).returncode == 0

    canonical = tmp_path / "canonical" / "ChainOrder.csv"
    assert canonical.is_file()
    import pandas as pd
    df = pd.read_csv(canonical)
    assert "chi_11_binder" in df.columns
    assert "T_K" in df.columns
