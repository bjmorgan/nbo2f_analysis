# nbo2f_analysis

NbO2F-specific infrastructure library for the chirality-emergence
paper. Provides:

- `ce_tools` — atoms builders, F-mask manipulation, orbit-rep loading,
  ground-state construction.
- `chain_geometry` — anion-chain decomposition and chain-aware MC moves
  for the ReO3 topology.
- `rewl` — Replica-Exchange Wang-Landau driver for finite-size scaling
  studies.
- Bundled production CE and 12 N=3 orbit representatives under
  `nbo2f_analysis/data/`.

## Install

From inside a clone of this repo:

    pip install -e .

From anywhere (e.g. ARCHER2):

    pip install git+https://github.com/bjmorgan/nbo2f_analysis.git

Four dependencies are fetched directly from their git remotes rather
than from PyPI: the `icet` fork, `mchammer-pt`, `mchammer-moves`, and
`chainorder`.

**If you have any of those four installed editably from a local
checkout, install this package with `--no-deps`** to avoid pip
replacing those editable installs with fresh wheels:

    pip install -e . --no-deps

Verify the editable installs are intact afterwards with
`pip list | grep -iE "icet|mchammer|chainorder"`.

## REWL driver

    rewl run         [--seed N] [--out-dir DIR] [--force] <config.yaml>
    rewl resume      [--extra-cycles N] [--out-dir DIR] <config.yaml>
    rewl postprocess [--out-dir DIR] <config.yaml>

The driver writes the checkpoint (`rewl_state.h5`), three run-summary
CSVs (`convergence.csv`, `exchange_rates.csv`,
`per_move_rejection_rates.csv`), and a 4-panel WL-health diagnostic
figure (`rewl_diagnostics.png`). Stitching and canonical reweighting
are separate post-processing steps provided by `mchammer-pt`:

    mchammer-pt-stitch rewl_state.h5 -o stitched_dos.csv
    mchammer-pt-reweight stitched_dos.csv --T-min 200 --T-max 800 --T-step 2.0 -o canonical_reweighted.csv

See `nbo2f_analysis/rewl/configs/template.yaml` for the full config
schema.
