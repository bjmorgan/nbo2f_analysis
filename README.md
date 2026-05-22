# nbo2f_analysis

Analysis tools for NbO2F finite-size scaling. Currently provides
the `rewl` driver (Replica-Exchange Wang-Landau).

## Install

Local development install:

    pip install -e nbo2f_analysis/

ARCHER2 or other deployment:

    pip install /path/to/nbo2f_analysis/

The `icet` fork, `mchammer-pt`, `mchammer-moves`, and `chainorder`
are fetched directly from their git remotes; `bsym` comes from PyPI.

## REWL driver

    rewl run         [--seed N] [--out-dir DIR] [--force] <config.yaml>
    rewl resume      [--extra-cycles N] [--out-dir DIR] <config.yaml>
    rewl postprocess [--out-dir DIR] <config.yaml>

The driver writes the checkpoint, per-window CSVs, the stitched DOS,
and a 4-panel WL-health diagnostic figure. Canonical reweighting is
a separate post-processing step provided by `mchammer-pt`:

    mchammer-pt-reweight stitched_dos.csv --T-min 200 --T-max 800 --T-step 2.0 -o canonical_reweighted.csv

See `nbo2f_analysis/rewl/configs/template.yaml` for the full
config schema.
