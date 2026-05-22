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

**If you have any of `icet`, `mchammer-pt`, `mchammer-moves`, or
`chainorder` installed editably from a local checkout, install this
package with `--no-deps` to avoid replacing those editable installs
with fresh wheels:**

    pip install -e nbo2f_analysis/ --no-deps

Then verify the editable installs are intact with `pip list | grep
-iE "icet|mchammer|chainorder"`.

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

See `nbo2f_analysis/rewl/configs/template.yaml` for the full
config schema.
