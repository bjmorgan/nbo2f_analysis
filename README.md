# nbo2f_analysis

NbO2F-specific infrastructure library for the chirality-emergence
paper. Provides:

- `ce_tools` — atoms builders, F-mask manipulation, orbit-rep loading,
  ground-state construction.
- `chain_geometry` — anion-chain decomposition and chain-aware MC moves
  for the ReO3 topology.
- `cell_symmetry` — whole-cell `<100>` reflection move (`CellReflect`)
  for bridging the enantiomeric chiral basins.
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
    rewl measure     [--extra-cycles N] [--out-dir DIR] <config.yaml>

The driver writes the checkpoint (`rewl_state.h5`), three run-summary
CSVs (`convergence.csv`, `exchange_rates.csv`,
`per_move_rejection_rates.csv`), and a 4-panel WL-health diagnostic
figure (`rewl_diagnostics.png`). Stitching and canonical reweighting
are separate post-processing steps provided by `mchammer-pt`:

    mchammer-pt-stitch rewl_state.h5 -o stitched_dos.csv
    mchammer-pt-reweight stitched_dos.csv --T-min 200 --T-max 800 --T-step 2.0 -o canonical_reweighted.csv

See `nbo2f_analysis/rewl/configs/template.yaml` for the full config
schema.

### Structural measurement

`rewl measure <config>` runs a frozen-g measurement pass on a converged
REWL checkpoint, recording the NbO2F order parameters as microcanonical
moments. It writes its own checkpoint (`measurement.checkpoint_filename`)
and resumes/chains off it, so statistics build up across jobs; restart by
deleting that checkpoint. It requires a `measurement` section in the
config (see the template).

The canonical reduction is a separate manual step, run from job scripts
via the `mchammer-pt` console scripts:

    mchammer-pt-stitch rewl_state.h5 -o dos.csv
    mchammer-pt-stitch-observables rewl_measure.h5 -o observables/
    mchammer-pt-reweight-observables observables/ dos.csv --T-min 350 --T-max 600 --T-step 1 -o canonical/

The reweight emits a coverage warning if the sampled bins miss canonical
weight at the requested temperatures.
