# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- ``rewl resume`` now counts completed cycles from the restored walker
  MC step (via ``mchammer_pt.completed_cycles``) instead of the
  pre-allocated per-cycle history shape, which always equalled the
  target. Previously an under-target checkpoint reported completion and
  no-opped, breaking checkpoint-chaining; large-L production runs killed
  by walltime can now be resumed to convergence. The completed count is
  divided by the checkpoint's own ``block_size`` rather than the run
  config's, so it stays correct across chained resumes and tolerates
  walkers that converged mid-block. The ``--extra-cycles`` manual
  override is unchanged.
- Requires ``mchammer-pt`` 0.23.0 or newer (for ``completed_cycles`` and
  the trimmed-history checkpoints).

## [0.6.0] - 2026-06-11

### Added

- The run YAML's `wl` section accepts an optional `one_over_t_entry`
  knob, forwarded to `WangLandauParallelTempering.process_pool`. It
  selects how a window's fill factor enters the `1/t` phase at the
  switch: `"window_clock"` (the default) follows the
  Belardinelli-Pereyra clock, with the fill factor jumping to `1/t`
  at the phase switch; `"f_continuous"` records a schedule-clock
  origin so the fill factor is continuous across the switch.
  Validated at config load. Omitting it reproduces the previous
  `window_clock` behaviour, so existing configs parse and behave
  unchanged. On resume the knob is read back from the checkpoint
  rather than the config.
- Requires `mchammer-pt` 0.22.0 or newer.

## [0.5.0] - 2026-06-09

### Added

- The run YAML's `wl` section accepts two optional knobs, forwarded to
  `WangLandauParallelTempering.process_pool`: `one_over_t_gate`
  (`"visit_once"`, the default, or `"flatness"`) selects the
  halving-phase gate under the `1_over_t` schedule, and
  `bp_stall_multiple` (default `4.0`) sets the stall threshold consulted
  under the `"flatness"` gate. Both are validated at config load.
  Omitting them reproduces the previous `visit_once` schedule, so
  existing configs parse and behave unchanged. On resume the knobs are
  read back from the checkpoint rather than the config.
- Requires `mchammer-pt` 0.21.0 or newer.

## [0.4.0] - 2026-06-04

### Changed

- BREAKING: the `config_search` section of the run YAML has a new schema.
  The annealing knobs (`temperature_high`, `temperature_low`,
  `n_temperature_levels`, `sweeps_per_level`, `harvest_interval_sweeps`,
  `max_anneals_per_worker`, `backstop_temperature`, `backstop_sweeps`) are
  removed and replaced by `window_search_penalty`, `walk_sweeps`,
  `max_walks_per_window`, and `n_workers`. Update existing configs to the
  new keys (see the shipped example configs).
- The starting-configuration search now delegates to the material-agnostic
  `mchammer_pt.seed_window_configs` (requires mchammer-pt >= 0.18.0),
  replacing the in-tree parallel-anneal search with the upstream
  bidirectional confined-walk search. Narrow low-energy and high-energy
  windows that the anneal could not fill are now seeded reliably.

## [0.3.1] - 2026-06-04

### Added

- The starting-configuration search now logs progress: a line each time a
  configuration is harvested into a window (showing that window's
  found/target count, plus a windows-filled tally when it completes), a
  periodic heartbeat listing which windows are still short, a notice when
  the lingering backstop runs, and a completion line. A long search is now
  legible instead of silent.

## [0.3.0] - 2026-06-04

### Added

- Per-walker starting structures: multiwalker REWL windows are now
  seeded with distinct, decorrelated configurations (one per walker)
  rather than replicating a single structure across the window.

### Changed

- The starting-configuration search is now a parallel simulated anneal
  driven by the production move set (Metropolis on the cluster-expansion
  energy), replacing the blind O/F-swap sampler. Each anneal harvests
  distinct in-window configurations on the way down a geometric
  temperature schedule; a fixed-temperature lingering backstop tops up
  any window the anneals leave short, and the ground state is seeded once
  into the lowest window. All search randomness derives from the run's
  ``random_seed``.
- BREAKING: the ``config_search`` YAML section changed. The blind-sampler
  knobs (``max_swaps``, ``attempts_per_swap_count``, ``random_attempts``)
  are replaced by annealing knobs (``temperature_high``,
  ``temperature_low``, ``n_temperature_levels``, ``sweeps_per_level``,
  ``harvest_interval_sweeps``, ``max_anneals_per_worker``,
  ``backstop_temperature``, ``backstop_sweeps``). Existing config files
  must be updated; see the bundled example configs under
  ``nbo2f_analysis/rewl/configs/``.
- Requires ``mchammer-pt`` 0.17.0 or newer, which adds the per-window
  single-or-sequence ``atoms`` argument to
  ``WangLandauParallelTempering.process_pool`` that per-walker seeding
  relies on. The git dependency tracks ``main``; ensure the installed
  ``mchammer-pt`` is at least 0.17.0.

## [0.2.0] - 2026-06-01

### Added

- ``cell_symmetry`` — whole-cell ``<100>`` mirror geometry and the
  ``CellReflect`` move, which reflects the anion sublattice across a
  ``<100>`` plane to bridge the degenerate enantiomeric chiral basins
  of the ordered phase (iso-energetic; permutes anions only).
- ``cell_reflect`` registered as a move type in the move registry,
  selectable from the YAML ``moves:`` section like the other moves.

## [0.1.0] - 2026-05-22

### Added

- ``ce_tools`` — atoms builders, F-mask manipulation, orbit-rep
  loading, and ground-state construction for the tiled chiral OOF
  N=3 ground state.
- ``chain_geometry`` — anion-chain decomposition for ReO3-topology
  supercells, plus chain-aware MC moves: ``RowShift``,
  ``MotifShift``, ``RowReflect``, ``PlaneShift``.
- ``rewl`` driver — Replica-Exchange Wang-Landau orchestrator with
  ``run``/``resume``/``postprocess`` CLI subcommands, YAML config
  schema with validation, spawn-based parallel
  starting-configuration search, and a 4-panel WL-health diagnostic
  figure. The driver writes the checkpoint (``rewl_state.h5``)
  plus three run-summary CSVs; stitching and canonical reweighting
  are external post-processing via mchammer-pt's
  ``mchammer-pt-stitch`` and ``mchammer-pt-reweight``.
- Bundled production CE (``data/ces/paircut9_5_5_ardr_n96.ce``)
  and the 12 N=3 orbit representatives
  (``data/orbit_representatives/``).
- Three example configs under
  ``nbo2f_analysis/rewl/configs/``: ``template.yaml``,
  ``L9_production.yaml``, ``L12_pilot.yaml``.
- Move-registry-driven configurable move set: the YAML config's
  ``moves:`` section selects from a registered set of move types
  (``pair_swap``, ``row_shift``, ``motif_shift``, ``chain_swap``,
  ``row_reflect``) with per-entry weights.
