# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  ``row_reflect``, ``cell_reflect``) with per-entry weights.
- ``cell_symmetry`` — whole-cell ``<100>`` mirror geometry and the
  ``CellReflect`` move, which reflects the anion sublattice across a
  ``<100>`` plane to bridge the degenerate enantiomeric chiral basins
  of the ordered phase (iso-energetic; permutes anions only).
