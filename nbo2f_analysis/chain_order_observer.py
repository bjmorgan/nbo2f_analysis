"""NbO2F chain-ordering observer for frozen-g REWL measurement.

Provides :class:`ChainOrderObserver`, an mchammer ``BaseObserver`` that
evaluates the NbO2F order parameters used to locate and characterise the
chirality-emergence transition, and :func:`build_chain_order_observer`,
which loads the chiral-orbit references the observer needs from the
bundled package data.

The observer is attached to a frozen-g REWL measurement run via
``WangLandauParallelTempering.record_observable``. The per-replica
recorders accumulate ``count``/``sum``/``sum2``/``sum4`` of the requested
scalars per energy bin; downstream stitching and reweighting turn those
moments into canonical ``<O>(T)`` and Binder cumulants.

``chi_11`` and ``closest_chi`` are recorded *signed*: chirality is a
Z2-symmetric broken-symmetry order parameter (``<chi> ~ 0`` in both
phases), so the magnitude and Binder information live in the even moments
the recorder accumulates.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from ase import Atoms
from chainorder import SublatticeOccupation, order_params
from mchammer.observers.base_observer import BaseObserver

from nbo2f_analysis.ce_tools import cis_fraction, load_orbit_rep, nbo4f2_fraction

# Recordable order parameters. ``closest_orbit`` (a categorical orbit
# label) is excluded: its sum/sum2/sum4 moments would be meaningless.
ALLOWED_OPS: frozenset[str] = frozenset({
    "chi_11", "closest_chi", "closest_sim", "icoh_global",
    "icoh_nn", "oof_amp", "cis_frac", "nbo4f2_frac", "collinear_ff",
    "chirality", "circ_coherence",
})

# Order parameters that require the (expensive) per-orbit similarity loop.
_SIMILARITY_OPS: frozenset[str] = frozenset({
    "chi_11", "closest_chi", "closest_sim",
})

# Every similarity op must also be a recordable op: a name added to
# _SIMILARITY_OPS but not ALLOWED_OPS would be silently unrequestable, since
# it could never pass the constructor's ops-subset check. Fail at import.
if not (_SIMILARITY_OPS <= ALLOWED_OPS):
    raise RuntimeError(
        "_SIMILARITY_OPS contains names absent from ALLOWED_OPS: "
        f"{sorted(_SIMILARITY_OPS - ALLOWED_OPS)}"
    )

# Order parameters that require the (expensive) Reynolds projection over the
# 48 cubic point operations performed by
# ``chainorder.order_params.circulation_invariants``.
_PROJECTION_OPS: frozenset[str] = frozenset({
    "chirality", "circ_coherence",
})

# Every projection op must also be a recordable op, by the same argument as the
# _SIMILARITY_OPS check above: a name here but not in ALLOWED_OPS would be
# silently unrequestable. Fail at import.
if not (_PROJECTION_OPS <= ALLOWED_OPS):
    raise RuntimeError(
        "_PROJECTION_OPS contains names absent from ALLOWED_OPS: "
        f"{sorted(_PROJECTION_OPS - ALLOWED_OPS)}"
    )

_N_ORBITS = 12

# Label of the observed chiral broken-symmetry ground state (orbit 11);
# ``chi_11`` is the signed similarity to this orbit specifically.
_GROUND_STATE_ORBIT = "11"

@dataclass(frozen=True)
class OrbitReference:
    """Symmetry-distinct sub-cell images of one chiral orbit.

    ``proper`` holds the images that preserve the orbit's handedness;
    ``improper`` holds the enantiomer images (related by an
    orientation-reversing operation). Naming the two fields keeps the
    chirality-defining distinction from riding on tuple position.
    """

    proper: tuple[np.ndarray, ...]
    improper: tuple[np.ndarray, ...]


# Per-orbit reference store: orbit label ("00".."11") -> its images.
OrbitRefs = dict[str, OrbitReference]


def _apply_spacegroup_op(
    occ: np.ndarray, perm: tuple[int, ...], signs: tuple[int, ...],
) -> np.ndarray:
    """Apply a cubic space-group operation to an anion-sublattice occupation.

    The anion sublattice has three edge-midpoint sub-lattices. Sublattice
    *s* sits at half-integer coordinate along axis *s* and integer
    coordinates along the other two. A rotation (``perm``, ``signs``)
    permutes and possibly negates the axes. Because of the half-integer vs
    integer distinction, a negated axis maps as ``c -> N-1-c`` when it is the
    sublattice's own (half-integer) axis and as ``c -> -c mod N`` for the
    other (integer) axes, where ``c`` is the coordinate along that axis.

    Args:
        occ: Occupation array, shape ``(3, N, N, N)``.
        perm: Axis permutation -- output axis *d* comes from input axis
            ``perm[d]``.
        signs: Per-axis sign flip (+1 or -1).

    Returns:
        Transformed occupation array, same shape.
    """
    N = occ.shape[1]
    result = np.empty_like(occ)
    for s_new in range(3):
        s_old = perm[s_new]
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    new = (i, j, k)
                    old = [0, 0, 0]
                    for d in range(3):
                        src = perm[d]
                        if signs[d] == 1:
                            old[src] = new[d]
                        elif src == s_old:
                            old[src] = (N - 1 - new[d]) % N
                        else:
                            old[src] = (-new[d]) % N
                    result[s_new, i, j, k] = occ[s_old, old[0], old[1], old[2]]
    return result


def _generate_orbit_references(n_sc_orbit: int = 3) -> OrbitRefs:
    """Pre-compute symmetry-distinct sub-cell images for all 12 chiral orbits.

    For each of the 12 bundled orbit representatives, applies all 48
    rotations of the cubic parent (24 proper + 24 improper) combined with
    all lattice translations within the ``n_sc_orbit`` cell, then
    deduplicates. Representatives are loaded from the package data
    directory via :func:`nbo2f_analysis.ce_tools.load_orbit_rep`.

    Args:
        n_sc_orbit: Sub-cell size of the orbit representatives. Defaults to
            3 (period-3 ordering).

    Returns:
        Dict mapping orbit label (``"00"``..``"11"``) to
        ``(proper, improper)`` tuples of
        ``(3, n_sc_orbit, n_sc_orbit, n_sc_orbit)`` occupation arrays.
        Proper images preserve handedness; improper images are the
        enantiomer.
    """
    rots: list[tuple[tuple[int, ...], tuple[int, int, int], int]] = []
    for perm in permutations((0, 1, 2)):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    signs = (sx, sy, sz)
                    p = list(perm)
                    parity = sum(
                        1 for i in range(3) for j in range(i + 1, 3)
                        if p[i] > p[j]
                    )
                    det = (-1) ** parity * sx * sy * sz
                    rots.append((perm, signs, det))

    result: OrbitRefs = {}
    for index in range(_N_ORBITS):
        atoms = load_orbit_rep(index)
        occ = SublatticeOccupation.from_atoms(
            atoms, N=n_sc_orbit, species="F",
        ).occupation
        N = occ.shape[1]
        seen_p: set[bytes] = set()
        seen_i: set[bytes] = set()
        proper: list[np.ndarray] = []
        improper: list[np.ndarray] = []
        for perm, signs, det in rots:
            rotated = _apply_spacegroup_op(occ, perm, signs)
            for tx in range(N):
                for ty in range(N):
                    for tz in range(N):
                        img = np.roll(
                            np.roll(
                                np.roll(rotated, tx, axis=1), ty, axis=2,
                            ),
                            tz, axis=3,
                        )
                        key = img.tobytes()
                        if det == 1 and key not in seen_p:
                            seen_p.add(key)
                            proper.append(img)
                        elif det == -1 and key not in seen_i:
                            seen_i.add(key)
                            improper.append(img)
        result[f"{index:02d}"] = OrbitReference(tuple(proper), tuple(improper))
    return result


class ChainOrderObserver(BaseObserver):
    """Observe NbO2F chain-ordering diagnostics and chiral similarity.

    Per observation, computes (averaged over the three chain directions
    where applicable):

    - ``oof_amp``: mean period-3 Fourier amplitude across chains.
    - ``icoh_nn``: nearest-neighbour inter-chain phase coherence.
    - ``icoh_global``: global inter-chain phase coherence.
    - ``cis_frac``: fraction of Nb with cis F coordination.
    - ``nbo4f2_frac``: fraction of Nb with the local NbO4F2 stoichiometry.
    - ``collinear_ff``: per-(cation, axis) density of collinear (trans)
      F--Nb--F (the adjacent-F chain motif), measured across all
      coordinations. It is 0 in the chain-ordered states (F-spacing
      >= 2) and rises towards 1/9 in the fully disordered f_F = 1/3
      limit.
    - ``chi_11``: signed enantiomeric similarity to orbit 11 (the observed
      broken-symmetry ground state), ``P_max - I_max`` over proper vs
      improper tilings.
    - ``closest_sim``: maximum similarity to any of the 12 orbits.
    - ``closest_chi``: signed enantiomeric similarity of the closest orbit.
      At an exact similarity tie the lowest-labelled orbit wins, so
      ``closest_chi`` then reflects only that one orbit's split: its
      magnitude depends on which tied orbit wins, and its sign additionally
      flips when the tied orbits are opposite-handed. Use ``chi_11`` (fixed
      to one orbit) for Binder-cumulant analysis.
    - ``chirality``: signed <111> circulation pseudoscalar
      (``|E_+|^2 - |E_-|^2``), the Reynolds projection over the 48 cubic
      operations computed by
      ``chainorder.order_params.circulation_invariants``. The tiled P3_121
      ground state gives ``1/4`` (orbit-11 handedness; its enantiomer gives
      ``-1/4``); ``chi_11`` remains the independent template cross-check.
    - ``circ_coherence``: the companion ordering strength
      (``|E_+|^2 + |E_-|^2``) from the same projection, ``1/4`` at the ground
      state.

    ``get_observable`` returns only the order parameters named in ``ops``.
    The structure must be in stable Nb-first ordering (as produced by
    :func:`nbo2f_analysis.ce_tools.atoms_from_f_mask_stable` /
    ``build_tiled_groundstate_atoms``): the cis/NbO4F2 branch reads
    ``structure.numbers[n_sc**3:]``.

    Args:
        n_sc: Supercell size (must be a multiple of ``n_sc_orbit``).
        interval: Observation interval in MC trial steps.
        orbit_refs: Per-orbit reference store, as returned by
            :func:`_generate_orbit_references`.
        ops: Order parameters to emit; each must be in :data:`ALLOWED_OPS`.
        n_sc_orbit: Sub-cell size of the orbit representatives. Must divide
            ``n_sc``. Defaults to 3.

    Raises:
        ValueError: if ``n_sc`` is not a multiple of ``n_sc_orbit``, if
            ``ops`` is empty, or if ``ops`` contains a name outside
            :data:`ALLOWED_OPS`.
    """

    def __init__(
        self,
        n_sc: int,
        interval: int,
        orbit_refs: OrbitRefs,
        ops: tuple[str, ...],
        n_sc_orbit: int = 3,
    ) -> None:
        if n_sc % n_sc_orbit != 0:
            raise ValueError(
                f"n_sc ({n_sc}) must be a multiple of n_sc_orbit "
                f"({n_sc_orbit})"
            )
        ops = tuple(ops)
        if not ops:
            raise ValueError("ops must be a non-empty sequence")
        unknown = set(ops) - ALLOWED_OPS
        if unknown:
            raise ValueError(
                f"ops {sorted(unknown)} not in ALLOWED_OPS "
                f"{sorted(ALLOWED_OPS)}"
            )
        super().__init__(return_type=dict, interval=interval, tag="ChainOrder")
        self._n_sc = n_sc
        self._n_sc_orbit = n_sc_orbit
        self._tile = n_sc // n_sc_orbit
        self._fft_idx = n_sc // 3
        self._orbit_refs = orbit_refs
        self._n_anion = 3 * n_sc**3
        self._ops = ops
        self._needs_similarity = bool(set(ops) & _SIMILARITY_OPS)
        self._needs_projection = bool(set(ops) & _PROJECTION_OPS)

    def _per_cell_counts(self, occ_array: np.ndarray) -> np.ndarray:
        """Sum of '1' entries per sub-cell position across all tile blocks."""
        n_o = self._n_sc_orbit
        t = self._tile
        return (
            occ_array.reshape(3, t, n_o, t, n_o, t, n_o)
            .transpose(0, 2, 4, 6, 1, 3, 5)
            .sum(axis=(4, 5, 6))
        )

    def _best_similarity(
        self,
        counts1: np.ndarray,
        counts0: np.ndarray,
        refs: tuple[np.ndarray, ...],
    ) -> float:
        best = 0
        for ref in refs:
            sim = int(np.where(ref, counts1, counts0).sum())
            if sim > best:
                best = sim
        return best / self._n_anion

    def get_observable(self, structure: Atoms) -> dict[str, float]:
        """Evaluate the requested order parameters on ``structure``."""
        occ = SublatticeOccupation.from_atoms(
            structure, N=self._n_sc, species="F",
        )
        oof_amps, icoh_nns, icoh_globals, ff_motifs = [], [], [], []
        for direction in (occ.x, occ.y, occ.z):
            fft = order_params.chain_fft(direction)
            oof_amps.append(float(np.abs(fft[..., self._fft_idx]).mean()))

            # Collinear F--Nb--F is an adjacent F--F pair along the chain: a
            # cation's two same-axis anions are consecutive chain sites. The
            # FF (1, 1) 2-motif frequency is its per-(cation, axis) density,
            # measured across all coordinations (cf. cis_frac, NbO4F2-only).
            # motif_frequencies omits patterns that never occur, so a
            # direction with no adjacent FF anywhere has no (1, 1) key --
            # treat that absence as a zero frequency.
            motifs = order_params.motif_frequencies(direction, window_length=2)
            ff = motifs.get((1, 1))
            ff_motifs.append(float(ff.mean()) if ff is not None else 0.0)

            G = order_params.inter_chain_correlation(direction, period=3)
            nn = (
                np.abs(G[1, 0]) + np.abs(G[0, 1])
                + np.abs(G[-1, 0]) + np.abs(G[0, -1])
            ) / 4
            off_origin = np.ones_like(G, dtype=bool)
            off_origin[0, 0] = False
            icoh_nns.append(float(nn))
            icoh_globals.append(float(np.abs(G[off_origin]).mean()))

        f_mask = structure.numbers[self._n_sc**3:] == 9
        computed: dict[str, float] = {
            "oof_amp": float(np.mean(oof_amps)),
            "icoh_nn": float(np.mean(icoh_nns)),
            "icoh_global": float(np.mean(icoh_globals)),
            "cis_frac": float(cis_fraction(f_mask, self._n_sc)),
            "nbo4f2_frac": float(nbo4f2_fraction(f_mask, self._n_sc)),
            "collinear_ff": float(np.mean(ff_motifs)),
        }
        if self._needs_similarity:
            counts1 = self._per_cell_counts(occ.occupation)
            counts0 = self._tile**3 - counts1
            best_sim = 0.0
            best_p = 0.0
            best_i = 0.0
            for label, ref in self._orbit_refs.items():
                p_max = self._best_similarity(counts1, counts0, ref.proper)
                i_max = self._best_similarity(counts1, counts0, ref.improper)
                if label == _GROUND_STATE_ORBIT:
                    computed["chi_11"] = p_max - i_max
                if p_max > best_sim or i_max > best_sim:
                    best_p = p_max
                    best_i = i_max
                    best_sim = max(p_max, i_max)
            computed["closest_sim"] = best_sim
            computed["closest_chi"] = best_p - best_i
        if self._needs_projection:
            ci = order_params.circulation_invariants(occ, period=3)
            computed["chirality"] = float(ci.chirality)
            computed["circ_coherence"] = float(ci.coherence)
        return {k: computed[k] for k in self._ops}


def build_chain_order_observer(
    n_sc: int,
    interval: int,
    ops: tuple[str, ...],
    n_sc_orbit: int = 3,
) -> ChainOrderObserver:
    """Build a :class:`ChainOrderObserver`, loading orbit references.

    The chiral-orbit references are built once from the bundled package
    data; the returned observer is picklable and ready to attach via
    ``record_observable``.

    Args:
        n_sc: Supercell size (must be a multiple of ``n_sc_orbit``).
        interval: Observation interval in MC trial steps.
        ops: Order parameters to record; each must be in
            :data:`ALLOWED_OPS`.
        n_sc_orbit: Orbit-representative sub-cell size. Defaults to 3.

    Returns:
        A configured :class:`ChainOrderObserver`.
    """
    orbit_refs = _generate_orbit_references(n_sc_orbit=n_sc_orbit)
    return ChainOrderObserver(n_sc, interval, orbit_refs, ops, n_sc_orbit=n_sc_orbit)
