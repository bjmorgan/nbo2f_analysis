"""Anion-chain geometry and MC moves for the ReO3-type NbO2F supercell.

Provides :func:`build_anion_chains` for enumerating anion chains, and
two ``mchammer_moves.CyclicShift`` subclasses that target the
inter-chain phase degree of freedom:

- :class:`RowShift`: shifts an entire anion row by +-1 site.
  Preserves OOF ordering globally but costs scale with ``n_sc``
  (every site on the chain changes its inter-chain environment).
- :class:`MotifShift`: shifts a single OOF period (3 sites) within
  a row. Creates two domain walls at the segment boundaries but the
  energy cost is constant regardless of ``n_sc``.

The decomposition is deterministic, derived from
:func:`ce_tools.anion_index`'s known site-ordering convention; no
position-based on-lattice classification is performed.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .ce_tools import anion_index
from mchammer_moves import CyclicShift
from mchammer_moves.moves.base import Move
from mchammer_moves.moves.cyclic_reflection import CyclicReflection

if TYPE_CHECKING:
    from mchammer.configuration_manager import ConfigurationManager


def build_anion_chains(n_sc: int) -> list[list[int]]:
    """Return all anion chains as lists of global atom indices.

    For each anion sublattice ``s`` in ``{0, 1, 2}`` (x-, y-, z-edge
    midpoints respectively) and each pair of orthogonal cell indices,
    the sites along the corresponding cubic axis form a length-``n_sc``
    chain. There are ``3 * n_sc**2`` chains in total, each of length
    ``n_sc``.

    Site indices are emitted as global indices into the atoms array
    produced by :func:`ce_tools.atoms_from_f_mask_stable`, where the
    first ``n_sc**3`` sites are cations and the remaining
    ``3 * n_sc**3`` are anions in :func:`ce_tools.anion_index` order.
    The cation offset is added here so the chains can be passed
    directly to ``mchammer_moves.CyclicShift``.

    Args:
        n_sc: Cubic supercell repeat count along each axis.

    Returns:
        ``3 * n_sc**2`` chains, each a length-``n_sc`` list of global
        atom indices.
    """
    cation_offset = n_sc**3
    chains: list[list[int]] = []
    for s in range(3):
        for a in range(n_sc):
            for b in range(n_sc):
                if s == 0:
                    chain = [
                        cation_offset + anion_index(n_sc, s, p, a, b)
                        for p in range(n_sc)
                    ]
                elif s == 1:
                    chain = [
                        cation_offset + anion_index(n_sc, s, a, p, b)
                        for p in range(n_sc)
                    ]
                else:
                    chain = [
                        cation_offset + anion_index(n_sc, s, a, b, p)
                        for p in range(n_sc)
                    ]
                chains.append(chain)
    return chains


class RowShift(CyclicShift):
    """Shift the species pattern along an anion row by +-1 site.

    Wraps ``mchammer_moves.CyclicShift`` with the anion-chain
    geometry for an ``n_sc x n_sc x n_sc`` ReO3-type supercell.
    Each proposal translates one chain's occupation by +-1 site with
    periodic boundaries, preserving any period-3 (OOF) ordering on
    that chain while changing its phase relative to neighbouring
    chains.

    Args:
        n_sc: Cubic supercell repeat count along each axis.
    """

    def __init__(self, n_sc: int) -> None:
        super().__init__(cycles=build_anion_chains(n_sc))
        self.name = "row_shift"


class MotifShift(CyclicShift):
    """Shift a single OOF period within an anion row by +-1 site.

    Each cycle is a length-3 segment of a chain, aligned to the
    period-3 grid. Rotating a segment creates two domain walls at the
    boundaries with neighbouring periods, but the energy cost is
    local and independent of supercell size. This makes ``MotifShift``
    competitive at large ``n_sc`` where :class:`RowShift` becomes
    prohibitively expensive.

    Args:
        n_sc: Cubic supercell repeat count along each axis.
            Must be divisible by 3.
    """

    def __init__(self, n_sc: int) -> None:
        if n_sc % 3 != 0:
            raise ValueError(f"n_sc must be divisible by 3, got {n_sc}")
        full_chains = build_anion_chains(n_sc)
        segments = []
        for chain in full_chains:
            for start in range(0, len(chain), 3):
                segments.append(chain[start : start + 3])
        super().__init__(cycles=segments)
        self.name = "motif_shift"


class RowReflect(CyclicReflection):
    """Reflect an anion chain around a random pivot.

    Wraps ``mchammer_moves.CyclicReflection`` with the anion-chain
    geometry for an ``n_sc x n_sc x n_sc`` ReO3-type supercell. Each
    proposal picks one chain and one pivot site within it, then
    reflects the chain's species pattern around that pivot (long-range
    species moves: a species at one end can hop to the other end in a
    single accepted move).

    Complements :class:`RowShift`'s nearest-neighbour shifts by opening
    reflection-symmetric pathways through chain configuration space.

    Args:
        n_sc: Cubic supercell repeat count along each axis. Must be at
            least 3 for the reflection to be non-trivial.
    """

    def __init__(self, n_sc: int) -> None:
        super().__init__(cycles=build_anion_chains(n_sc))
        self.name = "row_reflect"


def build_plane_chain_groups(n_sc: int) -> list[list[list[int]]]:
    """Return groups of co-planar parallel chains for ``PlaneShift``.

    For each anion sublattice ``s`` (chains run along axis ``s``) and each
    of the two perpendicular axes, ``n_sc`` groups are built — one per
    cell-coordinate value along the perpendicular axis. Each group
    contains ``n_sc`` chains sharing that perpendicular coordinate (and
    varying the other perpendicular coordinate). Shifting every chain in
    such a group by the same amount along its own chain axis preserves
    the in-phase relationship within the group while shifting it
    coherently relative to the rest of the system.

    Args:
        n_sc: Cubic supercell repeat count along each axis.

    Returns:
        ``6 * n_sc`` plane groups, each a list of ``n_sc`` chains
        (each chain a list of ``n_sc`` global anion indices).
    """
    cation_offset = n_sc**3
    groups: list[list[list[int]]] = []
    for s in range(3):
        perp_axes = tuple(a for a in range(3) if a != s)
        for fixed_pa_idx, fixed_pa in enumerate(perp_axes):
            other_pa = perp_axes[1 - fixed_pa_idx]
            for fixed_val in range(n_sc):
                group = []
                for other_val in range(n_sc):
                    coords = [0, 0, 0]
                    coords[fixed_pa] = fixed_val
                    coords[other_pa] = other_val
                    chain = []
                    for p in range(n_sc):
                        coords[s] = p
                        chain.append(
                            cation_offset
                            + anion_index(n_sc, s, coords[0], coords[1], coords[2])
                        )
                    group.append(chain)
                groups.append(group)
    return groups


class PlaneShift(Move):
    """Shift all anion chains in a 2D plane by +-1 site in concert.

    A "plane group" is a set of ``n_sc`` parallel chains sharing one
    perpendicular cell coordinate within an anion sublattice. Per
    proposal, picks one plane group uniformly at random and one
    direction in {+1, -1} uniformly, then shifts every chain in the
    group by that direction along its own chain axis. Preserves the
    in-phase relationship within the plane while shifting it coherently
    relative to the rest of the system — bridges configurations
    differing by a plane-level phase offset that ``RowShift`` can only
    reach via energetically-unfavourable single-chain intermediates.

    Detailed balance: each (group, direction) pair is selected with
    probability ``1 / (2 * n_groups)`` regardless of configuration.
    The reverse of a +1 shift along group ``g`` is a -1 shift along the
    same group, with the same selection probability. Standard
    Metropolis acceptance preserves detailed balance.

    Args:
        n_sc: Cubic supercell repeat count along each axis.
    """

    def __init__(self, n_sc: int, name: str = "plane_shift") -> None:
        super().__init__(name)
        self._n_sc = n_sc
        self._groups = build_plane_chain_groups(n_sc)

    @property
    def n_groups(self) -> int:
        return len(self._groups)

    def propose(
        self,
        configuration: "ConfigurationManager",
        next_random_number: Callable[[], float],
    ) -> tuple[list[int], list[int]] | None:
        group_idx = int(next_random_number() * len(self._groups))
        direction = 1 if next_random_number() < 0.5 else -1
        group = self._groups[group_idx]
        occupations = configuration.occupations
        L = self._n_sc
        # Direction +1: each position i receives species from i-1.
        offset = -1 if direction == 1 else 1
        sites: list[int] = []
        new_species: list[int] = []
        for chain in group:
            current = [int(occupations[s]) for s in chain]
            shifted = [current[(i + offset) % L] for i in range(L)]
            sites.extend(chain)
            new_species.extend(shifted)
        # Skip identity proposals so the per-move acceptance rate stays
        # meaningful (same convention as CyclicShift).
        current_all = [int(occupations[s]) for s in sites]
        if current_all == new_species:
            return None
        return sites, new_species


def build_anion_planes(n_sc: int) -> list[list[int]]:
    """Return all anion planes perpendicular to each of the three axes.

    For each axis ``a`` and each plane coordinate ``p`` in ``[0, n_sc)``,
    the corresponding plane contains every anion site whose cell coordinate
    along axis ``a`` is ``p``, across all three anion sublattices and all
    other cell coords. Each anion belongs to exactly one plane per axis,
    so the returned list contains ``3 * n_sc`` planes total, each of
    length ``3 * n_sc**2``.

    The plane index sets are pairwise disjoint within each axis (each
    anion has one cell coordinate per axis), but a given anion appears
    in three different planes (one per axis) across the full list.
    Passing the full list to :class:`IndexSetSwap` therefore violates
    its disjointness constraint; pass one axis at a time, or use the
    per-axis helper below.

    Args:
        n_sc: Cubic supercell repeat count along each axis.

    Returns:
        ``3 * n_sc`` planes, each a list of ``3 * n_sc**2`` global
        anion indices.
    """
    return [
        plane
        for axis in range(3)
        for plane in build_anion_planes_axis(n_sc, axis)
    ]


def build_anion_planes_axis(n_sc: int, axis: int) -> list[list[int]]:
    """Return the ``n_sc`` anion planes perpendicular to a single axis.

    Args:
        n_sc: Cubic supercell repeat count.
        axis: 0, 1, or 2 for x-, y-, z-perpendicular planes.

    Returns:
        ``n_sc`` pairwise-disjoint planes, each a list of
        ``3 * n_sc**2`` global anion indices.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    cation_offset = n_sc**3
    planes: list[list[int]] = []
    for p in range(n_sc):
        plane: list[int] = []
        for s in range(3):
            for j in range(n_sc):
                for k in range(n_sc):
                    if axis == 0:
                        idx = anion_index(n_sc, s, p, j, k)
                    elif axis == 1:
                        idx = anion_index(n_sc, s, j, p, k)
                    else:
                        idx = anion_index(n_sc, s, j, k, p)
                    plane.append(cation_offset + idx)
        planes.append(plane)
    return planes


def build_anion_blocks(n_sc: int, block_size: int = 3) -> list[list[int]]:
    """Return all 3D anion sub-blocks of side ``block_size`` tiling the supercell.

    The supercell is tiled into ``(n_sc / block_size) ** 3`` cubic
    sub-blocks. Each sub-block collects every anion site whose cell
    coordinates fall within the block, across all three anion
    sublattices, giving ``3 * block_size ** 3`` indices per block.

    For ``n_sc = 9`` with the default ``block_size = 3``, this gives
    27 blocks of 81 anion sites each — the natural N=3 chiral-orbit
    tiling unit for this system.

    Args:
        n_sc: Cubic supercell repeat count along each axis. Must be
            divisible by ``block_size``.
        block_size: Side length of each sub-block in cell units.

    Returns:
        ``(n_sc / block_size) ** 3`` pairwise-disjoint blocks, each a
        list of ``3 * block_size ** 3`` global anion indices.

    Raises:
        ValueError: If ``n_sc`` is not divisible by ``block_size``.
    """
    if n_sc % block_size != 0:
        raise ValueError(
            f"n_sc ({n_sc}) must be divisible by block_size ({block_size})"
        )
    cation_offset = n_sc**3
    n_blocks_per_axis = n_sc // block_size
    blocks: list[list[int]] = []
    for bi in range(n_blocks_per_axis):
        for bj in range(n_blocks_per_axis):
            for bk in range(n_blocks_per_axis):
                block: list[int] = []
                for s in range(3):
                    for di in range(block_size):
                        for dj in range(block_size):
                            for dk in range(block_size):
                                i = bi * block_size + di
                                j = bj * block_size + dj
                                k = bk * block_size + dk
                                idx = anion_index(n_sc, s, i, j, k)
                                block.append(cation_offset + idx)
                blocks.append(block)
    return blocks
