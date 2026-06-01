"""Whole-cell ``<100>`` mirror geometry on the NbO2F anion sublattice.

Provides :func:`build_reflection_permutation`, which reduces a ``<100>``
reflection to an anion-site index map, and :class:`CellReflect`, a
``mchammer_moves.SitePermutation`` subclass that offers the three
axis-orientation reflections as a trial move. A ``<100>`` mirror is an
exact symmetry of the parent ReO3 lattice and of the cluster expansion,
so reflecting any configuration yields an iso-energetic configuration in
the enantiomeric chiral basin; the move bridges the two degenerate
handednesses of the ordered phase that local moves separate by a large
barrier.

Cations are a single species and absent from the cluster expansion, so
the reflection permutes anion occupations only.
"""
from __future__ import annotations

from mchammer_moves import SitePermutation

from .ce_tools import anion_index


def build_reflection_permutation(n_sc: int, axis: int) -> dict[int, int]:
    """Map each anion site to its image under a ``<100>`` reflection.

    Reflects each anion site to its mirror image across the plane
    perpendicular to ``axis`` through the origin: the fractional
    coordinate along ``axis`` is negated (modulo the cell) and the other
    two are unchanged. The sublattice ``s`` is preserved.

    The sublattice whose edge-midpoint offset lies along ``axis``
    (``s == axis``) has a half-integer coordinate there and maps as
    ``c -> n_sc - 1 - c``; the other two sublattices have an integer
    coordinate and map as ``c -> (n_sc - c) % n_sc``.

    Indices are global (the ``n_sc**3`` cation offset is included), so the
    returned map can be passed straight to :class:`SitePermutation`. Sites
    lying on the mirror plane map to themselves and are omitted, since a
    fixed point is not a valid ``SitePermutation`` entry.

    Args:
        n_sc: Cubic supercell repeat count along each axis.
        axis: Reflection-plane normal: 0, 1, or 2 for x, y, z.

    Returns:
        ``{global_site: global_image}`` over moved anion sites only.

    Raises:
        ValueError: If ``axis`` is not 0, 1, or 2.
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    cation_offset = n_sc**3
    perm: dict[int, int] = {}
    for s in range(3):
        for i in range(n_sc):
            for j in range(n_sc):
                for k in range(n_sc):
                    image = [i, j, k]
                    c = image[axis]
                    image[axis] = (
                        n_sc - 1 - c if s == axis else (n_sc - c) % n_sc
                    )
                    src = cation_offset + anion_index(n_sc, s, i, j, k)
                    dst = cation_offset + anion_index(
                        n_sc, s, image[0], image[1], image[2]
                    )
                    if src != dst:
                        perm[src] = dst
    return perm
