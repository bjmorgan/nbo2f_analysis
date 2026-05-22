"""Anion-chain enumeration for the ReO3-type NbO2F supercell.

Provides a single function, :func:`build_anion_chains`, that lists the
sites of every anion chain in a cubic ``n_sc x n_sc x n_sc`` supercell
as global atom indices into the array produced by
:func:`ce_tools.atoms_from_f_mask_stable`. The output is suitable for
passing directly to ``mchammer_moves.CyclicShift(cycles=...)``.

The decomposition is deterministic, derived from
:func:`ce_tools.anion_index`'s known site-ordering convention; no
position-based on-lattice classification is performed.
"""

from __future__ import annotations

from .ce_tools import anion_index


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
