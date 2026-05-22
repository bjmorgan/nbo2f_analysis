"""Shared utilities for cluster expansion analysis of NbO2F.

Provides atoms builders, F/O mask manipulation, CE loading, and
structure generators for the ReO3-type NbO2F anion sublattice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from icet import ClusterExpansion

A = 3.902  # cubic lattice parameter (angstrom), matching CE primitive
CE_PATH = Path(__file__).parent / "paircut9_rfe.ce"


def _resolve_ce_path(ce: str | Path | None = None) -> Path:
    """Resolve a CE path argument, falling back to the module-level default."""
    if ce is None:
        return CE_PATH
    p = Path(ce)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    return p


def anion_index(n_sc: int, s: int, i: int, j: int, k: int) -> int:
    """Linear index for anion at sublattice s with cell coords i, j, k.

    Anion ordering: s=0 sites first (x-axis edge midpoints),
    then s=1 (y-axis), then s=2 (z-axis). Within each sublattice,
    indices run over (i, j, k) in row-major order with i as the
    slowest index. Total anion count is 3 * n_sc**3.
    """
    return s * n_sc**3 + i * n_sc**2 + j * n_sc + k


def index_to_anion(n_sc: int, idx: int) -> tuple[int, int, int, int]:
    """Invert `anion_index`."""
    s, rem = divmod(idx, n_sc**3)
    i, rem = divmod(rem, n_sc**2)
    j, k = divmod(rem, n_sc)
    return s, i, j, k


def nb_anion_neighbours(n_sc: int) -> np.ndarray:
    """Return a (n_sc**3, 6) array giving the 6 anion neighbours of each Nb.

    Nb at cell (i, j, k) has anion neighbours at:
        +x: (s=0, i,            j, k)
        -x: (s=0, (i-1) % n_sc, j, k)
        +y: (s=1, i, j,            k)
        -y: (s=1, i, (j-1) % n_sc, k)
        +z: (s=2, i, j, k)
        -z: (s=2, i, j, (k-1) % n_sc)

    Neighbour column order: [+x, -x, +y, -y, +z, -z].
    """
    result = np.empty((n_sc**3, 6), dtype=int)
    for i in range(n_sc):
        for j in range(n_sc):
            for k in range(n_sc):
                nb = i * n_sc**2 + j * n_sc + k
                result[nb, 0] = anion_index(n_sc, 0, i, j, k)
                result[nb, 1] = anion_index(n_sc, 0, (i - 1) % n_sc, j, k)
                result[nb, 2] = anion_index(n_sc, 1, i, j, k)
                result[nb, 3] = anion_index(n_sc, 1, i, (j - 1) % n_sc, k)
                result[nb, 4] = anion_index(n_sc, 2, i, j, k)
                result[nb, 5] = anion_index(n_sc, 2, i, j, (k - 1) % n_sc)
    return result


def atoms_from_f_mask(n_sc: int, f_mask: np.ndarray) -> Atoms:
    """Build an NbO2F supercell from an anion F/O occupation mask.

    Species ordering in the output: all Nb, then all O, then all F.
    This matches the POSCAR convention used in
    ``structures/orbit_representatives/``.

    Args:
        n_sc: supercell side in primitive cells.
        f_mask: length-(3 * n_sc**3) boolean array; True -> F, False -> O.

    Returns:
        ASE Atoms with cubic cell of side n_sc * A, pbc=True.
    """
    assert f_mask.shape == (3 * n_sc**3,)
    positions: list[list[float]] = []
    symbols: list[str] = []

    for i in range(n_sc):
        for j in range(n_sc):
            for k in range(n_sc):
                positions.append([i * A, j * A, k * A])
                symbols.append("Nb")

    o_positions: list[list[float]] = []
    f_positions: list[list[float]] = []
    for s in range(3):
        for i in range(n_sc):
            for j in range(n_sc):
                for k in range(n_sc):
                    pos = [i * A, j * A, k * A]
                    pos[s] += 0.5 * A
                    is_F = bool(f_mask[anion_index(n_sc, s, i, j, k)])
                    (f_positions if is_F else o_positions).append(pos)

    positions.extend(o_positions)
    symbols.extend(["O"] * len(o_positions))
    positions.extend(f_positions)
    symbols.extend(["F"] * len(f_positions))

    return Atoms(
        symbols=symbols,
        positions=positions,
        cell=np.diag([n_sc * A] * 3),
        pbc=True,
    )


def atoms_from_f_mask_stable(n_sc: int, f_mask: np.ndarray) -> Atoms:
    """Like ``atoms_from_f_mask`` but with mask-independent site indexing.

    Anions are placed in a fixed ``(s, i, j, k)`` iteration order with
    species determined by ``f_mask`` (rather than grouped into an O-block
    followed by an F-block). Site indices therefore map to fixed
    spatial positions regardless of the mask contents, which is
    required for mchammer-based MC work.
    """
    assert f_mask.shape == (3 * n_sc**3,)
    positions: list[list[float]] = []
    symbols: list[str] = []
    for i in range(n_sc):
        for j in range(n_sc):
            for k in range(n_sc):
                positions.append([i * A, j * A, k * A])
                symbols.append("Nb")
    for s in range(3):
        for i in range(n_sc):
            for j in range(n_sc):
                for k in range(n_sc):
                    pos = [i * A, j * A, k * A]
                    pos[s] += 0.5 * A
                    positions.append(pos)
                    is_F = bool(f_mask[anion_index(n_sc, s, i, j, k)])
                    symbols.append("F" if is_F else "O")
    return Atoms(
        symbols=symbols,
        positions=positions,
        cell=np.diag([n_sc * A] * 3),
        pbc=True,
    )


def _f_mask_from_atoms(atoms: Atoms, n_sc: int) -> np.ndarray:
    """Invert ``atoms_from_f_mask``: recover the F-occupation mask.

    Walks every canonical anion site (s, i, j, k), computes its
    fractional coordinate, and checks whether an F atom sits at
    that site (modulo PBC).
    """
    mask = np.zeros(3 * n_sc**3, dtype=bool)

    symbols = atoms.get_chemical_symbols()
    scaled = atoms.get_scaled_positions(wrap=True)
    f_scaled = np.array(
        [scaled[a] for a, sym in enumerate(symbols) if sym == "F"]
    )

    for s in range(3):
        for i in range(n_sc):
            for j in range(n_sc):
                for k in range(n_sc):
                    expected_cart = np.array(
                        [i * A, j * A, k * A], dtype=float
                    )
                    expected_cart[s] += 0.5 * A
                    expected_frac = expected_cart / (n_sc * A)
                    expected_frac = np.mod(expected_frac, 1.0)
                    if len(f_scaled) == 0:
                        continue
                    diff = f_scaled - expected_frac
                    diff -= np.round(diff)
                    hit = np.any(np.all(np.abs(diff) < 1e-6, axis=1))
                    if hit:
                        mask[anion_index(n_sc, s, i, j, k)] = True
    return mask


def make_random(n_sc: int, rng: np.random.Generator) -> Atoms:
    """Uniformly random F/O occupation at NbO2F stoichiometry."""
    n_anion = 3 * n_sc**3
    n_f = n_sc**3
    f_sites = rng.choice(n_anion, size=n_f, replace=False)
    mask = np.zeros(n_anion, dtype=bool)
    mask[f_sites] = True
    return atoms_from_f_mask(n_sc, mask)


def make_oof_random_phase(n_sc: int, rng: np.random.Generator) -> Atoms:
    """Period-3 F placement along each anion chain, random per-chain phase.

    Requires ``n_sc`` divisible by 3.
    """
    if n_sc % 3 != 0:
        raise ValueError(f"n_sc must be divisible by 3, got {n_sc}")
    mask = np.zeros(3 * n_sc**3, dtype=bool)
    for s in range(3):
        for a in range(n_sc):
            for b in range(n_sc):
                phase = int(rng.integers(0, 3))
                for p in range(n_sc):
                    if p % 3 != phase:
                        continue
                    if s == 0:
                        idx = anion_index(n_sc, 0, p, a, b)
                    elif s == 1:
                        idx = anion_index(n_sc, 1, a, p, b)
                    else:
                        idx = anion_index(n_sc, 2, a, b, p)
                    mask[idx] = True
    return atoms_from_f_mask(n_sc, mask)


# Opposite-axis-pair positions in the 6-vector [+x, -x, +y, -y, +z, -z]:
_TRANS_PAIRS: frozenset[frozenset[int]] = frozenset({
    frozenset({0, 1}),
    frozenset({2, 3}),
    frozenset({4, 5}),
})

_ALL_CIS_N2_CACHE: list[tuple[int, ...]] | None = None


def _enumerate_all_cis_N2() -> list[tuple[int, ...]]:
    """Return every all-cis F-site choice at n_sc = 2 as 8-tuples."""
    from itertools import combinations

    neighbours = nb_anion_neighbours(2)  # (8, 6)
    valid: list[tuple[int, ...]] = []
    f_mask = np.zeros(24, dtype=bool)
    for f_sites in combinations(range(24), 8):
        f_mask[:] = False
        f_mask[list(f_sites)] = True
        f_per_nb = f_mask[neighbours].sum(axis=1)
        if np.any(f_per_nb != 2):
            continue
        ok = True
        for b in range(8):
            positions = np.where(f_mask[neighbours[b]])[0]
            pair = frozenset({int(positions[0]), int(positions[1])})
            if pair in _TRANS_PAIRS:
                ok = False
                break
        if ok:
            valid.append(f_sites)
    return valid


def make_all_cis_N2(rng: np.random.Generator) -> Atoms:
    """Sample uniformly from the all-cis configurations at n_sc = 2."""
    global _ALL_CIS_N2_CACHE
    if _ALL_CIS_N2_CACHE is None:
        _ALL_CIS_N2_CACHE = _enumerate_all_cis_N2()
        if not _ALL_CIS_N2_CACHE:
            raise RuntimeError(
                "no all-cis configurations found at n_sc = 2; "
                "enumeration or cis definition is wrong."
            )
    idx = int(rng.integers(len(_ALL_CIS_N2_CACHE)))
    f_sites = _ALL_CIS_N2_CACHE[idx]
    mask = np.zeros(24, dtype=bool)
    mask[list(f_sites)] = True
    return atoms_from_f_mask(2, mask)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_ORBIT_DIR = _REPO_ROOT / "structures" / "orbit_representatives"

_ORBIT_FILENAMES = {
    0: "orbit_00_sg001_P1_stab01.vasp",
    1: "orbit_01_sg001_P1_stab01.vasp",
    2: "orbit_02_sg144_P3_1_stab09.vasp",
    3: "orbit_03_sg001_P1_stab01.vasp",
    4: "orbit_04_sg005_C2_stab02.vasp",
    5: "orbit_05_sg005_C2_stab02.vasp",
    6: "orbit_06_sg001_P1_stab01.vasp",
    7: "orbit_07_sg001_P1_stab01.vasp",
    8: "orbit_08_sg001_P1_stab01.vasp",
    9: "orbit_09_sg001_P1_stab01.vasp",
    10: "orbit_10_sg005_C2_stab06.vasp",
    11: "orbit_11_sg152_P3_121_stab54.vasp",
}


def load_orbit_rep(index: int) -> Atoms:
    """Load orbit representative ``index`` and rescale to a = A (3.902)."""
    from ase.io import read

    filename = _ORBIT_FILENAMES[index]
    atoms = read(_ORBIT_DIR / filename, format="vasp")
    assert len(atoms) == 108, (
        f"orbit {index}: expected 108 atoms (27 Nb + 54 O + 27 F), got {len(atoms)}"
    )
    cell_diag = np.diag(atoms.cell.array)
    assert np.allclose(cell_diag, 3 * 3.9, atol=1e-6) and np.allclose(
        atoms.cell.array, np.diag(cell_diag)
    ), (
        f"orbit {index}: expected cubic cell with a = 3*3.9 = 11.7 A, "
        f"got cell {atoms.cell.array}"
    )
    scaled = atoms.get_scaled_positions()
    n_sc = 3
    atoms.set_cell(np.diag([n_sc * A] * 3), scale_atoms=False)
    atoms.set_scaled_positions(scaled)
    return atoms


def tile(atoms: Atoms, m: int) -> Atoms:
    """Return an m x m x m supercell tiling of ``atoms``."""
    return atoms.repeat((m, m, m))


def energy_per_atom(ce: ClusterExpansion, atoms: Atoms) -> float:
    """CE energy per atom for ``atoms``."""
    return float(ce.predict(atoms))


def load_ce(ce: str | Path | None = None) -> ClusterExpansion:
    """Load the cluster expansion model.

    Args:
        ce: Optional path to a ``.ce`` file. If ``None``, uses ``CE_PATH``.
            Relative paths are resolved against this script's directory.
    """
    return ClusterExpansion.read(str(_resolve_ce_path(ce)))
