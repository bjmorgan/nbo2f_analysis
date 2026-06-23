"""Reference values for the NbO2F structural order parameters.

Closed-form random/limiting and exact ground-state reference values for the
order parameters emitted by :class:`nbo2f_analysis.chain_order_observer.
ChainOrderObserver`. These back the Methods OP definitions in the
chirality-emergence paper and ship as tested, citable code rather than a
loose analysis script.

For each referenced OP the module provides, where it exists, the exact
independent-site random limit (with its combinatorial derivation in the
function docstring), the finite-size scaling where the limit is zero, the
exact P3_121 ground-state anchor, and a Monte-Carlo random reference
computed through the *production* observer. ``main`` emits the whole table
as a provenance-headed CSV.

Random reference = fixed composition f_F = 1/3 (one third of anions are F)
with F placed at random; ground state = the tiled chiral P3_121 ordering.
"""
from __future__ import annotations

import csv
import importlib.metadata
import math
import sys
from fractions import Fraction
from typing import TextIO

import numpy as np

from nbo2f_analysis.ce_tools import (
    atoms_from_f_mask_stable,
    build_tiled_groundstate_atoms,
)
from nbo2f_analysis.chain_order_observer import build_chain_order_observer

# The OPs this module references, in CSV row order: the four manuscript
# structural OPs, plus collinear_ff and the chiral chi_11.
REFERENCE_OPS: tuple[str, ...] = (
    "oof_amp", "icoh_global", "cis_frac", "nbo4f2_frac", "collinear_ff",
    "chi_11",
)

# NbO2F composition: one third of the anions are F.
DEFAULT_F: Fraction = Fraction(1, 3)


def random_local_limits(f: Fraction = DEFAULT_F) -> dict[str, Fraction]:
    """Exact independent-site random limits of the local-coordination OPs.

    At fixed composition f_F = f with F placed at random, each octahedral
    vertex is independently F with probability f in the large-cell limit
    (fixed-composition finite-size corrections are O(1/L) and vanish). The
    closed forms follow from counting:

    - ``nbo4f2_frac`` = P(exactly 2 of the 6 octahedral vertices are F)
      = C(6, 2) f^2 (1 - f)^4.
    - ``cis_frac`` = that, restricted to cis placements. The 6 vertices form
      3 trans (opposite-axis) pairs, so of the C(6, 2) = 15 two-F placements
      3 are trans and 12 are cis: cis_frac = 12 f^2 (1 - f)^4.
    - ``collinear_ff`` = P(two adjacent chain sites are both F) = f^2 (the
      (1, 1) 2-motif frequency along a chain).

    At f = 1/3 these are 80/243, 64/243 and 1/9.

    Args:
        f: F fraction, an exact ``Fraction``. Defaults to 1/3.

    Returns:
        Exact ``Fraction`` values keyed by observer OP name
        (``nbo4f2_frac``, ``cis_frac``, ``collinear_ff``).
    """
    f = Fraction(f)
    placement = f**2 * (1 - f) ** 4   # probability of one specific 2-F-of-6 set
    n_two_f = math.comb(6, 2)         # 15 ways to place 2 F on 6 vertices
    n_cis = n_two_f - 3               # 12 cis (3 of the 15 are trans pairs)
    return {
        "nbo4f2_frac": n_two_f * placement,
        "cis_frac": n_cis * placement,
        "collinear_ff": f**2,
    }


def oof_amp_random(n_sc: int, f: float = 1 / 3) -> float:
    """Rayleigh floor of the random period-3 Fourier magnitude ``oof_amp``.

    ``chainorder.order_params.chain_fft`` normalises by the chain length N,
    so the period-3 (k = N/3) coefficient of a random chain has zero-mean
    real and imaginary parts, each with variance f(1 - f)/(2N) and zero
    covariance. Its magnitude is therefore Rayleigh-distributed with mean
    sigma * sqrt(pi/2), i.e.

        <oof_amp> = 0.5 * sqrt(pi * f * (1 - f) / N),

    which decays to 0 as N -> inf. Here N is the supercell side ``n_sc``.
    Checks against the production observer: L6 0.171 (MC 0.166), L9 0.139
    (0.139), L12 0.121 (0.119).

    Args:
        n_sc: Supercell side (the chain length N).
        f: F fraction. Defaults to 1/3.

    Returns:
        The expected random ``oof_amp`` at this size.
    """
    return 0.5 * math.sqrt(math.pi * f * (1 - f) / n_sc)


def ground_state_reference(n_sc: int) -> dict[str, float]:
    """Exact P3_121 ground-state values of the reference OPs.

    Builds the tiled chiral ground state and evaluates the production
    observer, so the anchors are pinned against the same code the manuscript
    runs. The values are L-independent:

        chi_11 = 4/9, icoh_global = 1, oof_amp = 1/3,
        cis_frac = nbo4f2_frac = 1, collinear_ff = 0.

    ``chi_11 = 4/9`` (not 1) because an orbit and its enantiomer share 5/9 of
    sites.

    Args:
        n_sc: Supercell side (a multiple of 3).

    Returns:
        Observer output for :data:`REFERENCE_OPS` on the tiled ground state.
    """
    observer = build_chain_order_observer(n_sc, interval=1, ops=REFERENCE_OPS)
    return observer.get_observable(build_tiled_groundstate_atoms(n_sc=n_sc))


def _random_fixed_composition_mask(
    n_sc: int, rng: np.random.Generator
) -> np.ndarray:
    """Boolean F-mask with exactly n_sc**3 F on 3 * n_sc**3 anion sites."""
    n_anion = 3 * n_sc**3
    mask = np.zeros(n_anion, dtype=bool)
    mask[rng.choice(n_anion, size=n_sc**3, replace=False)] = True
    return mask


def monte_carlo_random_reference(
    n_sc: int,
    n_samples: int,
    *,
    seed: int,
    ops: tuple[str, ...] = REFERENCE_OPS,
) -> dict[str, tuple[float, float]]:
    """Monte-Carlo random reference: (mean, SEM) per OP via the observer.

    Draws ``n_samples`` random fixed-composition (f_F = 1/3) F-masks, builds
    each as a stable-ordered structure (the ordering the observer's
    cis/NbO4F2 branch requires), and evaluates the production observer.
    ``chi_11`` is signed and averages to ~0 by Z2 symmetry.

    Args:
        n_sc: Supercell side (a multiple of 3).
        n_samples: Number of random configurations; >= 2 for a defined SEM.
        seed: Seed for the configuration RNG.
        ops: OPs to evaluate; defaults to :data:`REFERENCE_OPS`.

    Returns:
        ``{op: (mean, sem)}`` over the samples, sem = std / sqrt(n_samples).

    Raises:
        ValueError: if ``n_samples`` < 2.
    """
    if n_samples < 2:
        raise ValueError(
            f"n_samples must be >= 2 for a defined SEM, got {n_samples}"
        )
    observer = build_chain_order_observer(n_sc, interval=1, ops=ops)
    rng = np.random.default_rng(seed)
    samples = {op: np.empty(n_samples) for op in ops}
    for s in range(n_samples):
        mask = _random_fixed_composition_mask(n_sc, rng)
        out = observer.get_observable(atoms_from_f_mask_stable(n_sc, mask))
        for op in ops:
            samples[op][s] = out[op]
    return {
        op: (float(v.mean()), float(v.std(ddof=1) / math.sqrt(n_samples)))
        for op, v in samples.items()
    }


def _analytic_random(op: str, n_sc: int, local: dict[str, Fraction]) -> float:
    """Analytic random reference for ``op`` at size ``n_sc``.

    Local-coordination OPs use their exact (L-independent) limit; ``oof_amp``
    uses the Rayleigh floor; ``icoh_global`` and ``chi_11`` have analytic
    limit 0 (their finite-size value shows up in the MC column).
    """
    if op in local:
        return float(local[op])
    if op == "oof_amp":
        return oof_amp_random(n_sc)
    if op in ("icoh_global", "chi_11"):
        return 0.0
    raise KeyError(f"no analytic random reference for {op!r}")


def reference_table(
    sizes: tuple[int, ...], n_samples: int, *, seed: int
) -> list[dict[str, object]]:
    """Long-format reference rows: one per (OP, size).

    Columns: ``op, n_sc, analytic_random, mc_mean, mc_sem, ground_state``.
    ``ground_state`` is L-independent and repeated per size for a tidy table.

    Args:
        sizes: Supercell sides to tabulate (each a multiple of 3).
        n_samples: MC samples per size.
        seed: MC RNG seed, shared across sizes.

    Returns:
        Rows ordered by size, then :data:`REFERENCE_OPS`.
    """
    local = random_local_limits()
    rows: list[dict[str, object]] = []
    for n_sc in sizes:
        gs = ground_state_reference(n_sc)
        mc = monte_carlo_random_reference(n_sc, n_samples, seed=seed)
        for op in REFERENCE_OPS:
            mean, sem = mc[op]
            rows.append({
                "op": op,
                "n_sc": n_sc,
                "analytic_random": _analytic_random(op, n_sc, local),
                "mc_mean": mean,
                "mc_sem": sem,
                "ground_state": gs[op],
            })
    return rows


_CSV_FIELDS = ("op", "n_sc", "analytic_random", "mc_mean", "mc_sem",
               "ground_state")

# Defaults for the standalone CSV script. The size grid and sample count are
# module constants (no CLI): edit here or call reference_table directly for
# other grids.
DEFAULT_SIZES = (6, 9, 12)
DEFAULT_N_SAMPLES = 500
DEFAULT_SEED = 0


def provenance_lines(
    sizes: tuple[int, ...], n_samples: int, seed: int
) -> list[str]:
    """Commented header lines recording versions and run parameters.

    Embedding the provenance as leading ``#`` lines keeps the version and
    parameter record travelling with the citable CSV.
    """
    return [
        f"# nbo2f-analysis {importlib.metadata.version('nbo2f-analysis')}",
        f"# chainorder {importlib.metadata.version('chainorder')}",
        f"# f_F=1/3 sizes={list(sizes)} n_samples={n_samples} seed={seed}",
    ]


def write_csv(
    rows: list[dict[str, object]], file: TextIO, *, provenance: list[str]
) -> None:
    """Write the provenance comment lines, then the CSV table, to ``file``."""
    for line in provenance:
        file.write(line + "\n")
    writer = csv.DictWriter(file, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    """Emit the reference table as a provenance-headed CSV on stdout."""
    rows = reference_table(DEFAULT_SIZES, DEFAULT_N_SAMPLES, seed=DEFAULT_SEED)
    write_csv(
        rows,
        sys.stdout,
        provenance=provenance_lines(
            DEFAULT_SIZES, DEFAULT_N_SAMPLES, DEFAULT_SEED
        ),
    )


if __name__ == "__main__":
    main()
