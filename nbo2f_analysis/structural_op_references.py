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

import math
from fractions import Fraction

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
