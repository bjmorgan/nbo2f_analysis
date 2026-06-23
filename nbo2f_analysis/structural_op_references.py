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
DEFAULT_F = Fraction(1, 3)
