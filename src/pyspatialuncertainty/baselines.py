"""Naive baselines for the standard deviation of a block average.

Both ignore the actual spatial correlation of the prediction errors:

* :func:`sd_average` averages the point standard deviations over the block. This is what
  one gets by assuming perfectly correlated errors (ρ ≡ 1); it overestimates the
  uncertainty of block averages (Wadoux & Heuvelink, 2023, Section 1).
* :func:`sd_independent` assumes that the errors in different cells are uncorrelated
  (ρ = 0 for distinct cells), giving sqrt(Σ σ_i²) / n; it underestimates the uncertainty,
  increasingly so as the number of cells grows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pyspatialuncertainty._types import FloatArray
from pyspatialuncertainty.blocks import ResolvedBlock


def sd_average(blocks: Sequence[ResolvedBlock], sd: FloatArray) -> FloatArray:
    """Block mean of the point standard deviations (perfect-correlation baseline)."""
    return np.array([float(np.mean(sd[b.cells])) for b in blocks])


def sd_independent(blocks: Sequence[ResolvedBlock], sd: FloatArray) -> FloatArray:
    """sqrt(Σ σ_i²) / n over the cells of each block (independence baseline)."""
    return np.array([float(np.sqrt(np.sum(sd[b.cells] ** 2)) / b.cells.size) for b in blocks])
