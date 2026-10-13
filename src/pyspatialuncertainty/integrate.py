"""Monte Carlo integration of the block-average error variance (steps 5-6 of the method).

For a block B, the variance of the block-averaged prediction error is (Eq. 7 of
Wadoux & Heuvelink, 2023)

.. math::

    \\operatorname{var}\\Big(\\frac{1}{|B|}\\int_B \\varepsilon(s)\\,ds\\Big)
      = \\frac{1}{|B|^2}\\int_B\\int_B \\sigma(s)\\,\\sigma(u)\\,\\rho(|s-u|)\\,ds\\,du ,

which equals the expectation of g(S, U) = σ(S)σ(U)ρ(|S − U|) for S and U independent and
uniformly distributed on B. It is estimated by the mean of g over m independent pairs of
uniform points, exactly as in the paper and the authors' R script. The Monte Carlo standard
error of the estimate, sd(g)/√m, is reported alongside it.

Several integrands (pairs of σ maps and correlation functions) can be evaluated on the
same random point pairs, for instance one per class in categorical mode, or the fitted and
the true correlation function in a simulation study. Sharing the pairs (common random
numbers) makes differences between integrands less noisy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pyspatialuncertainty._types import FloatArray, SeedLike
from pyspatialuncertainty.blocks import ResolvedBlock
from pyspatialuncertainty.correlation import CorrelationFunction


@dataclass(frozen=True)
class Integrand:
    """σ(s)σ(u)ρ(|s − u|) for a σ map and a correlation function.

    Attributes
    ----------
    sd
        Flat (row-major) array of point-support prediction-error standard deviations per cell.
    correlation
        Correlation function of the standardized errors.
    """

    sd: FloatArray
    correlation: CorrelationFunction


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimates of block-average error variances.

    Attributes
    ----------
    variance
        Array of shape (n_integrands, n_blocks): estimated variance of the block-average error.
    mcse
        Monte Carlo standard error of ``variance``.
    n_mc
        Number of point pairs per block.
    """

    variance: FloatArray
    mcse: FloatArray
    n_mc: int


def seed_sequence(seed: SeedLike) -> np.random.SeedSequence:
    """Convert any accepted seed to a ``SeedSequence``."""
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if isinstance(seed, np.random.Generator):
        return np.random.SeedSequence(int(seed.integers(0, 2**63 - 1)))
    return np.random.SeedSequence(seed)


def mc_block_variance(
    blocks: Sequence[ResolvedBlock],
    integrands: Sequence[Integrand],
    *,
    n_mc: int = 10_000,
    seed: SeedLike = None,
    chunk: int = 200_000,
) -> MCResult:
    """Estimate the variance of the block-average error by Monte Carlo integration.

    Parameters
    ----------
    blocks
        Blocks resolved against the validity mask (see
        :func:`~pyspatialuncertainty.blocks.resolve_blocks`).
    integrands
        One or more (σ map, correlation function) pairs, evaluated on the same point pairs.
    n_mc
        Number of independent pairs of uniform points per block. The paper uses 10,000.
    seed
        Seed for the random number generator. Each block gets its own stream derived from
        it, so results do not depend on how the work is chunked.
    chunk
        Maximum number of pairs drawn at once.

    Returns
    -------
    MCResult
        Variance estimates and their Monte Carlo standard errors, shape
        (len(integrands), len(blocks)).
    """
    if n_mc < 2:
        raise ValueError("n_mc must be at least 2")
    k = len(integrands)
    if k == 0:
        raise ValueError("at least one integrand is required")
    variance = np.empty((k, len(blocks)))
    mcse = np.empty((k, len(blocks)))
    children = seed_sequence(seed).spawn(len(blocks))
    for b, (block, child) in enumerate(zip(blocks, children, strict=True)):
        rng = np.random.default_rng(child)
        count = 0
        mean = np.zeros(k)
        m2 = np.zeros(k)
        while count < n_mc:
            m = min(chunk, n_mc - count)
            x1, y1, c1 = block.sampler(m, rng)
            x2, y2, c2 = block.sampler(m, rng)
            dist = np.hypot(x1 - x2, y1 - y2)
            rho_cache: dict[int, FloatArray] = {}
            for j, ig in enumerate(integrands):
                key = id(ig.correlation)
                if key not in rho_cache:
                    rho_cache[key] = np.asarray(ig.correlation(dist), dtype=np.float64)
                g = ig.sd[c1] * ig.sd[c2] * rho_cache[key]
                # Chan et al. parallel update of mean and sum of squared deviations
                g_mean = float(g.mean())
                g_m2 = float(np.sum((g - g_mean) ** 2))
                delta = g_mean - mean[j]
                total = count + m
                mean[j] += delta * m / total
                m2[j] += g_m2 + delta**2 * count * m / total
            count += m
        if not np.all(np.isfinite(mean)):
            raise ValueError(
                f"non-finite integrand in block {block.id!r}; check for NaN standard deviations"
            )
        variance[:, b] = mean
        mcse[:, b] = np.sqrt(m2 / (count - 1) / count)
    return MCResult(variance=variance, mcse=mcse, n_mc=n_mc)
