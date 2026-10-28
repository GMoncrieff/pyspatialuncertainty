"""Estimated mean of the standardized errors: the correction applied by default.

The method of Wadoux & Heuvelink (2023) assumes that the standardized prediction errors
z = e/σ have mean zero: the map is unbiased, or its class probabilities are calibrated.
A systematic offset then never averages out, so block intervals shrink with block size
while the error they should cover does not.

By default the package treats the mean of z as unknown, as ordinary kriging does:

    z(s) = μ + w(s),  w a zero-mean field with correlogram ρ.

* μ is estimated by generalised least squares from the validation errors,
  μ̂ = 1ᵀK⁻¹z / 1ᵀK⁻¹1, where K is the correlation matrix of the errors at the validation
  points (nugget on the diagonal). Its variance in units of z is A = (1ᵀK⁻¹1)⁻¹.
* The block prediction is shifted by x_B·μ̂, where x_B is the block mean of σ(s).
* The variance of the block-average error becomes

      V_B + x_B²·A − 2·x_B·c_B,

  with V_B the Monte Carlo integral of σ(s)σ(u)ρ(|s − u|) over B × B (as in the paper),
  x_B²·A the uncertainty of μ̂, and c_B = Σ_i G_i · (block mean of σ(s)ρ(|s − x_i|)),
  G = A·1ᵀK⁻¹, the part of the block error that the validation errors already reveal
  (the cross term).

With μ fixed at zero (``mean="zero"``) the shift and the two added terms vanish and the
method is exactly that of the paper. The correlogram should then be fitted with the mean
fixed at zero, and with an estimated mean by REML (``mean="constant"`` in
`fit_correlation`), so that the fit does not count the offset as correlation.

Simulation studies with known truth (the ``improvements`` experiments in the repository)
found the correction calibrated for continuous maps, with or without a systematic offset,
and much better than the zero-mean method for class maps, rare classes included. It adds
the noise of μ̂ to the block estimates, so with a small validation set (less than about
200 points, or a handful of clusters) on a map that is already calibrated it is slightly
less accurate than the zero-mean method.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
from scipy import linalg

from pyspatialuncertainty._types import FloatArray
from pyspatialuncertainty.blocks import ResolvedBlock
from pyspatialuncertainty.grid import GridSpec
from pyspatialuncertainty.variogram.models import VariogramModel

_CHUNK_ENTRIES = 4_000_000  # cells × points per distance-matrix chunk of the cross term

MeanTreatment = Literal["constant", "zero"]
"""How the aggregation functions treat the mean of the standardized errors."""


def variogram_of(correlation: object) -> VariogramModel | None:
    """The variogram model behind a model, fit or correlation object, if there is one."""
    if isinstance(correlation, VariogramModel):
        return correlation
    fitted = getattr(correlation, "model", None)
    return fitted if isinstance(fitted, VariogramModel) else None


def validation_points(
    correlation: object, validation: pl.DataFrame | None, *, what: str = "the correlation"
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validation coordinates and standardized errors for estimating the mean.

    Taken from ``validation`` (columns ``x``, ``y`` and ``std_error``) when given, otherwise
    from the points stored on a fit (`CorrelationFit`, `ClassVariogram`).
    """
    if validation is not None:
        missing = [c for c in ("x", "y", "std_error") if c not in validation.columns]
        if missing:
            raise ValueError(f"validation is missing the columns {missing}")
        x, y, z = (validation[c].cast(pl.Float64).to_numpy() for c in ("x", "y", "std_error"))
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        return x[ok], y[ok], z[ok]
    pts = tuple(getattr(correlation, k, None) for k in ("x", "y", "z"))
    if any(p is None for p in pts):
        raise ValueError(
            f"estimating the mean of the errors (mean='constant', the default) needs the "
            f"validation points: pass {what} as returned by fit_correlation or "
            "fit_class_variograms, or give validation=, or use mean='zero' to assume "
            "mean-zero errors as in Wadoux & Heuvelink (2023)"
        )
    return pts  # type: ignore[return-value]


def check_fit_mean(correlation: object) -> None:
    """Warn when a correlogram fitted with the mean fixed at zero meets an estimated mean."""
    details = getattr(correlation, "details", None) or getattr(correlation, "fit", None)
    if getattr(details, "mean", None) == "zero":
        warnings.warn(
            "the correlogram was fitted with the mean fixed at zero (ML) but the mean is "
            "estimated here; fit it with mean='constant' (REML) so that an offset of the "
            "errors is not also counted as correlation",
            stacklevel=3,
        )


def structured_correlation(model: VariogramModel, h: FloatArray) -> FloatArray:
    """Correlation between distinct points, with the limit 1 − (nugget share) at h = 0.

    The nugget is uncorrelated measurement-scale noise, so two distinct points at the
    same location (or a block point and a validation point) share only the structured
    part of the variance.
    """
    if not model.sill > 0:
        return np.zeros_like(h)
    rho = np.asarray(model.correlation(h), dtype=np.float64)
    return np.where(h > 0, rho, 1.0 - model.nugget / model.sill)


@dataclass(frozen=True)
class MeanEstimate:
    """GLS estimate of the constant mean of the standardized errors.

    Attributes
    ----------
    beta
        μ̂, the estimated mean of the standardized errors.
    var_beta
        A = (1ᵀK⁻¹1)⁻¹, the variance of μ̂ in units of z.
    weights
        G = A·1ᵀK⁻¹, the GLS weights of the validation points (μ̂ = G·z; they sum to one).
    x, y
        Coordinates of the validation points.
    model
        The variogram model whose correlogram defines K.
    """

    beta: float
    var_beta: float
    weights: FloatArray = field(repr=False)
    x: FloatArray = field(repr=False)
    y: FloatArray = field(repr=False)
    model: VariogramModel = field(repr=False)

    @property
    def se(self) -> float:
        """Standard error of μ̂."""
        return float(np.sqrt(self.var_beta))


def estimate_mean(
    model: VariogramModel, x: FloatArray, y: FloatArray, z: FloatArray
) -> MeanEstimate:
    """GLS estimate of a constant mean of standardized errors under a fitted correlogram.

    Parameters
    ----------
    model
        Variogram model of the standardized errors (ideally fitted by REML,
        ``fit_correlation(..., mean="constant")``).
    x, y
        Coordinates of the validation points.
    z
        Standardized errors at the validation points.

    Returns
    -------
    MeanEstimate
        μ̂, its variance and the GLS weights.

    Examples
    --------
    >>> import numpy as np
    >>> from pyspatialuncertainty.variogram.models import vgm
    >>> rng = np.random.default_rng(0)
    >>> x, y = rng.uniform(0, 100, (2, 50))
    >>> z = 0.5 + rng.normal(size=50)
    >>> est = estimate_mean(vgm(1.0, "Nug"), x, y, z)
    >>> # independent errors: the GLS mean is the plain mean and A = 1/n
    >>> bool(np.isclose(est.beta, z.mean())), round(est.var_beta * 50, 6)
    (True, 1.0)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    if not (x.size == y.size == z.size):
        raise ValueError("x, y and z must have the same length")
    if z.size < 2:
        raise ValueError("at least two validation points are needed to estimate the mean")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(z))):
        raise ValueError("validation coordinates and errors must be finite")
    D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    K = structured_correlation(model, D)
    K[np.diag_indices_from(K)] = 1.0 + 1e-10
    try:
        c = linalg.cho_factor(K, lower=True, check_finite=False)
    except linalg.LinAlgError as err:
        raise ValueError(
            "the correlation matrix of the validation points is not positive definite"
        ) from err
    k1 = linalg.cho_solve(c, np.ones(z.size), check_finite=False)
    var_beta = 1.0 / float(np.sum(k1))
    weights = var_beta * k1
    return MeanEstimate(
        beta=float(weights @ z), var_beta=var_beta, weights=weights, x=x, y=y, model=model
    )


@dataclass(frozen=True)
class BlockTerms:
    """Per-block terms of the estimated-mean correction.

    Attributes
    ----------
    sd_mean
        x_B, the block mean of σ.
    shift
        x_B·μ̂, added to the block prediction.
    var_beta
        x_B²·A, the variance added by the uncertainty of μ̂.
    cross
        2·x_B·c_B, the cross term subtracted from the variance (zero when not used).
    """

    sd_mean: FloatArray
    shift: FloatArray
    var_beta: FloatArray
    cross: FloatArray


def block_terms(
    estimate: MeanEstimate,
    sd: FloatArray,
    grid: GridSpec,
    blocks: Sequence[ResolvedBlock],
    *,
    cross_term: bool = True,
) -> BlockTerms:
    """Shift and variance terms of the corrected block estimates.

    Block means are taken over the block's cells (cell centres for the cross term), which
    is exact for blocks made of whole cells with σ constant in each cell.

    Parameters
    ----------
    estimate
        The GLS mean of the standardized errors (`estimate_mean`).
    sd
        Flat (row-major) σ per cell of ``grid``.
    grid
        The grid of the blocks.
    blocks
        Blocks resolved against the validity mask.
    cross_term
        Subtract the cross term 2·x_B·c_B (default). Without it the variance is larger.

    Returns
    -------
    BlockTerms
        x_B, the shift and the added variance terms, one value per block.
    """
    xb = np.array([float(np.mean(sd[b.cells])) if b.cells.size else np.nan for b in blocks])
    shift = xb * estimate.beta
    var_beta = xb**2 * estimate.var_beta
    cross = np.zeros(len(blocks))
    if cross_term and len(blocks):
        cells = np.unique(np.concatenate([b.cells for b in blocks]))
        cx, cy = grid.cell_centres(cells)
        v = np.empty(cells.size)
        step = max(1, _CHUNK_ENTRIES // max(1, estimate.x.size))
        for a in range(0, cells.size, step):
            b_ = min(a + step, cells.size)
            h = np.hypot(cx[a:b_, None] - estimate.x[None, :], cy[a:b_, None] - estimate.y[None, :])
            v[a:b_] = structured_correlation(estimate.model, h) @ estimate.weights
        field_ = np.full(sd.size, np.nan)
        field_[cells] = sd[cells] * v
        cb = np.array([float(np.mean(field_[b.cells])) if b.cells.size else np.nan for b in blocks])
        cross = 2.0 * xb * cb
    return BlockTerms(sd_mean=xb, shift=shift, var_beta=var_beta, cross=cross)
