"""Synthetic scenarios with known truth for continuous maps.

* `ErrorModelScenario` -- a fixed map with spatially correlated prediction errors
  e(s) = σ(s)·ε*(s) of known structure: σ is non-stationary and ε* is stationary with a
  known correlation function (nugget + structured part). This isolates steps 3-8 of the
  method: the true σ is known, can be misspecified on purpose, and the exact variance of
  every block-average error can be computed.
* `PipelineScenario` -- the design of the authors' R example (a linear trend plus a
  Gaussian random field, sampled at n points, mapped with a quantile regression forest and
  10-fold cross-validation), re-implemented in Python. Each realization draws a new field
  and sample, so the whole pipeline (steps 1-8) is evaluated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import polars as pl
import xarray as xr

from pyspatialuncertainty._types import FloatArray, IntArray
from pyspatialuncertainty.correlation import CorrelationFunction, VariogramCorrelation
from pyspatialuncertainty.standardize import sd_from_quantiles
from pyspatialuncertainty.synthetic.grf import CirculantEmbedding
from pyspatialuncertainty.synthetic.layout import EvaluationBlocks, SubGridLayout
from pyspatialuncertainty.synthetic.truth import exact_block_variance
from pyspatialuncertainty.variogram.models import VariogramModel, vgm


def unit_sill(model: VariogramModel) -> VariogramModel:
    """Rescale a variogram model to unit sill (a correlogram)."""
    s = model.sill
    return model.replace_parameters(
        [st.psill / s for st in model.structures], [st.range for st in model.structures]
    )


def without_nugget(model: VariogramModel) -> VariogramModel:
    """The model with its nugget structures removed."""
    return VariogramModel(tuple(s for s in model.structures if s.model != "Nug"))


def default_sigma_surface(
    layout: SubGridLayout, sd_min: float = 0.5, sd_max: float = 2.0
) -> FloatArray:
    """A smooth, non-stationary standard deviation surface on the cell grid (north-up)."""
    x = layout.cell_x / layout.extent
    y = layout.cell_y / layout.extent
    xx, yy = np.meshgrid(x, y)
    u = 0.5 + 0.5 * np.sin(2 * math.pi * xx + 0.3) * np.cos(math.pi * yy)
    return sd_min + (sd_max - sd_min) * u


@dataclass(frozen=True)
class ErrorRealization:
    """One realization of the prediction errors of `ErrorModelScenario`.

    Attributes
    ----------
    error_sub
        True prediction error at the sub-grid nodes (the point-scale nugget excluded).
    validation
        Validation observations: coordinates ``x``, ``y``, the flat index of their ``cell``
        and the observed prediction ``error`` (including the point-scale nugget).
    """

    error_sub: FloatArray
    validation: pl.DataFrame


@dataclass(frozen=True)
class ErrorModelScenario:
    """Known spatially correlated prediction errors on a fixed map.

    Parameters
    ----------
    layout
        Cell grid and sub-grid.
    correlation
        Correlogram of the standardized errors ε* as a variogram model (rescaled to unit sill),
        e.g. ``vgm(0.8, "Exp", 8, nugget=0.2)``. The nugget is point-scale variation present
        in observations only.
    sd_min, sd_max
        Range of the true (non-stationary) error standard deviation σ(s).
    n_validation
        Number of validation points (uniform at random over the sub-grid nodes).
    """

    layout: SubGridLayout = field(default_factory=SubGridLayout)
    correlation: VariogramModel = field(default_factory=lambda: vgm(0.8, "Exp", 8.0, nugget=0.2))
    sd_min: float = 0.5
    sd_max: float = 2.0
    n_validation: int = 300

    @cached_property
    def correlogram(self) -> VariogramModel:
        """The correlation model rescaled to unit sill."""
        return unit_sill(self.correlation)

    @property
    def nugget(self) -> float:
        """Nugget share of the standardized-error variance."""
        return self.correlogram.nugget

    @cached_property
    def structured(self) -> VariogramModel:
        """The structured (non-nugget) part, with sill 1 − nugget."""
        return without_nugget(self.correlogram)

    @cached_property
    def sigma_true(self) -> FloatArray:
        """True error standard deviation on the cell grid (n × n, north-up)."""
        return default_sigma_surface(self.layout, self.sd_min, self.sd_max)

    def sigma_map(self, scale: float = 1.0, shape: str = "true") -> xr.DataArray:
        """The σ map handed to the method: ``scale`` × the true σ, or a constant map
        (``shape="constant"``, the domain mean of the true σ times ``scale``)."""
        if shape == "true":
            values = scale * self.sigma_true
        elif shape == "constant":
            values = np.full_like(self.sigma_true, scale * float(self.sigma_true.mean()))
        else:
            raise ValueError("shape must be 'true' or 'constant'")
        return self.layout.cell_dataarray(values, name="sd")

    def true_correlation(self) -> CorrelationFunction:
        """The true correlation function of the standardized errors (with nugget)."""
        return VariogramCorrelation(self.correlogram)

    @cached_property
    def sampler(self) -> CirculantEmbedding:
        """Exact sampler of the structured part on the sub-grid."""
        ns = self.layout.n_sub
        return CirculantEmbedding((ns, ns), self.layout.sub_spacing, self.structured.covariance)

    def simulate(self, rng: np.random.Generator) -> ErrorRealization:
        """Draw one realization of the errors and of the validation data."""
        lay = self.layout
        s = self.sampler.sample(rng, 1)[0]
        error_sub = lay.upsample(self.sigma_true) * s
        idx = rng.choice(lay.n_sub * lay.n_sub, size=self.n_validation, replace=False)
        x, y = lay.sub_coords(idx)
        cell = lay.cell_of_sub(idx)
        sig = self.sigma_true.ravel()[cell]
        nug = math.sqrt(self.nugget) * rng.standard_normal(idx.size)
        err = sig * (s.ravel()[idx] + nug)
        val = pl.DataFrame({"x": x, "y": y, "cell": cell, "error": err})
        return ErrorRealization(error_sub=error_sub, validation=val)

    def exact_block_sd(self, blocks: EvaluationBlocks) -> FloatArray:
        """Exact standard deviation of the block-average error for each block."""
        lay = self.layout
        weights = lay.upsample(self.sigma_true)
        out = np.empty(len(blocks.sub_index))
        for b, idx in enumerate(blocks.sub_index):
            mask = np.zeros(lay.n_sub * lay.n_sub, dtype=bool)
            mask[idx] = True
            var = exact_block_variance(
                weights,
                mask.reshape(lay.n_sub, lay.n_sub),
                lay.sub_spacing,
                self.structured.covariance,
            )
            out[b] = math.sqrt(max(var, 0.0))
        return out


# ---------------------------------------------------------------------------
# The authors' R example as a full pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineRealization:
    """One realization of `PipelineScenario`.

    Attributes
    ----------
    truth
        The true field at the grid nodes (y ascending, x ascending).
    prediction, sd
        QRF predictive mean and standard deviation at the grid nodes, as DataArrays.
    validation
        Cross-validation table with ``x``, ``y``, ``observed``, ``predicted``, ``sd`` and ``fold``.
    sample_index
        Flat indices of the sampled nodes.
    """

    truth: xr.DataArray
    prediction: xr.DataArray
    sd: xr.DataArray
    validation: pl.DataFrame
    sample_index: IntArray


@dataclass(frozen=True)
class PipelineScenario:
    """The synthetic example of the authors' R script, as a full mapping pipeline.

    Defaults reproduce the script: a 100 × 100 grid on [0, 100]², trend
    5 + 0.1·x + 0.05·y, residual covariance 5·exp(−h/25), 200 sample points, 10-fold
    cross-validation and a 250-tree quantile regression forest on the coordinates.

    Parameters
    ----------
    n_nodes, extent
        Grid nodes per side and side length.
    intercept, slope_x, slope_y
        Linear trend.
    residual_sill, residual_range
        Exponential covariance of the residual field.
    n_train
        Number of sample points.
    n_folds
        Cross-validation folds (fold sizes are random, as in the script).
    n_trees, min_node_size, max_features
        Random forest settings (ranger defaults for two covariates: mtry = 1, minimal node
        size 5 to split at).
    n_sim
        Number of draws from the conditional distribution per location (the script uses 500).
    qrf
        ``"ranger"`` (default) re-implements ranger's quantile regression as used in the
        script -- one random training response per terminal node and tree -- on top of
        scikit-learn's random forest. ``"quantile-forest"`` uses Meinshausen's weighted
        quantiles from the ``quantile-forest`` package instead.
    n_quantiles
        Number of quantile levels used with ``qrf="quantile-forest"``.
    """

    n_nodes: int = 100
    extent: float = 100.0
    intercept: float = 5.0
    slope_x: float = 0.1
    slope_y: float = 0.05
    residual_sill: float = 5.0
    residual_range: float = 25.0
    n_train: int = 200
    n_folds: int = 10
    n_trees: int = 250
    min_node_size: int = 5
    max_features: int = 1
    n_sim: int = 500
    qrf: str = "ranger"
    n_quantiles: int = 100

    @classmethod
    def with_trend_r2(cls, r2: float, **kwargs: object) -> PipelineScenario:
        """Scale the trend so that it explains a fraction ``r2`` of the total variance."""
        base = cls(**kwargs)  # type: ignore[arg-type]
        var_xy = base.extent**2 / 12.0
        trend_var = (base.slope_x**2 + base.slope_y**2) * var_xy
        scale = math.sqrt(r2 * base.residual_sill / ((1 - r2) * trend_var))
        return cls(**{**kwargs, "slope_x": base.slope_x * scale, "slope_y": base.slope_y * scale})  # type: ignore[arg-type]

    @property
    def coords(self) -> FloatArray:
        """Node coordinates along each axis (as ``seq(0, extent, length.out = n_nodes)``)."""
        return np.linspace(0.0, self.extent, self.n_nodes)

    @property
    def trend_r2(self) -> float:
        """Share of the total variance explained by the trend."""
        tv = (self.slope_x**2 + self.slope_y**2) * self.extent**2 / 12.0
        return tv / (tv + self.residual_sill)

    @cached_property
    def sampler(self) -> CirculantEmbedding:
        """Exact sampler of the residual field."""
        spacing = self.extent / (self.n_nodes - 1)
        return CirculantEmbedding(
            (self.n_nodes, self.n_nodes),
            spacing,
            lambda h: self.residual_sill * np.exp(-h / self.residual_range),
        )

    def _fit_predict(
        self,
        x_train: FloatArray,
        z_train: FloatArray,
        x_new: FloatArray,
        rng: np.random.Generator,
        *,
        cv: bool,
    ) -> tuple[FloatArray, FloatArray]:
        """Fit a quantile regression forest and return the predictive mean and σ at ``x_new``."""
        seed = int(rng.integers(0, 2**31 - 1))
        if self.qrf == "quantile-forest":
            from quantile_forest import RandomForestQuantileRegressor

            qf = RandomForestQuantileRegressor(
                n_estimators=self.n_trees,
                max_features=self.max_features,
                min_samples_split=self.min_node_size,
                random_state=seed,
                n_jobs=1,
            )
            qf.fit(x_train, z_train)
            levels = (np.arange(self.n_quantiles) + 0.5) / self.n_quantiles
            q = np.asarray(qf.predict(x_new, quantiles=list(levels)))
            return q.mean(axis=1), np.asarray(sd_from_quantiles(q, levels, dim=1))
        # ranger's quantile regression: one random training response per terminal node and
        # tree, then `n_sim` draws with replacement from the per-tree values (as in the script)
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(
            n_estimators=self.n_trees,
            max_features=self.max_features,
            min_samples_split=self.min_node_size,
            bootstrap=True,
            random_state=seed,
            n_jobs=1,
        )
        rf.fit(x_train, z_train)
        leaves_train = rf.apply(x_train)
        leaves_new = rf.apply(x_new)
        node_rng = np.random.default_rng(seed)
        values = np.empty(leaves_new.shape)
        for t in range(self.n_trees):
            lookup = np.full(int(leaves_train[:, t].max()) + 1, np.nan)
            perm = node_rng.permutation(z_train.size)
            lookup[leaves_train[perm, t]] = z_train[perm]
            values[:, t] = lookup[leaves_new[:, t]]
        draws = values[
            np.arange(values.shape[0])[:, None],
            node_rng.integers(0, self.n_trees, (values.shape[0], self.n_sim)),
        ]
        mean = draws.mean(axis=1)
        var = draws.var(axis=1, ddof=1)
        if cv:
            var = var * self.n_sim / (self.n_sim - 1)  # the script's extra factor in the CV loop
        return mean, np.sqrt(var)

    def simulate(self, rng: np.random.Generator) -> PipelineRealization:
        """Simulate the field, sample it, cross-validate and map it with a QRF."""
        c = self.coords
        xx, yy = np.meshgrid(c, c)  # rows: y, columns: x
        resid = self.sampler.sample(rng, 1)[0]
        truth = self.intercept + self.slope_x * xx + self.slope_y * yy + resid
        n_cells = self.n_nodes * self.n_nodes
        idx = rng.choice(n_cells, size=self.n_train, replace=False)
        feats_all = np.column_stack([xx.ravel(), yy.ravel()])
        feats = feats_all[idx]
        z = truth.ravel()[idx]
        folds = rng.integers(0, self.n_folds, size=self.n_train)
        pred_cv = np.empty(self.n_train)
        sd_cv = np.empty(self.n_train)
        for k in range(self.n_folds):
            test = folds == k
            if not test.any():
                continue
            pred_cv[test], sd_cv[test] = self._fit_predict(
                feats[~test], z[~test], feats[test], rng, cv=True
            )
        mean_all, sd_all = self._fit_predict(feats, z, feats_all, rng, cv=False)
        coords = {"y": c, "x": c}
        val = pl.DataFrame(
            {
                "x": feats[:, 0],
                "y": feats[:, 1],
                "observed": z,
                "predicted": pred_cv,
                "sd": sd_cv,
                "fold": folds,
            }
        )
        return PipelineRealization(
            truth=xr.DataArray(truth, coords=coords, dims=("y", "x")),
            prediction=xr.DataArray(mean_all.reshape(truth.shape), coords=coords, dims=("y", "x")),
            sd=xr.DataArray(sd_all.reshape(truth.shape), coords=coords, dims=("y", "x")),
            validation=val,
            sample_index=idx.astype(np.int64),
        )
