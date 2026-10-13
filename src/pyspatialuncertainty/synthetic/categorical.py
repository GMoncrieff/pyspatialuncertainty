"""Synthetic class maps with known truth: thresholded latent Gaussian fields.

A latent variable Y(s) = μ(s) + η(s) is cut by thresholds t_1 < ... < t_{K−1} into K
classes. μ is the part of Y that the map "knows" (constant within each cell, fixed across
realizations), and η is unpredictable, spatially correlated noise with unit variance,
made of a structured Gaussian field and a point-scale nugget. The calibrated class
probabilities of the map are then exactly

    p_k(s) = Φ(t_k − μ(s)) − Φ(t_{k−1} − μ(s)),

and the realizations of η give the true classes. The block truth is the fraction of the
block area in class k: with a nugget, the point-scale variation averages out and the true
fraction is the block mean of P(class k | μ, structured part of η); without a nugget it is
the mean of the class indicator over the sub-grid nodes.

The signal-to-noise ratio Var(μ)/Var(η) controls how sharp the probabilities are, and
temperature scaling of the probabilities simulates miscalibration.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import polars as pl
import xarray as xr
from scipy import optimize, special

from pyspatialuncertainty._types import FloatArray
from pyspatialuncertainty.categorical import temperature_scale
from pyspatialuncertainty.correlation import CallableCorrelation
from pyspatialuncertainty.synthetic.continuous import unit_sill, without_nugget
from pyspatialuncertainty.synthetic.grf import CirculantEmbedding
from pyspatialuncertainty.synthetic.layout import SubGridLayout
from pyspatialuncertainty.synthetic.truth import bvn_rectangle, indicator_block_variance
from pyspatialuncertainty.variogram.models import VariogramModel, vgm


def interval_probability(lower: FloatArray, upper: FloatArray) -> FloatArray:
    """P(lower < Z ≤ upper) for a standard normal Z, accurate in both tails."""
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    upper_tail = special.ndtr(-lo) - special.ndtr(-hi)
    lower_tail = special.ndtr(hi) - special.ndtr(lo)
    out: FloatArray = np.where(lo > 0, upper_tail, lower_tail)
    return out


@dataclass(frozen=True)
class ClassRealization:
    """One realization of `LatentClassScenario`.

    Attributes
    ----------
    fraction_sub
        Array (K, n_sub, n_sub): the true class membership at the sub-grid nodes (a class
        probability given the structured noise when there is a nugget, else 0/1).
    validation
        Validation points with ``x``, ``y``, ``cell`` (flat cell index) and ``label``.
    """

    fraction_sub: FloatArray
    validation: pl.DataFrame


@dataclass(frozen=True)
class LatentClassScenario:
    """Thresholded latent Gaussian field with exactly calibrated class probabilities.

    Parameters
    ----------
    layout
        Cell grid and sub-grid.
    latent
        Correlogram of the latent noise η as a variogram model (rescaled to unit sill); its
        nugget is point-scale noise.
    snr
        Signal-to-noise ratio Var(μ) / Var(η).
    signal_range
        Range (in map units) of the exponential Gaussian field used to generate μ.
    prevalence
        Target overall class proportions.
    classes
        Class labels.
    n_validation
        Number of validation points.
    seed
        Seed of the fixed signal μ.
    """

    layout: SubGridLayout = field(default_factory=SubGridLayout)
    latent: VariogramModel = field(default_factory=lambda: vgm(0.8, "Exp", 8.0, nugget=0.2))
    snr: float = 1.0
    signal_range: float = 16.0
    prevalence: tuple[float, ...] = (0.6, 0.3, 0.1)
    classes: tuple[Hashable, ...] = ("a", "b", "c")
    n_validation: int = 300
    seed: int = 1320

    def __post_init__(self) -> None:
        if len(self.prevalence) != len(self.classes):
            raise ValueError("prevalence and classes must have the same length")
        if not math.isclose(sum(self.prevalence), 1.0, abs_tol=1e-9):
            raise ValueError("prevalences must sum to one")

    @cached_property
    def correlogram(self) -> VariogramModel:
        """Latent noise correlogram (unit sill)."""
        return unit_sill(self.latent)

    @property
    def nugget(self) -> float:
        """Nugget share of the latent noise variance."""
        return self.correlogram.nugget

    @cached_property
    def structured(self) -> VariogramModel:
        """Structured part of the latent noise (sill 1 − nugget)."""
        return without_nugget(self.correlogram)

    @cached_property
    def mu(self) -> FloatArray:
        """The fixed signal μ on the cell grid (n × n, north-up), with variance ``snr``."""
        n = self.layout.n
        emb = CirculantEmbedding(
            (n, n), self.layout.cell_size, lambda h: np.exp(-h / self.signal_range)
        )
        field_ = emb.sample(np.random.default_rng(self.seed), 1)[0]
        field_ = (field_ - field_.mean()) / field_.std()
        out: FloatArray = math.sqrt(self.snr) * field_
        return out

    @cached_property
    def thresholds(self) -> FloatArray:
        """Class thresholds t_1 < ... < t_{K−1} giving the target prevalences over the domain."""
        mu = self.mu.ravel()
        cum = np.cumsum(self.prevalence)[:-1]
        out = []
        for target in cum:
            out.append(
                optimize.brentq(
                    lambda t, c=target: float(np.mean(special.ndtr(t - mu))) - c, -20.0, 20.0
                )
            )
        return np.array(out)

    @property
    def limits(self) -> FloatArray:
        """Thresholds with −∞ and +∞ added: class k lies in (limits[k], limits[k+1]]."""
        return np.concatenate([[-np.inf], self.thresholds, [np.inf]])

    @cached_property
    def p_true(self) -> FloatArray:
        """Calibrated class probabilities (K, n, n)."""
        lim = self.limits
        p = np.stack(
            [
                interval_probability(lim[k] - self.mu, lim[k + 1] - self.mu)
                for k in range(len(self.classes))
            ]
        )
        p = np.clip(p, 1e-12, 1.0)
        out: FloatArray = p / p.sum(axis=0, keepdims=True)
        return out

    def probabilities(self, temperature: float = 1.0) -> xr.DataArray:
        """The probability map handed to the method, optionally temperature-scaled."""
        p = (
            self.p_true
            if temperature == 1.0
            else temperature_scale(self.p_true, temperature, class_dim=0)
        )
        lay = self.layout
        return xr.DataArray(
            p,
            coords={"class": list(self.classes), "y": lay.cell_y, "x": lay.cell_x},
            dims=("class", "y", "x"),
        )

    @cached_property
    def sampler(self) -> CirculantEmbedding:
        """Exact sampler of the structured latent noise on the sub-grid."""
        ns = self.layout.n_sub
        return CirculantEmbedding((ns, ns), self.layout.sub_spacing, self.structured.covariance)

    def simulate(self, rng: np.random.Generator) -> ClassRealization:
        """Draw one realization of the true classes and of the validation data."""
        lay = self.layout
        s = self.sampler.sample(rng, 1)[0]
        y_struct = lay.upsample(self.mu) + s
        lim = self.limits
        k_classes = len(self.classes)
        if self.nugget > 0:
            sd_n = math.sqrt(self.nugget)
            frac = np.stack(
                [
                    interval_probability((lim[k] - y_struct) / sd_n, (lim[k + 1] - y_struct) / sd_n)
                    for k in range(k_classes)
                ]
            )
        else:
            idx_class = np.searchsorted(self.thresholds, y_struct, side="left")
            frac = np.stack([(idx_class == k).astype(np.float64) for k in range(k_classes)])
        idx = rng.choice(lay.n_sub * lay.n_sub, size=self.n_validation, replace=False)
        x, y = lay.sub_coords(idx)
        cell = lay.cell_of_sub(idx)
        y_point = y_struct.ravel()[idx] + math.sqrt(self.nugget) * rng.standard_normal(idx.size)
        label_idx = np.searchsorted(self.thresholds, y_point, side="left")
        labels = [self.classes[i] for i in label_idx]
        val = pl.DataFrame({"x": x, "y": y, "cell": cell, "label": pl.Series(labels)})
        return ClassRealization(fraction_sub=frac, validation=val)

    def latent_correlation(self, h: FloatArray) -> FloatArray:
        """Correlation of the latent variable at two distinct points at distance h > 0."""
        return self.structured.covariance(h)

    def average_correlogram(
        self, k: int, distances: FloatArray, *, n_pairs: int = 4000, seed: int = 0
    ) -> FloatArray:
        """Domain-averaged correlation of the standardized indicator errors of class k.

        For each distance d, random pairs of points at distance d are drawn and the exact
        correlation of (I_k − p_k)/σ_k at the two points is averaged. This is the best
        stationary approximation of the (non-stationary) indicator error correlation.
        """
        rng = np.random.default_rng(seed)
        lay = self.layout
        lim = self.limits
        out = np.empty(distances.size)
        g = lay.grid
        mu = self.mu.ravel()
        for i, d in enumerate(distances):
            if d == 0:
                out[i] = 1.0
                continue
            c1_list: list[np.ndarray] = []
            c2_list: list[np.ndarray] = []
            got = 0
            for _ in range(50):
                ang = rng.uniform(0, 2 * math.pi, n_pairs)
                x1 = rng.uniform(0, lay.extent, n_pairs)
                y1 = rng.uniform(0, lay.extent, n_pairs)
                x2 = x1 + d * np.cos(ang)
                y2 = y1 + d * np.sin(ang)
                ok = (x2 >= 0) & (x2 < lay.extent) & (y2 >= 0) & (y2 < lay.extent)
                c1_list.append(g.flat_index_of(x1[ok], y1[ok]))
                c2_list.append(g.flat_index_of(x2[ok], y2[ok]))
                got += int(ok.sum())
                if got >= n_pairs:
                    break
            if got == 0:
                out[i] = out[i - 1] if i > 0 else 1.0
                continue
            c1 = np.concatenate(c1_list)[:n_pairs]
            c2 = np.concatenate(c2_list)[:n_pairs]
            a1, b1 = lim[k] - mu[c1], lim[k + 1] - mu[c1]
            a2, b2 = lim[k] - mu[c2], lim[k + 1] - mu[c2]
            p1 = interval_probability(a1, b1)
            p2 = interval_probability(a2, b2)
            r = float(self.latent_correlation(np.array([d]))[0])
            joint = bvn_rectangle(a1, b1, a2, b2, r)
            denom = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
            ok_d = denom > 1e-12
            if ok_d.any():
                out[i] = float(np.mean((joint[ok_d] - p1[ok_d] * p2[ok_d]) / denom[ok_d]))
            else:
                out[i] = 0.0
        return out

    def oracle_correlation(
        self, k: int, *, max_distance: float | None = None, n: int = 80
    ) -> CallableCorrelation:
        """Interpolated domain-averaged correlogram of class k (see `average_correlogram`)."""
        dmax = max_distance if max_distance is not None else self.layout.extent
        d = np.concatenate([[0.0], np.geomspace(1e-3 * self.layout.cell_size, dmax, n)])
        rho = self.average_correlogram(k, d)
        rho_pos = rho[1:]

        def corr(h: FloatArray) -> FloatArray:
            hh = np.asarray(h, dtype=np.float64)
            vals = np.interp(hh, d[1:], rho_pos, left=rho_pos[0], right=rho_pos[-1])
            return np.where(hh > 0, vals, 1.0)

        return CallableCorrelation(corr)

    def exact_block_sd(self, k: int, sub_index: np.ndarray) -> float:
        """Exact SD of the true block fraction of class k over the given sub-grid nodes."""
        lay = self.layout
        cells = lay.cell_of_sub(sub_index)
        mu = self.mu.ravel()[cells]
        lim = self.limits
        x, y = lay.sub_coords(sub_index)
        coords = np.column_stack([x, y])
        rho_struct = self.structured  # covariance = (1 − nugget)·ρ_s(h); at h = 0: 1 − nugget

        def corr(dist: FloatArray) -> FloatArray:
            return rho_struct.covariance(dist)

        var = indicator_block_variance(lim[k] - mu, lim[k + 1] - mu, coords, corr)
        return math.sqrt(max(var, 0.0))
