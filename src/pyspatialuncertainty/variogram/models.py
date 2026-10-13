"""Variogram models in gstat's parameterisation.

All structures follow the conventions of the R package gstat (``vgm()``), so that a
model written as ``vgm(0.46, "Exp", 2, 0.05)`` in R is written as
``vgm(0.46, "Exp", 2, nugget=0.05)`` here and evaluates to the same semivariance.
With partial sill c and range parameter a, the semivariance for h > 0 is:

* ``Nug``: c (the range is not used);
* ``Exp``: c·(1 − exp(−h/a)); the practical range is about 3a;
* ``Sph``: c·(1.5·h/a − 0.5·(h/a)³) for h ≤ a and c beyond; the range is a;
* ``Gau``: c·(1 − exp(−(h/a)²)); the practical range is about √3·a;
* ``Mat``: c·(1 − 2^(1−κ)/Γ(κ)·(h/a)^κ·K_κ(h/a)), with smoothness κ;
* ``Ste``: as ``Mat`` with h/a replaced by 2·√κ·h/a (M. Stein's parameterisation).

Every structure has γ(0) = 0; a nugget therefore only applies to strictly positive lags.
A :class:`VariogramModel` is a sum (nested model) of such structures.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from scipy import special

from pyspatialuncertainty._types import FloatArray

if TYPE_CHECKING:
    from pyspatialuncertainty.correlation import VariogramCorrelation

ModelType = Literal["Nug", "Exp", "Sph", "Gau", "Mat", "Ste"]
"""Supported structure types (gstat model codes)."""

MODEL_TYPES: tuple[ModelType, ...] = get_args(ModelType)


def _as_lags(h: ArrayLike) -> FloatArray:
    lags = np.asarray(h, dtype=np.float64)
    if np.any(lags < 0):
        raise ValueError("lag distances must be non-negative")
    return lags


def _matern_part(x: FloatArray, kappa: float) -> FloatArray:
    """Return 2^(1-κ)/Γ(κ) x^κ K_κ(x), the Matérn correlation, with value 1 at x = 0."""
    out = np.ones_like(x)
    pos = x > 0
    xp = x[pos]
    if xp.size:
        log_val = (
            (1.0 - kappa) * math.log(2.0)
            - special.gammaln(kappa)
            + kappa * np.log(xp)
            + np.log(special.kve(kappa, xp))
            - xp
        )
        out[pos] = np.exp(log_val)
    return out


def _unit_nug(h: FloatArray, a: float, kappa: float) -> FloatArray:
    return (h > 0).astype(np.float64)


def _unit_exp(h: FloatArray, a: float, kappa: float) -> FloatArray:
    return -np.expm1(-h / a)


def _unit_sph(h: FloatArray, a: float, kappa: float) -> FloatArray:
    r = np.minimum(h / a, 1.0)
    return 1.5 * r - 0.5 * r**3


def _unit_gau(h: FloatArray, a: float, kappa: float) -> FloatArray:
    return -np.expm1(-((h / a) ** 2))


def _unit_mat(h: FloatArray, a: float, kappa: float) -> FloatArray:
    return 1.0 - _matern_part(h / a, kappa)


def _unit_ste(h: FloatArray, a: float, kappa: float) -> FloatArray:
    return 1.0 - _matern_part(2.0 * math.sqrt(kappa) * h / a, kappa)


_UNIT_GAMMA: dict[str, Callable[[FloatArray, float, float], FloatArray]] = {
    "Nug": _unit_nug,
    "Exp": _unit_exp,
    "Sph": _unit_sph,
    "Gau": _unit_gau,
    "Mat": _unit_mat,
    "Ste": _unit_ste,
}


@dataclass(frozen=True, slots=True)
class Structure:
    """One elementary structure of a (nested) variogram model.

    Parameters
    ----------
    model
        gstat model code: ``"Nug"``, ``"Exp"``, ``"Sph"``, ``"Gau"``, ``"Mat"`` or ``"Ste"``.
    psill
        Partial sill (contribution to the total sill); must be non-negative.
    range
        Range parameter *a* in gstat's parameterisation (ignored for ``"Nug"``).
    kappa
        Smoothness parameter of the ``"Mat"`` and ``"Ste"`` models.
    """

    model: ModelType
    psill: float
    range: float = 0.0
    kappa: float = 0.5

    def __post_init__(self) -> None:
        if self.model not in _UNIT_GAMMA:
            raise ValueError(f"unknown model {self.model!r}; expected one of {MODEL_TYPES}")
        if not math.isfinite(self.psill) or self.psill < 0:
            raise ValueError(f"psill must be finite and non-negative, got {self.psill}")
        if self.model != "Nug" and not (math.isfinite(self.range) and self.range > 0):
            raise ValueError(f"range of a {self.model} structure must be positive")
        if self.model in ("Mat", "Ste") and not self.kappa > 0:
            raise ValueError("kappa must be positive")

    def unit_gamma(self, h: ArrayLike) -> FloatArray:
        """Semivariance of the structure with unit partial sill."""
        lags = _as_lags(h)
        out = _UNIT_GAMMA[self.model](lags, self.range, self.kappa)
        return np.where(lags > 0, out, 0.0)

    def gamma(self, h: ArrayLike) -> FloatArray:
        """Semivariance of the structure."""
        return self.psill * self.unit_gamma(h)


@dataclass(frozen=True)
class VariogramModel:
    """A nested variogram model: the sum of one or more :class:`Structure` objects.

    Models are usually built with :func:`vgm` or the convenience constructors
    (:func:`Nugget`, :func:`Exponential`, ...) and combined with ``+``::

        model = vgm(0.46, "Exp", 2, nugget=0.05) + vgm(0.08, "Sph", 15) + vgm(0.06, "Exp", 60)
    """

    structures: tuple[Structure, ...]

    def __post_init__(self) -> None:
        if len(self.structures) == 0:
            raise ValueError("a variogram model needs at least one structure")
        object.__setattr__(self, "structures", tuple(self.structures))

    def __add__(self, other: VariogramModel) -> VariogramModel:
        if not isinstance(other, VariogramModel):
            return NotImplemented
        return VariogramModel(self.structures + other.structures)

    def __len__(self) -> int:
        return len(self.structures)

    @property
    def sill(self) -> float:
        """Total sill: the sum of all partial sills, including the nugget."""
        return float(sum(s.psill for s in self.structures))

    @property
    def nugget(self) -> float:
        """Sum of the partial sills of the nugget structures."""
        return float(sum(s.psill for s in self.structures if s.model == "Nug"))

    def gamma(self, h: ArrayLike) -> FloatArray:
        """Semivariance γ(h). By convention γ(0) = 0."""
        lags = _as_lags(h)
        out = np.zeros_like(lags)
        for s in self.structures:
            if s.psill > 0:
                out = out + s.gamma(lags)
        return out

    def covariance(self, h: ArrayLike) -> FloatArray:
        """Covariance C(h) = sill − γ(h)."""
        return self.sill - self.gamma(h)

    def correlation(self, h: ArrayLike) -> FloatArray:
        """Correlation ρ(h) = (sill − γ(h)) / sill, the conversion used in the paper (step 4)."""
        sill = self.sill
        if not sill > 0:
            raise ValueError("the correlation function is undefined for a model with zero sill")
        return (sill - self.gamma(h)) / sill

    def to_correlation(self) -> VariogramCorrelation:
        """Return the correlation function of this model as a callable object."""
        from pyspatialuncertainty.correlation import VariogramCorrelation

        return VariogramCorrelation(self)

    def to_polars(self) -> pl.DataFrame:
        """Tabulate the structures (columns ``model``, ``psill``, ``range``, ``kappa``)."""
        return pl.DataFrame(
            {
                "model": [s.model for s in self.structures],
                "psill": [s.psill for s in self.structures],
                "range": [s.range for s in self.structures],
                "kappa": [s.kappa for s in self.structures],
            },
            schema={
                "model": pl.String,
                "psill": pl.Float64,
                "range": pl.Float64,
                "kappa": pl.Float64,
            },
        )

    @classmethod
    def from_table(cls, table: pl.DataFrame | Mapping[str, Sequence[Any]]) -> VariogramModel:
        """Build a model from a table with columns ``model``, ``psill``, ``range`` and
        optionally ``kappa``, such as ``as.data.frame()`` of a gstat ``variogramModel``."""
        df = table if isinstance(table, pl.DataFrame) else pl.DataFrame(dict(table))
        kappas = df["kappa"].to_list() if "kappa" in df.columns else [0.5] * df.height
        structures = []
        for model, psill, rng, kappa in zip(
            df["model"].to_list(), df["psill"].to_list(), df["range"].to_list(), kappas, strict=True
        ):
            code = str(model).strip()
            if code not in _UNIT_GAMMA:
                raise ValueError(f"unknown model {code!r}")
            structures.append(
                Structure(
                    cast(ModelType, code),
                    float(psill),
                    float(rng) if code != "Nug" else 0.0,
                    float(kappa) if code in ("Mat", "Ste") else 0.5,
                )
            )
        return cls(tuple(structures))

    def replace_parameters(
        self, psills: Iterable[float], ranges: Iterable[float]
    ) -> VariogramModel:
        """Return a copy with new partial sills and ranges (one value per structure)."""
        new = [
            replace(s, psill=float(p), range=float(r) if s.model != "Nug" else 0.0)
            for s, p, r in zip(self.structures, psills, ranges, strict=True)
        ]
        return VariogramModel(tuple(new))

    def __str__(self) -> str:
        lines = [f"VariogramModel (sill = {self.sill:.6g})"]
        for s in self.structures:
            extra = "" if s.model == "Nug" else f"  range={s.range:.6g}"
            if s.model in ("Mat", "Ste"):
                extra += f"  kappa={s.kappa:.6g}"
            lines.append(f"  {s.model}  psill={s.psill:.6g}{extra}")
        return "\n".join(lines)


def vgm(
    psill: float,
    model: ModelType,
    range: float = 0.0,
    nugget: float = 0.0,
    kappa: float = 0.5,
) -> VariogramModel:
    """Create a variogram model, mirroring gstat's ``vgm(psill, model, range, nugget, kappa)``.

    Parameters
    ----------
    psill
        Partial sill of the structure.
    model
        gstat model code (``"Nug"``, ``"Exp"``, ``"Sph"``, ``"Gau"``, ``"Mat"``, ``"Ste"``).
    range
        Range parameter in gstat's parameterisation.
    nugget
        Optional nugget added as a separate ``"Nug"`` structure (placed first, as in gstat).
    kappa
        Smoothness of the ``"Mat"`` and ``"Ste"`` models.

    Returns
    -------
    VariogramModel
        The (possibly two-structure) model.

    Examples
    --------
    >>> m = vgm(0.46, "Exp", 2, nugget=0.05) + vgm(0.08, "Sph", 15) + vgm(0.06, "Exp", 60)
    >>> round(m.sill, 2)
    0.65
    """
    if model == "Nug":
        return Nugget(float(psill) + float(nugget))
    structures: list[Structure] = []
    if nugget > 0:
        structures.append(Structure("Nug", float(nugget)))
    structures.append(Structure(model, float(psill), float(range), float(kappa)))
    return VariogramModel(tuple(structures))


def Nugget(psill: float) -> VariogramModel:  # noqa: N802 - constructor-style name
    """Pure nugget model: γ(h) = psill for h > 0."""
    return VariogramModel((Structure("Nug", float(psill)),))


def Exponential(psill: float, range: float) -> VariogramModel:  # noqa: N802
    """Exponential model, γ(h) = psill·(1 − exp(−h/range))."""
    return VariogramModel((Structure("Exp", float(psill), float(range)),))


def Spherical(psill: float, range: float) -> VariogramModel:  # noqa: N802
    """Spherical model reaching its sill at h = range."""
    return VariogramModel((Structure("Sph", float(psill), float(range)),))


def Gaussian(psill: float, range: float) -> VariogramModel:  # noqa: N802
    """Gaussian model, γ(h) = psill·(1 − exp(−(h/range)²))."""
    return VariogramModel((Structure("Gau", float(psill), float(range)),))


def Matern(psill: float, range: float, kappa: float = 0.5) -> VariogramModel:  # noqa: N802
    """Matérn model in gstat's ``"Mat"`` parameterisation."""
    return VariogramModel((Structure("Mat", float(psill), float(range), float(kappa)),))


def MaternStein(psill: float, range: float, kappa: float = 0.5) -> VariogramModel:  # noqa: N802
    """Matérn model in M. Stein's parameterisation (gstat's ``"Ste"``)."""
    return VariogramModel((Structure("Ste", float(psill), float(range), float(kappa)),))
