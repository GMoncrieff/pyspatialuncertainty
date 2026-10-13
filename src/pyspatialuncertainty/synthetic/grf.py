"""Exact simulation of stationary Gaussian random fields on regular grids.

`CirculantEmbedding` implements the circulant embedding method (Dietrich & Newsam,
1997; Wood & Chan, 1994): the covariance matrix of the grid is embedded in a larger
block-circulant matrix whose eigenvalues are obtained by one FFT. If all eigenvalues are
non-negative (the embedding grid is enlarged until they are), fields drawn this way have
*exactly* the requested covariance. Each complex FFT yields two independent realizations.

`cholesky_sample` draws fields by Cholesky decomposition of the covariance matrix and
is used for small grids and to cross-check the FFT sampler.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import fft, linalg

from pyspatialuncertainty._types import FloatArray

Covariance = Callable[[FloatArray], FloatArray]


class CirculantEmbedding:
    """Exact sampler for a stationary isotropic Gaussian field on a regular grid.

    Parameters
    ----------
    shape
        Number of grid nodes (rows, columns).
    spacing
        Distance between neighbouring nodes (the same in both directions).
    covariance
        Covariance function C(h) of the field, evaluated at arrays of distances.
    tol
        Relative tolerance below which negative eigenvalues are treated as round-off.
    max_size
        Maximum size of the embedding grid along each axis.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        spacing: float,
        covariance: Covariance,
        *,
        tol: float = 1e-10,
        max_size: int = 16384,
    ) -> None:
        ny, nx = shape
        if ny < 1 or nx < 1 or spacing <= 0:
            raise ValueError("invalid grid")
        self.shape = (int(ny), int(nx))
        self.spacing = float(spacing)
        my = fft.next_fast_len(max(2 * ny, 2))
        mx = fft.next_fast_len(max(2 * nx, 2))
        while True:
            iy = np.minimum(np.arange(my), my - np.arange(my)) * spacing
            ix = np.minimum(np.arange(mx), mx - np.arange(mx)) * spacing
            base = covariance(np.hypot(iy[:, None], ix[None, :]))
            eig = fft.fft2(base).real
            if eig.min() >= -tol * eig.max():
                break
            if max(my, mx) * 2 > max_size:
                raise ValueError(
                    "could not find a non-negative definite circulant embedding; the "
                    "correlation range is probably too large relative to the grid"
                )
            my = fft.next_fast_len(2 * my)
            mx = fft.next_fast_len(2 * mx)
        self.embedding_shape = (my, mx)
        self._sqrt_eig = np.sqrt(np.clip(eig, 0.0, None) / (my * mx))
        self.variance = float(base[0, 0])

    def sample(self, rng: np.random.Generator, n: int = 1) -> FloatArray:
        """Draw ``n`` independent realizations, returned with shape (n, rows, cols)."""
        ny, nx = self.shape
        my, mx = self.embedding_shape
        out = np.empty((n, ny, nx))
        k = 0
        while k < n:
            z = rng.standard_normal((my, mx)) + 1j * rng.standard_normal((my, mx))
            y = fft.fft2(self._sqrt_eig * z)
            out[k] = y.real[:ny, :nx]
            if k + 1 < n:
                out[k + 1] = y.imag[:ny, :nx]
            k += 2
        return out


def cholesky_sample(
    coords: FloatArray, covariance: Covariance, rng: np.random.Generator, n: int = 1
) -> FloatArray:
    """Draw ``n`` realizations at arbitrary points by Cholesky decomposition.

    Parameters
    ----------
    coords
        Array of shape (m, 2) with the point coordinates.
    covariance
        Covariance function C(h).
    rng
        Random number generator.
    n
        Number of realizations.

    Returns
    -------
    ndarray
        Array of shape (n, m).
    """
    d = np.hypot(coords[:, None, 0] - coords[None, :, 0], coords[:, None, 1] - coords[None, :, 1])
    cov = covariance(d)
    chol = linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]) * cov[0, 0], lower=True)
    return (chol @ rng.standard_normal((coords.shape[0], n))).T
