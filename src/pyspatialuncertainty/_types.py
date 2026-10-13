"""Shared type aliases."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

SeedLike = int | np.random.SeedSequence | np.random.Generator | None
"""A random seed: an integer, a ``SeedSequence``, a ``Generator`` or ``None``."""
