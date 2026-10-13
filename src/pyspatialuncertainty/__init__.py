"""Uncertainty of spatial averages and totals of maps.

A Python implementation of the Monte Carlo integration approach of
Wadoux & Heuvelink (2023), *Uncertainty of spatial averages and totals of natural
resource maps*, Methods in Ecology and Evolution 14:1320-1332,
https://doi.org/10.1111/2041-210X.14106.
"""

from importlib.metadata import version as _version

__version__ = _version("pyspatialuncertainty")

__all__ = ["__version__"]
