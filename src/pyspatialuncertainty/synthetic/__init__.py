"""Synthetic data with known truth for verifying and stress-testing the method."""

from pyspatialuncertainty.synthetic.categorical import ClassRealization, LatentClassScenario
from pyspatialuncertainty.synthetic.continuous import (
    ErrorModelScenario,
    ErrorRealization,
    PipelineRealization,
    PipelineScenario,
    default_sigma_surface,
    unit_sill,
    without_nugget,
)
from pyspatialuncertainty.synthetic.grf import CirculantEmbedding, cholesky_sample
from pyspatialuncertainty.synthetic.layout import EvaluationBlocks, SubGridLayout, evaluation_blocks
from pyspatialuncertainty.synthetic.truth import (
    bvn_cdf,
    bvn_rectangle,
    exact_block_variance,
    indicator_block_variance,
)

__all__ = [
    "CirculantEmbedding",
    "ClassRealization",
    "ErrorModelScenario",
    "ErrorRealization",
    "EvaluationBlocks",
    "LatentClassScenario",
    "PipelineRealization",
    "PipelineScenario",
    "SubGridLayout",
    "bvn_cdf",
    "bvn_rectangle",
    "cholesky_sample",
    "default_sigma_surface",
    "evaluation_blocks",
    "exact_block_variance",
    "indicator_block_variance",
    "unit_sill",
    "without_nugget",
]
