"""Shared matplotlib style for the docs and the report (validated reference palette)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

# categorical slots in fixed order (never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ESTIMATOR_COLOR = {
    "method": SERIES[0],
    "oracle": SERIES[1],
    "naive_avg": SERIES[2],
    "naive_indep": SERIES[3],
}
ESTIMATOR_MARKER = {"method": "o", "oracle": "s", "naive_avg": "^", "naive_indep": "v"}
ESTIMATOR_LABEL = {
    "method": "method (fitted correlogram)",
    "oracle": "oracle (true σ and ρ)",
    "naive_avg": "naive: average of σ",
    "naive_indep": "naive: independent errors",
}

BLUES = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#c2410c", "#eb6834", "#f2f1ee", "#3987e5", "#184f95"]
)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "font.size": 9.5,
        }
    )


apply_style()
__all__ = [
    "BLUES",
    "DIVERGING",
    "ESTIMATOR_COLOR",
    "ESTIMATOR_LABEL",
    "ESTIMATOR_MARKER",
    "SERIES",
    "apply_style",
    "plt",
]
