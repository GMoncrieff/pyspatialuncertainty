"""Monte Carlo convergence on the R example (the paper's trial-and-error check, made quantitative).

Uses the R script's own exported σ map and variogram model, integrates over the whole
domain [0, 100]² with m point pairs for several m, and repeats each with independent seeds.

Usage: ``uv run --group experiments python -m experiments.run_mc_convergence``
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xarray as xr

import pyspatialuncertainty as psu
from experiments.common import ROOT, load_config, write


def main() -> None:
    cfg = load_config()
    sec = cfg["mc_convergence"]
    ref = ROOT.parent / "tests" / "data" / "r_reference"
    grid = pl.read_csv(ref / "grid.csv", infer_schema_length=None)
    c = np.sort(grid["x1"].unique().to_numpy())
    sd_da = xr.DataArray(
        np.sqrt(grid["predVarRF"].to_numpy()).reshape(100, 100),
        coords={"y": c, "x": c},
        dims=("y", "x"),
    )
    model = psu.VariogramModel.from_table(pl.read_csv(ref / "variogram_model.csv"))
    g = psu.GridSpec.from_dataarray(sd_da)
    sd = g.values(sd_da)
    blocks = psu.resolve_blocks(psu.blocks.box(0, 0, 100, 100, g), np.isfinite(sd))
    integrand = psu.Integrand(sd, psu.from_variogram(model))
    rows = []
    for m in sec["m"]:
        for rep in range(sec["replicates"]):
            res = psu.mc_block_variance(
                blocks, [integrand], n_mc=int(m), seed=[cfg["seed"], int(m), rep]
            )
            rows.append(
                {
                    "m": int(m),
                    "replicate": rep,
                    "var_average": float(res.variance[0, 0]),
                    "mcse": float(res.mcse[0, 0]),
                }
            )
        print(f"m={m} done", flush=True)
    write(pl.DataFrame(rows), "mc_convergence.parquet")


if __name__ == "__main__":
    main()
