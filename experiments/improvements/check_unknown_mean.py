"""Exact check of the unknown-mean correction and of its implementation (round 3).

On a 30 × 30 grid the structured error field at the cell centres and the validation
errors (structured part plus nugget) are simulated exactly by Cholesky decomposition.
The map predicts zero, so the block error is the block mean of σ(s)·(x(s)ᵀβ + w(s)).
For a constant mean and for stratum means (stratified validation sample, calibration
differing by stratum), the script checks:

1. the corrected block estimate is unbiased;
2. the variance from ``candidates.unknown_mean_terms`` plus the exact discrete V_B
   matches the simulated variance of the corrected error;
3. the Monte Carlo V_B of ``psu.mc_block_variance`` (continuous block) agrees with the
   discrete V_B up to discretisation;
4. REML with X = 1 reproduces ``mean="constant"``.

Usage: ``uv run python -m experiments.improvements.check_unknown_mean``
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import pyspatialuncertainty as psu
from experiments.improvements import candidates as cand

N_SIM = 4000
TOL_VAR = 0.06  # relative tolerance for simulated vs formula variance (4,000 draws)


def grid_setup() -> tuple[psu.GridSpec, list[psu.ResolvedBlock], np.ndarray, np.ndarray]:
    c = np.arange(30) + 0.5
    template = xr.DataArray(np.zeros((30, 30)), coords={"y": c[::-1], "x": c}, dims=("y", "x"))
    grid = psu.GridSpec.from_dataarray(template)
    domain = psu.blocks.domain(grid).blocks[0]
    tile = psu.blocks.tiles(grid, 10).blocks[0]  # top-left (north-west) 10 x 10 tile
    bs = psu.blocks.BlockSet(grid, (domain, tile), "custom")
    resolved = psu.resolve_blocks(bs, np.ones(grid.size, dtype=bool))
    cx, cy = grid.cell_centres(np.arange(grid.size))
    return grid, resolved, cx, cy


def check(
    name: str,
    X_val: np.ndarray,
    X_cells: np.ndarray,
    beta_true: np.ndarray,
    val_cells: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    grid, resolved, cx, cy = grid_setup()
    model = psu.vgm(0.75, "Exp", 4.0, nugget=0.25)
    sd = 0.5 + 1.5 * cx / 30.0
    xv, yv = cx[val_cells], cy[val_cells]
    n_cells, n = grid.size, val_cells.size
    px = np.concatenate([cx, xv])
    py = np.concatenate([cy, yv])
    C = cand.structured_correlation(model, cand.distance_matrix(px, py))
    L = np.linalg.cholesky(C + 1e-9 * np.eye(C.shape[0]))
    bmat = cand.block_average_matrix(resolved, n_cells)
    gls = cand.gls_terms(model, xv, yv, np.zeros(n), X_val)
    d_cells = np.hypot(cx[:, None] - xv[None, :], cy[:, None] - yv[None, :])
    _, added = cand.unknown_mean_terms(bmat, sd, X_cells, model, d_cells, gls)
    xb = np.asarray(bmat @ (sd[:, None] * X_cells))
    eta = model.nugget / model.sill
    ok = True
    for b, blk in enumerate(resolved):
        cells = blk.cells
        vb = float(np.mean(np.outer(sd[cells], sd[cells]) * C[np.ix_(cells, cells)]))
        formula = vb + added[b]
        errs = np.empty(N_SIM)
        for r in range(N_SIM):
            w = L @ rng.standard_normal(C.shape[0])
            truth = np.mean(sd[cells] * (X_cells[cells] @ beta_true + w[:n_cells][cells]))
            z = X_val @ beta_true + w[n_cells:] + np.sqrt(eta) * rng.standard_normal(n)
            errs[r] = truth - xb[b] @ (gls.G @ z)
        mc = psu.mc_block_variance(
            [blk], [psu.Integrand(sd, cand.correlation(model))], n_mc=200_000, seed=1
        )
        rel = abs(errs.var() - formula) / formula
        se_bias = errs.std() / np.sqrt(N_SIM)
        good = rel < TOL_VAR and abs(errs.mean()) < 4 * se_bias
        ok &= good
        print(
            f"{name:7s} block {blk.id!s:8s}: bias {errs.mean():+.4f} (se {se_bias:.4f}), "
            f"variance simulated {errs.var():.5f} vs formula {formula:.5f} "
            f"(rel. diff {rel:.3f}); V_B discrete {vb:.5f} vs MC {mc.variance[0, 0]:.5f} "
            f"{'ok' if good else 'FAIL'}"
        )
    return ok


def main() -> None:
    rng = np.random.default_rng(20261016)
    grid, _, _, cy = grid_setup()
    ok = True
    # constant mean, simple random validation sample, calibration offset 0.4
    val = rng.choice(grid.size, 50, replace=False)
    ok &= check("X = 1", np.ones((50, 1)), np.ones((grid.size, 1)), np.array([0.4]), val, rng)
    # stratum means, stratified validation sample (30 per stratum), offsets 0 and 0.6
    stratum = (cy > 20).astype(int)
    val = np.concatenate(
        [rng.choice(np.flatnonzero(stratum == h), 30, replace=False) for h in (0, 1)]
    )
    kept = np.array([1])
    ok &= check(
        "strata",
        cand.design_rows(stratum[val], kept),
        cand.design_rows(stratum, kept),
        np.array([0.0, 0.6]),
        val,
        rng,
    )
    # REML with X = 1 equals mean="constant"
    z = rng.standard_normal(60)
    x, y = rng.uniform(0, 30, 60), rng.uniform(0, 30, 60)
    m1 = cand.likelihood_fit(x, y, z, mean="constant")
    m2 = cand.likelihood_fit(x, y, z, X=np.ones((60, 1)))
    same = np.allclose(m1.sill, m2.sill, rtol=1e-4) and np.allclose(
        m1.nugget, m2.nugget, rtol=1e-3, atol=1e-6
    )
    ok &= bool(same)
    print(
        f"REML X = 1 vs mean='constant': sill {m1.sill:.5f} / {m2.sill:.5f}, "
        f"nugget {m1.nugget:.5f} / {m2.nugget:.5f} {'ok' if same else 'FAIL'}"
    )
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
