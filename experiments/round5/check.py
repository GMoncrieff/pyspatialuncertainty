"""Stage 0 checks of the round-5 harness.

1. Exact point-support V_B by FFT against the package's Monte Carlo integral (common
   pairs, 20,000 pairs per small block and 1,000,000 per large block), for four correlograms
   including a very short one.
2. The GLS of the standardized errors (X = 1) against `mean_correction.estimate_mean`, and
   the FFT cross term against `mean_correction.block_terms` (brute force).
3. Reductions: X = σ with constant σ is X = 1; the empirical-Bayes correction at λ = 1 is
   the estimated-mean correction and at λ = 0 has variance V_B.
4. Exact simulation on a 30 × 30 grid: corrected block errors are unbiased and their
   variance matches the formula for designs X = σ, X = 1/σ and X = [1, m], and for the
   empirical-Bayes correction with known τ.

Usage: ``uv run python -m experiments.round5.check [--quick]``
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import pyspatialuncertainty as psu
from experiments.round5 import arms as ar
from experiments.round5 import shared as sh
from pyspatialuncertainty.mean_correction import block_terms, estimate_mean

N_SIM = 4000
TOL_VAR = 0.06


def check_vb(quick: bool) -> bool:
    ls = sh.build_class_landscape()
    layout = ls.layout
    sd = np.sqrt(ls.p_true[2] * (1 - ls.p_true[2]))
    models = {
        "Exp 32": psu.vgm(0.8, "Exp", 32.0, nugget=0.2),
        "Sph 150": psu.vgm(0.7, "Sph", 150.0, nugget=0.3),
        "Gau 20": psu.vgm(0.9, "Gau", 20.0, nugget=0.1),
        "Exp 0.4": psu.vgm(1.0, "Exp", 0.4),
    }
    exact = {
        k: sh.exact_vb(layout, sd, sh.make_kernels(layout, sh.corr_of_model(m)))
        for k, m in models.items()
    }
    ok = True
    for name, bset, n_mc in (
        ("small", ls.bsets["small"], 5_000 if quick else 20_000),
        ("large", ls.bsets["large"], 200_000 if quick else 1_000_000),
    ):
        resolved = psu.resolve_blocks(bset, np.ones(sh.N * sh.N, dtype=bool))
        res = psu.mc_block_variance(
            resolved,
            [psu.Integrand(sd, sh.corr_of_model(m).rho) for m in models.values()],
            n_mc=n_mc,
            seed=5,
        )
        idx = (
            np.flatnonzero(layout.side_of_block <= 64)
            if name == "small"
            else np.flatnonzero(layout.side_of_block > 64)
        )
        for k, key in enumerate(models):
            e = exact[key][idx]
            z = (e - res.variance[k]) / res.mcse[k]
            rel = np.abs(e / res.variance[k] - 1)
            for side in sorted(set(layout.side_of_block[idx])):
                sel = layout.side_of_block[idx] == side
                frac = float(np.mean(np.abs(z[sel]) <= 3))
                if key == "Exp 0.4" and side > 4:
                    # Monte Carlo is unreliable here: almost no pair falls within a cell or two of
                    # each other. Compare with mean(σ²)·∫ρ/|B| (∫ρ = 2πa²), which ignores the
                    # block edges and the variation of σ between neighbouring cells (a few %; a
                    # 20M-pair Monte Carlo on one 64-cell tile agreed with the exact value
                    # within 1.8 standard errors).
                    cells = [sh.tile_cells(sh.N, side, int(t)) for t in layout.tiles[side]]
                    approx = (
                        np.array([np.mean(sd[c] ** 2) for c in cells])
                        * 2
                        * math.pi
                        * 0.4**2
                        / side**2
                    )
                    rel_a = np.abs(e[sel] / approx - 1)
                    good = float(np.max(rel_a)) < 0.10
                    ok &= good
                    print(
                        f"V_B {key:8s} side {side:5d}: vs analytic approximation max rel. diff "
                        f"{np.max(rel_a):.4f} {'ok' if good else 'FAIL'}"
                    )
                    continue
                good = frac >= 0.97 or float(np.max(rel[sel])) < 0.01
                ok &= good
                print(
                    f"V_B {key:8s} side {side:5d}: |z| ≤ 3 in {frac:.1%} of {sel.sum()} blocks, "
                    f"max |z| {np.max(np.abs(z[sel])):.1f}, max rel. diff {np.max(rel[sel]):.4f} "
                    f"{'ok' if good else 'FAIL'}"
                )
    return ok


def check_gls_cross() -> bool:
    ls = sh.build_class_landscape()
    layout = ls.layout
    rng = np.random.default_rng(3)
    cells = rng.choice(sh.N * sh.N, 300, replace=False)
    x, y = layout.cx[cells], layout.cy[cells]
    model = psu.vgm(0.6, "Exp", 40.0, nugget=0.4)
    corr = sh.corr_of_model(model)
    z = 0.3 + rng.standard_normal(300)
    est = estimate_mean(model, x, y, z)
    g = ar.gls(ar.gls_factor(corr, x, y), np.ones(300), z)
    d_beta = abs(est.beta - g.beta[0])
    d_w = float(np.max(np.abs(est.weights - g.G[0])))
    sd = np.sqrt(ls.p_true[0] * (1 - ls.p_true[0]))
    kern = sh.make_kernels(layout, corr)
    fft_terms = ar.linear_correction(
        layout,
        kern,
        sd,
        np.zeros(layout.n_blocks),
        cells,
        [np.ones(sh.N * sh.N)],
        g,
        np.zeros(layout.n_blocks),
    )
    resolved = psu.resolve_blocks(ls.bsets["large"], np.ones(sh.N * sh.N, dtype=bool))
    bt = block_terms(est, sd, ls.grid, resolved, cross_term=True)
    large = np.flatnonzero(layout.side_of_block > 64)
    d_cross = float(
        np.max(
            np.abs(2 * fft_terms.xb[large, 0] * fft_terms.C[0, large] - bt.cross) / np.abs(bt.cross)
        )
    )
    d_shift = float(np.max(np.abs(fft_terms.est[large] - bt.shift)))
    good = d_beta < 1e-10 and d_w < 1e-10 and d_cross < 1e-9 and d_shift < 1e-12
    print(
        f"GLS vs estimate_mean: |Δβ| {d_beta:.1e}, max |Δw| {d_w:.1e}; FFT cross term vs "
        f"block_terms: max rel. diff {d_cross:.1e}; shift {d_shift:.1e} {'ok' if good else 'FAIL'}"
    )
    return good


def small_layout(n: int = 30) -> sh.Layout:
    return sh.make_layout(n, (10, n), {10: np.arange(9), n: np.arange(1)})


def check_reductions() -> bool:
    lay = small_layout()
    rng = np.random.default_rng(4)
    model = psu.vgm(0.75, "Exp", 4.0, nugget=0.25)
    corr = sh.corr_of_model(model)
    kern = sh.make_kernels(lay, corr)
    cells = rng.choice(900, 60, replace=False)
    x, y = lay.cx[cells], lay.cy[cells]
    chol = ar.gls_factor(corr, x, y)
    z = 0.2 + rng.standard_normal(60)
    sd = np.full(900, 0.7)
    vb = sh.exact_vb(lay, sd, kern)
    share = np.zeros(lay.n_blocks)
    one = ar.linear_correction(
        lay, kern, sd, share, cells, [np.ones(900)], ar.gls(chol, np.ones(60), z), vb
    )
    lin = ar.linear_correction(lay, kern, sd, share, cells, [sd], ar.gls(chol, sd[cells], z), vb)
    e1, v1 = ar.eb_correction(one, share, 1.0)
    e0, v0 = ar.eb_correction(one, share, 0.0)
    good = (
        np.allclose(one.est, lin.est, atol=1e-12)
        and np.allclose(one.var, lin.var, atol=1e-12)
        and np.allclose(e1, one.est)
        and np.allclose(v1, one.var)
        and np.allclose(v0, vb)
        and np.allclose(e0, share)
    )
    print(
        "reductions (X = σ with constant σ is X = 1; EB at λ = 1 and λ = 0): "
        + ("ok" if good else "FAIL")
    )
    return bool(good)


def check_exact_simulation(rng: np.random.Generator) -> bool:
    """Gaussian errors simulated exactly at cell centres (discrete truth, centre-support V_B)."""
    lay = small_layout()
    n_cells = 900
    model = psu.vgm(0.75, "Exp", 4.0, nugget=0.25)
    corr = sh.corr_of_model(model)
    kern = sh.make_kernels(lay, corr)
    centre_kern = sh.Kernels({s: sh.Kernel(s, corr, "centre") for s in lay.sides}, kern.centre)
    cx, cy = lay.cx, lay.cy
    sd = 0.5 + 1.5 * cx / 30.0
    m = np.sin(cx / 5.0) + np.cos(cy / 7.0)
    m = (m - m.mean()) / m.std()
    val = rng.choice(n_cells, 60, replace=False)
    xv, yv = cx[val], cy[val]
    px, py = np.concatenate([cx, xv]), np.concatenate([cy, yv])
    C = corr.struct(np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :]))
    L = np.linalg.cholesky(C + 1e-9 * np.eye(C.shape[0]))
    eta = model.nugget / model.sill
    vb = sh.exact_vb(lay, sd, centre_kern)
    chol = ar.gls_factor(corr, xv, yv)
    ok = True
    designs = {
        "X = σ": ([sd], np.array([0.5])),
        "X = 1/σ": ([1.0 / sd], np.array([0.4])),
        "X = [1, m]": ([np.ones(n_cells), m], np.array([0.3, -0.4])),
    }
    w_all = np.array([L @ rng.standard_normal(C.shape[0]) for _ in range(N_SIM)])
    nug = math.sqrt(eta) * rng.standard_normal((N_SIM, val.size))
    blocks = [lay.n_blocks - 1, 0]
    for name, (cols, beta) in designs.items():
        Xc = np.column_stack(cols)
        Xv = Xc[val]
        g = ar.gls(chol, Xv, np.zeros(val.size))
        corr_terms = ar.linear_correction(
            lay, centre_kern, sd, np.zeros(lay.n_blocks), val, cols, g, vb
        )
        mean_cells = Xc @ beta
        for b in blocks:
            bc = lay.bmat[b].indices
            errs = np.empty(N_SIM)
            for r in range(N_SIM):
                w = w_all[r]
                truth = np.mean(sd[bc] * (mean_cells[bc] + w[:n_cells][bc]))
                z = Xv @ beta + w[n_cells:] + nug[r]
                errs[r] = truth - corr_terms.xb[b] @ (g.G @ z)
            formula = corr_terms.var[b]
            rel = abs(errs.var() - formula) / formula
            se = errs.std() / math.sqrt(N_SIM)
            good = rel < TOL_VAR and abs(errs.mean()) < 4 * se
            ok &= good
            print(
                f"{name:10s} block {b:2d}: bias {errs.mean():+.4f} (se {se:.4f}), variance "
                f"simulated {errs.var():.5f} vs formula {formula:.5f} (rel {rel:.3f}) "
                f"{'ok' if good else 'FAIL'}"
            )
    # empirical Bayes with known τ: μ ~ N(0, τ²) per draw
    tau = 0.5
    g1 = ar.gls(chol, np.ones(val.size), np.zeros(val.size))
    one = ar.linear_correction(
        lay, centre_kern, sd, np.zeros(lay.n_blocks), val, [np.ones(n_cells)], g1, vb
    )
    lam, _ = ar.eb_lambda(0.0, float(g1.A[0, 0]), tau**2)
    _, var_eb = ar.eb_correction(one, np.zeros(lay.n_blocks), lam)
    mus = tau * rng.standard_normal(N_SIM)
    for b in blocks:
        bc = lay.bmat[b].indices
        errs = np.empty(N_SIM)
        for r in range(N_SIM):
            w = w_all[r]
            truth = np.mean(sd[bc] * (mus[r] + w[:n_cells][bc]))
            z = mus[r] + w[n_cells:] + nug[r]
            errs[r] = truth - lam * one.xb[b, 0] * float(g1.G[0] @ z)
        rel = abs(errs.var() - var_eb[b]) / var_eb[b]
        se = errs.std() / math.sqrt(N_SIM)
        good = rel < TOL_VAR and abs(errs.mean()) < 4 * se
        ok &= good
        print(
            f"EB τ = {tau}  block {b:2d}: λ = {lam:.3f}, bias {errs.mean():+.4f} (se {se:.4f}), "
            f"variance simulated {errs.var():.5f} vs formula {var_eb[b]:.5f} (rel {rel:.3f}) "
            f"{'ok' if good else 'FAIL'}"
        )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-vb", action="store_true")
    a = ap.parse_args()
    ok = check_reductions()
    ok &= check_exact_simulation(np.random.default_rng(20261029))
    ok &= check_gls_cross()
    if not a.skip_vb:
        ok &= check_vb(a.quick)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
