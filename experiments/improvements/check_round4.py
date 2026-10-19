"""Checks of the round-4 shortcuts against the round-3 code on a 64 × 64 landscape.

* GLS weights with X = 1 equal ``candidates.gls_terms``;
* the FFT cross term and shift equal ``candidates.unknown_mean_terms`` (cell sums with a
  cells × points distance matrix), for a variogram model and for a callable correlogram
  (``round3_ideal.terms_callable``);
* the FFT block variances equal the direct cell double sums (and agree with the package's
  Monte Carlo integral up to its error and the cell discretisation);
* the vectorized PPI and Olofsson estimators equal ``candidates.ppi_blocks`` and
  ``candidates.olofsson_blocks`` wherever the latter are defined;
* the exact class-map oracle (``round4.exact_class_variance``) equals the scenario's exact
  block SD (``LatentClassScenario.exact_block_sd``) where it uses all cell pairs, and is
  within Monte Carlo error of it elsewhere;
* holdout samples are distinct cells, nested, and clusters stay within their radius.

Usage::

    uv run --group experiments python -m experiments.improvements.check_round4
"""

from __future__ import annotations

import numpy as np

import pyspatialuncertainty as psu
from experiments.improvements import candidates as cand
from experiments.improvements import round3_ideal as ideal
from experiments.improvements import round4 as r4
from pyspatialuncertainty.synthetic import LatentClassScenario, SubGridLayout
from pyspatialuncertainty.synthetic.continuous import default_sigma_surface

N = 64
SIDES = (4, 16, 64)


def close(name: str, a: np.ndarray, b: np.ndarray, rtol: float = 1e-8) -> None:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    err = float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))
    print(f"{name:<44} max rel. diff {err:.2e}")
    assert err < rtol, name


def main() -> None:
    rng = np.random.default_rng(4)
    lay = SubGridLayout(n=N, f=1)
    blocks = r4.make_blocks(N, SIDES, 64)
    bmat = blocks.bmat
    cx, cy = lay.grid.cell_centres(np.arange(N * N))
    sd = default_sigma_surface(lay).ravel()
    model = psu.vgm(0.8, "Exp", 8.0, nugget=0.2)
    cells = r4.sample_cells("random", N, 150, rng)
    x, y = cx[cells], cy[cells]
    z = rng.standard_normal(cells.size) + 0.4
    D = cand.distance_matrix(x, y)
    d_cells = np.hypot(cx[:, None] - x[None, :], cy[:, None] - y[None, :])
    ones_v, ones_c = np.ones((cells.size, 1)), np.ones((N * N, 1))
    x_b = np.asarray(bmat @ sd)

    # GLS and cross term, variogram model
    ref = cand.gls_terms(model, x, y, z, ones_v)
    g = r4.gls_ones(cand.structured_correlation(model, D))
    close("GLS A", g.A, ref.A[0, 0])
    close("GLS G", g.G, ref.G[0])
    shift_ref, added_ref = cand.unknown_mean_terms(
        bmat, sd, ones_c, model, d_cells, ref, cross=True
    )
    kern = r4.Kernel(N, lambda d: cand.structured_correlation(model, d))
    cross = np.asarray(bmat @ (sd * r4.cross_field(kern, cells, g.G)))
    shift, _ = r4.corrected(x_b, np.zeros_like(x_b), g, z, cross)
    close("shift (variogram model)", shift, shift_ref)
    close("added variance with cross term", x_b**2 * g.A - 2 * x_b * cross, added_ref)
    _, added_nc = cand.unknown_mean_terms(bmat, sd, ones_c, model, d_cells, ref, cross=False)
    close("added variance without cross term", x_b**2 * g.A, added_nc)

    # callable correlogram (the class-map oracle path)
    corr = psu.CallableCorrelation(lambda h: np.where(h > 0, 0.5 * np.exp(-h / 5.0), 1.0))
    rho0 = 0.55

    def fn(d: np.ndarray) -> np.ndarray:
        return np.where(d > 0, np.asarray(corr(d)), rho0)

    ref_c = ideal.gls_callable(corr, rho0, x, y, z, ones_v)
    g_c = r4.gls_ones(fn(D))
    close("GLS G (callable)", g_c.G, ref_c.G[0])
    shift_ref, added_ref = ideal.terms_callable(bmat, sd, ones_c, corr, rho0, d_cells, ref_c)
    cross_c = np.asarray(bmat @ (sd * r4.cross_field(r4.Kernel(N, fn), cells, g_c.G)))
    shift_c, _ = r4.corrected(x_b, np.zeros_like(x_b), g_c, z, cross_c)
    close("shift (callable)", shift_c, shift_ref)
    close("added variance (callable)", x_b**2 * g_c.A - 2 * x_b * cross_c, added_ref)

    # exact block variance
    def direct(fun) -> np.ndarray:
        out = []
        for blk in blocks.resolved:
            c = blk.cells
            dd = np.hypot(cx[c][:, None] - cx[c][None, :], cy[c][:, None] - cy[c][None, :])
            out.append(float(np.mean(np.outer(sd[c], sd[c]) * fun(dd))))
        return np.array(out)

    v_model = r4.exact_block_variance(blocks, sd, lambda d: cand.structured_correlation(model, d))
    close(
        "exact V_B (variogram model)",
        v_model,
        direct(lambda d: cand.structured_correlation(model, d)),
    )
    close("exact V_B (callable)", r4.exact_block_variance(blocks, sd, fn), direct(fn))
    mc = psu.mc_block_variance(
        blocks.resolved, [psu.Integrand(sd, cand.correlation(model))], n_mc=200_000, seed=1
    )
    rel = np.abs(mc.variance[0] - v_model) / v_model
    for side in SIDES:
        sel = np.asarray(blocks.groups) == str(side)
        print(
            f"  MC (continuous) vs cell sums, {side:>2}-cell tiles: median rel. diff "
            f"{np.median(rel[sel]):.3f} (median MC rel. SE "
            f"{np.median(mc.mcse[0][sel] / v_model[sel]):.3f})"
        )

    # design-based arms
    sc = LatentClassScenario(
        layout=lay,
        latent=psu.vgm(0.8, "Exp", 8.0, nugget=0.2),
        signal_range=8.0,
        prevalence=r4.PREVALENCE,
        classes=r4.CLASSES,
    )
    p = sc.p_true.reshape(3, -1)
    map_class = np.argmax(p, axis=0)
    vc = r4.sample_cells("clustered", N, 1500, rng)
    label = np.array([rng.choice(3, p=p[:, c]) for c in vc])
    pts, blk = blocks.membership(vc)
    weights = np.asarray(bmat @ np.eye(3)[map_class])
    for k in range(3):
        mine, mine_sd = r4.olofsson(weights, blk, map_class[vc][pts], (label == k)[pts])
        mine = mine - np.asarray(bmat @ p[k])
        ref_s, ref_sd = cand.olofsson_blocks(blocks.resolved, map_class, p[k], vc, label == k, 3)
        ok = np.isfinite(ref_s)
        n_b = np.bincount(blk, minlength=len(blocks.groups))
        assert np.all(np.isfinite(mine) == (n_b >= r4.N_MIN)), "Olofsson availability"
        close(f"Olofsson shift, class {k} ({ok.sum()} blocks)", mine[ok], ref_s[ok], 1e-10)
        close(f"Olofsson SE, class {k}", mine_sd[ok], ref_sd[ok], 1e-10)
        print(f"  merged strata make {int((np.isfinite(mine) & ~ok).sum())} more blocks available")
    resid = rng.standard_normal(vc.size)
    s_ppi, sd_ppi = r4.ppi(len(blocks.groups), blk, resid[pts])
    ref_s, ref_sd = cand.ppi_blocks(blocks.resolved, vc, resid)
    assert np.array_equal(np.isfinite(s_ppi), np.isfinite(ref_s)), "PPI availability"
    ok = np.isfinite(ref_s)
    close("PPI shift", s_ppi[ok], ref_s[ok], 1e-10)
    close("PPI SE", sd_ppi[ok], ref_sd[ok], 1e-10)

    # samples
    # random: a shuffled draw without replacement, so each prefix is a simple random sample;
    # clustered: whole clusters in the order drawn
    for design in r4.DESIGNS:
        a = r4.sample_cells(design, 256, 1000, np.random.default_rng(9))
        assert np.unique(a).size == a.size, f"{design}: duplicate cells"
    b = r4.sample_cells("clustered", 256, 100, np.random.default_rng(9))
    assert np.array_equal(a[:100], b), "clustered samples are not nested"
    c = r4.sample_cells("clustered", 256, 1000, np.random.default_rng(9)).reshape(
        -1, r4.CLUSTER_SIZE
    )
    row, col = np.divmod(c, 256)
    spread = np.hypot(row[:, :, None] - row[:, None, :], col[:, :, None] - col[:, None, :]).max()
    assert spread <= 2 * r4.CLUSTER_RADIUS, "cluster wider than its diameter"
    print(f"samples: distinct, nested; largest within-cluster distance {spread:.1f} cells")
    assert r4.landscape_groups() == r4.make_blocks(r4.N_SIDE, r4.SIDES, r4.N_SMALL).groups

    # exact class-map oracle: a 4-cell and a 16-cell tile (all pairs), and the domain (MC)
    ls = r4.landscape("categorical", N, SIDES)
    v, se = r4.exact_class_variance(ls, 2, np.random.default_rng(0))
    for b in (5, 256 + 3):
        ref = ls["sc"].exact_block_sd(2, ls["blocks"].resolved[b].cells) ** 2
        close(f"exact oracle variance, block {b} (all pairs)", v[b], ref, 1e-8)
    ref = ls["sc"].exact_block_sd(2, ls["blocks"].resolved[-1].cells) ** 2
    print(f"exact oracle variance, domain: {(v[-1] - ref) / se[-1]:+.2f} Monte Carlo SEs off")
    assert abs(v[-1] - ref) < 4 * se[-1], "exact oracle variance, domain"
    print("all round-4 checks passed")


if __name__ == "__main__":
    main()
