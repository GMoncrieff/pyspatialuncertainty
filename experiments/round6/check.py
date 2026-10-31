"""Stage 0 checks of the round-6 harness.

1. Pairing with round 5 at 1024²: the blocks and realization 0's truth, samples and labels
   equal round 5's, and the ``ok`` arm reproduces round 5's estimates and variances.
2. Threshold-shaped correlation against scipy's bivariate normal, and (2/π)·asin at t = 0;
   the threshold REML profile against a direct implementation.
3. Calibrator scores and Hessians against finite differences; the dependence-aware sandwich
   against a Monte Carlo of clustered labels (where the iid sandwich is too small).
4. Continuous maps: the design X = [1, 1/σ, m/σ] leaves no calibration variance after a
   linear recalibration, and its variance formula matches an exact simulation.
5. Interval rules on edge values, and the polars expressions against numpy.
6. The 256² world: blocks, shares and samples.

Usage: ``uv run python -m experiments.round6.check [--quick]``
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import polars as pl
from scipy import integrate, special, stats

import pyspatialuncertainty as psu
from experiments.round5 import arms as ar
from experiments.round5 import shared as sh
from experiments.round6 import calib as cb
from experiments.round6 import cont, fits, run, summarize
from experiments.round6 import world as wd


def report(name: str, good: bool, detail: str) -> bool:
    print(f"{name}: {detail} {'ok' if good else 'FAIL'}", flush=True)
    return good


def check_pairing() -> bool:
    from experiments.round5 import run as r5

    w = wd.class_world(1024)
    ls = r5.landscape()
    good = all(
        np.array_equal(w.layout.tiles[s], ls.layout.tiles[s]) for s in sh.SIDES
    ) and np.array_equal(w.p_true, ls.p_true)
    d6, d5 = wd.class_draws(w, 0), sh.class_draws(ls, 0)
    for name in ("cells", "clusters", "labels", "cal_cells", "cal_labels"):
        good &= all(np.array_equal(getattr(d6, name)[d], getattr(d5, name)[d]) for d in sh.DESIGNS)
    good &= np.array_equal(d6.truth, d5.truth)
    good &= all(np.array_equal(d6.perms[k], d5.perms[k]) for k in d5.perms)
    ok = report("pairing: blocks, truth, samples, labels", bool(good), "identical to round 5")
    o5, _ = r5.setting(
        ls, d5, "random", 50, ls.p_true, "M1", {"ok"}, calib=None, boot=0, vmax=0,
        rng=np.random.default_rng(0),
    )  # fmt: skip
    o6, _ = run.setting(
        w, d6, "random", 50, w.p_true, ("ok",), calib=None, rng=np.random.default_rng(0)
    )
    diff = max(
        max(np.max(np.abs(o5[("ok", c)][k] - o6[("ok", c)][k])) for k in (0, 1)) for c in sh.CLASSES
    )
    return ok & report("pairing: ok arm (M1, random, n = 50)", diff < 1e-12, f"max |Δ| {diff:.1e}")


def check_threshold() -> bool:
    good = True
    worst = 0.0
    for p in (0.67, 0.30, 0.03):
        thr = fits.Threshold.at(p)
        for r in (0.05, 0.3, 0.6, 0.9, 0.99):
            # P(Y₁ > t, Y₂ > t) by quadrature over y₁ (independent of Owen's T)
            p11 = integrate.quad(
                lambda y1, t=thr.t, r=r: (
                    stats.norm.pdf(y1) * stats.norm.sf((t - r * y1) / math.sqrt(1 - r * r))
                ),
                thr.t,
                np.inf,
                epsabs=1e-13,
                epsrel=1e-11,
            )[0]
            exact = (p11 - p * p) / (p * (1 - p))
            worst = max(worst, abs(float(thr(np.array([r]))[0]) - exact))
    good &= worst < 1e-5
    ok = report("threshold correlation vs bivariate normal", good, f"max |Δ| {worst:.1e}")
    half = fits.Threshold.at(0.5)
    r = np.linspace(0, 1, 101)
    d_asin = float(np.max(np.abs(half(r) - 2 / math.pi * np.arcsin(r))))
    ok &= report("threshold correlation at t = 0 vs (2/π)·asin", d_asin < 1e-6, f"{d_asin:.1e}")
    # REML profile against a direct implementation
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 300, 120), rng.uniform(0, 300, 120)
    z = 0.2 + rng.standard_normal(120)
    D = fits.distances(x, y)
    iu = np.triu_indices(120, 1)
    thr = fits.Threshold.at(0.3)
    prof = fits.threshold_profile(z, D[iu], iu, thr)
    worst = 0.0
    for theta in (np.array([-1.0, 3.0]), np.array([0.5, 4.0]), np.array([2.0, 2.0])):
        corr = thr.corr(theta)
        K = corr.struct(D)
        K[np.diag_indices(120)] = 1.0 + 1e-10
        Ki = np.linalg.inv(K)
        one = np.ones(120)
        s11 = one @ Ki @ one
        q = z @ Ki @ z - (one @ Ki @ z) ** 2 / s11
        direct = 0.5 * (119 * math.log(q) + np.linalg.slogdet(K)[1] + math.log(s11))
        worst = max(worst, abs(prof(theta)[0] - direct))
    return ok & report("threshold REML profile vs direct", worst < 1e-7, f"max |Δ| {worst:.1e}")


def _nll(kind: str, theta: np.ndarray, lp: np.ndarray, labels: np.ndarray, ridge: bool) -> float:
    k, m = lp.shape
    if kind == "vector":
        z = theta[:k, None] * lp + theta[k:, None]
        pen = 0.5 * sh.VEC_RIDGE * (((theta[:k] - 1) ** 2).sum() + (theta[k:] ** 2).sum())
    else:
        z = lp / math.exp(theta[0])
        pen = 0.0
    val = float((special.logsumexp(z, axis=0) - z[labels, np.arange(m)]).sum())
    return val + (pen if ridge else 0.0)


def check_calibrator() -> bool:
    rng = np.random.default_rng(2)
    p = rng.dirichlet([2.0, 1.0, 0.5], 300).T
    lp = np.log(p)
    labels = np.array(
        [rng.choice(3, p=p[:, i] ** 1.5 / (p[:, i] ** 1.5).sum()) for i in range(300)]
    )
    ok = True
    for kind in ("vector", "temperature"):
        par = sh.CALIBRATORS[kind][0](lp, labels)
        th = cb.class_theta(kind, par)
        s, _, H = cb.class_parts(kind, th, lp, labels)
        h = 1e-5
        q = th.size
        grad = np.array(
            [
                (
                    _nll(kind, th + h * e, lp, labels, False)
                    - _nll(kind, th - h * e, lp, labels, False)
                )
                / (2 * h)
                for e in np.eye(q)
            ]
        )
        Hn = np.empty((q, q))
        for i, ei in enumerate(np.eye(q)):
            for j, ej in enumerate(np.eye(q)):
                Hn[i, j] = (
                    _nll(kind, th + h * ei + h * ej, lp, labels, True)
                    - _nll(kind, th + h * ei - h * ej, lp, labels, True)
                    - _nll(kind, th - h * ei + h * ej, lp, labels, True)
                    + _nll(kind, th - h * ei - h * ej, lp, labels, True)
                ) / (4 * h * h)
        d_g = float(np.max(np.abs(s.sum(axis=0) - grad)) / max(1.0, np.max(np.abs(grad))))
        d_h = float(np.max(np.abs(H - Hn)) / np.max(np.abs(Hn)))
        ok &= report(
            f"calibrator derivatives ({kind})", d_g < 1e-5 and d_h < 1e-4,
            f"score rel. {d_g:.1e}, Hessian rel. {d_h:.1e}",
        )  # fmt: skip
    return ok


def check_sandwich_mc(n_sim: int) -> bool:
    """Clustered labels with a shared-uniform coupling, Cov(Yᵢ, Yⱼ) = q²·W within a cluster:
    the model sandwich (known ρ) and the pair-correlation sandwich (from the locations only)
    against the spread of the refitted calibrator; the iid sandwich is too small."""
    rng = np.random.default_rng(3)
    n_cl, size, q, side = 30, 10, 0.6, 1024
    cells, cid = wd.sample_cells(side, "clustered", n_cl * size, rng)  # the harness's sampler
    probs = rng.dirichlet([3.0, 2.0, 1.0], n_cl)
    p = probs[cid].T  # (3, 300): one probability vector per cluster
    lp = np.log(p)
    x, y = (cells % side) + 0.5, side - (cells // side) - 0.5
    pairs = cb.pair_weights(x, y, side)
    iu = np.triu_indices(cid.size, 1)
    in_unit = cid[iu[0]] == cid[iu[1]]
    same = (iu[0][in_unit], iu[1][in_unit], np.ones(int(in_unit.sum())))  # the cluster meat
    R = q * q * (cid[:, None] == cid[None, :]).astype(float)
    np.fill_diagonal(R, 0.0)
    cum = np.cumsum(p, axis=0)
    thetas, model, geo, iid, clu = [], [], [], [], []
    for _ in range(n_sim):
        u_c = rng.uniform(size=n_cl)[cid]
        own = rng.uniform(size=cid.size)
        u = np.where(rng.uniform(size=cid.size) < q, u_c, own)
        labels = np.minimum((u[None, :] > cum).sum(axis=0), 2)
        th = cb.class_theta("vector", sh.fit_vector(lp, labels))
        thetas.append(th)
        model.append(np.diag(cb.class_sandwich("vector", th, lp, labels, R=R)))
        geo.append(np.diag(cb.class_sandwich("vector", th, lp, labels, pairs=pairs)))
        iid.append(np.diag(cb.class_sandwich("vector", th, lp, labels)))
        clu.append(np.diag(cb.class_sandwich("vector", th, lp, labels, pairs=same)))
    emp = np.std(np.array(thetas), axis=0)
    r_model = emp / np.sqrt(np.mean(model, axis=0))
    r_geo = np.sqrt(np.mean(clu, axis=0) / np.mean(geo, axis=0))
    r_iid = emp / np.sqrt(np.mean(iid, axis=0))
    good = bool(
        np.all((r_model > 0.8) & (r_model < 1.25))
        and np.all((r_geo > 0.9) & (r_geo < 1.1))
        and np.median(r_iid) > 1.3
    )
    return report(
        "sandwiches vs Monte Carlo (30 clusters of 10, ρ = 0.36)",
        good,
        f"empirical / model SD {r_model.min():.2f}-{r_model.max():.2f}; cluster-ID / "
        f"pair-correlation sandwich SD {r_geo.min():.2f}-{r_geo.max():.2f}; "
        f"empirical / iid SD {np.median(r_iid):.2f}",
    )


def check_pair_weights() -> bool:
    """A random sample gets (almost) no weighted pairs; a clustered one gets its clusters."""
    rng = np.random.default_rng(6)
    good = True
    detail = []
    for side, k in ((1024, 5000), (256, 5000), (1024, 500)):
        for design in sh.DESIGNS:
            cells, clus = wd.sample_cells(side, design, k, rng)
            x, y = (cells % side) + 0.5, side - (cells // side) - 0.5
            _, _, wt = cb.pair_weights(x, y, side)
            counts = np.bincount(clus)
            n_same = float((counts * (counts - 1) / 2).sum())
            detail.append(f"{side}² {design[:4]} n={k}: Σw {wt.sum():.0f} / {n_same:.0f}")
            if design == "random":
                good &= wt.sum() < 0.01 * k
            elif side == 1024:  # sparse clusters; dense overlapping ones (256²) are reported
                good &= 0.7 * n_same < wt.sum() < 1.4 * n_same
    return report("pair-correlation weights (Σw / same-unit pairs)", bool(good), "; ".join(detail))


def check_cont_absorbs() -> bool:
    arms = ("ok+cs", "ok+cg", "ok+x3+cs", "ok+x3+cg", "ok+x3+ci")
    df, dg = cont.realization(256, 0, "C2l", arms)
    land = df.filter((pl.col("bside") == 256) & (pl.col("arm") != "truth"))
    worst = 0.0
    for arm in ("ok+x3+cs", "ok+x3+cg", "ok+x3+ci"):
        v = land.filter(pl.col("arm") == arm).select("design", "n", "var")
        calv = dg.select("design", "n", pl.col(f"{arm}_calvar_land").alias("cv"))
        j = v.join(calv, on=["design", "n"])
        worst = max(worst, float((j["cv"] / j["var"]).max()))
    pos = float(dg["ok+cs_calvar_land"].min())
    return report(
        "continuous X = [1, 1/σ, m/σ] absorbs the linear recalibration",
        worst < 1e-10 and pos > 0,
        f"calibration share of the variance {worst:.1e}; with X = [1] it is {pos:.1e} or more",
    )


def check_x3_simulation(n_sim: int) -> bool:
    """Gaussian errors at cell centres on a 30 × 30 grid with mean X3·β (centre-support V_B)."""
    rng = np.random.default_rng(4)
    lay = sh.make_layout(30, (10, 30), {10: np.arange(9), 30: np.arange(1)})
    model = psu.vgm(0.75, "Exp", 4.0, nugget=0.25)
    corr = sh.corr_of_model(model)
    kern = sh.make_kernels(lay, corr)
    centre = sh.Kernels({s: sh.Kernel(s, corr, "centre") for s in lay.sides}, kern.centre)
    cx, cy = lay.cx, lay.cy
    sd = 0.5 + 1.5 * cx / 30.0
    m = np.sin(cx / 5.0) + np.cos(cy / 7.0)
    m = (m - m.mean()) / m.std()
    cols = [np.ones(900), 1.0 / sd, m / sd]
    beta = np.array([0.3, -0.2, 0.4])
    val = rng.choice(900, 80, replace=False)
    px, py = np.concatenate([cx, cx[val]]), np.concatenate([cy, cy[val]])
    C = corr.struct(np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :]))
    L = np.linalg.cholesky(C + 1e-9 * np.eye(C.shape[0]))
    vb = sh.exact_vb(lay, sd, centre)
    Xv = np.column_stack([c[val] for c in cols])
    g = ar.gls(ar.gls_factor(corr, cx[val], cy[val]), Xv, np.zeros(80))
    terms = ar.linear_correction(lay, centre, sd, np.zeros(lay.n_blocks), val, cols, g, vb)
    mean_cells = np.column_stack(cols) @ beta
    eta = model.nugget / model.sill
    ok = True
    for b in (lay.n_blocks - 1, 0):
        bc = lay.bmat[b].indices
        errs = np.empty(n_sim)
        for r in range(n_sim):
            wv = L @ rng.standard_normal(C.shape[0])
            truth = np.mean(sd[bc] * (mean_cells[bc] + wv[:900][bc]))
            z = Xv @ beta + wv[900:] + math.sqrt(eta) * rng.standard_normal(80)
            errs[r] = truth - terms.xb[b] @ (g.G @ z)
        # the simulated variance has a relative SE of about √(2/n_sim)
        z_var = (errs.var() - terms.var[b]) / (terms.var[b] * math.sqrt(2.0 / n_sim))
        se = errs.std() / math.sqrt(n_sim)
        good = abs(z_var) < 3 and abs(errs.mean()) < 4 * se
        ok &= report(
            f"X = [1, 1/σ, m/σ] exact simulation, block {b}", good,
            f"bias {errs.mean():+.4f} (se {se:.4f}), variance {errs.var():.5f} vs "
            f"formula {terms.var[b]:.5f}",
        )  # fmt: skip
    return ok


def check_intervals() -> bool:
    est = np.array([0.0, 1.0, -0.01, 1.01, 0.3, 0.3, 0.02, 0.98])
    sd = np.array([0.05, 0.05, 0.05, 0.05, 0.0, 0.05, 0.03, 0.03])
    iv = summarize.intervals(est, sd)
    nlo, nhi = np.clip(iv["normal"][0], 0, 1), np.clip(iv["normal"][1], 0, 1)
    good = bool(
        np.all(iv["hybrid"][0] <= np.minimum(nlo, iv["logit"][0]) + 1e-15)
        and np.all(iv["hybrid"][1] >= np.maximum(nhi, iv["logit"][1]) - 1e-15)
        and all(np.all((lo >= 0) & (hi <= 1)) for k, (lo, hi) in iv.items() if k != "normal")
        and iv["hybrid"][0][0] == 0.0
        and iv["arcsine"][0][0] == 0.0
        and iv["logit"][0][6] > 0.0
        and iv["hybrid"][0][6] == 0.0
        and abs(iv["logit"][0][4] - 0.3) < 1e-12
        and abs(iv["arcsine"][1][4] - 0.3) < 1e-12
    )
    rng = np.random.default_rng(5)
    e, s = rng.uniform(-0.05, 1.05, 2000), rng.uniform(0, 0.2, 2000)
    df = pl.DataFrame({"est": e, "sd": s})
    worst = 0.0
    for rule, (lo, hi) in summarize.intervals(e, s).items():
        plo, phi = summarize._bounds(rule)
        got = df.select(plo.alias("lo"), phi.alias("hi"))
        worst = max(worst, float(np.max(np.abs(got["lo"].to_numpy() - lo))))
        worst = max(worst, float(np.max(np.abs(got["hi"].to_numpy() - hi))))
    return report(
        "interval rules (edge values; polars vs numpy)", good and worst < 1e-12, f"{worst:.1e}"
    )


def check_small_world() -> bool:
    w = wd.class_world(256)
    lay = w.layout
    dr = wd.class_draws(w, 0)
    good = (
        lay.sides == (4, 64, 256)
        and [lay.tiles[s].size for s in lay.sides] == [256, 16, 1]
        and np.allclose(w.p_true.sum(axis=0), 1.0)
        and np.allclose(np.asarray(lay.bmat.sum(axis=1)).ravel(), 1.0)
        and np.allclose(dr.truth.sum(axis=0), 1.0)
        and all(np.unique(dr.cells[d]).size == 5000 for d in sh.DESIGNS)
        and all(dr.cells[d].max() < 256 * 256 for d in sh.DESIGNS)
    )
    cw = wd.cont_world(256)
    good &= cw.layout.n_blocks == lay.n_blocks and abs(float(cw.m.std()) - 1.0) < 1e-12
    return report("256² world", bool(good), f"{lay.n_blocks} blocks, shares and samples")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ok = check_intervals()
    ok &= check_threshold()
    ok &= check_calibrator()
    ok &= check_small_world()
    ok &= check_x3_simulation(1500 if a.quick else 4000)
    ok &= check_pair_weights()
    ok &= check_sandwich_mc(150 if a.quick else 400)
    ok &= check_cont_absorbs()
    ok &= check_pairing()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
