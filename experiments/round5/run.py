"""Round 5 driver for class maps: every arm on one realization of one map.

Maps (``shared.CLASS_MAPS``): M1 calibrated; M2 / M3 Part 2's combined miscalibration
calibrated by vector scaling / temperature on its own n points; M4-M6 round 4's offset,
slope and scale miscalibrations with vector scaling. Realization r reuses the Landscape
Check's truth, samples, labels and correlogram subsets, so the baselines are paired with
Parts 1 and 2.

Usage::

    uv run python -m experiments.round5.run <out_dir> --maps M1,M2,M3 --start 0 --stop 50
        [--workers 4] [--boot 50] [--vparam-max-n 5000] [--arms current,ok,...]

Writes ``<out_dir>/<map>/r###.parquet`` (block estimates and variances per arm, plus the
truth) and ``<out_dir>/<map>/d###.parquet`` (fits and diagnostics); existing files are
skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import math
import resource
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
from scipy import special

from experiments.improvements.candidates import pairs, sigma_weighted, wls
from experiments.round5 import arms as ar
from experiments.round5 import shared as sh
from pyspatialuncertainty.variogram.likelihood import fit_likelihood

ALL_ARMS = (
    "current", "ok", "ok_lin", "ok_cil", "ok_cil_n", "ok_eb", "ok_eb_pool", "fam_es", "cap", "sw",
    "oracle_zero", "oracle_ok", "vparam", "current_cb", "ok_cb", "ppi", "olofsson", "A", "B", "D",
)  # fmt: skip
CAP_FRACTION = 0.25  # practical range ≤ ¼ of the fit sample's bounding-box diagonal

_LS: sh.ClassLandscape | None = None
_ORACLE: dict[int, sh.Corr] = {}
_MIS: dict[str, np.ndarray] = {}


def landscape() -> sh.ClassLandscape:
    global _LS
    if _LS is None:
        _LS = sh.build_class_landscape()
    return _LS


def oracle_corr(j: int) -> sh.Corr:
    if j not in _ORACLE:
        _ORACLE[j] = sh.corr_of_callable(landscape().sc.oracle_correlation(j), "oracle")
    return _ORACLE[j]


def log_mis(kind: str) -> np.ndarray:
    if kind not in _MIS:
        _MIS[kind] = np.log(sh.miscalibrate(landscape().p_raw, kind))
    return _MIS[kind]


def fit(z: np.ndarray, x: np.ndarray, y: np.ndarray, **kw: object) -> tuple[object, str]:
    """The package's likelihood fit; pure nugget when the errors carry no variation."""
    if np.ptp(z) == 0:
        return sh.NUGGET_ONLY, "fallback"
    try:
        return fit_likelihood(None, z, x, y, **kw).model, ""  # type: ignore[arg-type]
    except ValueError as err:
        return sh.NUGGET_ONLY, f"fallback: {err}"


def describe(prefix: str, model: object) -> dict[str, object]:
    return {
        f"{prefix}_family": sh.family(model),  # type: ignore[arg-type]
        f"{prefix}_range": sh.practical_range(model),  # type: ignore[arg-type]
        f"{prefix}_nugget": sh.nugget_share(model),  # type: ignore[arg-type]
    }


def setting(
    ls: sh.ClassLandscape,
    dr: sh.Draws,
    design: str,
    n: int,
    pf: np.ndarray,
    map_name: str,
    arms: set[str],
    *,
    calib: tuple[str, np.ndarray] | None,
    boot: int,
    vmax: int,
    rng: np.random.Generator,
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    """All arms for one (design, n) on the map ``pf`` (K × cells)."""
    layout = ls.layout
    bmat = layout.bmat
    nb = layout.n_blocks
    ones_cells = np.ones(pf.shape[1])
    cells = dr.cells[design][:n]
    lab = dr.labels[design][:n]
    clus = dr.clusters[design][:n]
    x, y = layout.cx[cells], layout.cy[cells]
    sub = dr.perms[(design, n)]
    xs, ys = x[sub], y[sub]
    share = np.stack([bmat @ pf[j] for j in range(len(sh.CLASSES))])
    sd = np.sqrt(pf * (1.0 - pf))
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    diags: list[dict[str, object]] = []
    oks: dict[int, ar.Corrected] = {}
    cur_var: dict[int, np.ndarray] = {}
    cil_maps: dict[int, np.ndarray] = {}
    cil_var: dict[int, np.ndarray] = {}
    combo_g: dict[int, dict[str, np.ndarray]] = {}  # GLS weights per combo, for the bootstrap

    def ok_like(
        j: int, corr: sh.Corr, z: np.ndarray, kern: sh.Kernels | None = None
    ) -> ar.Corrected:
        kern = kern or sh.make_kernels(layout, corr)
        vb = sh.exact_vb(layout, sd[j], kern)
        chol = ar.gls_factor(corr, x, y)
        g = ar.gls(chol, np.ones(n), z)
        return ar.linear_correction(layout, kern, sd[j], share[j], cells, [ones_cells], g, vb)

    for j, c in enumerate(sh.CLASSES):
        t0 = time.time()
        d: dict[str, object] = {"design": design, "n": n, "cls": c}
        yj = (lab == j).astype(np.float64)
        pj, sj = pf[j, cells], sd[j, cells]
        z = (yj - pj) / sj
        zs = z[sub]
        m0, f0 = fit(zs, xs, ys, mean="zero")
        m1, f1 = fit(zs, xs, ys, mean="constant")
        d |= (
            describe("f0", m0)
            | describe("f1", m1)
            | {"fallback": bool(f0 or f1), "fallback_msg": f0 or f1}
        )
        c0, c1 = sh.corr_of_model(m0), sh.corr_of_model(m1)
        k0, k1 = sh.make_kernels(layout, c0), sh.make_kernels(layout, c1)
        vb0, vb1 = sh.exact_vb(layout, sd[j], k0), sh.exact_vb(layout, sd[j], k1)
        out[("current", c)] = (share[j], vb0)
        cur_var[j] = vb0
        chol1 = ar.gls_factor(c1, x, y)
        g_ok = ar.gls(chol1, np.ones(n), z)
        ok = ar.linear_correction(layout, k1, sd[j], share[j], cells, [ones_cells], g_ok, vb1)
        oks[j] = ok
        out[("ok", c)] = (ok.est, ok.var)
        d |= {"mu": float(g_ok.beta[0]), "A": float(g_ok.A[0, 0])}

        if arms & {"ok_lin", "ok_cil", "ok_cil_n"}:
            g_lin = ar.gls(chol1, sj, z)
            lin = ar.linear_correction(layout, k1, sd[j], share[j], cells, [sd[j]], g_lin, vb1)
            out[("ok_lin", c)] = (lin.est, lin.var)
            d |= {"delta_lin": float(g_lin.beta[0]), "A_lin": float(g_lin.A[0, 0])}
            lp_pts = special.logit(pj)
            delta, bracketed = ar.cil_delta(chol1, lp_pts, yj, float(g_lin.beta[0]))
            pd_map = special.expit(special.logit(pf[j]) + delta)
            sdd = np.sqrt(pd_map * (1.0 - pd_map))
            pd_pts = pd_map[cells]
            s_pts = np.sqrt(pd_pts * (1.0 - pd_pts))
            g_cil = ar.gls(chol1, s_pts, (yj - pd_pts) / s_pts, prior_prec=1.0 / ar.CIL_PRIOR_VAR)
            vbd = sh.exact_vb(layout, sdd, k1)
            cil = ar.linear_correction(layout, k1, sdd, bmat @ pd_map, cells, [sdd], g_cil, vbd)
            out[("ok_cil", c)] = (bmat @ pd_map, cil.var)
            cil_maps[j], cil_var[j] = pd_map, cil.var
            d |= {"delta_cil": delta, "cil_bracketed": bracketed}

        lam, tau2 = ar.eb_lambda(float(g_ok.beta[0]), float(g_ok.A[0, 0]))
        out[("ok_eb", c)] = ar.eb_correction(ok, share[j], lam)
        d |= {"eb_tau2": tau2, "eb_lambda": lam}

        if "fam_es" in arms:
            if sh.family(m1) in ("Exp", "Sph", "Nug"):
                out[("fam_es", c)] = (ok.est, ok.var)
                d |= {"fam_es": "same"}
            else:
                m4, f4 = fit(zs, xs, ys, mean="constant", models=("Exp", "Sph"))
                r4 = ok_like(j, sh.corr_of_model(m4), z)
                out[("fam_es", c)] = (r4.est, r4.var)
                d |= describe("f4", m4) | {"fam_es": "refit" + (f" {f4}" if f4 else "")}
        if "cap" in arms:
            cap = CAP_FRACTION * math.hypot(float(np.ptp(xs)), float(np.ptp(ys)))
            d |= {"cap": cap}
            if not sh.practical_range(m1) > cap:
                out[("cap", c)] = (ok.est, ok.var)
                d |= {"cap_status": "same"}
            else:
                m5, f5 = fit(zs, xs, ys, mean="constant", max_practical_range=cap)
                r5 = ok_like(j, sh.corr_of_model(m5), z)
                out[("cap", c)] = (r5.est, r5.var)
                d |= describe("f5", m5) | {"cap_status": "refit" + (f" {f5}" if f5 else "")}
        if "sw" in arms:
            try:
                ev = sigma_weighted(pairs(xs, ys), (yj - pj)[sub], sj[sub])
                m7 = wls(ev)
                r7 = ok_like(j, sh.corr_of_model(m7), z)
                out[("sw", c)] = (r7.est, r7.var)
                d |= describe("f7", m7) | {"sw_status": "fit"}
            except Exception as err:  # degenerate sample variograms: keep ok, flag it
                out[("sw", c)] = (ok.est, ok.var)
                d |= {"sw_status": f"failed: {type(err).__name__}"}
        if map_name == "M1" and arms & {"oracle_zero", "oracle_ok"}:
            co = oracle_corr(j)
            ko = sh.make_kernels(layout, co)
            vbo = sh.exact_vb(layout, sd[j], ko)
            out[("oracle_zero", c)] = (share[j], vbo)
            ro = ok_like(j, co, z, ko)
            out[("oracle_ok", c)] = (ro.est, ro.var)
        if "vparam" in arms:
            status = "skipped"
            if n <= vmax and not (f0 or f1):
                pts, w, code, status = ar.sigma_points(zs, xs, ys, m1)
                if len(pts) > 1:
                    ests, vars_ = [ok.est], [ok.var]
                    for pt in pts[1:]:
                        rj = ok_like(j, sh.corr_of_model(ar.model_of(code, pt)), z)
                        ests.append(rj.est)
                        vars_.append(rj.var)
                    E, V = np.stack(ests), np.stack(vars_)
                    var = w @ V + w @ (E - ok.est) ** 2
                    out[("vparam", c)] = (ok.est, var)
            if ("vparam", c) not in out:
                out[("vparam", c)] = (ok.est, ok.var)
            d |= {"vparam_status": status}
        if arms & {"A", "B", "D"}:
            # A: Exp/Sph-only REML fit, with the correlogram-parameter uncertainty (vparam)
            if sh.family(m1) in ("Exp", "Sph", "Nug"):
                m_a, r_a = m1, ok
            else:
                m_a, _ = fit(zs, xs, ys, mean="constant", models=("Exp", "Sph"))
                r_a = ok_like(j, sh.corr_of_model(m_a), z)
            var_a = r_a.var
            st_a = "skipped"
            if n <= vmax and not (f0 or f1) and sh.family(m_a) != "Nug":
                pts, w, code, st_a = ar.sigma_points(zs, xs, ys, m_a)
                if len(pts) > 1:
                    res = [r_a] + [
                        ok_like(j, sh.corr_of_model(ar.model_of(code, pt)), z) for pt in pts[1:]
                    ]
                    E, V = np.stack([q.est for q in res]), np.stack([q.var for q in res])
                    var_a = w @ V + w @ (E - r_a.est) ** 2
            out[("A", c)] = (r_a.est, var_a)
            combo_g[j] = {"A": r_a.gls.G[0]}
            d |= describe("fA", m_a) | {"A_vparam": st_a}
            # D: the σσ-weighted correlogram for blocks of 256 cells and more, A below
            try:
                m_d = wls(sigma_weighted(pairs(xs, ys), (yj - pj)[sub], sj[sub]))
                r_d = ok_like(j, sh.corr_of_model(m_d), z)
                d |= describe("fD", m_d) | {"D_status": "fit"}
            except Exception as err:  # degenerate sample variograms: keep A, flag it
                r_d = r_a
                d |= {"D_status": f"failed: {type(err).__name__}"}
            large = layout.side_of_block >= 256
            out[("D", c)] = (
                np.where(large, r_d.est, r_a.est),
                np.where(large, r_d.var, var_a),
            )
            combo_g[j]["D"] = r_d.gls.G[0]
            # B: A plus the calibrator bootstrap (added below on recalibrated maps)
            out[("B", c)] = out[("A", c)]
        if "ppi" in arms:
            est = np.full(nb, np.nan)
            var = np.full(nb, np.nan)
            resid = yj - pj
            row, col = cells // sh.N, cells % sh.N
            tile = (row // 256) * (sh.N // 256) + col // 256
            blocks = [(layout.n_blocks - 1, np.ones(n, dtype=bool))] + [
                (layout.slices[256].start + t, tile == t) for t in range((sh.N // 256) ** 2)
            ]
            for b, inside in blocks:
                rbar, se = ar.ppi_block(resid[inside], clus[inside])
                est[b], var[b] = share[j][b] + rbar, se**2
            out[("ppi", c)] = (est, var)
        if "olofsson" in arms and design == "random":
            mc = np.argmax(pf, axis=0)
            e, se = sh.olofsson_landscape(
                np.bincount(mc, minlength=3) / mc.size, mc[cells], lab == j
            )
            est = np.full(nb, np.nan)
            var = np.full(nb, np.nan)
            est[-1], var[-1] = e, se**2
            out[("olofsson", c)] = (est, var)
        d["seconds"] = time.time() - t0
        diags.append(d)

    # pooled empirical Bayes (one τ² for the three classes)
    tau2 = max(0.0, float(np.mean([oks[j].gls.beta[0] ** 2 - oks[j].gls.A[0, 0] for j in oks])))
    for j, c in enumerate(sh.CLASSES):
        lam, _ = ar.eb_lambda(float(oks[j].gls.beta[0]), float(oks[j].gls.A[0, 0]), tau2)
        out[("ok_eb_pool", c)] = ar.eb_correction(oks[j], share[j], lam)
        diags[j] |= {"eb_pool_tau2": tau2, "eb_pool_lambda": lam}
    if cil_maps and len(cil_maps) == len(sh.CLASSES):  # shares renormalised to sum to 1
        tot = sum(cil_maps.values())
        for j, c in enumerate(sh.CLASSES):
            out[("ok_cil_n", c)] = (bmat @ (cil_maps[j] / tot), cil_var[j])

    # bootstrap of the calibrator (calibrated-after-miscalibration maps)
    if calib is not None and boot > 1 and arms & {"current_cb", "ok_cb", "B", "D"}:
        cal, lp = calib
        fit_cal, apply_cal = sh.CALIBRATORS[cal]
        cc = dr.cal_cells[design][:n]
        cl = dr.cal_labels[design][:n]
        _, inv = np.unique(dr.cal_clusters[design][:n], return_inverse=True)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv)
        groups = np.split(order, np.cumsum(counts)[:-1])  # calibration points per cluster
        bsh = np.empty((boot, len(sh.CLASSES), nb))
        es = np.empty((boot, len(sh.CLASSES), nb))
        names = ["ok"] + (["A", "D"] if combo_g else [])
        ec = {k: np.empty((boot, len(sh.CLASSES), nb)) for k in names}
        for b in range(boot):
            idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
            pb = sh.clip_norm(apply_cal(fit_cal(lp[:, cc[idx]], cl[idx]), lp))
            for j in range(len(sh.CLASSES)):
                sdb = np.sqrt(pb[j] * (1.0 - pb[j]))
                bsh[b, j] = bmat @ pb[j]
                zb = ((lab == j) - pb[j, cells]) / sdb[cells]
                xb = bmat @ sdb
                es[b, j] = bsh[b, j] + xb * float(oks[j].gls.G[0] @ zb)
                for k in names[1:]:
                    ec[k][b, j] = bsh[b, j] + xb * float(combo_g[j][k] @ zb)
        for j, c in enumerate(sh.CLASSES):
            v_sh, v_es = bsh[:, j].var(axis=0, ddof=1), es[:, j].var(axis=0, ddof=1)
            out[("current_cb", c)] = (share[j], cur_var[j] + v_sh)
            out[("ok_cb", c)] = (oks[j].est, oks[j].var + v_es)
            if combo_g:
                v_a = ec["A"][:, j].var(axis=0, ddof=1)
                v_d = ec["D"][:, j].var(axis=0, ddof=1)
                large = layout.side_of_block >= 256
                est_a, var_a = out[("A", c)]
                out[("B", c)] = (est_a, var_a + v_a)
                est_d, var_d = out[("D", c)]
                out[("D", c)] = (est_d, var_d + np.where(large, v_d, v_a))
            diags[j] |= {
                "boot_var_share_land": float(v_sh[-1]),
                "boot_var_ok_land": float(v_es[-1]),
            }
    return out, diags


def realization(
    r: int, map_name: str, arms: set[str], boot: int, vmax: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ls = landscape()
    dr = sh.class_draws(ls, r)
    mis, cal = sh.CLASS_MAPS[map_name]
    lp = log_mis(mis) if mis else None
    frames, diag_rows = [], []
    for j, c in enumerate(sh.CLASSES):
        frames.append(pl.DataFrame({
            "design": "-", "n": 0, "arm": "truth", "cls": c, "block": np.arange(ls.layout.n_blocks),
            "est": dr.truth[j], "var": np.zeros(ls.layout.n_blocks),
        }))  # fmt: skip
    for di, design in enumerate(sh.DESIGNS):
        for ni, n in enumerate(sh.SIZES):
            t0 = time.time()
            calib = None
            extra: dict[str, object] = {}
            if cal is not None:
                assert lp is not None
                fit_cal, apply_cal = sh.CALIBRATORS[cal]
                par = fit_cal(lp[:, dr.cal_cells[design][:n]], dr.cal_labels[design][:n])
                pf = sh.clip_norm(apply_cal(par, lp))
                calib = (cal, lp)
                extra = {"calibrator": repr(par)}
            else:
                pf = ls.p_true
            rng = np.random.default_rng([sh.SEED, 9, r, di, ni])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out, diags = setting(
                    ls,
                    dr,
                    design,
                    n,
                    pf,
                    map_name,
                    arms,
                    calib=calib,
                    boot=boot,
                    vmax=vmax,
                    rng=rng,
                )
            for (arm, c), (est, var) in out.items():
                if arm not in arms:
                    continue
                keep = np.flatnonzero(np.isfinite(est))
                frames.append(pl.DataFrame({
                    "design": design, "n": n, "arm": arm, "cls": c, "block": keep,
                    "est": est[keep], "var": var[keep],
                }))  # fmt: skip
            for d in diags:
                d |= extra | {"warnings": len(caught), "setting_seconds": time.time() - t0}
            diag_rows += diags
    df = pl.concat(frames).with_columns(
        pl.lit(r).cast(pl.Int16).alias("r"),
        pl.lit(map_name).alias("map"),
        pl.col("n").cast(pl.Int16),
        pl.col("block").cast(pl.Int16),
        pl.col("est").cast(pl.Float64),
        pl.col("var").cast(pl.Float32),
    )
    dg = pl.from_dicts(diag_rows, infer_schema_length=None).with_columns(
        pl.lit(r).alias("r"), pl.lit(map_name).alias("map")
    )
    return df, dg


def task(args: tuple[int, str, str, tuple[str, ...], int, int]) -> tuple[int, str, float]:
    r, map_name, out_dir, arms, boot, vmax = args
    t0 = time.time()
    df, dg = realization(r, map_name, set(arms), boot, vmax)
    d = Path(out_dir) / map_name
    dg.write_parquet(d / f"d{r:03d}.parquet")
    df.write_parquet(d / f"r{r:03d}.parquet")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  peak memory r{r} {map_name}: {peak:.0f} MB", flush=True)
    return r, map_name, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--maps", default="M1,M2,M3")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--boot", type=int, default=50)
    ap.add_argument("--vparam-max-n", type=int, default=5000)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    a = ap.parse_args()
    out = Path(a.out_dir)
    maps = a.maps.split(",")
    arms = tuple(a.arms.split(","))
    for m in maps:
        (out / m).mkdir(parents=True, exist_ok=True)
    todo = [
        (r, m, str(out), arms, a.boot, a.vparam_max_n)
        for r in range(a.start, a.stop)
        for m in maps
        if not (out / m / f"r{r:03d}.parquet").exists()
    ]
    with ProcessPoolExecutor(
        a.workers, mp_context=get_context("spawn"), max_tasks_per_child=1
    ) as ex:  # a fresh process per task keeps memory from creeping up
        for r, m, sec in ex.map(task, todo):
            print(f"realization {r} {m}: {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
