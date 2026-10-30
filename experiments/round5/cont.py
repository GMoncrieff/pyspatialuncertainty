"""Round 5 driver for continuous maps (the error-model landscape of round 4).

The prediction error is e(s) = σ_true(s)·(w(s) + nugget) + b(s), with w an exponential field
(Nug 0.2 + Exp 0.8, range 32 cells). The map's σ is σ_user. Maps:

* ``C1``: calibrated (b = 0, σ_user = σ_true);
* ``C2``: b = −0.3·σ_true (round 4's offset) and σ_user = 0.75·σ_true, then recalibrated on
  its own n points: ``C2s`` by a σ factor only, ``C2l`` by a linear correction of the
  prediction (intercept and slope on m, the standardized prediction field) plus a σ factor;
* ``C3``: a constant offset b = −0.3·mean(σ_true) and σ_user = 0.75·σ_true, recalibrated
  as ``C3s`` / ``C3l``.

Arms: ``current``, ``ok`` (X = 1, a bias proportional to σ), ``ok_e`` (X = 1/σ, a constant
bias), ``ok_m`` (X = [1, m], a slope), ``ok_eb``, ``fam_es``, ``cap``, ``sw``, ``vparam``,
``ok_cb`` / ``current_cb`` (bootstrap of the recalibration), ``oracle_zero`` / ``oracle_ok``
(C1) and ``ppi``.

Usage::

    uv run python -m experiments.round5.cont <out_dir> --maps C1,C2s,C2l,C3s,C3l --start 0 --stop 50
"""

from __future__ import annotations

import argparse
import math
import resource
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from experiments.improvements.candidates import pairs, sigma_weighted, wls
from experiments.round5 import arms as ar
from experiments.round5 import shared as sh
from experiments.round5.run import CAP_FRACTION, describe, fit
from pyspatialuncertainty.synthetic import ErrorModelScenario, SubGridLayout
from pyspatialuncertainty.synthetic.grf import CirculantEmbedding

BIAS, SD_SCALE = 0.3, 0.75
CONT_MAPS = {
    "C1": (None, None),
    "C2s": ("prop", "sigma"),
    "C2l": ("prop", "linear"),
    "C3s": ("const", "sigma"),
    "C3l": ("const", "linear"),
}
ALL_ARMS = (
    "current", "ok", "ok_e", "ok_m", "ok_eb", "fam_es", "cap", "sw", "vparam", "current_cb",
    "ok_cb", "oracle_zero", "oracle_ok", "ppi",
)  # fmt: skip


@dataclass
class ContLandscape:
    sc: ErrorModelScenario
    layout: sh.Layout
    sd_true: np.ndarray
    m: np.ndarray  # standardized prediction field (round 4's m)


_LS: ContLandscape | None = None
_ORACLE: sh.Corr | None = None


def landscape() -> ContLandscape:
    global _LS
    if _LS is None:
        cls = sh.build_class_landscape()  # the same blocks
        lay = SubGridLayout(n=sh.N, f=1)
        sc = ErrorModelScenario(layout=lay, correlation=sh.CORRELOGRAM)
        sd_true = np.asarray(sc.sigma_true).ravel()
        emb = CirculantEmbedding((sh.N, sh.N), 1.0, lambda h: np.exp(-h / sh.SIGNAL_RANGE))
        f = emb.sample(np.random.default_rng(sh.SEED + 1), 1)[0].ravel()
        m = (f - f.mean()) / f.std()
        _LS = ContLandscape(sc, cls.layout, sd_true, m)
    return _LS


def oracle() -> sh.Corr:
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = sh.corr_of_model(landscape().sc.correlogram, "oracle")
    return _ORACLE


def _corrected(
    layout: sh.Layout,
    sd: np.ndarray,
    share: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    cols: list[np.ndarray],
    kern: sh.Kernels,
    vb: np.ndarray,
    chol: tuple[np.ndarray, bool],
) -> ar.Corrected:
    """The estimated-mean correction with design columns ``cols`` (given on all cells)."""
    Xp = np.column_stack([col[c] for col in cols])
    return ar.linear_correction(layout, kern, sd, share, c, cols, ar.gls(chol, Xp, z), vb)


def _ok_with(
    layout: sh.Layout,
    sd: np.ndarray,
    share: np.ndarray,
    c: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    model: object,
) -> ar.Corrected:
    """``ok`` (X = 1) with another correlogram."""
    corr = sh.corr_of_model(model)  # type: ignore[arg-type]
    kern = sh.make_kernels(layout, corr)
    vb = sh.exact_vb(layout, sd, kern)
    return _corrected(
        layout, sd, share, c, z, [np.ones(sd.size)], kern, vb, ar.gls_factor(corr, x, y)
    )


def recalibrate(
    kind: str, e: np.ndarray, sd: np.ndarray, m: np.ndarray
) -> tuple[float, float, float]:
    """(b0, b1, c): prediction correction b0 + b1·m (``linear`` only) and σ factor c."""
    if kind == "linear":
        X = np.column_stack([np.ones(e.size), m])
        b, *_ = np.linalg.lstsq(X, e, rcond=None)
        r = e - X @ b
        return float(b[0]), float(b[1]), math.sqrt(float(np.mean((r / sd) ** 2)))
    return 0.0, 0.0, math.sqrt(float(np.mean((e / sd) ** 2)))


def realization(
    r: int, map_name: str, arms: set[str], boot: int, vmax: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ls = landscape()
    layout = ls.layout
    bmat = layout.bmat
    nb = layout.n_blocks
    kind, recal = CONT_MAPS[map_name]
    rng = np.random.default_rng([sh.SEED, 10, r])
    s = ls.sc.sampler.sample(rng, 1)[0].ravel()
    n_max = max(sh.SIZES)
    cells, clusters, eps = {}, {}, {}
    for d in sh.DESIGNS:
        cells[d], clusters[d] = sh.sample_cells(d, n_max, rng)
    for d in sh.DESIGNS:
        eps[d] = rng.standard_normal(n_max)
    perms = {(d, n): rng.permutation(n)[: sh.N_FIT] for d in sh.DESIGNS for n in sh.SIZES}
    rng_c = np.random.default_rng([sh.SEED, 11, r])
    cal_cells, cal_clusters, cal_eps = {}, {}, {}
    for d in sh.DESIGNS:
        cal_cells[d], cal_clusters[d] = sh.sample_cells(d, n_max, rng_c)
    for d in sh.DESIGNS:
        cal_eps[d] = rng_c.standard_normal(n_max)
    sd_n = math.sqrt(ls.sc.nugget)
    if kind == "prop":
        bias = -BIAS * ls.sd_true
    elif kind == "const":
        bias = np.full(ls.sd_true.size, -BIAS * float(ls.sd_true.mean()))
    else:
        bias = np.zeros(ls.sd_true.size)
    sd_user = (SD_SCALE if kind else 1.0) * ls.sd_true
    struct = ls.sd_true * s
    frames, diags = [], []
    for di, design in enumerate(sh.DESIGNS):
        for ni, n in enumerate(sh.SIZES):
            t0 = time.time()
            c = cells[design][:n]
            x, y = layout.cx[c], layout.cy[c]
            e_pts = struct[c] + ls.sd_true[c] * sd_n * eps[design][:n] + bias[c]
            d: dict[str, object] = {"design": design, "n": n, "cls": "-"}
            b0 = b1 = 0.0
            cfac = 1.0
            if recal:
                cc = cal_cells[design][:n]
                e_cal = struct[cc] + ls.sd_true[cc] * sd_n * cal_eps[design][:n] + bias[cc]
                b0, b1, cfac = recalibrate(recal, e_cal, sd_user[cc], ls.m[cc])
                d |= {"b0": b0, "b1": b1, "c": cfac}
            shift_map = b0 + b1 * ls.m  # added to the prediction: the error becomes e − shift
            sd = cfac * sd_user
            truth = bmat @ (struct + bias - shift_map)
            # the truth depends on the setting when the recalibration moves the prediction
            frames.append(
                pl.DataFrame(
                    {
                        "design": design,
                        "n": n,
                        "arm": "truth",
                        "cls": "-",
                        "block": np.arange(nb),
                        "est": truth,
                        "var": np.zeros(nb),
                    }
                )
            )
            ev = e_pts - (b0 + b1 * ls.m[c])
            z = ev / sd[c]
            sub = perms[(design, n)]
            zs, xs, ys = z[sub], x[sub], y[sub]
            share = np.zeros(nb)
            out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m0, f0 = fit(zs, xs, ys, mean="zero")
                m1, f1 = fit(zs, xs, ys, mean="constant")
                d |= describe("f0", m0) | describe("f1", m1)
                k0, k1 = (
                    sh.make_kernels(layout, sh.corr_of_model(m0)),
                    sh.make_kernels(layout, sh.corr_of_model(m1)),
                )
                vb0, vb1 = sh.exact_vb(layout, sd, k0), sh.exact_vb(layout, sd, k1)
                out["current"] = (share, vb0)
                chol1 = ar.gls_factor(sh.corr_of_model(m1), x, y)

                corrected = partial(_corrected, layout, sd, share, c, z)
                ok_with = partial(_ok_with, layout, sd, share, c, z, x, y)
                ok = corrected([np.ones(ls.m.size)], k1, vb1, chol1)
                out["ok"] = (ok.est, ok.var)
                d |= {"mu": float(ok.gls.beta[0]), "A": float(ok.gls.A[0, 0])}
                r_e = corrected([1.0 / sd], k1, vb1, chol1)
                out["ok_e"] = (r_e.est, r_e.var)
                r_m = corrected([np.ones(ls.m.size), ls.m], k1, vb1, chol1)
                out["ok_m"] = (r_m.est, r_m.var)
                lam, _ = ar.eb_lambda(float(ok.gls.beta[0]), float(ok.gls.A[0, 0]))
                out["ok_eb"] = ar.eb_correction(ok, share, lam)
                d |= {"eb_lambda": lam}

                if "fam_es" in arms:
                    if sh.family(m1) in ("Exp", "Sph", "Nug"):
                        out["fam_es"] = (ok.est, ok.var)
                    else:
                        m4, _ = fit(zs, xs, ys, mean="constant", models=("Exp", "Sph"))
                        r4 = ok_with(m4)
                        out["fam_es"] = (r4.est, r4.var)
                if "cap" in arms:
                    cap = CAP_FRACTION * math.hypot(float(np.ptp(xs)), float(np.ptp(ys)))
                    if not sh.practical_range(m1) > cap:
                        out["cap"] = (ok.est, ok.var)
                    else:
                        m5, _ = fit(zs, xs, ys, mean="constant", max_practical_range=cap)
                        r5 = ok_with(m5)
                        out["cap"] = (r5.est, r5.var)
                if "sw" in arms:
                    try:
                        r7 = ok_with(wls(sigma_weighted(pairs(xs, ys), ev[sub], sd[c][sub])))
                        out["sw"] = (r7.est, r7.var)
                    except Exception:
                        out["sw"] = (ok.est, ok.var)
                if "vparam" in arms:
                    out["vparam"] = (ok.est, ok.var)
                    if n <= vmax and not (f0 or f1):
                        pts, w, code, status = ar.sigma_points(zs, xs, ys, m1)
                        d |= {"vparam_status": status}
                        if len(pts) > 1:
                            res = [ok] + [ok_with(ar.model_of(code, pt)) for pt in pts[1:]]
                            E = np.stack([q.est for q in res])
                            V = np.stack([q.var for q in res])
                            out["vparam"] = (ok.est, w @ V + w @ (E - ok.est) ** 2)
                if map_name == "C1" and arms & {"oracle_zero", "oracle_ok"}:
                    ko = sh.make_kernels(layout, oracle())
                    vbo = sh.exact_vb(layout, sd, ko)
                    out["oracle_zero"] = (share, vbo)
                    ro = corrected([np.ones(ls.m.size)], ko, vbo, ar.gls_factor(oracle(), x, y))
                    out["oracle_ok"] = (ro.est, ro.var)
                if "ppi" in arms:
                    est = np.full(nb, np.nan)
                    var = np.full(nb, np.nan)
                    row, col = c // sh.N, c % sh.N
                    tile = (row // 256) * (sh.N // 256) + col // 256
                    blocks = [(nb - 1, np.ones(n, dtype=bool))] + [
                        (layout.slices[256].start + t, tile == t) for t in range(16)
                    ]
                    for b, inside in blocks:
                        rbar, se = ar.ppi_block(ev[inside], clusters[design][:n][inside])
                        est[b], var[b] = rbar, se**2
                    out["ppi"] = (est, var)
                if recal and boot > 1 and arms & {"current_cb", "ok_cb"}:
                    cc = cal_cells[design][:n]
                    e_cal = struct[cc] + ls.sd_true[cc] * sd_n * cal_eps[design][:n] + bias[cc]
                    _, inv = np.unique(cal_clusters[design][:n], return_inverse=True)
                    groups = np.split(
                        np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1]
                    )
                    rngb = np.random.default_rng([sh.SEED, 12, r, di, ni])
                    bsh, es = np.empty((boot, nb)), np.empty((boot, nb))
                    xb_unit = bmat @ sd_user
                    for b in range(boot):
                        idx = np.concatenate(
                            [groups[g] for g in rngb.integers(0, len(groups), len(groups))]
                        )
                        bb0, bb1, cb = recalibrate(
                            recal, e_cal[idx], sd_user[cc][idx], ls.m[cc][idx]
                        )
                        # change of the true block error when the prediction moves by bb0 + bb1·m
                        bsh[b] = (bmat @ shift_map) - (bmat @ (bb0 + bb1 * ls.m))
                        zb = (e_pts - (bb0 + bb1 * ls.m[c])) / (cb * sd_user[c])
                        es[b] = bsh[b] - cb * xb_unit * float(ok.gls.G[0] @ zb)  # error − estimate
                    out["current_cb"] = (share, vb0 + bsh.var(axis=0, ddof=1))
                    out["ok_cb"] = (ok.est, ok.var + es.var(axis=0, ddof=1))
            d |= {"warnings": len(caught), "seconds": time.time() - t0}
            diags.append(d)
            for arm, (est, var) in out.items():
                if arm not in arms:
                    continue
                keep = np.flatnonzero(np.isfinite(est))
                frames.append(
                    pl.DataFrame(
                        {
                            "design": design,
                            "n": n,
                            "arm": arm,
                            "cls": "-",
                            "block": keep,
                            "est": est[keep],
                            "var": var[keep],
                        }
                    )
                )
    df = pl.concat(frames).with_columns(
        pl.lit(r).cast(pl.Int16).alias("r"),
        pl.lit(map_name).alias("map"),
        pl.col("n").cast(pl.Int16),
        pl.col("block").cast(pl.Int16),
        pl.col("var").cast(pl.Float32),
    )
    dg = pl.from_dicts(diags, infer_schema_length=None).with_columns(
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
    ap.add_argument("--maps", default="C1,C2s,C2l,C3s,C3l")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--boot", type=int, default=50)
    ap.add_argument("--vparam-max-n", type=int, default=5000)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    a = ap.parse_args()
    out = Path(a.out_dir)
    maps = a.maps.split(",")
    for m in maps:
        (out / m).mkdir(parents=True, exist_ok=True)
    todo = [
        (r, m, str(out), tuple(a.arms.split(",")), a.boot, a.vparam_max_n)
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
