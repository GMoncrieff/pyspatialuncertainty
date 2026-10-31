"""Round 6 driver for continuous maps (round 4's error-model landscape, any side length).

The prediction error is e(s) = σ_true(s)·(w(s) + nugget) + b(s), with w an exponential
field (Nug 0.2 + Exp 0.8, range 32 cells). Maps (``world.CONT_MAPS``):

* ``C1``: calibrated;
* ``C2s`` / ``C2l``: b = −0.3·σ_true and σ_user = 0.75·σ_true, recalibrated on its own n
  points by a σ factor (``s``) or by b0 + b1·m and a σ factor (``l``);
* ``C3s`` / ``C3l``: a constant b = −0.3·mean σ_true, recalibrated in the same two ways;
* ``C4``: as C2 but not recalibrated (σ is 25 % too small, the bias is left in).

Arms are ``fit+mod+...``: fits ``cur`` (zero-mean ML), ``ok`` (REML, free family), ``es``
(REML, Exp/Sph); modifiers

* ``x2`` / ``x3``: mean design X = [1, 1/σ] / [1, 1/σ, m/σ] (in z units: a bias ∝ σ, a
  constant data-unit bias, and a bias linear in the prediction m) instead of X = [1];
* ``vp`` / ``vg``: parameter uncertainty by Laplace / grid-posterior sigma points;
* ``scale``: variances times the REML sill ŝ² of the standardized errors;
* ``ci`` / ``cg`` / ``cs`` / ``cc``: uncertainty of the linear recalibration by iid bootstrap /
  pair-correlation sandwich draws / model sandwich draws / cluster bootstrap (reference; see
  ``calib``). The σ factor does not move the estimate, so it is held at its fitted value.

Usage::

    uv run python -m experiments.round6.cont <out_dir> --side 1024 --maps C1,C2l --stop 30
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import resource
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from experiments.round5 import arms as ar
from experiments.round5 import shared as sh
from experiments.round5.cont import BIAS, SD_SCALE, recalibrate
from experiments.round6 import calib as cb
from experiments.round6 import fits
from experiments.round6 import world as wd

FITS = ("cur", "ok", "es")
MODS = ("x2", "x3", "vp", "vg", "scale", "ci", "cg", "cs", "cc")
CAL = ("ci", "cg", "cs", "cc")
SCREEN = (
    "cur", "ok", "es", "ok+x2", "ok+x3", "ok+vp", "ok+vg", "ok+scale", "ok+ci", "ok+cg",
    "ok+cs", "ok+cc", "ok+x3+ci", "ok+x3+cg", "ok+x3+cc",
)  # fmt: skip


@dataclass(frozen=True)
class Spec:
    fit: str
    design: str = "1"  # "1", "x2" or "x3"
    unc: str = ""
    scale: bool = False
    cal: str = ""


def parse(arm: str) -> Spec:
    fit, *mods = arm.split("+")
    if fit not in FITS or any(m not in MODS for m in mods) or len(set(mods)) < len(mods):
        raise ValueError(f"unknown arm {arm!r}")
    des = [m for m in mods if m in ("x2", "x3")]
    unc = [m for m in mods if m in ("vp", "vg")]
    cal = [m for m in mods if m in CAL]
    if len(des) > 1 or len(unc) > 1 or len(cal) > 1 or (fit == "cur" and mods):
        raise ValueError(f"inconsistent arm {arm!r}")
    return Spec(
        fit, des[0] if des else "1", unc[0] if unc else "", "scale" in mods, cal[0] if cal else ""
    )


_W: dict[int, wd.ContWorld] = {}


def world(n: int) -> wd.ContWorld:
    if n not in _W:
        _W[n] = wd.cont_world(n)
    return _W[n]


class ContSetting:
    """Fits and corrections for one (design, n), cached across arms."""

    def __init__(
        self,
        w: wd.ContWorld,
        sd: np.ndarray,
        cells: np.ndarray,
        z: np.ndarray,
        sub: np.ndarray,
        designs: dict[str, tuple[str, ...]],
    ):
        self.w, self.layout, self.sd, self.cells, self.z = w, w.layout, sd, cells, z
        self.x, self.y = self.layout.cx[cells], self.layout.cy[cells]
        self.xs, self.ys, self.zs = self.x[sub], self.y[sub], z[sub]
        self.Ds = fits.distances(self.xs, self.ys)
        self.D = fits.distances(self.x, self.y)
        self.designs = designs  # mean designs needed per fit, evaluated together
        self.share = np.zeros(self.layout.n_blocks)
        self.fits: dict[str, fits.Fit] = {}
        self.base: dict[tuple[str, str], ar.Corrected] = {}
        self.unc: dict[tuple[str, str, str], tuple[np.ndarray, str]] = {}
        self.cur_vb: np.ndarray | None = None

    def fit(self, label: str) -> fits.Fit:
        if label not in self.fits:
            zs, xs, ys, D = self.zs, self.xs, self.ys, self.Ds
            if label == "cur":
                f = fits.package("cur", zs, xs, ys, D, mean="zero")
            elif label == "ok":
                f = fits.package("ok", zs, xs, ys, D)
            else:
                ok = self.fit("ok")
                f = (
                    dataclasses.replace(ok, label="es")
                    if ok.family in ("Exp", "Sph", "Nug")
                    else fits.package("es", zs, xs, ys, D, models=("Exp", "Sph"))
                )
            self.fits[label] = f
        return self.fits[label]

    def cols(self, design: str) -> list[np.ndarray]:
        one = np.ones(self.sd.size)
        if design == "x2":
            return [one, 1.0 / self.sd]
        if design == "x3":
            return [one, 1.0 / self.sd, self.w.m / self.sd]
        return [one]

    def evaluate(self, corr: sh.Corr, designs: list[str]) -> list[ar.Corrected]:
        kern = sh.make_kernels(self.layout, corr)
        vb = sh.exact_vb(self.layout, self.sd, kern)
        chol = fits.gls_factor(corr, self.D)
        out = []
        for des in designs:
            cols = self.cols(des)
            Xp = np.column_stack([c[self.cells] for c in cols])
            g = ar.gls(chol, Xp, self.z)
            out.append(
                ar.linear_correction(
                    self.layout, kern, self.sd, self.share, self.cells, cols, g, vb
                )
            )
        return out

    def corrected(self, label: str, design: str) -> ar.Corrected:
        key = (label, design)
        if key not in self.base:
            designs = sorted(set(self.designs.get(label, ())) | {design})
            for des, res in zip(designs, self.evaluate(self.fit(label).corr, designs), strict=True):
                self.base[(label, des)] = res
        return self.base[key]

    def current(self) -> np.ndarray:
        if self.cur_vb is None:
            kern = sh.make_kernels(self.layout, self.fit("cur").corr)
            self.cur_vb = sh.exact_vb(self.layout, self.sd, kern)
        return self.cur_vb

    def with_uncertainty(self, label: str, design: str, unc: str) -> tuple[np.ndarray, str]:
        key = (label, design, unc)
        if key in self.unc:
            return self.unc[key]
        f = self.fit(label)
        base = self.corrected(label, design)
        pts, w, status = (fits.laplace if unc == "vp" else fits.grid_posterior)(f)
        if len(pts) <= 1 or f.corr_of is None:
            self.unc[key] = (base.var, status)
            return self.unc[key]
        first = 1 if unc == "vp" else 0
        res = [base] * first + [self.evaluate(f.corr_of(pt), [design])[0] for pt in pts[first:]]
        E = np.stack([q.est for q in res])
        V = np.stack([q.var for q in res])
        self.unc[key] = (w @ V + w @ (E - base.est) ** 2, status)
        return self.unc[key]


def realization(
    n_side: int, r: int, map_name: str, arms: tuple[str, ...]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    w = world(n_side)
    layout = w.layout
    bmat = layout.bmat
    nb = layout.n_blocks
    bside = layout.side_of_block
    specs = {a: parse(a) for a in arms}
    designs = {
        f: tuple(sorted({s.design for s in specs.values() if s.fit == f}))
        for f in {s.fit for s in specs.values()}
    }
    kind, recal = wd.CONT_MAPS[map_name]
    dr = wd.cont_draws(w, r)
    sd_n = math.sqrt(w.sc.nugget)
    if kind == "prop":
        bias = -BIAS * w.sd_true
    elif kind == "const":
        bias = np.full(w.sd_true.size, -BIAS * float(w.sd_true.mean()))
    else:
        bias = np.zeros(w.sd_true.size)
    sd_user = (SD_SCALE if kind else 1.0) * w.sd_true
    struct = w.sd_true * dr.s
    frames, diags = [], []
    for di, design in enumerate(sh.DESIGNS):
        for ni, n in enumerate(sh.SIZES):
            t0 = time.time()
            c = dr.cells[design][:n]
            e_pts = struct[c] + w.sd_true[c] * sd_n * dr.eps[design][:n] + bias[c]
            d: dict[str, object] = {"design": design, "n": n, "cls": "-"}
            b0 = b1 = 0.0
            cfac = 1.0
            cc_ = dr.cal_cells[design][:n]
            e_cal = struct[cc_] + w.sd_true[cc_] * sd_n * dr.cal_eps[design][:n] + bias[cc_]
            if recal:
                b0, b1, cfac = recalibrate(recal, e_cal, sd_user[cc_], w.m[cc_])
                d |= {"b0": b0, "b1": b1, "c": cfac}
            shift_map = b0 + b1 * w.m
            sd = cfac * sd_user
            truth = bmat @ (struct + bias - shift_map)
            frames.append(pl.DataFrame({
                "design": design, "n": n, "arm": "truth", "cls": "-", "block": np.arange(nb),
                "bside": bside, "est": truth, "var": np.zeros(nb),
            }))  # fmt: skip
            ev = e_pts - (b0 + b1 * w.m[c])
            z = ev / sd[c]
            st = ContSetting(w, sd, c, z, dr.perms[(design, n)], designs)
            out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for arm, s in specs.items():
                    if s.fit == "cur":
                        out[arm] = (st.share, st.current())
                        continue
                    base = st.corrected(s.fit, s.design)
                    var = base.var
                    if s.unc:
                        var, status = st.with_uncertainty(s.fit, s.design, s.unc)
                        d[f"{s.fit}_{s.design}_{s.unc}"] = status
                    if s.scale:
                        var = var * st.fit(s.fit).sill
                    out[arm] = (base.est, var)
                cal_arms = {a: s for a, s in specs.items() if s.cal}
                if recal == "linear" and cal_arms:
                    rng = np.random.default_rng(wd.seed(n_side, 12, r, di, ni))
                    clus = dr.cal_clusters[design][:n]
                    xc, yc = layout.cx[cc_], layout.cy[cc_]
                    for ck in sorted({s.cal for s in cal_arms.values()}):
                        group = {a: s for a, s in cal_arms.items() if s.cal == ck}
                        by_fit: dict[str, np.ndarray] = {}
                        for fl in sorted({s.fit for s in group.values()}):
                            if ck == "cs":
                                R = cb.correlation_matrix(st.fit(fl).corr.rho, xc, yc)
                                by_fit[fl] = cb.linear_draws(e_cal, w.m[cc_], sd[cc_], rng, R=R)
                                del R
                            elif ck == "cg":
                                pairs = cb.pair_weights(xc, yc, n_side)
                                by_fit[fl] = cb.linear_draws(
                                    e_cal, w.m[cc_], sd[cc_], rng, pairs=pairs
                                )
                            else:
                                bs = []
                                for _ in range(cb.N_DRAWS):
                                    idx = cb.resample(ck, clus, rng)
                                    bb0, bb1, _c = recalibrate(
                                        recal, e_cal[idx], sd_user[cc_][idx], w.m[cc_][idx]
                                    )
                                    bs.append((bb0, bb1))
                                by_fit[fl] = np.array(bs)
                        for a, s in group.items():
                            g = st.corrected(s.fit, s.design).gls
                            xb = [bmat @ (sd * col) for col in st.cols(s.design)]
                            err = np.empty((cb.N_DRAWS, nb))
                            for b, (bb0, bb1) in enumerate(by_fit[s.fit]):
                                # change of the true block error when the prediction moves
                                bsh = bmat @ shift_map - bmat @ (bb0 + bb1 * w.m)
                                zb = (e_pts - (bb0 + bb1 * w.m[c])) / sd[c]
                                est_b = sum(xb[q] * float(g.G[q] @ zb) for q in range(len(xb)))
                                err[b] = bsh - est_b
                            v = err.var(axis=0, ddof=1)
                            out[a] = (out[a][0], out[a][1] + v)
                            d[f"{a}_calvar_land"] = float(v[-1])
                for f in st.fits.values():
                    d |= f.describe() | {f"{f.label}_sill": f.sill}
            d |= {"warnings": len(caught), "seconds": time.time() - t0}
            diags.append(d)
            for arm, (est, var) in out.items():
                keep = np.flatnonzero(np.isfinite(est))
                frames.append(pl.DataFrame({
                    "design": design, "n": n, "arm": arm, "cls": "-", "block": keep,
                    "bside": bside[keep], "est": est[keep], "var": var[keep],
                }))  # fmt: skip
    df = pl.concat(frames).with_columns(
        pl.lit(r).cast(pl.Int16).alias("r"),
        pl.lit(map_name).alias("map"),
        pl.lit(n_side).cast(pl.Int16).alias("side"),
        pl.col("n").cast(pl.Int16),
        pl.col("block").cast(pl.Int16),
        pl.col("bside").cast(pl.Int16),
        pl.col("est").cast(pl.Float64),
        pl.col("var").cast(pl.Float64),
    )
    dg = pl.from_dicts(diags, infer_schema_length=None).with_columns(
        pl.lit(r).alias("r"), pl.lit(map_name).alias("map"), pl.lit(n_side).alias("side")
    )
    return df, dg


def task(args: tuple[int, int, str, str, tuple[str, ...]]) -> tuple[int, str, float]:
    n_side, r, map_name, out_dir, arms = args
    t0 = time.time()
    df, dg = realization(n_side, r, map_name, arms)
    d = Path(out_dir) / str(n_side) / map_name
    dg.write_parquet(d / f"d{r:03d}.parquet")
    df.write_parquet(d / f"r{r:03d}.parquet")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  peak memory {n_side} r{r} {map_name}: {peak:.0f} MB", flush=True)
    return r, map_name, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--maps", default="C1,C2s,C2l,C3l,C4")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arms", default=",".join(SCREEN))
    a = ap.parse_args()
    arms = tuple(a.arms.split(","))
    for arm in arms:
        parse(arm)
    out = Path(a.out_dir)
    maps = a.maps.split(",")
    for m in maps:
        (out / str(a.side) / m).mkdir(parents=True, exist_ok=True)
    todo = [
        (a.side, r, m, str(out), arms)
        for r in range(a.start, a.stop)
        for m in maps
        if not (out / str(a.side) / m / f"r{r:03d}.parquet").exists()
    ]
    with ProcessPoolExecutor(
        a.workers, mp_context=get_context("spawn"), max_tasks_per_child=1
    ) as ex:  # a fresh process per task keeps memory from creeping up
        for r, m, sec in ex.map(task, todo):
            print(f"realization {r} {m}: {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
