"""Round 6 driver for class maps: recipes and their components on one realization.

Arms are named ``fit+mod+mod...``. The fit is one of

* ``cur``: zero-mean ML, free family (the paper's method, no mean correction);
* ``ok``: REML, free family (Exp/Sph/Gau), the estimated-mean correction of branch ``ok``;
* ``es``: REML, Exp/Sph; ``exp``: REML, Exp only;
* ``thr``: threshold-shaped REML (``fits.threshold``),

and the modifiers, applied in a fixed order whatever their order in the name, are

* ``x2``: mean design X = [1, σ] instead of X = [1];
* ``vp`` / ``vg``: parameter uncertainty by Laplace / grid-posterior sigma points;
* ``disp``: variances times the pooled Brier ratio of the validation points;
* ``ci`` / ``cg`` / ``cs`` / ``cc``: calibrator uncertainty by iid bootstrap / pair-correlation
  sandwich draws / model sandwich draws / cluster bootstrap (``calib``; the last needs the
  design and is a reference only).

``olofsson`` adds Olofsson's estimator of the landscape share (random samples only).
Nothing depends on the block size, the landscape size, n or the design.

Usage::

    uv run python -m experiments.round6.run <out_dir> --side 1024 --maps M1,M2 --start 0
        --stop 30 [--workers 4] [--arms ok,thr,thr+vg,...]

Writes ``<out_dir>/<side>/<map>/r###.parquet`` (block estimates and variances per arm, and
the truth) and ``d###.parquet`` (fits and diagnostics); existing files are skipped.
"""

from __future__ import annotations

import argparse
import dataclasses
import resource
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from experiments.round5 import arms as ar
from experiments.round5 import shared as sh
from experiments.round6 import calib as cb
from experiments.round6 import fits
from experiments.round6 import world as wd

FITS = ("cur", "ok", "es", "exp", "thr")
MODS = ("x2", "vp", "vg", "disp", "ci", "cg", "cs", "cc")
CAL = ("ci", "cg", "cs", "cc")
SCREEN = (
    "cur", "ok", "es", "exp", "thr", "thr+vp", "thr+vg", "thr+x2", "thr+disp", "thr+ci",
    "thr+cg", "thr+cs", "thr+cc", "olofsson",
)  # fmt: skip


@dataclass(frozen=True)
class Spec:
    fit: str
    x2: bool = False
    unc: str = ""  # "", "vp" or "vg"
    disp: bool = False
    cal: str = ""  # "", "ci", "cs" or "cc"


def parse(arm: str) -> Spec:
    fit, *mods = arm.split("+")
    if fit not in FITS or any(m not in MODS for m in mods) or len(set(mods)) < len(mods):
        raise ValueError(f"unknown arm {arm!r}")
    unc = [m for m in mods if m in ("vp", "vg")]
    cal = [m for m in mods if m in CAL]
    if len(unc) > 1 or len(cal) > 1 or (fit == "cur" and mods):
        raise ValueError(f"inconsistent arm {arm!r}")
    return Spec(fit, "x2" in mods, unc[0] if unc else "", "disp" in mods, cal[0] if cal else "")


_W: dict[int, wd.ClassWorld] = {}
_MIS: dict[tuple[int, str], np.ndarray] = {}


def world(n: int) -> wd.ClassWorld:
    if n not in _W:
        _W[n] = wd.class_world(n)
    return _W[n]


def log_mis(n: int, kind: str) -> np.ndarray:
    if (n, kind) not in _MIS:
        _MIS[(n, kind)] = np.log(sh.miscalibrate(world(n).p_raw, kind))
    return _MIS[(n, kind)]


class ClassSetting:
    """Fits and corrections of one class for one (design, n), cached across arms."""

    def __init__(
        self,
        layout: sh.Layout,
        sd: np.ndarray,
        share: np.ndarray,
        cells: np.ndarray,
        z: np.ndarray,
        sub: np.ndarray,
        p_bar: float,
        designs: dict[str, tuple[bool, ...]],
    ):
        self.layout, self.sd, self.share, self.cells, self.z = layout, sd, share, cells, z
        self.x, self.y = layout.cx[cells], layout.cy[cells]
        self.sub = sub
        self.xs, self.ys, self.zs = self.x[sub], self.y[sub], z[sub]
        self.Ds = fits.distances(self.xs, self.ys)
        self.D = fits.distances(self.x, self.y)
        self.p_bar = p_bar
        self.designs = designs  # mean designs needed per fit, evaluated together
        self.fits: dict[str, fits.Fit] = {}
        self.base: dict[tuple[str, bool], ar.Corrected] = {}
        self.unc: dict[tuple[str, bool, str], tuple[np.ndarray, str]] = {}
        self.cur_vb: np.ndarray | None = None

    def fit(self, label: str) -> fits.Fit:
        if label in self.fits:
            return self.fits[label]
        zs, xs, ys, D = self.zs, self.xs, self.ys, self.Ds
        if label == "cur":
            f = fits.package("cur", zs, xs, ys, D, mean="zero")
        elif label == "ok":
            f = fits.package("ok", zs, xs, ys, D)
        elif label == "es":
            ok = self.fit("ok")
            f = (
                dataclasses.replace(ok, label="es")
                if ok.family in ("Exp", "Sph", "Nug")
                else fits.package("es", zs, xs, ys, D, models=("Exp", "Sph"))
            )
        elif label == "exp":
            es = self.fit("es")
            f = (
                dataclasses.replace(es, label="exp")
                if es.family in ("Exp", "Nug")
                else fits.package("exp", zs, xs, ys, D, models=("Exp",))
            )
        else:
            f = fits.threshold("thr", zs, xs, ys, D, self.p_bar)
        self.fits[label] = f
        return f

    def cols(self, x2: bool) -> list[np.ndarray]:
        one = np.ones(self.sd.size)
        return [one, self.sd] if x2 else [one]

    def evaluate(self, corr: sh.Corr, designs: list[bool]) -> list[ar.Corrected]:
        kern = sh.make_kernels(self.layout, corr)
        vb = sh.exact_vb(self.layout, self.sd, kern)
        chol = fits.gls_factor(corr, self.D)
        out = []
        for x2 in designs:
            cols = self.cols(x2)
            Xp = np.column_stack([c[self.cells] for c in cols])
            g = ar.gls(chol, Xp, self.z)
            out.append(
                ar.linear_correction(
                    self.layout, kern, self.sd, self.share, self.cells, cols, g, vb
                )
            )
        return out

    def corrected(self, label: str, x2: bool) -> ar.Corrected:
        key = (label, x2)
        if key not in self.base:
            designs = sorted(set(self.designs.get(label, ())) | {x2})
            for des, res in zip(designs, self.evaluate(self.fit(label).corr, designs), strict=True):
                self.base[(label, des)] = res
        return self.base[key]

    def current(self) -> np.ndarray:
        if self.cur_vb is None:
            kern = sh.make_kernels(self.layout, self.fit("cur").corr)
            self.cur_vb = sh.exact_vb(self.layout, self.sd, kern)
        return self.cur_vb

    def with_uncertainty(self, label: str, x2: bool, unc: str) -> tuple[np.ndarray, str]:
        """Variance averaged over sigma points: Σ wᵢ (Vᵢ + (Eᵢ − E)²), E the fit's estimate."""
        key = (label, x2, unc)
        if key in self.unc:
            return self.unc[key]
        f = self.fit(label)
        designs = sorted(set(self.designs.get(label, ())) | {x2})  # one factorization each
        bases = {des: self.corrected(label, des) for des in designs}
        pts, w, status = (fits.laplace if unc == "vp" else fits.grid_posterior)(f)
        if len(pts) <= 1 or f.corr_of is None:
            for des in designs:
                self.unc[(label, des, unc)] = (bases[des].var, status)
            return self.unc[key]
        first = 1 if unc == "vp" else 0  # Laplace points start at the fit itself
        res = [self.evaluate(f.corr_of(pt), designs) for pt in pts[first:]]
        for k, des in enumerate(designs):
            per = [bases[des]] * first + [r[k] for r in res]
            E = np.stack([q.est for q in per])
            V = np.stack([q.var for q in per])
            self.unc[(label, des, unc)] = (w @ V + w @ (E - bases[des].est) ** 2, status)
        return self.unc[key]


def pooled_rho(
    rhos: list[Callable[[np.ndarray], np.ndarray]],
) -> Callable[[np.ndarray], np.ndarray]:
    return lambda d: np.mean([np.asarray(r(d), dtype=np.float64) for r in rhos], axis=0)


def setting(
    w: wd.ClassWorld,
    dr: sh.Draws,
    design: str,
    n: int,
    pf: np.ndarray,
    arms: tuple[str, ...],
    *,
    calib: tuple[str, np.ndarray, dict] | None,
    rng: np.random.Generator,
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], list[dict[str, object]]]:
    layout = w.layout
    bmat = layout.bmat
    nb = layout.n_blocks
    specs = {a: parse(a) for a in arms if a != "olofsson"}
    designs = {
        f: tuple(sorted({s.x2 for s in specs.values() if s.fit == f}))
        for f in {s.fit for s in specs.values()}
    }
    cells = dr.cells[design][:n]
    lab = dr.labels[design][:n]
    sub = dr.perms[(design, n)]
    sd_all = np.sqrt(pf * (1.0 - pf))
    phi = fits.brier_ratio(lab, pf[:, cells]) if any(s.disp for s in specs.values()) else 1.0
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    diags: list[dict[str, object]] = []
    settings: list[ClassSetting] = []
    for j, c in enumerate(sh.CLASSES):
        t0 = time.time()
        d: dict[str, object] = {"design": design, "n": n, "cls": c, "phi": phi}
        sd = sd_all[j]
        z = ((lab == j) - pf[j, cells]) / sd[cells]
        cs = ClassSetting(layout, sd, bmat @ pf[j], cells, z, sub, float(pf[j].mean()), designs)
        settings.append(cs)
        for arm, s in specs.items():
            if s.fit == "cur":
                out[(arm, c)] = (cs.share, cs.current())
                continue
            base = cs.corrected(s.fit, s.x2)
            var = base.var
            if s.unc:
                var, status = cs.with_uncertainty(s.fit, s.x2, s.unc)
                d[f"{s.fit}_{s.unc}"] = status
            if s.disp:
                var = var * phi
            out[(arm, c)] = (base.est, var)
        for f in cs.fits.values():
            d |= f.describe()
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

    cal_arms = {a: s for a, s in specs.items() if s.cal}
    if calib is None or not cal_arms:
        return out, diags
    t0 = time.time()
    kind, lp, par0 = calib
    fit_cal, apply_cal = sh.CALIBRATORS[kind]
    cc_ = dr.cal_cells[design][:n]
    cl = dr.cal_labels[design][:n]
    clus = dr.cal_clusters[design][:n]
    xc, yc = layout.cx[cc_], layout.cy[cc_]
    for ck in sorted({s.cal for s in cal_arms.values()}):
        group = {a: s for a, s in cal_arms.items() if s.cal == ck}
        fls = sorted({s.fit for s in group.values()})
        by_fit: dict[str, list[dict]] = {}
        if ck == "cs":  # the model sandwich depends on each fit's correlogram
            for fl in fls:
                rho = pooled_rho([st.fit(fl).corr.rho for st in settings])
                R = cb.correlation_matrix(rho, xc, yc)
                by_fit[fl] = cb.class_draws(kind, par0, lp[:, cc_], cl, rng, R=R)
                del R
        elif ck == "cg":
            pairs = cb.pair_weights(xc, yc, w.n)
            pars = cb.class_draws(kind, par0, lp[:, cc_], cl, rng, pairs=pairs)
            by_fit = dict.fromkeys(fls, pars)
        else:
            pars = [
                fit_cal(lp[:, cc_[idx]], cl[idx])
                for idx in (cb.resample(ck, clus, rng) for _ in range(cb.N_DRAWS))
            ]
            by_fit = dict.fromkeys(fls, pars)
        draws = {a: np.empty((cb.N_DRAWS, len(sh.CLASSES), nb)) for a in group}
        for fl, plist in by_fit.items():
            members = {a: s for a, s in group.items() if s.fit == fl}
            for b, par in enumerate(plist):
                pb = sh.clip_norm(apply_cal(par, lp))
                for j, cs in enumerate(settings):
                    sdb = np.sqrt(pb[j] * (1.0 - pb[j]))
                    bsh = bmat @ pb[j]
                    zb = ((lab == j) - pb[j, cells]) / sdb[cells]
                    for a, s in members.items():
                        g = cs.corrected(s.fit, s.x2).gls
                        cols = [np.ones(sdb.size), sdb] if s.x2 else [np.ones(sdb.size)]
                        shift = sum(
                            (bmat @ (sdb * col)) * float(g.G[q] @ zb) for q, col in enumerate(cols)
                        )
                        draws[a][b, j] = bsh + shift
        for a in group:
            for j, c in enumerate(sh.CLASSES):
                v = draws[a][:, j].var(axis=0, ddof=1)
                est, var = out[(a, c)]
                out[(a, c)] = (est, var + v)
                diags[j][f"{a}_calvar_land"] = float(v[-1])
    for dg in diags:
        dg["cal_seconds"] = time.time() - t0
    return out, diags


def realization(
    n_side: int, r: int, map_name: str, arms: tuple[str, ...]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    w = world(n_side)
    dr = wd.class_draws(w, r)
    mis, cal = wd.CLASS_MAPS[map_name]
    lp = log_mis(n_side, mis) if mis else None
    nb = w.layout.n_blocks
    bside = w.layout.side_of_block
    frames, diag_rows = [], []
    for j, c in enumerate(sh.CLASSES):
        frames.append(pl.DataFrame({
            "design": "-", "n": 0, "arm": "truth", "cls": c, "block": np.arange(nb),
            "bside": bside, "est": dr.truth[j], "var": np.zeros(nb),
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
                calib = (cal, lp, par)
                extra = {"calibrator": repr(par)}
            elif mis is not None:
                assert lp is not None
                pf = np.exp(lp)
            else:
                pf = w.p_true
            rng = np.random.default_rng(wd.seed(n_side, 9, r, di, ni))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out, diags = setting(w, dr, design, n, pf, arms, calib=calib, rng=rng)
            for (arm, c), (est, var) in out.items():
                keep = np.flatnonzero(np.isfinite(est))
                frames.append(pl.DataFrame({
                    "design": design, "n": n, "arm": arm, "cls": c, "block": keep,
                    "bside": bside[keep], "est": est[keep], "var": var[keep],
                }))  # fmt: skip
            for d in diags:
                d |= extra | {"warnings": len(caught), "setting_seconds": time.time() - t0}
            diag_rows += diags
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
    dg = pl.from_dicts(diag_rows, infer_schema_length=None).with_columns(
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
    ap.add_argument("--maps", default="M1,M2,M3,M7")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arms", default=",".join(SCREEN))
    a = ap.parse_args()
    arms = tuple(a.arms.split(","))
    for arm in arms:
        if arm != "olofsson":
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
