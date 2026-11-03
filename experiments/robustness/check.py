"""Stage 0 checks of the robustness harness.

1. Round-6 identity: ``base`` gives round 6's world and draws, and the driver's ``base`` ×
   ``offset`` realization equals round 6's M4 output saved before the plumbing edits
   (``--ref``, a parquet written by ``round6.run.realization(256, 0, "M4", arms)``).
2. Calibration: per generator, label frequencies against the map (the oracle softmax(u)
   for the classifier worlds), by probability bin, with t statistics over realizations.
3. Truth: the mean label of a simple random sample against the landscape truth, per
   realization (the truth is the expected point label of the cells).
4. Samplers: anisotropic, non-stationary and nested correlations against their targets.
5. Potts: prevalences after tuning and the Gibbs plateau; classifier maps after vector scaling.
6. Layouts at 128², 512² and 2048²; the ranking verdicts of the scorecard on toy data.

Usage: ``uv run python -m experiments.robustness.check [--quick] [--ref ref.parquet]``
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import polars as pl

from experiments.robustness import gens as G  # noqa: N812
from experiments.robustness import run
from experiments.robustness import summarize as sm
from experiments.round5 import shared as sh
from experiments.round6 import world as wd


def report(name: str, good: bool, detail: str) -> bool:
    print(f"{name}: {detail} {'ok' if good else 'FAIL'}", flush=True)
    return good


def check_round6(sides: tuple[int, ...], ref: str | None) -> bool:
    ok = True
    for n in sides:
        w6 = wd.class_world(n)
        w = G.world(G.GENS["base"], n)
        good = np.array_equal(w.p_raw, w6.p_raw) and all(
            np.array_equal(w.layout.tiles[s], w6.layout.tiles[s]) for s in w6.layout.sides
        )
        d6 = wd.class_draws(w6, 3)
        d, _ = G.draws(w, 3)
        good &= np.array_equal(d.truth, d6.truth)
        for name in ("cells", "labels", "cal_cells", "cal_labels", "clusters"):
            good &= all(
                np.array_equal(getattr(d, name)[k], getattr(d6, name)[k]) for k in sh.DESIGNS
            )
        ok &= report(f"base = round 6 world and draws ({n}²)", bool(good), "identical")
    if ref is None:
        return ok
    old = pl.read_parquet(ref)
    new, _ = run.realization(256, "base", "offset", 0, arms=run.ARMS)
    key = ["design", "n", "arm", "cls", "block"]
    j = old.join(new.select(*key, pl.col("est").alias("e2"), pl.col("var").alias("v2")), on=key)
    diff = float(
        j.select(
            pl.max_horizontal(
                (pl.col("est") - pl.col("e2")).abs(), (pl.col("var") - pl.col("v2")).abs()
            )
        )
        .max()
        .item()
    )
    good = j.height == old.height == new.height and diff == 0.0
    return ok & report(
        "driver base × offset = round 6 M4 (256², r 0, all arms)",
        good,
        f"{j.height} rows matched of {old.height}; max |Δ| {diff:.1e}",
    )


def check_calibration(n: int, n_real: int) -> bool:
    ok = True
    for name, g in G.GENS.items():
        w = G.world(g, n)
        k_cls = len(g.classes)
        res: list[np.ndarray] = []  # (real, class, bin) mean residual
        counts = np.zeros((k_cls, 10))
        land = []
        for r in range(n_real):
            dr, p_map = G.draws(w, 1000 + r)
            p = w.p_star if g.kind == "classifier" else p_map
            cells, lab = dr.cells["random"], dr.labels["random"]
            Y = np.eye(k_cls)[lab].T
            pc = p[:, cells]
            bins = np.minimum((pc * 10).astype(int), 9)
            m = np.full((k_cls, 10), np.nan)
            for k in range(k_cls):
                for b in range(10):
                    sel = bins[k] == b
                    counts[k, b] += sel.sum()
                    if sel.any():
                        m[k, b] = float((Y[k, sel] - pc[k, sel]).mean())
            res.append(m)
            land.append(Y.mean(axis=1) - dr.truth[:, -1])
        R = np.array(res)
        mean = np.nanmean(R, axis=0)
        se = np.nanstd(R, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(R), axis=0))
        enough = (counts >= 50 * n_real) & (np.sum(np.isfinite(R), axis=0) >= n_real // 2)
        t = np.where(enough, np.abs(mean) / np.maximum(se, 1e-12), 0.0)
        L = np.array(land)
        t_land = np.abs(L.mean(axis=0)) / (L.std(axis=0, ddof=1) / math.sqrt(n_real) + 1e-12)
        exact = name != "potts"  # the Potts field is not calibrated to softmax(u)
        bias = float(np.nanmax(np.abs(np.where(enough, mean, 0.0))))
        good = (float(t.max()) < 4.0 or not exact) and float(t_land.max()) < 4.0
        ok &= report(
            f"calibration and truth: {name}",
            bool(good),
            f"max |t| over bins {t.max():.1f} (max |bias| {bias:.3f}"
            f"{'' if exact else ', not required'}); sample mean label vs truth max |t| "
            f"{t_land.max():.1f}",
        )
    return ok


def _corr(fields: np.ndarray, dy: int, dx: int, cols: slice = slice(None)) -> float:
    a = fields[:, : fields.shape[1] - dy, : fields.shape[2] - dx][:, :, cols]
    b = fields[:, dy:, dx:][:, :, cols]
    return float(np.mean(a * b) / math.sqrt(np.mean(a * a) * np.mean(b * b)))


def check_samplers() -> bool:
    rng = np.random.default_rng(7)
    n = 512
    ok = True
    an = G.AnisoEmbedding(n, 1.0, 64.0, 16.0, 30.0).sample(rng, 16)
    a = math.radians(30.0)
    # lag of length 8 along the major axis and across it, in (row, col) = (dy, dx)
    lags = {
        "major": (8 * math.sin(a), 8 * math.cos(a)),
        "minor": (8 * math.cos(a), -8 * math.sin(a)),
    }
    detail = []
    for k, (dy, dx) in lags.items():
        dyi, dxi = round(dy), round(dx)
        u = (dxi * math.cos(a) + dyi * math.sin(a)) / 64.0
        v = (-dxi * math.sin(a) + dyi * math.cos(a)) / 16.0
        target = math.exp(-math.hypot(u, v))
        # flip the columns for a negative dx
        got = _corr(an[:, :, ::-1], dyi, -dxi) if dxi < 0 else _corr(an, dyi, dxi)
        ok &= abs(got - target) < 0.06
        detail.append(f"{k} {got:.2f} vs {target:.2f}")
    ns = G.BlendEmbedding(n, 1.0, 8.0, 128.0).sample(rng, 16)
    for name, cols, x in (("west", slice(0, 32), 16), ("east", slice(n - 32, n), n - 16)):
        th = 0.5 * math.pi * (x + 0.5) / n
        target = math.cos(th) ** 2 * math.exp(-4 / 8) + math.sin(th) ** 2 * math.exp(-4 / 128)
        got = _corr(ns, 4, 0, cols)
        ok &= abs(got - target) < 0.06
        detail.append(f"nonstat {name} lag 4 {got:.2f} vs {target:.2f}")
    sc = G.world(G.GENS["nested"], n).sc
    assert sc is not None
    ne = sc.sampler.sample(rng, 16) / math.sqrt(0.8)
    for lag in (2, 32):
        target = 0.5 * math.exp(-lag / 4) + 0.5 * math.exp(-lag / 128)
        got = _corr(ne, 0, lag)
        ok &= abs(got - target) < 0.06
        detail.append(f"nested lag {lag} {got:.2f} vs {target:.2f}")
    return report("samplers", bool(ok), "; ".join(detail))


def check_potts_and_maps(n: int) -> bool:
    g = G.GENS["potts"]
    w = G.world(g, n)
    assert w.u is not None
    prev = []
    for r in range(4):
        S, agree = G.gibbs_potts(w.u, n, np.random.default_rng(r), trace=True)
        prev.append(np.bincount(S, minlength=3) / S.size)
    P = np.mean(prev, axis=0)
    plateau = abs(np.mean(agree[-50:]) - np.mean(agree[100:150]))
    ok = report(
        f"potts prevalence and Gibbs plateau ({n}²)",
        bool(np.max(np.abs(P - np.array(g.prevalence))) < 0.03 and plateau < 0.005),
        f"prevalence {np.round(P, 3)}; equal-neighbour share {agree[-1]:.3f}, drift {plateau:.4f}",
    )
    for name in ("classifier", "potts"):
        w = G.world(G.GENS[name], n)
        dr, p = G.draws(w, 0)
        lp = G.map_logits(p, "offset")
        fit, apply = sh.CALIBRATORS["vector"]
        cells, lab = dr.cal_cells["random"], dr.cal_labels["random"]
        pf = sh.clip_norm(apply(fit(lp[:, cells], lab), lp))
        f_lab = np.bincount(dr.labels["random"], minlength=3) / dr.labels["random"].size
        gap = np.abs(pf.mean(axis=1) - f_lab)
        ok &= report(
            f"{name} map recalibrated (offset, 5000 random calibration points)",
            bool(gap.max() < 0.02),
            f"map shares {np.round(pf.mean(axis=1), 3)} vs labels {np.round(f_lab, 3)}",
        )
    return ok


def check_layouts() -> bool:
    good = True
    detail = []
    for n, sides, counts in (
        (128, (4, 64, 128), (256, 4, 1)),
        (512, (4, 64, 256, 512), (256, 64, 4, 1)),
        (2048, (4, 64, 256, 2048), (256, 256, 64, 1)),
    ):
        lay = wd.make_layout(n)
        good &= lay.sides == sides and tuple(lay.tiles[s].size for s in sides) == counts
        good &= bool(np.allclose(np.asarray(lay.bmat.sum(axis=1)).ravel(), 1.0))
        detail.append(f"{n}²: {lay.n_blocks} blocks")
    return report("layouts", bool(good), "; ".join(detail))


def check_verdicts() -> bool:
    rng = np.random.default_rng(8)
    rows = []
    for arm, c in (("cur", 0.6), ("ok", 0.8), (run.R6, 0.89)):
        for b in (4, 256):
            for r in range(30):
                rows.append({
                    "arm": arm, "side": 256, "gen": "base", "state": "cal", "design": "random",
                    "n": 50, "n_cal": 50, "n_train": 0, "cls": "a", "role": "common", "bside": b,
                    "r": r, "cov_normal": float(rng.uniform(c - 0.05, c + 0.05)),
                    "cov_clip": 0.9,
                })  # fmt: skip
    pr = pl.DataFrame(rows)
    cl = pr.group_by(["arm", *sm.DIMS]).agg(
        pl.col("cov_normal").mean(), pl.col("r").n_unique().alias("n_real")
    )
    v1 = sm.ranking(cl, pr)["verdict"]
    swap = pr.with_columns(
        arm=pl.when(pl.col("arm") == "ok").then(pl.lit(run.R6))
        .when(pl.col("arm") == run.R6).then(pl.lit("ok")).otherwise(pl.col("arm"))
    )  # fmt: skip
    cl2 = swap.group_by(["arm", *sm.DIMS]).agg(
        pl.col("cov_normal").mean(), pl.col("r").n_unique().alias("n_real")
    )
    v2 = sm.ranking(cl2, swap)["verdict"]
    return report("ranking verdicts on toy data", v1 == "clear" and v2 == "reversed", f"{v1}, {v2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    ok = check_verdicts()
    ok &= check_layouts()
    ok &= check_samplers()
    ok &= check_round6((256,) if a.quick else (256, 1024), a.ref)
    ok &= check_potts_and_maps(256)
    ok &= check_calibration(128, 12 if a.quick else 40)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
