# Round 6: one fixed recipe for class maps, one for continuous maps

Round 5's best methods switched by block size (logit intervals and the σσ-weighted fit only
from 256×256), by sample size (parameter uncertainty only up to 500 points) or by design
(cluster bootstrap). Round 6 looked for one configuration per map type that runs unchanged at
every block size, landscape size, sample size and design, without cluster IDs, on calibrated,
recalibrated and uncorrected maps. Full tables are in [`scorecard.md`](scorecard.md); result
data and the HTML report ("Round 6 Recipes") stay outside the repository.

## Files

- `world.py`: the test bed at any landscape side n: class and continuous landscapes, tiles
  (4, 64, 256 and the landscape; at most 256 tiles per size), samplers and draws. At n = 1024
  every draw reproduces round 5 exactly; other sides append n to the seeds.
- `fits.py`: package REML fits (free family, Exp/Sph, Exp, zero-mean ML), the
  threshold-shaped REML fit, Laplace and grid-posterior sigma points, the Brier ratio.
- `calib.py`: calibrator draws: iid bootstrap, pair-correlation sandwich (cluster-process
  fit to the sample's pair distances), model sandwich, cluster bootstrap (reference).
- `run.py` (class maps M1–M7) and `cont.py` (continuous maps C1–C4): arms named
  `fit+modifier+...`, one parquet per landscape side × map × realization; resumable.
- `summarize.py`: interval rules, cells, the universal scorecard, marginals, the noise floor
  and the paired advance rule. `check.py`: Stage 0 checks (all pass).

```sh
uv run python -m experiments.round6.check [--quick]
uv run python -m experiments.round6.run OUT --side 1024 --maps M1,...,M7 --start 30 --stop 60 \
    --arms cur,ok,es+vp+cc,thr+vp+disp+cg,thr+x2+vp+disp+cg,olofsson
uv run python -m experiments.round6.cont OUT --side 256 --maps C1,C2s,C2l,C3s,C3l,C4 --start 30 --stop 50 \
    --arms cur,ok,es+vp+cc,ok+x3+scale,ok+x3+vp+scale
uv run python -m experiments.round6.summarize OUT out.json --kind class --sides 1024,256 \
    --maps M1,...,M7 --scorecard score.md --compare "arm|rule:base|rule,..."
```

Cost: a class task (one map × realization, 6 settings, 3 classes) takes 6–10 min at 1024²
and 3–5 min at 256² with these arms; the n = 5,000 GLS (one 5,000 × 5,000 Cholesky, about
2.5 s per correlogram) dominates. Peak memory about 1.6 GB per task; 4 workers on 4 cores.

## Test bed

- Class maps: M1 calibrated; M2/M3 combined miscalibration recalibrated by vector scaling /
  temperature; M4/M5/M6 rare-class offset / rare-class slope / overall scale, each recalibrated
  by vector scaling; M7 the combined miscalibration left uncorrected. All come from one
  simulator: a thresholded latent Gaussian field (Nug 0.2 + Exp 0.8, range 32; signal range
  64; prevalences 67/30/3 %).
- Continuous maps: round 4's error model; C1 calibrated; C2 (bias ∝ σ) and C3 (constant bias)
  with σ × 0.75, recalibrated by a σ factor (s) or a linear correction (l); C4 uncorrected.
- Landscapes 1024² and 256²; n = 50, 500, 5,000; random and clustered (clusters of 10 within
  16 cells); calibration samples drawn separately with the same design and size.

## Results

Universal error (mean |coverage − 0.90| over all cells; noise floor of a landscape cell 0.042
at 30 realizations, 0.051 at 20):

| class maps, 882 cells | error | 1024² / 256² | landscape blocks | worst cell |
|---|---|---|---|---|
| `cur` (paper) | 0.127 | 0.144 / 0.104 | 0.293 | 0 % |
| `ok` | 0.089 | 0.096 / 0.079 | 0.155 | 3 % |
| round-5 B `es+vp+cc` | 0.076 | 0.080 / 0.072 | 0.130 | 3 % |
| `thr+vp+disp+cg`, hybrid interval | 0.052 | 0.052 / 0.053 | 0.091 | 7 % |
| **`thr+x2+vp+disp+cg`, normal interval** | **0.042** | 0.043 / 0.041 | 0.063 | 50 % |

| continuous maps, 252 cells | error | 1024² / 256² | maps |
|---|---|---|---|
| `ok` | 0.070 | 0.075 / 0.062 | 0.034–0.155 |
| round-5 B | 0.059 | 0.064 / 0.053 | 0.030–0.140 |
| **`ok+x3+scale`** | **0.028** | 0.033 / 0.021 | 0.023–0.030 |

- The class recipe is flat across designs (0.049 clustered, 0.035 random), n (0.038–0.045),
  maps (0.037–0.066) and landscapes. It breaks one pre-registered rule: 55 of 6,300 landscape
  estimates fall outside [0, 1] (`ok`: 0). Without X = [1, σ] it meets every rule (0.052).
  Clipping estimates to [0, 1] is an open decision.
- Its worst cells are the rare class on the uncorrected map M7 (50–67 %).
- Continuous: parameter uncertainty adds nothing to `ok+x3+scale` (0.029 with it).

## What each component did (screen, 1024², 30 realizations)

- Fits (vs `ok`, 0.112): threshold-shaped −0.019, Exp −0.015, Exp/Sph −0.004.
- On the threshold fit: X = [1, σ] −0.028 (out-of-range +6); Brier-ratio dispersion −0.017;
  Laplace sigma points −0.008; grid-posterior sigma points +0.006 (rejected); calibrator draws
  −0.004 to −0.006 (iid, pair-correlation, cluster bootstrap all alike); hybrid interval
  −0.019; logit +0.090 and arcsine +0.007 (rejected).
- Continuous (vs `ok`, 0.065): REML scale −0.033; X = [1, 1/σ, m/σ] −0.013 (calibrator draws
  then add exactly 0); X = [1, 1/σ] −0.012; Laplace −0.007; calibrator draws −0.003 to −0.008.

## Open issues

- The simulator is itself a thresholded latent Gaussian, which favours the threshold-shaped
  correlogram. Robustness to other class-map generators, landscape structures, sizes and
  calibration-sample sizes is untested.
- Rare class on uncorrected maps (M7); dense, overlapping clusters get only part of their
  calibrator variance from the pair-correlation weights.
- Proposed for the package, pending approval: the class recipe (and clipping) and the
  continuous recipe. Package defaults are unchanged.
