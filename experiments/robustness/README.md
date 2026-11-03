# Robustness study: the three class-map approaches on other landscapes

Rounds 1–6 all used one simulator: a thresholded latent Gaussian field (Nug 0.2 + Exp 0.8,
range 32; signal range 64; 67/30/3 %). That family also suits round 6's threshold-shaped
correlogram. This study freezes the three approaches and changes the world: landscape
generators, landscape sizes, calibration vs validation sample sizes and the amount of data
the map was trained on. Result data and the HTML report stay outside the repository.

## Approaches (frozen, `experiments/round6/run.py`)

- `cur`: the paper (zero-mean ML fit, free family, no mean correction).
- `ok`: REML free family, GLS mean X = [1], estimated-mean correction.
- R6 = `thr+x2+vp+disp+cg`: threshold-shaped REML, X = [1, σ], Laplace sigma points,
  Brier-ratio dispersion, pair-correlation calibrator draws.
- All with normal 90 % intervals. `olofsson` (landscape, random samples) is an unranked
  reference.

## Files

- `gens.py`: the generators (worlds and per-realization draws).
- `run.py`: driver; tasks are side × generator × map state × realization; resumable.
- `summarize.py`: the pre-registered scorecards. `check.py`: Stage 0 checks (all pass).
- The round-6 modules gained plumbing only: `setting()` takes `n_cal` and the class names,
  `class_draws` takes K from the scenario, the summary functions take the cell dimensions.
  Defaults are unchanged; the driver's `base` × `offset` output equals round 6's M4 bit for bit.

```sh
uv run python -m experiments.robustness.check [--quick] [--ref m4_ref.parquet]
uv run python -m experiments.robustness.run OUT --side 256 --gens base,potts --states cal,offset,scale \
    --start 0 --stop 30 [--grid diag|offdiag] [--ntrain 1000] [--workers 4]
uv run python -m experiments.robustness.summarize OUT out.json --scorecard scorecard.md
```

## Test bed

Map states (every map, the calibrated one included, is recalibrated by vector scaling fitted
on a separate calibration sample of the same design): `cal` (no miscalibration), `offset`
(rare-class logit −0.7, round 6's M4) and `scale` (logits × 1/0.7, round 6's M6). Designs:
simple random and clustered (10 per cluster within 16 cells). Blocks: tiles of 4, 64 and 256
cells per side (those smaller than the landscape, at most 256 each) and the landscape.

Generators (distance from round 6's simulator declared in advance):

| generator | distance | what changes |
|---|---|---|
| `base` | anchor | round 6's world (identical draws) |
| `smooth` | middle | latent correlogram Gaussian (a = 55.4, same practical range) |
| `nested` | middle | Exp(4) 0.4 + Exp(128) 0.4 |
| `aniso` | middle | Exp ranges 64 along 30° and 16 across |
| `nonstat` | middle | range 8 in the west blending to 128 in the east |
| `rare6` | near | 6 classes, 40/25/15/10/9/1 % |
| `sharp` / `blurry` | near | signal-to-noise 4 / 0.25 (map accuracy) |
| `classifier` | far | nonlinear class process on 4 covariates with Gumbel noise fields; map from a linear multinomial logistic regression trained on 1,000 labels of the same landscape |
| `potts` | far | as `classifier` but the classes are a Potts Markov field (β = 0.8) |

The latent generators keep round 6's truth (block mean of the class probability given the
structured noise) and the same map signal μ. The classifier worlds add point-scale
salt-and-pepper noise (a label is redrawn from softmax(u) with probability 0.1) and the truth
is the block mean of 0.9·1{class} + 0.1·softmax(u). Their maps are refitted every realization
and are never exactly calibrated.

## Pre-registered evaluation

Written before the main runs.

- A cell is side × generator × state × design × n_val × n_cal × n_train × class × block
  size. Universal error = mean |coverage − 0.90| over cells, every cell weighted equally
  (round 6's scorecard). The noise floor of a landscape cell is reported.
- Scorecards: **core** (256², all generators, n_cal = n_val ∈ {50, 500, 5000}, n_train
  1,000; the headline); **sizes** (`base`, `nonstat`, `classifier`, `potts` at 128²–1024²
  and `base`, `potts` at 2048²); **cross** (the same four at 256², n_val × n_cal ∈ {50, 500,
  5000}²); **training** (`classifier`, `potts` at 256², n_train 200 / 1,000 / 5,000);
  **all** (every cell of every run). Satellites use realizations 0–19.
- Reported per scorecard: universal error per arm, overall (mean over cells) and as the mean
  of per-generator errors; per-generator error; marginals by block size, n_val, n_cal,
  n_train, design, side, state, generator and class role (common / mid / rare); the worst cell
  per arm, overall and per generator.
- Ranking per generator and per satellite level: d1 = err(R6) − err(ok) and
  d2 = err(ok) − err(cur), each with a paired bootstrap 90 % CI over realizations (400
  resamples). Verdict: **clear** if both CIs lie below 0; **holds** if both differences are
  negative; **reversed** if either CI lies above 0; otherwise **tie**. The far generators
  (`potts`, `classifier`) are reported first.
- R6 is called **robust** if it beats `ok` on every generator, no verdict is reversed, and no
  marginal level of R6 is worse than `ok`'s by more than 0.02 (round 6's rule). Its
  degradation from the anchor, err_g(R6) − err_base(R6), is reported per generator.
- Estimates outside [0, 1] are counted per arm and generator (landscape and all blocks).
  Coverage with estimates clipped to [0, 1] is a diagnostic only; clipping stays open.

## Runs

Fixed from the pilot before the main runs (4 workers; minutes per task under 4-way load: 3.7
at 256², 6.8 for `rare6`, 7.5 at 1024², about 7 for the off-diagonal grid, about 20 at
2048² with 2.5 GB peak memory), in this order, about 46 hours in all:

| stage | side | generators | grid | realizations |
|---|---|---|---|---|
| core | 256² | all 10 | n_cal = n_val, n_train 1,000 | 0–29 |
| sizes | 1024² | `base`, `nonstat`, `classifier`, `potts` | n_cal = n_val | 0–19 |
| cross | 256² | the same four | the six n_cal ≠ n_val pairs | 0–19 |
| training | 256² | `classifier`, `potts` | n_train 200 and 5,000 | 0–19 |
| sizes | 512², 128² | the same four | n_cal = n_val | 0–19 |
| sizes | 2048² | `base`, `potts` | n_cal = n_val | 0–9 (indicative) |

Every task runs all three map states. Noise floor of a landscape cell: 0.042 at 30
realizations, 0.051 at 20, 0.07 at 10.
