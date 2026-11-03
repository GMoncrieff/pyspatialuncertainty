# Project memory

Notes for Claude Code sessions on this repository: what the package is, how we work on
it, and what the experiments and a real-data study have taught us. Keep it current when
a decision or finding changes.

## The package

A Python implementation of Wadoux & Heuvelink (2023): uncertainty of block averages and
totals of maps by Monte Carlo integration of σ(s)σ(u)ρ(|s − u|), for continuous maps and
per class for class-probability maps (σ_k = √(p_k(1 − p_k))).

- Core modules: `continuous.py`, `categorical.py`, `integrate.py` (MC engine),
  `blocks.py`, `grid.py`, `variogram/` (gstat-compatible variograms, WLS and likelihood
  fits), `mean_correction.py` (estimated-mean correction), `synthetic/` (truth for tests).
- Checks: `uv run pytest` (unit, doctests, R parity on committed reference data),
  `uv run ruff check . && uv run ruff format --check .`, `uv run mypy` (src only).
- Docs: `docs/` Quarto site with committed `_site`; rebuild with
  `cd docs && uv run quartodoc build && uv run quarto render`.

## Branches

- `main`: the paper's method, with a zero-mean ML likelihood fit as the step-4 default.
- `ok`: the estimated-mean ("ordinary kriging") correction is the default for continuous
  and class maps. REML fit (`mean="constant"`), GLS mean, shift x_B·μ̂, variance
  V_B + x_B²A − 2x_B·c_B (cross term on). `fit_correlation(..., method="wls")` with
  `aggregate_continuous(..., mean="zero")` is exactly the paper and passes R parity.
- `claude/improvement-experiments`: experiment rounds 1–2 (correlogram fits).
- `claude/unknown-mean-experiments`: rounds 3–4 (`experiments/improvements/round3*.py`,
  `round4.py`, `candidates.py` with the `ok`, `ok_nc`, `ok_strata`, `ok_logit`, oracle
  and design-based arms) and the write-up in `experiments/improvements/README.md`.
- `claude/round5-experiments` (off `ok`): round 5, `experiments/round5/` (harness with exact
  FFT V_B, `README.md`, `scorecard.md`). Results and the "Round 5 Ideas" report are not in the repo.
- `claude/round6-experiments` (off round 5; the latest branch, start new work here): round 6,
  `experiments/round6/` (landscape-size-generic test bed, threshold-shaped fit, design-free
  calibrator draws, universal scorecard; `README.md`, `scorecard.md`). Results and the
  "Round 6 Recipes" report (claude.ai artifact) are not in the repo. Package unchanged.

## Current state (after round 6)

Three class-map approaches are now the reference set (definitions in
`experiments/round6/run.py`, arm grammar `fit+modifier+...`):

- **`current`** (`cur`): the paper; zero-mean ML fit (free family), no mean correction,
  variance V_B.
- **`ok`** (`ok`): branch `ok`'s default; REML (free family Exp/Sph/Gau), GLS mean X = [1],
  shift and added variance with cross term.
- **R6 class** (`thr+x2+vp+disp+cg`, normal interval): threshold-shaped REML correlogram
  (indicator correlation of a latent Exp field cut at the class's mean threshold,
  t = Φ⁻¹(1 − p̄), via Owen's T; 2 parameters: latent nugget, latent range); GLS mean on
  X = [1, σ]; Laplace sigma points of the REML fit at every n; all variances × the pooled
  Brier ratio Σ(y − p)²/Σp(1 − p); calibrator uncertainty by 50 pair-correlation sandwich
  draws (`calib.pair_weights`: Thomas-process fit to the excess of the calibration sample's
  pair distances over a simple random sample of cells; no cluster IDs). Fit on ≤ 500 points;
  GLS and exact FFT integrals on all points. Open: clip estimates to [0, 1]?
- Strict variant meeting every pre-registered rule: `thr+vp+disp+cg` with the hybrid
  interval (union of clipped normal and logit intervals).
- Continuous recipe: `ok+x3+scale` (GLS mean on X = [1, 1/σ, m/σ], variances × REML sill).

Round 6 universal error (mean |coverage − 0.90| over all cells; floor ≈ 0.04): class
`cur` 0.127, `ok` 0.089, R6 class 0.042 (strict 0.052); continuous `ok` 0.070, R6 0.028.
All of it on one simulator family, so robustness is the open question.

## Working conventions (from the user)

- Develop and push only on the branch the user names; never push elsewhere or open a PR
  unless asked. Commit messages end with the session's attribution trailer.
- Keep the repository clean: no stray outputs, quick-run files or reports in commits.
  Scratch work, private data and one-off analysis code go in the session scratchpad.
- Data shared privately by colleagues (e.g. the thicket bundle) never goes into the
  repository, nor do scripts that read it.
- Don't change package defaults without being asked ("don't adopt yet" was the standing
  rule until the `ok` branch was requested).
- Reports are standalone HTML artifacts with many figures; prose plain and short,
  numbers checked against the data before publishing.
- No model identifiers in commits, docs or artifacts.
- Methods must be universal (round 6 onward): one fixed configuration for class maps and one
  for continuous maps; nothing may switch on block size, landscape size, sample size,
  sampling design or map scenario. The recipe may not use cluster IDs or design labels.
- Evaluate with the universal scorecard (`experiments/round6/summarize.py`): every cell
  equally weighted, marginals per dimension, worst cell, the binomial noise floor, paired
  bootstrap CIs over shared realizations; pre-register the advance rule.

## Operations (cloud sessions)

- 4 cores, 15 GB. A round-6 class task (one map × realization, 6 settings) takes 6–10 min at
  1024² and 3–5 min at 256²; the n = 5,000 GLS Cholesky (~2.5 s per correlogram) dominates,
  so landscape size matters less than the number of correlograms evaluated at n = 5,000.
- Run long jobs detached (`setsid nohup`), resumable (skip existing parquets), one process
  per task (`max_tasks_per_child=1`), BLAS pinned to one thread. Keep a harness-tracked
  background waiter (≤ 2 h, re-armed when it expires) so the container is not reclaimed.
  Worker restarts happen; detached runs survive them.
- The scratchpad is per session: results do not carry over to a new session. Commit the
  scorecard markdown (not data) and rerun if raw results are needed.

## Lessons from simulations with known truth

- **Correlogram fit (rounds 1–2).** Likelihood fits beat the paper's WLS fit; ML with
  the mean fixed at zero roughly halved block-SD error. gstat `fit.method = 1` is the
  cheapest improvement within WLS. No correlogram fit fixes a mis-scaled σ, or the rare
  class at large blocks.
- **Estimated mean (round 3).** Treating the mean of z as unknown (REML + GLS + shift +
  added variance) fixes most class-map undercoverage. The cross term is exact with the
  true correlogram; with an estimated one, class maps covered better without it at a few
  hundred points. How the holdout is collected matters more than any estimator: it must
  be independent of training data; informative sampling breaks every method. A global
  one-parameter recalibration (σ factor or temperature) fitted on training CV helps.
  Size rule: the correction pays off from about 200 points (or when the map is clearly
  miscalibrated); below that, on a calibrated map, the zero-mean method is more accurate.
  `ok_logit` is fragile (collinear design under clipped probabilities).
- **Landscape scale (round 4, 1024² cells).** `ok` holds for continuous maps and absorbs
  a constant offset exactly. With thousands of points, MC noise in V_B is large relative
  to the corrected variance: compute V_B exactly (FFT cell sums) or raise n_mc. A σ scale
  error passes straight through. A stationary correlogram understates a rare class's
  landscape-share SD by more than a third even on a calibrated map; use Olofsson there
  with a random holdout. `ok_nc` overstates large-block SDs 1.2–2.4× with ≥1000 points.
  REML on a subsample sometimes finds a near-flat long-range component that wrecks the
  GLS mean. Design-based estimators need random holdouts and enough points per block.

## Lessons from a real stratified dataset (colleague's thicket condition map)

Shared privately; the data and analysis code are not in the repository. Stratified
random sample (vegetation type × earlier map class), very unequal design weights, a
design-weighted Dirichlet calibrator with spatially cross-fitted out-of-fold p.

- The paper's method on the calibrated map gave a severe-class national interval about
  a seventh of Olofsson's; `ok` widened it to roughly Olofsson's width, close to the
  colleague's calibrator bootstrap. `current` ignores calibration uncertainty.
- The GLS mean is unweighted: with unequal-probability sampling it can miss a bias the
  design-weighted mean shows. A strata-aware mean (`ok_strata`: intercept + stratum
  covariates known for every pixel) recovered the raw-map estimates; a design-weighted
  GLS mean is the obvious next test.
- `ok` vs `ok_nc` changed little (same estimates, widths 3–13 % larger without the cross
  term).
- With few clustered labels, cross-fitting the calibrator over a dozen clusters gives
  near-zero out-of-fold probabilities (|z| in the hundreds) and every model-based method
  breaks. Calibrating on one half and fitting the error model on the other fixes it:
  `ok`/`ok_strata` then cover ~90 % even at 10 % of the labels, while Olofsson/PPI on a
  clustered subsample undercover (their SEs assume random sampling). With half the
  labels, design-based estimators on that half were the most reliable.
- `current` stays too narrow at any label count and can even narrow as labels are
  removed.
- REML again found domain-scale correlograms (ranges of hundreds of km) in some
  half-sample fits, giving very wide intervals.

## Lessons from round 5 (1024² class landscape, paired with the landscape check)

- Calibration uncertainty is the biggest missing term on recalibrated maps: a calibrator
  bootstrap (50 refits, clusters resampled for clustered samples) cut `ok`'s landscape
  coverage error by 35–46 % on vector-scaled maps. On continuous maps it helps after a
  linear recalibration; a σ factor alone does not move the estimate, so it adds nothing.
- REML on indicator errors picks Gaussian ~75 % of the time with 50 points; Exp/Sph only
  is a small, consistent gain. Sigma-point parameter uncertainty is another small gain.
- Logit intervals help the landscape (B → 0.034–0.049, near Olofsson) but cover ~60 % at
  4×4; use them only for very large blocks.
- Large-block undercoverage is mostly the correlogram fit (true correlogram: 256×256
  error 0.082 → 0.027). A σσ-weighted fit fixes 256×256 but puts estimates outside [0, 1].
- New mean models (logit intercept, X = σ, EB shrinkage) and a range cap did not help.
  X = 1/σ fixes a constant continuous offset but breaks a σ-shaped one.
- Temperature recalibration (M3) stays the worst case: rare class at 5,000 random points
  covers 6 % at the landscape.
- Superseded by round 6's universality rule: block-size-specific logit intervals and
  σσ-weighted fits, and the n-capped parameter uncertainty, are no longer allowed.

## Lessons from round 6 (universal recipes; 1024² and 256², M1–M7, C1–C4)

- **Correlogram shape.** The class-indicator correlogram falls fast over the first cells
  and then decays with the latent range. With clustered samples REML fitted the fast part
  and missed the tail (fitted V_B at 256×256 ≈ 0.5 × truth). The threshold-shaped family has
  both parts built in (0.5 → 0.8 of truth); Exp alone is better than Sph or Exp/Sph; a free
  second exponential overshoots up to 16×. With 50 points every fit overstates V_B 2–4×.
- **Mean shape.** X = [1, σ] (logit-offset shape) was the largest class gain (−0.028) and
  cut wild estimates sharply, but gives ~1 % of landscape estimates outside [0, 1]. For
  continuous maps X = [1, 1/σ, m/σ] absorbs a linear recalibration exactly, so calibrator
  draws add nothing on top of it.
- **Variance scale.** REML sill (continuous, −0.033) and pooled Brier ratio (class, −0.017)
  fix mis-scaled σ / overconfident probabilities, including uncorrected maps.
- **Calibrator draws.** The model sandwich using the fitted field correlation between
  calibration points overstates the variance 4–9× (the field also drives the truth). The
  conditional (design) variance is what helps; the pair-correlation sandwich recovers the
  cluster bootstrap within 3 % from coordinates alone. Pitfalls found on the way: compare
  against the lattice (discrete) null, not continuous CSR; pair counts vary like a
  U-statistic, so simulate the null covariance; restrict the cluster scale to ≤ 1/16 of the
  side. Dense overlapping clusters (5,000 points on 256²) are undercounted (~¼). All
  calibrator methods gave similar coverage gains (−0.004 to −0.008).
- **Rejected:** grid-posterior sigma points (lower small-block coverage), logit and arcsine
  intervals (shares at or near 0/1 in small tiles). The hybrid interval helps without X = [1, σ]
  and is neutral with it.
- **Residual failures:** rare class on the uncorrected map M7 (worst cells 50–67 % for R6
  class, 3 % for `ok`). Olofsson stays strong at the landscape with random samples (0.060).
- **Caveat:** the class simulator is itself a thresholded latent Gaussian, which favours the
  threshold-shaped fit. Robustness to other generators is untested.

## Open next steps

- Robustness study (next): `current`, `ok` and R6 class under other class-map generators
  (not thresholded Gaussians), landscape structures, landscape sizes and amounts of
  calibration/training data.
- Package proposals awaiting approval (defaults unchanged): the R6 class recipe (with or
  without clipping estimates to [0, 1]) and the continuous recipe `ok+x3+scale`.
- Rare class on uncorrected maps; design-weighted or strata-aware GLS mean for unequal-
  probability samples (thicket lesson); exact V_B (FFT) in the package.
