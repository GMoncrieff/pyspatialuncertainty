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

## Open next steps

- Package proposals awaiting approval: calibrator bootstrap variance, Exp/Sph-only REML
  for class maps, a sigma-point parameter-uncertainty option, logit intervals for large blocks.
- A large-block correlogram fit that keeps the GLS mean stable.
- Exact V_B (FFT cell sums) in the package for large validation sets.
- A design-weighted or strata-aware mean in the package (`X` design in the GLS).
- Class-map `ok` with exact V_B; the σσ-weighted correlogram for rare classes.
