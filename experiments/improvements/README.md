# Improvement experiments

These experiments test changes to the method that could improve two things:

- the **coverage** of its prediction intervals;
- the **accuracy of its block standard deviations (SDs)**.

They are run on simulated maps with a known truth. The candidate estimators live in
`candidates.py`; nothing here is used by the package.

**Outcome:** the zero-mean likelihood fit (`ml_zero_mean`) was adopted as the package
default for step 4, in `fit_correlation` and `fit_class_variograms`. The package
implementation (`fit_likelihood`) gives the same fits as `candidates.likelihood_fit`.
`method="wls"` keeps the paper's procedure, which reproduces the authors' R script.

Round 3 tests an estimated calibration (an unknown mean) for a separate holdout sample; see
[Round 3](#round-3-estimated-calibration-an-unknown-mean). Its adoption is pending.
[Round 4](#round-4-the-corrections-at-landscape-scale) retests the chosen corrections on a
1024 × 1024 landscape with holdouts of up to 5000 points.

## How the variants are judged

Every variant of a scenario is run on the same setup, so differences between variants are
paired:

- 200 simulated maps (realizations);
- the same validation data;
- the same Monte Carlo point pairs.

Coverage and accuracy are measured on 155 blocks:

- square tiles of 2 to 128 cells, at most 32 per size;
- six polygons.

Two criteria decide.

- **Coverage of the 90 % interval** (prediction ± 1.645 · SD). Coverage is computed per
  cell, where a cell is a class (categorical maps) crossed with a block size. The table
  reports:
  - `coverage error`: the mean of |coverage − 0.90| over cells;
  - `worst cell`: the lowest coverage of any cell.
- **SD accuracy**: `SD log-error` is the RMS of log(estimated SD / true SD) over
  realizations and blocks. It combines bias with fit-to-fit noise; 0.10 means about 10 %
  error.

A variant counts as better or worse than `current` in a scenario if:

- its coverage error moves by more than 0.005; or
- its SD log-error moves by more than 0.01.

`Δ coverage` gives the paired difference, with a standard error over realizations.

The true SD comes from two sources:

- **Gaussian continuous scenarios:** the exact block SD.
- **Forest-pipeline and categorical scenarios:** the RMSE of the block error over the 200
  realizations.

## Scenarios

| mode | scenario | what changes from the baseline |
|---|---|---|
| continuous | baseline | known σ and correlated errors (`ErrorModelScenario`): Exp range 8, nugget 0.25, 300 validation points, 128 × 128 grid |
| | n100 | 100 validation points |
| | range64 | correlation range 64 (longer than the variogram cutoff of about 60) |
| | nugget0.9 | 90 % nugget |
| | sd_scale0.75 | the σ handed to the method is 0.75 × the true σ |
| | spherical | spherical correlation with the same practical range, so the true model is not exponential |
| | pipeline | the authors' R example as a full pipeline: GRF, sampling, ranger-style QRF and 10-fold CV, with 200 points (`PipelineScenario`) |
| categorical | baseline | thresholded latent field with calibrated probabilities (`LatentClassScenario`): 3 classes at 60/30/10 %, 300 validation points |
| | n800 | 800 validation points |
| | temperature0.7 | over-confident probabilities (temperature 0.7) |
| | range64 | latent correlation range 64 |

## Variants

Each variant changes one part of the current procedure. The current procedure:

1. a Matheron variogram of the standardized errors, with gstat defaults;
2. an automatic bounded WLS fit of Nug + Exp/Sph/Gau, `fit.method = 7`;
3. ρ = (sill − γ)/sill.

| variant | round | change |
|---|---|---|
| `current` | 1, 2 | the procedure above (the package default, as in the paper and gstat) |
| `wls_npairs` | 1, 2 | WLS weights N_j (gstat `fit.method = 1`) instead of N_j / h_j² |
| `wls_cressie` | 1 | Cressie's weights N_j / γ(h_j)² (gstat `fit.method = 2`) |
| `exp_only` | 1 | only Nug + Exp, with no model selection |
| `sigma_weighted` | 1 | variogram of the raw errors with σσ pair weights, Σ[(e_i − e_j)² − (σ_i − σ_j)²] / 2Σσ_iσ_j. This is the weighting the block integral uses. |
| `reml` | 1, 2 | Gaussian REML fit of Nug + Exp/Sph/Gau to the standardized errors, with an unknown constant mean |
| `ml_zero_mean` | 2 | Gaussian ML fit with the mean fixed at zero, as the method assumes |
| `rescale` | 1 | the current SD × √mean(z²), which recalibrates σ from the validation errors |
| `pooled` | 1 | categorical only: one correlogram fitted to the variogram pooled over classes |

Variants not run:

- **Bootstrap mixture intervals and beta intervals:** dropped at the user's request.
- **Cressie–Hawkins sample variogram:** settled by an earlier comparison. It is badly biased
  for indicator errors, and only helps continuous maps with outliers.
- **Latent-Gaussian (copula) model of class errors:** left out because it is built the same
  way as the simulator, so it would win for the wrong reason.

## Results

The full tables are in `results/round1/scorecard.md` and `results/round2/scorecard.md`.

### Round 1 (single changes)

| mode | variant | coverage error | worst cell | SD log-error | coverage better / worse | SD better / worse |
|---|---|---|---|---|---|---|
| continuous | `reml` | 0.050 | 0.700 | 0.416 | 5 / 0 | 5 / 2 |
| continuous | `wls_npairs` | 0.053 | 0.665 | 0.497 | 5 / 0 | 2 / 2 |
| continuous | `wls_cressie` | 0.058 | 0.685 | 0.550 | 6 / 0 | 3 / 2 |
| continuous | `exp_only` | 0.080 | 0.575 | 0.861 | 5 / 2 | 3 / 4 |
| continuous | `sigma_weighted` | 0.088 | 0.600 | 0.777 | 4 / 3 | 2 / 4 |
| continuous | `rescale` | 0.089 | 0.605 | 0.757 | 1 / 2 | 1 / 5 |
| continuous | `current` | 0.090 | 0.600 | 0.734 | | |
| categorical | `wls_npairs` | 0.077 | 0.560 | 1.122 | 4 / 0 | 4 / 0 |
| categorical | `wls_cressie` | 0.091 | 0.540 | 1.386 | 4 / 0 | 4 / 0 |
| categorical | `reml` | 0.097 | 0.375 | 0.563 | 3 / 0 | 4 / 0 |
| categorical | `sigma_weighted` | 0.115 | 0.330 | 1.126 | 3 / 1 | 3 / 1 |
| categorical | `exp_only` | 0.131 | 0.525 | 1.950 | 2 / 0 | 0 / 4 |
| categorical | `pooled` | 0.135 | 0.405 | 1.530 | 1 / 1 | 4 / 0 |
| categorical | `rescale` | 0.136 | 0.375 | 1.630 | 1 / 2 | 2 / 1 |
| categorical | `current` | 0.137 | 0.455 | 1.637 | | |

- **`exp_only`** wins only when the truth is exponential. With a spherical truth its
  coverage is 0.93 and its SDs are 9 % too wide, and it is worse than `current` at
  nugget 0.9.
- **`rescale`** fixes the case where σ is too small (σ × 0.75: coverage 0.74 → 0.87). It
  breaks the forest pipeline, though (0.86 → 0.74): there the calibration of σ at the
  validation points does not carry over to the map.
- **`sigma_weighted` and `pooled`** help in some categorical scenarios and hurt in others.

### Round 2 (round-1 leaders plus the zero-mean likelihood fit)

The round-1 analysis showed where coverage still fails: large blocks (tiles of 64 and 128
cells, and polygons). For continuous maps the cause was noise in the long-range part of the
correlogram. Both the variogram and REML ignore a common offset in the validation errors,
but that offset is direct evidence of the map-wide error that dominates large blocks.
Fixing the mean at zero, as the method assumes anyway, lets the fit use it.

| mode | variant | coverage error | worst cell | SD log-error | coverage better / worse | SD better / worse |
|---|---|---|---|---|---|---|
| continuous | `ml_zero_mean` | **0.050** | 0.705 | **0.373** | 6 / 0 | 5 / 1 |
| continuous | `reml` | 0.050 | 0.700 | 0.416 | 5 / 0 | 5 / 2 |
| continuous | `wls_npairs` | 0.053 | 0.665 | 0.497 | 5 / 0 | 2 / 2 |
| continuous | `current` | 0.090 | 0.600 | 0.734 | | |
| categorical | `wls_npairs` | **0.077** | 0.560 | 1.122 | 4 / 0 | 4 / 0 |
| categorical | `ml_zero_mean` | 0.080 | 0.380 | **0.553** | 4 / 0 | 4 / 0 |
| categorical | `reml` | 0.097 | 0.375 | 0.563 | 3 / 0 | 4 / 0 |
| categorical | `current` | 0.137 | 0.455 | 1.637 | | |

Coverage by scenario:

| scenario | `current` | `wls_npairs` | `reml` | `ml_zero_mean` |
|---|---|---|---|---|
| continuous: baseline | 0.882 | 0.889 | 0.884 | 0.888 |
| continuous: n100 | 0.842 | 0.873 | 0.861 | 0.866 |
| continuous: range64 | 0.818 | 0.828 | 0.869 | **0.912** |
| continuous: nugget0.9 | 0.645 | 0.835 | 0.819 | 0.804 |
| continuous: sd_scale0.75 | 0.744 | 0.755 | 0.752 | 0.755 |
| continuous: spherical | 0.911 | 0.909 | 0.910 | 0.909 |
| continuous: pipeline | 0.857 | 0.867 | 0.901 | 0.880 |
| categorical: baseline | 0.816 | 0.871 | 0.831 | 0.831 |
| categorical: n800 | 0.841 | 0.867 | 0.851 | 0.857 |
| categorical: range64 | 0.753 | 0.825 | 0.800 | **0.857** |
| categorical: temperature0.7 | 0.643 | 0.734 | 0.746 | 0.751 |

## Findings

1. **The zero-mean likelihood fit (`ml_zero_mean`) is the best all-round change.**
   - It never moves coverage further from 0.90 than `current` in the 11 scenarios.
   - It halves the SD error. The SD log-error falls from 0.73 to 0.37 (continuous) and from
     1.64 to 0.55 (categorical).
   - It almost entirely removes the collapses to SD = 0 that the WLS fit produces (2–6 % of
     class-map fits).
   - It fixes the long-range case (continuous 0.82 → 0.91, with every block size at 0.905 or
     better; categorical 0.75 → 0.86).
   - **REML** does almost as well. It is less accurate in the forest pipeline (SDs 26 % too
     high there, vs 9 % for the zero-mean fit) and in the long-range case.
2. **`wls_npairs` (gstat `fit.method = 1`) is the cheapest real improvement.**
   - It is a one-argument change, it keeps the variogram workflow, and it never moves
     coverage further from 0.90.
   - For categorical maps it has the best average coverage (0.077 coverage error vs 0.080
     for the zero-mean fit).
   - It gets there partly by overestimating SDs (median ratio up to 1.22 for class maps and
     1.34 for continuous maps). Its SD error is about twice that of the likelihood fits.
3. **What no correlogram fit fixes:**
   - **A mis-scaled σ** (σ × 0.75, temperature 0.7). The method trusts the scale of σ, and
     the one recalibration tried (`rescale`) is not safe in the forest pipeline.
   - **Class maps, rare classes, large blocks.** For the 10 % class, the likelihood fits
     give SDs about 0.6 × the truth at large blocks and 1.3 × at the smallest tiles. The
     correlation of indicator errors depends on the class probabilities: it is stronger
     where p is mid-range. A single stationary correlogram cannot represent that. A model
     whose correlation varies with p would be the next experiment.

## Costs of adopting a likelihood fit

- **Departs from the paper and gstat.** The default would no longer reproduce the authors'
  R script. The current WLS procedure would have to stay available, for example as an
  option, for R parity.
- **Computation.** Each likelihood evaluation is a dense Cholesky factorisation, O(n³) in
  the number of validation points. Single-threaded fit times:

  | validation points | likelihood fit | WLS fit |
  |---|---|---|
  | 300 | 0.7 s | 0.2 s |
  | 800 | 7 s | 0.4 s |
  | 1,500 | 32 s | 0.2 s |

  Several thousand points would need subsampling or an approximate likelihood.
- **Assumes zero-mean standardized errors.** A systematic prediction bias is read as
  long-range correlation, which widens the intervals of large blocks. That is arguably the
  right reaction, since bias is a block-level error that the current method ignores.

## Round 3: estimated calibration (an unknown mean)

### The question

Class maps undercover, and worst for rare classes at large blocks. On real data with a
stratified reference sample, our rare-class intervals were also much narrower than
Olofsson's and PPI's. The method assumes the map is calibrated on average: standardized
errors have mean exactly 0. A systematic miscalibration then never averages out, so the
interval shrinks with block size while the error does not.

### The change

Treat the mean of the standardized error as unknown: z(s) = x(s)ᵀβ + w(s).

- The correlogram is fitted by REML with the same design X.
- β is estimated by GLS from the holdout residuals: A = (XᵀK⁻¹X)⁻¹, β̂ = A XᵀK⁻¹z.
- The block estimate is shifted by x_Bᵀβ̂, where x_B is the block mean of σ(s)x(s).
- The variance gains the uncertainty of β̂, minus a cross term:
  V_B + x_BᵀA x_B − 2 x_BᵀA XᵀK⁻¹ c_B, where c_B,i is the block mean of σ(s)ρ(|s − x_i|).

This is ordinary or universal kriging logic; the current method is the simple-kriging
case. `check_unknown_mean.py` verifies the shift and variance by exact simulation on a
small grid.

| arm | design X | cross term |
|---|---|---|
| `ok` | 1 | yes |
| `ok_nc` | 1 | no |
| `ok_strata` / `ok_strata_nc` | mapped-class indicators (strata with < 5 points pooled) | yes / no |
| `ok_logit` / `ok_logit_nc` | [σ, σ·logit p], a linearised logistic recalibration; GLS on the `ok` correlogram | yes / no |
| `ppi` | design-based: block map mean + mean residual of points in the block (≥ 20 points) | — |
| `olofsson` | design-based: post-stratified by mapped class | — |

### Stages

| stage | script | what it tests | cells × realizations |
|---|---|---|---|
| 1 | `round3.py` | full mapping pipelines (random-forest learner, 100 × 100 grid): workflows holdout, holdout + refit, random CV, distance-matched CV | 12 × 100 |
| 1b | `round3_holdout.py` | a separate 100-point holdout: split from the campaign, or independent random, stratified or clustered; recalibration on training-data CV | 12 × 100 |
| 2 | `round3_ideal.py --stage core` | the round-1/2 scenarios plus `bias`, `rare3`, `rare3_shift`, `rare3_omission`; holdout sizes 50–400 | 39 × 100–200 |
| 3 | `round3_ideal.py --stage designs` | validation designs: stratified, clustered, roads, half, confident, informative; holdout sizes 50–400 | 48 × 100–150 |

Learners in Stages 1 and 1b are covariate-only or location-aware (covariates plus x and y).
Training samples are random, clustered or along roads. Scorecards are in
`results/round3/{pipeline,holdout,core,designs}/scorecard.md`.

### Results

**Stage 1b, class maps: coverage error with global recalibration on training-data CV**
(mean over learners and training designs; area-estimate RMSE ÷ `current` in brackets).

| holdout | `current` | `ok` | `ok_nc` | `ok_logit_nc` | `ppi` | `olofsson` |
|---|---|---|---|---|---|---|
| split from the campaign | 0.192 | 0.099 | 0.080 | 0.074 | 0.185 | 0.249 |
| independent random | 0.059 | 0.046 | 0.038 (1.18) | 0.043 (1.21) | 0.078 | 0.144 |
| independent stratified | 0.104 | 0.062 | 0.036 (1.33) | 0.035 (1.32) | 0.211 | 0.221 |
| independent clustered | 0.147 | 0.077 | 0.055 (1.47) | 0.058 (**4.04**) | 0.305 | 0.401 |

**Stage 1b, continuous maps** (same layout): independent random holdout `current` 0.017,
`ok` 0.014 (1.06), `ok_nc` 0.050; independent clustered `current` 0.072, `ok` 0.053,
`ok_nc` 0.030.

**Stage 2, class maps at 300 points: coverage error.**

| scenario | `current` | `ok` | `ok_nc` | `ok_strata_nc` | `ok_logit_nc` | `oracle_ok` | `ppi` |
|---|---|---|---|---|---|---|---|
| baseline | 0.072 | 0.044 | 0.031 | 0.030 | 0.029 | 0.021 | 0.022 |
| range64 | 0.047 | 0.037 | 0.037 | 0.040 | 0.035 | 0.020 | 0.018 |
| rare3 | 0.056 | 0.037 | 0.028 | 0.027 | 0.025 | 0.021 | 0.052 |
| rare3_shift | 0.112 | 0.071 | 0.050 | 0.041 | 0.038 | 0.045 | 0.067 |
| rare3_omission | 0.098 | 0.069 | 0.046 | 0.038 | 0.036 | 0.052 | 0.073 |
| temperature0.7 | 0.151 | 0.135 | 0.091 | 0.074 | 0.066 | 0.092 | 0.028 |

Rare class (3 %) at 128-cell tiles, coverage (estimated SD ÷ true SD):

| scenario | `current` | `ok` | `ok_nc` | `ok_strata_nc` | `ok_logit_nc` | `ppi` | `olofsson` |
|---|---|---|---|---|---|---|---|
| rare3 | 0.75 (1.33) | 0.85 (0.92) | 0.94 (1.43) | 0.94 (1.30) | 0.96 (1.36) | 0.85 (0.93) | 0.83 (0.69) |
| rare3_shift | 0.34 (0.44) | 0.63 (0.30) | 0.76 (0.49) | 0.90 (0.54) | 0.90 (1.06) | 0.90 (1.03) | 0.91 (1.04) |
| rare3_omission | 0.36 (0.51) | 0.63 (0.17) | 0.78 (0.27) | 0.89 (0.25) | 0.91 (1.17) | 0.89 (0.97) | — |

**Stage 2, continuous maps at 300 points: coverage error, `current` → `ok`.** baseline
0.012 → 0.012; bias (+0.3σ) 0.118 → 0.012; n100 0.034 → 0.019; nugget0.9 0.096 → 0.044;
range64 0.012 → 0.005; spherical 0.009 → 0.008; sd_scale0.75 0.145 → 0.135; pipeline 0.040
→ 0.032. `oracle_ok` (true correlogram, with the cross term) is at 0.003–0.008 wherever σ is
right.

**Holdout size (Stages 2 and 3): coverage error / area RMSE ÷ `current`.**

| case | n | `current` | correction |
|---|---|---|---|
| continuous baseline, random (`ok`) | 50 | 0.059 | 0.057 / 1.11 |
| | 100 | 0.024 | 0.032 / 0.99 |
| | 200 | 0.015 | 0.023 / 0.92 |
| | 300–400 | 0.009–0.012 | 0.009–0.012 / 0.90 |
| class map rare3_shift, clustered (`ok_nc`) | 50 | 0.165 | 0.049 / 1.45 |
| | 100 | 0.146 | 0.061 / 1.31 |
| | 200 | 0.150 | 0.055 / 1.05 |
| | 300–400 | 0.125–0.129 | 0.057–0.062 / 0.98 |
| class map rare3_shift, stratified (`ok_nc`) | 50–100 | 0.093–0.095 | 0.024–0.032 / 1.23–1.71 |
| | 200–400 | 0.104–0.120 | 0.049–0.052 / 0.90–0.95 |

**Stage 3, calibrated rare class at 300 points: coverage error by validation design.**

| design | `current` | `ok_nc` | `ok_logit_nc` | `ppi` | `olofsson` |
|---|---|---|---|---|---|
| clustered | 0.112 | 0.048 | 0.038 | 0.304 | 0.388 |
| roads | 0.115 | 0.046 | 0.046 | 0.359 | 0.424 |
| half (western half only) | 0.067 | 0.030 | 0.028 | 0.085 | 0.249 |
| confident (∝ max p⁴) | 0.108 | 0.059 | 0.052 | 0.233 | 0.377 |
| stratified (equal allocation) | 0.174 | 0.079 | 0.070 | 0.450 | 0.095 |
| informative (×5 where the truth is rare) | **0.047** | 0.133 | 0.175 | 0.429 | 0.405 |

For continuous maps under the four Stage-3 designs, `ok` beats `current` under clustered,
roads and half samples (0.030 / 0.022 / 0.017 vs 0.041 / 0.035 / 0.031) and is 0.003
worse under the confident design (0.014 vs 0.011).

### Findings

1. **The correction fixes the main weakness.** Class-map coverage error roughly halves in
   every stage and design except the informative one. The rare class at large blocks goes
   from 0.34–0.75 to 0.76–0.94 coverage with `ok_nc` (128-cell tiles, Stage 2).
2. **The cross term is right in principle but too generous for class maps in practice.**
   With the true correlogram (`oracle_ok`) it is calibrated in both modes. With an estimated
   correlogram, class maps undercover with it (`ok`) and cover well without it (`ok_nc`).
   Continuous maps are accurate with it and overcover without it.
3. **How the holdout is collected matters more than any estimator.** A holdout split off a
   clustered or road campaign sits next to training points, so its residuals understate the
   map's errors; every method fails (Stage 1b, coverage error 0.07–0.25). Any independent
   holdout, even a clustered one, removes most of that.
4. **Recalibration fitted on training-data CV helps both modes; one parameter is enough.**
   A σ factor or a single temperature lowers coverage error in both modes. Per-class
   recalibration (temperature plus class biases) is no better and slightly worse for the
   rare class, whose biases rest on about 8 training points.
5. **The correction costs accuracy with small holdouts.** It removes bias where the map is
   miscalibrated but adds the noise of β̂ everywhere. Area-estimate RMSE is 1.1–1.8 ×
   `current` at 50–100 points and about break-even from 200.
6. **`ok_logit_nc` is the best class-map arm on idealized maps but fragile on real ones.**
   With the right mean shape it gets the rare-class SD right (1.06–1.17) where the
   constant mean underestimates it. With clipped random-forest probabilities and a
   clustered holdout its two design columns become nearly collinear, and the GLS breaks
   (area RMSE 4 ×, Stage 1b). A REML refit with the logit design was also unstable (one
   realization in 100 fitted a sill of 8.8), so the arm reuses the `ok` correlogram.
7. **Design-based estimators need many points per block.** PPI and Olofsson are calibrated
   at large blocks with enough random or stratified points, and fail under clustered,
   road or opportunistic samples (coverage error 0.23–0.45).
8. **Limits no estimator here fixes:**
   - **Informative sampling.** When inclusion depends on the true class, the residuals
     misrepresent the map's errors, and the correction makes coverage worse than
     `current` (0.133 vs 0.047). The correction must not be used with such samples.
   - **A mis-scaled σ.** A mean correction cannot fix a scale error (sd_scale0.75,
     temperature0.7); recalibration does.
   - **Skewed rare-class shares at mid-size blocks.** At 64-cell tiles every method,
     the oracle included, covers the rare class 0.73–0.89, because a symmetric normal
     interval misfits a share that is mostly near zero.

### Against the round-3 hypotheses

| hypothesis | outcome |
|---|---|
| H1: out-of-fold residuals plus a refit map are as valid as a holdout | Not in general: random CV is optimistic under clustered samples, and location-aware learners break it. Superseded by the separate-holdout assumption. |
| H2: clustered samples need distance-matched folds | Supported: with recalibration, distance-matched CV cut continuous coverage error from 0.12 (random CV) to 0.06 (Stage 1). |
| H3: the cross term is exact for a holdout map and must be dropped after a location-aware refit | Exact with the true correlogram; with an estimated one, class maps need it dropped even with a separate holdout. |
| H4: one-parameter recalibration helps class maps and is safe for continuous maps | Supported when fitted on training-data CV; fitting it on the holdout itself did not help. |

### Recommended workflow

1. Train the map on the training data and keep the model; do not refit on the holdout.
2. Collect the holdout **independently** of the training data, with a probability design
   if possible (random or stratified). Clustered or opportunistic holdouts work with the
   correction; informative ones (inclusion depending on the outcome) do not.
3. Fit a **global** recalibration on cross-validated training predictions: σ × √mean(z²),
   or one temperature for class probabilities.
4. Estimate the correlogram on the holdout by REML with an unknown constant mean, and
   apply the correction:
   - **continuous maps:** `ok`, with the cross term;
   - **class maps:** `ok_nc`, without the cross term.
5. **Size rule:** use the correction with a holdout of **at least about 200 points**, or
   when the map is evidently miscalibrated (for example a clear offset between predicted
   and observed class shares). With fewer points and a calibrated map, `current` is more
   accurate.

## Round 4: the corrections at landscape scale

### The question

Round 3 chose `ok` for continuous maps and `ok_nc` for class maps on a 128 × 128 grid with
50–400 holdout points. Do those choices hold at the scale where maps are used? Round 4
compares them with the package as it is (`current`), two oracles and the design-based
estimators on a 1024 × 1024 landscape, with holdouts of 100 to 5000 points and blocks from
4 × 4 cells to the whole landscape.

### Design

| factor | levels |
|---|---|
| landscape | 1024 × 1024 cells; error correlogram Nug 0.2 + Exp 0.8 with range parameter 32 cells (practical range ≈ 96). Continuous maps: σ from 0.5 to 2 on a smooth surface. Class maps: classes a/b/c at 67/30/3 %, with a known signal of range 64 cells. |
| blocks | square tiles of side 4 (256 random tiles), 64 (all 256), 256 (all 16) and 1024 cells (the landscape) |
| holdout | 100, 1000 or 5000 points, random or clustered (clusters of 10 cells within 16 cells); nested, 100 ⊂ 1000 ⊂ 5000 |
| map, continuous | `calibrated`; `offset` (the map over-predicts by 0.3 σ); `slope` (mean error −0.3 σ·m(s), with m a standardized field of range 64: a map that over-stretches); `scale` (the σ map is 0.75 × the true σ) |
| map, class | `calibrated`; `offset` (rare-class logit − 0.7: under-predicted); `slope` (rare-class logit × 1.3: too steep); `scale` (temperature 0.7: over-confident). Probabilities are clipped to [0.001, 0.999]. |
| realizations | 100 per mode. The truth, the holdout and the Monte Carlo pairs are shared by every map, size and design within a realization. |

Correlograms are fitted on a random subsample of at most 500 holdout points, as the
package advises for large n. The GLS mean uses all points.

| arm | what it is |
|---|---|
| `current` | the package: zero-mean ML correlogram, Monte Carlo V_B (10,000 pairs), no shift |
| `ok` | REML correlogram, GLS constant mean, with the cross term |
| `ok_nc` | the same without the cross term (class maps) |
| `ok_exact` | `ok` with V_B as exact cell sums instead of Monte Carlo (continuous maps) |
| `oracle` | true σ and true covariance, no shift. For class maps this is the exact non-stationary covariance of the true class shares. |
| `oracle_avg` | class maps: true σ with the domain-averaged correlogram, the best a stationary correlogram can do |
| `oracle_ok` | `ok` with the true correlogram (class maps: domain-averaged) |
| `ppi` | continuous maps: block map mean plus the mean residual of the points in the block |
| `olofsson` | class maps: post-stratified by mapped class within the block. Strata with fewer than 2 points are merged into the block's largest stratum. |

The design-based arms need 20 points in a block.

The cross term enters only through the block mean of σ(s)·v(s), where
v(s) = Σ_i G_i ρ(|s − x_i|). v is one FFT convolution on the cell grid, so the exact cell
sums of round 3 need no cells × points distance matrix. The oracle variances are also
exact cell sums: FFT for stationary correlograms, and bivariate-normal covariances of the
latent field for the class oracle. `check_round4.py` checks all of these against the
round-3 code and the scenario's exact block SD; they agree to about 1e-15.

### Results

Coverage of the 90 % interval. Each cell gives the result for a random / a clustered
holdout. The scorecard (`results/round4/scorecard.md`) and `summary.csv` hold every
combination.

**Continuous maps, whole landscape.**

| map | n | `current` | `ok` | `ok_exact` | `ppi` | `oracle` | `oracle_ok` |
|---|---|---|---|---|---|---|---|
| calibrated | 100 | 0.82 / 0.69 | 0.95 / 0.88 | 0.95 / 0.88 | 0.92 / 0.52 | 0.91 | 0.92 / 0.90 |
| | 1000 | 0.87 / 0.87 | 0.87 / 0.93 | 0.87 / 0.94 | 0.91 / 0.60 | 0.91 | 0.87 / 0.95 |
| | 5000 | 0.87 / 0.89 | 0.82 / 0.90 | 0.90 / 0.88 | 0.95 / 0.50 | 0.91 | 0.91 / 0.91 |
| offset | 100 | 0.36 / 0.08 | 0.95 / 0.88 | 0.95 / 0.88 | 0.92 / 0.52 | 0.01 | 0.92 / 0.90 |
| | 1000 | 0.03 / 0.01 | 0.87 / 0.93 | 0.87 / 0.94 | 0.92 / 0.54 | 0.01 | 0.87 / 0.95 |
| | 5000 | 0.01 / 0.02 | 0.82 / 0.90 | 0.90 / 0.88 | 0.96 / 0.50 | 0.01 | 0.91 / 0.91 |
| slope | 100 | 0.86 / 0.76 | 0.91 / 0.87 | 0.92 / 0.87 | 0.90 / 0.48 | 0.91 | 0.91 / 0.88 |
| | 1000 | 0.91 / 0.88 | 0.86 / 0.89 | 0.87 / 0.89 | 0.91 / 0.60 | 0.91 | 0.86 / 0.90 |
| | 5000 | 0.91 / 0.91 | 0.86 / 0.91 | 0.89 / 0.92 | 0.94 / 0.52 | 0.91 | 0.92 / 0.94 |
| scale | 100 | 0.74 / 0.60 | 0.85 / 0.82 | 0.85 / 0.82 | 0.92 / 0.52 | 0.91 | 0.84 / 0.82 |
| | 1000 | 0.74 / 0.71 | 0.76 / 0.76 | 0.76 / 0.76 | 0.91 / 0.60 | 0.91 | 0.77 / 0.78 |
| | 5000 | 0.75 / 0.73 | 0.75 / 0.81 | 0.83 / 0.79 | 0.95 / 0.50 | 0.91 | 0.85 / 0.82 |

Area-estimate RMSE ÷ `current` at the landscape, random / clustered:

| map | n | `ok` | `ppi` |
|---|---|---|---|
| calibrated | 100 | 1.18 / 3.26 | 1.33 / 3.42 |
| | 1000 | 0.52 / 0.92 | 0.46 / 1.03 |
| | 5000 | 0.36 / 0.44 | 0.18 / 0.47 |
| offset | 100 | 0.29 / 0.81 | 0.33 / 0.86 |
| | 1000 | 0.13 / 0.23 | 0.11 / 0.26 |
| | 5000 | 0.09 / 0.11 | 0.04 / 0.12 |

At 256-cell tiles the pattern is the same but milder:
- **Offset:** `current` covers 0.59–0.79 and `ok` 0.83–0.89, with RMSE 0.63–0.67 × `current` (random holdout).
- **Scale:** every model-based arm covers 0.72–0.80.
- **`ppi`:** 0.89–0.90 with a random holdout and 0.47–0.53 with a clustered one.

**Class maps, the stationarity gap.** The SD of the domain-averaged correlogram
(`oracle_avg`) ÷ the exact SD (`oracle`), median over blocks:

| class | 4 cells | 64 | 256 | 1024 |
|---|---|---|---|---|
| a (67 %) | 0.97 | 0.94 | 0.93 | 0.93 |
| b (30 %) | 0.98 | 0.95 | 0.96 | 0.98 |
| c (3 %) | 1.00 | 0.83 | 0.64 | 0.61 |

**Class maps, rare class (c, 3 %), whole landscape.**

| map | n | `current` | `ok_nc` | `ok` | `olofsson` | `oracle` | `oracle_avg` |
|---|---|---|---|---|---|---|---|
| calibrated | 100 | 0.94 / 0.64 | 0.96 / 0.98 | 0.90 / 0.97 | 0.81 / 0.56 | 0.89 | 0.68 |
| | 1000 | 0.70 / 0.56 | 0.97 / 0.86 | 0.90 / 0.84 | 0.92 / 0.67 | 0.89 | 0.68 |
| | 5000 | 0.65 / 0.59 | 0.95 / 0.84 | 0.71 / 0.60 | 0.90 / 0.62 | 0.89 | 0.68 |
| offset | 100 | 0.39 / 0.29 | 0.90 / 0.83 | 0.71 / 0.81 | 0.82 / 0.57 | 0.16 | 0.03 |
| | 1000 | 0.07 / 0.06 | 0.46 / 0.50 | 0.27 / 0.43 | 0.92 / 0.70 | 0.16 | 0.03 |
| | 5000 | 0.07 / 0.06 | 0.44 / 0.37 | 0.16 / 0.11 | 0.93 / 0.62 | 0.16 | 0.03 |
| slope | 1000 | 0.10 / 0.12 | 0.66 / 0.60 | 0.49 / 0.56 | 0.92 / 0.67 | 0.29 | 0.13 |
| | 5000 | 0.13 / 0.12 | 0.59 / 0.53 | 0.29 / 0.36 | 0.91 / 0.62 | 0.29 | 0.13 |
| scale | 1000 | 0.10 / 0.13 | 0.66 / 0.58 | 0.49 / 0.52 | 0.92 / 0.67 | 0.26 | 0.08 |
| | 5000 | 0.11 / 0.09 | 0.57 / 0.51 | 0.28 / 0.36 | 0.90 / 0.62 | 0.26 | 0.08 |

**Rare class at 256-cell tiles.**

| map | n | `current` | `ok_nc` | `olofsson` | `oracle` | `oracle_avg` |
|---|---|---|---|---|---|---|
| calibrated | 1000 | 0.80 / 0.62 | 0.81 / 0.72 | 0.67 / 0.40 | 0.93 | 0.76 |
| | 5000 | 0.78 / 0.73 | 0.78 / 0.73 | 0.83 / 0.64 | 0.93 | 0.76 |
| offset | 1000 | 0.63 / 0.53 | 0.71 / 0.63 | 0.68 / 0.40 | 0.82 | 0.69 |
| | 5000 | 0.64 / 0.58 | 0.72 / 0.64 | 0.83 / 0.64 | 0.82 | 0.69 |

**Common class (a, 67 %), whole landscape:** coverage (SD ÷ true SD).

| map | n | `current` | `ok_nc` | `ok` | `olofsson` | `oracle` |
|---|---|---|---|---|---|---|
| calibrated | 100 | 0.82 / 0.58 (1.60) | 0.94 / 0.90 (1.52) | 0.86 / 0.88 (0.94) | 0.88 / 0.64 | 0.89 |
| | 1000 | 0.76 / 0.67 (0.89) | 1.00 / 0.95 (2.11) | 0.85 / 0.90 (0.99) | 0.91 / 0.63 | 0.89 |
| | 5000 | 0.78 / 0.75 (0.85) | 1.00 / 1.00 (1.20) | 0.74 / 0.81 (0.43) | 0.91 / 0.60 | 0.89 |
| scale | 1000 | 0.36 / 0.21 (0.48) | 0.89 / 0.81 (1.16) | 0.63 / 0.68 (0.58) | 0.91 / 0.63 | 0.52 |
| | 5000 | 0.33 / 0.28 (0.45) | 0.93 / 0.89 (1.47) | 0.42 / 0.59 (0.54) | 0.91 / 0.60 | 0.52 |

The SD ratios in brackets are for the random holdout.

### Findings

1. **Small blocks are unaffected.** At 4- and 64-cell tiles, within the error range, the
   model-based arms agree to within 0.05, so the correction neither helps nor hurts. What
   matters there is the holdout:
   - 100 random points under-cover a little at 4-cell tiles (continuous 0.84, common
     classes 0.81–0.86, against an oracle at 0.90–0.93);
   - 100 clustered points under-cover 64-cell tiles (0.76–0.82; class maps are still at
     0.83 with 1000 points).
2. **For continuous maps, `ok` holds up and gains accuracy with n.**
   - **Under the offset**, `current` collapses: 0.01–0.36 at the landscape and 0.59–0.79
     at 256-cell tiles. `ok` covers 0.82–0.95 and 0.83–0.89.
   - **Offset and calibrated give identical `ok` results**, because the correction
     absorbs a constant offset of z exactly.
   - **Even on a calibrated map**, with 1000 or more points the corrected landscape
     estimate is more accurate than the map's own (RMSE 0.36–0.52 × `current`, random
     holdouts). The holdout measures the realized mean error of this particular map.
3. **With thousands of points, V_B must be computed exactly.**
   - **The size problem:** at the landscape with 5000 random points, the corrected
     variance is about 15 % of V_B. V_B from 10,000 Monte Carlo pairs has a relative SD
     of about 10 %.
   - **The consequence:** the corrected variance came out at or below zero in 5 of 100
     realizations, and `ok` covered 0.82.
   - **The fix:** exact cell sums (`ok_exact`) restore 0.90 at no cost in accuracy.
   - **The rule:** Monte Carlo V_B is fine for `current` and for small holdouts. The
     corrected estimate needs V_B accurate to well below the corrected variance.
4. **A mis-scaled σ passes straight through.** Under `scale`, every model-based arm,
   `oracle_ok` included, covers 0.71–0.85 with 1000 or more points (0.59–0.85 overall),
   with SD ratios near 0.75. The method uses
   only the correlation of the standardized errors and discards their sill. Global
   recalibration (round-3 workflow, step 3) remains necessary. The design-based arms are
   unaffected.
5. **A spatially varying bias is mild.** The `slope` bias averages out over large
   blocks. At 256-cell tiles every arm, the oracle included, covers 0.84–0.87 with random
   holdouts. A
   constant mean cannot remove it, but the correction does not make it worse.
6. **Rare classes break the stationary correlogram.**
   - **The gap:** the exact oracle shows that even the best stationary correlogram
     understates the SD of a 3 % class's block share. The shortfall is 17 % at 64-cell
     tiles and 36–39 % at 256 cells and beyond; for the common classes it is 2–7 %.
   - **Why:** a rare class's indicator errors are correlated only where the latent
     signal is near its threshold, and those cells also have the largest σ.
   - **The effect:** on a calibrated map, `current` covers the rare class 0.65–0.80 at
     256 cells and above once n ≥ 1000 (random holdouts). The exact oracle covers
     0.89–0.93.
   - **Why round 3 missed it:** its largest blocks were barely beyond the error range.
7. **`ok_nc` covers by overstating the SD; `ok` is right for common classes but not
   robust.**
   - **`ok_nc`:** without the cross term, its landscape SD for class a is 1.2–2.4 × too
     large with 1000 or more points (coverage 0.95–1.00). For the rare class this
     inflation happens to offset the stationarity gap on a calibrated map (0.95–0.97,
     random holdouts).
   - **`ok` at 1000 points:** with the cross term it gets the SD right (0.95–0.99).
   - **`ok` at 5000 random points:** coverage falls to 0.74, for two reasons:
     - the Monte Carlo noise of finding 3;
     - one realization in which REML on the 500-point subsample found a near-flat
       long-range component (Gaussian, range 1224 cells, nugget 89 %). The GLS mean was
       0.60 where the oracle's was 0.02. That realization alone makes `ok`'s landscape
       RMSE 1.2 × `current` rather than the 0.32 × of `oracle_ok`.
8. **A miscalibrated rare class needs a design-based estimate at large blocks.**
   - **The model-based arms fail:** a constant mean in z cannot follow a bias that
     depends on p. With 1000 or more points the bias left over dominates the landscape
     estimate: `ok_nc` covers 0.37–0.66, `ok` 0.11–0.56 and `current` 0.06–0.13.
   - **Olofsson works with a random holdout:** 0.90–0.93 at the landscape, and 0.83 at
     256-cell tiles with 5000 points. Its area RMSE is 0.14–0.42 × `current`.
   - **The correction still helps accuracy:** it roughly halves the error (RMSE
     0.42–0.61 × `current` with 1000 or more random points).
9. **Design-based estimators need random holdouts and enough points per block.**
   - **With a random holdout:** `ppi` covers 0.89–0.96 wherever it can be computed, and
     gives the most accurate large-block estimates (continuous landscape RMSE 0.04–0.46 ×
     `current` with 1000 or more points). `olofsson` covers 0.88–0.93 at the landscape,
     but less for the rare class in smaller blocks: 0.67 at 256-cell tiles with about 60
     points per tile, 0.83 with about 300.
   - **With a clustered holdout:** both break (`ppi` 0.47–0.60, `olofsson` 0.30–0.70 at
     256 cells and above), because their SEs assume independent points.
   - **With about 20 points per block:** Olofsson's SE for the rare class is often
     exactly zero, since none of the points is rare. Its coverage at 64-cell tiles is
     0.21–0.25.
10. **With small holdouts the correction costs accuracy, more so when clustered.** With
    100 points, the corrected landscape estimate has 1.2 × the RMSE of `current` on a
    calibrated continuous map with random points, and 3.3 × with 10 clusters (class
    maps: 2.1–3.3 ×). The intervals stay honest (coverage 0.86–0.98), but the estimate
    is worse. The round-3 size rule stands, and for a clustered holdout it is the number
    of clusters that counts.

### What this changes

1. **Continuous maps:** keep `ok`. When the holdout is large (about 1000 points or more)
   or a block holds much of it, compute V_B exactly (cell sums by FFT, as here) or with
   far more pairs.
2. **Class maps, `ok` or `ok_nc`:**
   - With a few hundred points, keep `ok_nc` (round 3).
   - With about 1000 points or more, `ok_nc` overstates large-block SDs 1.2–2.4 ×.
     `ok` with exact V_B is the principled choice for common classes, but two things are
     not yet tested: the class-map runs had no exact-V_B arm, and the REML fit needs a
     guard against near-flat long-range components (for example a cap on the practical
     range).
3. **Rare classes at large blocks:** do not trust model-based intervals for their
   landscape-scale shares. With a random holdout, use Olofsson. The stationary
   correlogram understates their SD by more than a third even when the map is
   calibrated.
4. **Recalibrate σ or the temperature first** (round 3, unchanged): no mean correction
   fixes a scale error.
5. **Next candidates:**
   - a range cap in the REML fit;
   - class-map `ok` with exact V_B;
   - for rare classes, the σσ-weighted correlogram of round 1, which weights pairs as
     the block integral does.

## Reproducing

```bash
uv sync --group experiments
uv run --group experiments python -m experiments.improvements.run --mode continuous --round 1
uv run --group experiments python -m experiments.improvements.run --mode categorical --round 1
uv run python -m experiments.improvements.summarize --round 1
# the same with --round 2
```

Run times on 4 cores:

- **Round 1:** about 15 min (continuous) and 40 min (categorical).
- **Round 2:** about 15 min (continuous) and 52 min (categorical). The 800-point
  categorical scenario alone takes 34 min, mostly for the likelihood fits.

`--quick` runs 4 realizations as a smoke test.

Round 3:

```bash
uv run python -m experiments.improvements.check_unknown_mean
uv run --group experiments python -m experiments.improvements.round3 --stage pipeline
uv run --group experiments python -m experiments.improvements.round3_holdout --stage
uv run --group experiments python -m experiments.improvements.round3_ideal --stage core
uv run --group experiments python -m experiments.improvements.round3_ideal --stage designs
uv run python -m experiments.improvements.round3 --summarize pipeline
uv run python -m experiments.improvements.round3_holdout --summarize
uv run python -m experiments.improvements.round3_ideal --summarize core   # and designs
```

`--skip-existing` resumes a stage after an interruption. On 4 cores: Stage 1 about 3.5 h,
Stage 1b about 2 h, Stage 2 about 4 h, Stage 3 about 2.5 h.

Round 4:

```bash
uv run --group experiments python -m experiments.improvements.check_round4
uv run --group experiments python -m experiments.improvements.round4 --quick   # smoke test
uv run --group experiments python -m experiments.improvements.round4 --skip-existing --max-chunks 3
uv run --group experiments python -m experiments.improvements.round4 --oracle-pass
uv run python -m experiments.improvements.round4 --summarize
```

The runner writes chunks of 20 realizations. `--max-chunks` keeps each call within a time
limit, and `--skip-existing` resumes. On 4 cores: about 1.1 h for the continuous maps,
2.8 h for the class maps, and 5 min for the exact class-map oracle (`--oracle-pass`).
