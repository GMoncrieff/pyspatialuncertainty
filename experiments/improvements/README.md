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
