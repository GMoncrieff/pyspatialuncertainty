# Round 5: principled fixes for calibrated and miscalibrated maps

Fifteen changes to the estimated-mean method (`ok`), each aimed at a failure that the
landscape check found on the 1024 × 1024 class landscape. They were tested on the same
paired realizations, holdouts of 50 / 500 / 5,000 points (random and clustered), blocks
from 4×4 cells up to the landscape, and 90 % intervals. Full tables are in
[`scorecard.md`](scorecard.md). Result data and the HTML report stay outside the repository.

## Files

- `shared.py`: landscape, samples, class maps M1–M6, calibrators, Olofsson, and exact
  FFT block integrals (V_B and the cross term).
- `arms.py`: GLS with a design X, empirical-Bayes shrinkage, the logit intercept, REML
  sigma points (`vparam`), and PPI.
- `run.py`: the class-map driver, one parquet per realization and map, resumable.
  `cont.py` is the continuous-map driver.
- `check.py`: Stage 0 checks. They cover the reductions to `ok`/`current`, exact simulation
  of the variance formulas, GLS against `estimate_mean`, and the FFT cross term against
  brute force.
- `summarize.py`: coverage, SD ratio, accuracy, the scorecard with paired bootstrap CIs,
  and the advance rule.

```sh
uv run python -m experiments.round5.check
uv run python -m experiments.round5.run OUT --maps M1,M2,M3 --start 0 --stop 100 \
    --workers 4 --boot 50 --vparam-max-n 500 --arms current,ok,A,B,D,ppi,olofsson
uv run python -m experiments.round5.cont OUT --maps C1,C2s,C2l,C3s,C3l --stop 50
uv run python -m experiments.round5.summarize OUT summary.json --maps M1,M2,M3 --scorecard score.md
```

## Maps

| map | what was done to it |
|---|---|
| M1 | calibrated |
| M2 | combined miscalibration, then vector scaling fitted on the n validation points |
| M3 | combined miscalibration, then temperature scaling |
| M4 / M5 / M6 | rare-class logit offset / rare-class slope / overall temperature, then vector scaling |
| C1 / C2 / C3 | continuous: calibrated / bias −0.3σ with σ × 0.75 / constant bias with σ × 0.75; `s` recalibrates with a σ factor, `l` with intercept + slope + σ factor |

## Results

Coverage error is the mean of |coverage − 0.90| over the settings.

**Screen (Stage 1).** M1–M3, 50 realizations, all arms.

- Advanced:
  - `fam_es`: REML limited to Exp/Sph.
  - `vparam`: the variance averaged over unscented sigma points of the REML fit.
  - `ok_cb`: calibrator bootstrap; it helped on both recalibrated maps.
- Large blocks only:
  - `sw`: σσ-weighted correlogram.
  - Logit intervals: these collapse to about 60 % coverage at 4×4.
- Rejected: `ok_lin`, `ok_cil`, `ok_cil_n`, `ok_eb`, `ok_eb_pool` and `cap`.
- The true correlogram cuts the 256×256 error from 0.082 to 0.027 on M1.

**Combinations (Stage 2).**

- A = `ok` + `fam_es` + `vparam`.
- B = A + calibrator bootstrap.
- C = B with logit intervals from 256×256.
- D = B with the `sw` correlogram from 256×256.

Landscape / 256×256 coverage error:

| map | ok | A | B | C | D |
|---|---|---|---|---|---|
| M1 | 0.051 / 0.086 | 0.050 / 0.079 | = A | 0.041 / 0.088 | 0.046 / 0.053 |
| M2 | 0.107 / 0.111 | 0.092 / 0.100 | 0.058 / 0.089 | 0.034 / 0.089 | 0.067 / 0.053 |
| M3 | 0.163 / 0.105 | 0.159 / 0.096 | 0.147 / 0.091 | 0.116 / 0.088 | 0.136 / 0.049 |
| M4 | 0.097 / 0.104 | 0.082 / 0.089 | 0.058 / 0.077 | 0.040 / 0.083 | 0.066 / 0.056 |
| M5 | 0.096 / 0.106 | 0.081 / 0.091 | 0.062 / 0.079 | 0.047 / 0.085 | 0.079 / 0.056 |
| M6 | 0.106 / 0.110 | 0.090 / 0.096 | 0.067 / 0.083 | 0.049 / 0.086 | 0.081 / 0.060 |

- Olofsson: 0.040–0.047 at the landscape.
- A and B pass the advance rule on M1–M3 and on M4–M6, with accuracy within about 2 % of
  `ok`.
- D fails: 7–26 landscape estimates per map fall outside [0, 1]. B has 0–2.
- M3's rare class still covers only 6 % at the landscape with 5,000 random points.

**Continuous maps.** 50 realizations.

- B equals A where a σ factor recalibrates the map, because the σ factor does not move the
  estimate.
- After a linear recalibration, B helps: C2l 0.150 → 0.120 and C3l 0.073 → 0.050.
- `ok_e` (X = 1/σ) is right only for a constant offset: C3s 0.073 → 0.020, but C2s
  0.027 → 0.200.

## Proposed for the package (not adopted)

- Calibrator bootstrap variance.
- Exp/Sph-only REML for class maps.
- An option for correlogram-parameter uncertainty (sigma points).
- An option for logit intervals for very large blocks.
