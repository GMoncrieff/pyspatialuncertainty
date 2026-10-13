# Experiments

These are the simulation studies behind the report in `report/report.qmd`. All settings are in
`config.toml`. Every run is seeded, so re-running a script reproduces the committed
results in `results/`.

| Script | What it does | Run time (4 cores) |
|---|---|---|
| `run_continuous.py` | Verification (E1) and ablations (E2) for continuous maps, using known correlated errors (`ErrorModelScenario`) | ~15 min |
| `run_categorical.py` | Verification and ablations for class maps, using thresholded latent fields (`LatentClassScenario`) | ~45 min |
| `run_pipeline.py` | The authors' R example as a full pipeline: GRF, sampling, ranger-style QRF, 10-fold CV (`PipelineScenario`) | ~15 min |
| `run_mc_convergence.py` | Monte Carlo convergence on the R script's own σ map and variogram | <1 min |
| `summarize.py` | Coverage with cluster-bootstrap CIs, plus SD-ratio summaries, written to `results/summary.parquet` | <1 min |

`improvements/` holds separate experiments that test changes to the method against the
current procedure. See `improvements/README.md`.

Run them from the repository root, for example:

```bash
uv sync --group experiments
uv run python -m experiments.run_continuous      # add --quick for a smoke test
uv run python -m experiments.summarize
```

## Estimators

Each simulation study compares these estimators of the block-error SD:

| Estimator | How the block-error SD is estimated |
|---|---|
| `method` | The paper's procedure: σ map, variogram of the standardized validation errors (gstat defaults, automatic Nug + Exp/Sph/Gau fit), then MC integration with m = 10,000 |
| `oracle` | True σ and true correlation; for class maps, the exact domain-averaged correlogram of the standardized indicator errors |
| `naive_avg` | Average of σ over the block (assumes perfect correlation) |
| `naive_indep` | √(Σσ²)/n (assumes independence) |

The pipeline study has no oracle, because the true conditional error SD of a QRF is not known.

## Truth

The true block value is the mean over a sub-grid with 4 × 4 nodes per map cell. This
approximates the continuous block mean that the method targets. Point-scale nugget
variation enters only the point observations. The pipeline study follows the R script and
uses the mean over grid nodes.
