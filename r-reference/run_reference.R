#!/usr/bin/env Rscript
## Reference harness for the Python parity tests of pyspatialuncertainty.
##
## Runs the authors' script (Uncertainty_spatial_averages_and_totals.R, vendored
## unchanged from AlexandreWadoux/UncertaintySpatialAggregation@934481f) with only
## its two RStudio-specific lines disabled, then exports the intermediate objects
## that the Python implementation must reproduce:
##
##   grid.csv                 the 100 x 100 reference map, QRF predictions and variances
##   pt_all.csv               the 200 sample points with CV predictions and standardized errors
##   sample_variogram*.csv    gstat sample variograms (default; Cressie-Hawkins; custom cutoff/width)
##   variogram_model.csv      the (hand-set) nested variogram model of the script
##   variogram_line.csv       gamma(h), covariance and correlation of that model at test distances
##   gstat_fits.csv           gstat fits of a few models with fit.method 1, 2, 6 and 7
##   mc_pairs.csv             the exact 1000 point pairs drawn by the script's MC integration,
##                            with the per-pair integrand values computed by the script's ff()
##   mc_replicates.csv        200 independent MC estimates with m = 1000 (vectorised integrand)
##   mc_highm.csv             one MC estimate with m = 1e6 (vectorised integrand)
##   summary.csv              scalar results of the script (block mean, SD, interval, total)
##   manifest.csv             R / package versions and settings
##
## Usage (from the repository root):  Rscript r-reference/run_reference.R [output_dir]

args <- commandArgs(trailingOnly = TRUE)
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg)))
out_dir <- if (length(args) >= 1) args[1] else file.path(script_dir, "..", "tests", "data", "r_reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir)

## Plots of the original script go to a null device.
pdf(NULL)

write_full <- function(df, name) {
  df <- as.data.frame(df)
  for (j in seq_along(df)) {
    if (is.double(df[[j]])) df[[j]] <- sprintf("%.17g", df[[j]])
  }
  write.csv(df, file.path(out_dir, name), row.names = FALSE, quote = FALSE)
}

## ---------------------------------------------------------------------------
## 1. Run the authors' script
## ---------------------------------------------------------------------------
src <- readLines(file.path(script_dir, "Uncertainty_spatial_averages_and_totals.R"))
rstudio_lines <- grepl("rstudioapi::getActiveDocumentContext", src, fixed = TRUE) |
  grepl("^setwd\\(root_dir\\)", src)
stopifnot(sum(rstudio_lines) == 2)
src[rstudio_lines] <- paste("#", src[rstudio_lines])

mc_line <- grep("^var_average <- mcint2.mod\\(", src)
stopifnot(length(mc_line) == 1)

t0 <- Sys.time()
eval(parse(text = src[seq_len(mc_line - 1)]), envir = globalenv())
## RNG state immediately before the script's Monte Carlo integration.
seed_before_mc <- get(".Random.seed", envir = globalenv())
eval(parse(text = src[mc_line:length(src)]), envir = globalenv())
cat("Script finished in", format(Sys.time() - t0), "\n")

## Free the dense 10^4 x 10^4 matrices before continuing.
rm(list = intersect(c("distx", "disty", "Dist", "C", "Upper"), ls(globalenv())), envir = globalenv())
invisible(gc())

## ---------------------------------------------------------------------------
## 2. Re-draw the script's MC pairs from the saved RNG state
## ---------------------------------------------------------------------------
assign(".Random.seed", seed_before_mc, envir = globalenv())
win <- owin(xrange = c(0, 100), yrange = c(0, 100))
m_script <- 1000
pairs1 <- as.data.frame(runifpoint(m_script, win = win))
pairs2 <- as.data.frame(runifpoint(m_script, win = win))
z_script <- ff(pt1x = pairs1[, 1], pt1y = pairs1[, 2], pt2x = pairs2[, 1], pt2y = pairs2[, 2])
## Must reproduce the script's own result bit for bit.
stopifnot(identical(mean(z_script), var_average))

## Vectorised version of the script's integrand f(), used for the replicates.
sill_total <- sum(lzn.fit.RF.nested$psill)
grid_raster <- rasterFromXYZ(grid)
f_vec <- function(x1, y1, x2, y2) {
  d <- sqrt((x1 - x2)^2 + (y1 - y2)^2)
  rho <- (sill_total - variogramLine(lzn.fit.RF.nested, dist_vector = d, covariance = FALSE)$gamma) / sill_total
  sd1 <- sqrt(raster::extract(grid_raster, cbind(x1, y1))[, "predVarRF"])
  sd2 <- sqrt(raster::extract(grid_raster, cbind(x2, y2))[, "predVarRF"])
  data.frame(dist = d, rho = rho, sd1 = sd1, sd2 = sd2, integrand = sd1 * sd2 * rho)
}
vec_script <- f_vec(pairs1[, 1], pairs1[, 2], pairs2[, 1], pairs2[, 2])
stopifnot(max(abs(vec_script$integrand - z_script)) < 1e-12)

write_full(data.frame(x1 = pairs1[, 1], y1 = pairs1[, 2], x2 = pairs2[, 1], y2 = pairs2[, 2],
                      dist = vec_script$dist, rho = vec_script$rho, sd1 = vec_script$sd1,
                      sd2 = vec_script$sd2, integrand = z_script), "mc_pairs.csv")

## ---------------------------------------------------------------------------
## 3. Replicate MC estimates (m = 1000) and one high-precision estimate (m = 1e6)
## ---------------------------------------------------------------------------
set.seed(20231320)
n_rep <- 200
rep_est <- numeric(n_rep)
rep_sd <- numeric(n_rep)
for (r in seq_len(n_rep)) {
  p1 <- as.data.frame(runifpoint(m_script, win = win))
  p2 <- as.data.frame(runifpoint(m_script, win = win))
  g <- f_vec(p1[, 1], p1[, 2], p2[, 1], p2[, 2])$integrand
  rep_est[r] <- mean(g)
  rep_sd[r] <- sd(g)
}
write_full(data.frame(replicate = seq_len(n_rep), m = m_script, var_average = rep_est,
                      integrand_sd = rep_sd), "mc_replicates.csv")

set.seed(14106)
m_high <- 1e6
chunk <- 1e5
s1 <- 0
s2 <- 0
for (k in seq_len(m_high / chunk)) {
  p1 <- as.data.frame(runifpoint(chunk, win = win))
  p2 <- as.data.frame(runifpoint(chunk, win = win))
  g <- f_vec(p1[, 1], p1[, 2], p2[, 1], p2[, 2])$integrand
  s1 <- s1 + sum(g)
  s2 <- s2 + sum(g^2)
}
high_mean <- s1 / m_high
high_sd <- sqrt((s2 - m_high * high_mean^2) / (m_high - 1))
write_full(data.frame(m = m_high, var_average = high_mean, integrand_sd = high_sd,
                      mcse = high_sd / sqrt(m_high)), "mc_highm.csv")

## ---------------------------------------------------------------------------
## 4. Maps, samples, variograms
## ---------------------------------------------------------------------------
write_full(grid[, c("x1", "x2", "y", "predRF", "predVarRF")], "grid.csv")
write_full(data.frame(grid_row = as.integer(rownames(pt_all)), x1 = pt_all$x1, x2 = pt_all$x2,
                      y = pt_all$y, predRF = pt_all$predRF, predVarRF = pt_all$predVarRF,
                      interval90 = as.integer(pt_all$interval90), fold = as.integer(pt_all$sp),
                      stdErrRF = pt_all$stdErrRF), "pt_all.csv")

write_full(as.data.frame(lzn.vgm.RF)[, c("np", "dist", "gamma")], "sample_variogram.csv")
write_full(data.frame(boundary = attr(lzn.vgm.RF, "boundaries")), "sample_variogram_boundaries.csv")
vgm_cressie <- variogram(stdErrRF ~ 1, pt_all, locations = ~ x1 + x2, cressie = TRUE)
write_full(as.data.frame(vgm_cressie)[, c("np", "dist", "gamma")], "sample_variogram_cressie.csv")
vgm_custom <- variogram(stdErrRF ~ 1, pt_all, locations = ~ x1 + x2, cutoff = 30, width = 2)
write_full(as.data.frame(vgm_custom)[, c("np", "dist", "gamma")], "sample_variogram_cutoff30_width2.csv")

model_df <- as.data.frame(lzn.fit.RF.nested)
write_full(data.frame(model = as.character(model_df$model), psill = model_df$psill,
                      range = model_df$range, kappa = model_df$kappa), "variogram_model.csv")

test_dist <- c(0, 1e-8, 1e-3, 0.5, 1, 1.5, 2, 3, 5, 10, 14.9, 15, 15.1, 30, 59.9, 60, 61, 100, 141.4)
vl <- variogramLine(lzn.fit.RF.nested, dist_vector = test_dist, covariance = FALSE)
vc <- variogramLine(lzn.fit.RF.nested, dist_vector = test_dist, covariance = TRUE)
write_full(data.frame(dist = test_dist, gamma = vl$gamma, covariance = vc$gamma,
                      correlation = (sill_total - vl$gamma) / sill_total), "variogram_line.csv")
write_full(data.frame(dist = corRF$dist, correlation = corRF$gamma), "correlogram_script.csv")

## Single-structure variogramLine values for each model type (parameterisation check).
single <- list(
  Exp = vgm(1, "Exp", 10), Sph = vgm(1, "Sph", 10), Gau = vgm(1, "Gau", 10),
  Mat = vgm(1, "Mat", 10, kappa = 1.5), Ste = vgm(1, "Ste", 10, kappa = 1.5),
  Ste05 = vgm(1, "Ste", 10, kappa = 0.5), Nug = vgm(1, "Nug", 0)
)
single_out <- do.call(rbind, lapply(names(single), function(nm) {
  v <- variogramLine(single[[nm]], dist_vector = test_dist, covariance = FALSE)
  data.frame(case = nm, dist = test_dist, gamma = v$gamma)
}))
write_full(single_out, "variogram_line_single_models.csv")

## gstat fits for comparison with the Python WLS implementation.
fit_cases <- list(
  exp_m7 = list(model = vgm(0.5, "Exp", 10, 0.1), method = 7),
  exp_m6 = list(model = vgm(0.5, "Exp", 10, 0.1), method = 6),
  exp_m2 = list(model = vgm(0.5, "Exp", 10, 0.1), method = 2),
  exp_m1 = list(model = vgm(0.5, "Exp", 10, 0.1), method = 1),
  sph_m7 = list(model = vgm(0.5, "Sph", 20, 0.1), method = 7),
  ste_m7 = list(model = vgm(0.7, "Ste", 25, 0.1, kappa = 0.5), method = 7)
)
fit_out <- do.call(rbind, lapply(names(fit_cases), function(nm) {
  fc <- fit_cases[[nm]]
  fit <- fit.variogram(lzn.vgm.RF, fc$model, fit.method = fc$method, fit.kappa = FALSE)
  fd <- as.data.frame(fit)
  data.frame(case = nm, fit_method = fc$method, component = seq_len(nrow(fd)),
             model = as.character(fd$model), psill = fd$psill, range = fd$range,
             kappa = fd$kappa, sserr = attr(fit, "SSErr"),
             singular = as.integer(isTRUE(attr(fit, "singular"))))
}))
write_full(fit_out, "gstat_fits.csv")

## ---------------------------------------------------------------------------
## 5. Scalars and manifest
## ---------------------------------------------------------------------------
cv_res <- pt_all$predRF - pt_all$y
summary_df <- data.frame(
  key = c("mean_y", "var_y", "mean_predRF", "var_average", "sd_average", "pi90_lower_z164",
          "pi90_upper_z164", "sum_predRF", "B", "var_total", "sd_total", "cv_me", "cv_rmse",
          "cv_r2", "cv_picp90", "sill_total", "m_script", "mcse_script"),
  value = c(mean(grid$y), var(grid$y), mean(grid$predRF), var_average, sqrt(var_average),
            mean(grid$predRF) - 1.64 * sqrt(var_average), mean(grid$predRF) + 1.64 * sqrt(var_average),
            sum(grid$predRF), B, var_total, sqrt(var_total), mean(cv_res), sqrt(mean(cv_res^2)),
            1 - sum((pt_all$y - pt_all$predRF)^2) / sum((pt_all$y - mean(pt_all$y))^2),
            mean(pt_all$interval90), sill_total, m_script, sd(z_script) / sqrt(m_script))
)
write_full(summary_df, "summary.csv")

pkgs <- c("gstat", "ranger", "raster", "sp", "spatstat", "spatstat.geom", "spatstat.random",
          "matrixStats", "terra")
manifest <- data.frame(
  key = c("R.version", "upstream_repo", "upstream_commit", "script_sha256_note", "BLAS",
          "run_date", paste0("pkg.", pkgs)),
  value = c(R.version.string, "AlexandreWadoux/UncertaintySpatialAggregation",
            "934481f28e76f4b650130a8bf1b0c4b3e83ee954", "see r-reference/README.md",
            La_library(), format(Sys.time(), "%Y-%m-%d"),
            vapply(pkgs, function(p) as.character(packageVersion(p)), ""))
)
write.csv(manifest, file.path(out_dir, "manifest.csv"), row.names = FALSE)
cat("Reference outputs written to", out_dir, "\n")
cat("var_average =", var_average, " sqrt =", sqrt(var_average), "\n")
cat("high-m var_average =", high_mean, " (MCSE", high_sd / sqrt(m_high), ")\n")
