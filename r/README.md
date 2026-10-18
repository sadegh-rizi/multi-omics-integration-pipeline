# Optional R steps

The pipeline runs entirely in Python. These R scripts are extra checks from the
original analysis. Run them after the Python stages have written their inputs,
from the repository root. The `r/renv.lock` file pins the R package versions
(`renv::restore(lockfile = "r/renv.lock")`).

| Script | What it does | Inputs, from the Python pipeline | Packages |
|---|---|---|---|
| `differential_limma.R` | The same group x sex model per region, with limma's moderated variances | Called by `stages.differential.run_limma` when `Settings.use_r=True` | limma |
| `noisyr_crosscheck.R` | The original noisyR abundance-similarity method, as a check on the Python version | Called by `stages.noise_floor` when `Settings.use_r=True` | noisyR |
| `asca_crosscheck.R` | An independent base-R ASCA fit, as an estimation check on the Python ASCA+ | `results/<omics>/06_asca/<population>/inputs/` | base R (ALASCA if installed) |
| `diablo.R` | mixOmics DIABLO with mouse-grouped cross-validation, plus a control-only reference projection | `results/integration/00_inputs/` | mixOmics |
| `mfa.R` | FactoMineR multiple factor analysis | `results/integration/00_inputs/` | FactoMineR |
| `benchmark_blocks.R` | The R-side MFA / block.spls benchmark arms | `results/integration/00_inputs/` | FactoMineR, mixOmics |
| `test_differential_limma.R` | testthat test for `differential_limma.R` | - | testthat, limma |

```bash
Rscript r/diablo.R results/integration results/integration/r_diablo
Rscript r/mfa.R results/integration results/integration/r_mfa
Rscript r/benchmark_blocks.R results/integration results/integration/r_benchmark
Rscript r/asca_crosscheck.R results/lipidomics/06_asca/all60/inputs/complete_feature_matrix.csv \
        results/lipidomics/06_asca/all60/inputs/metadata.csv results/lipidomics/06_asca/all60/r_crosscheck.csv
Rscript r/test_differential_limma.R
```
