"""Generate walkthrough.ipynb from the cells below (keeps the notebook diff-able).

    python notebooks/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append(("md", text.strip("\n")))


def code(text: str) -> None:
    cells.append(("code", text.strip("\n")))


md("""
# MIA multi-omics pipeline: a guided run on toy data

Does **maternal immune activation (MIA)** during pregnancy change the lipids,
metabolites and proteins of the offspring's brain? The study has 20 mice
(10 MIA, 10 saline control) and 3 brain regions each, so 60 samples, with
lipidomics, metabolomics and proteomics measured on every sample.

This notebook runs the whole analysis pipeline from raw files to multi-omics
integration. **The data here are simulated.** `mia_multiomics.toy_data` writes
files with exactly the same layout as the real core-facility deliverables, and
plants a known truth in them:

| Planted in the toy data | What the pipeline should find |
|---|---|
| strong brain-region differences in all omics | region dominates every analysis (positive control) |
| **no MIA effect at all** | no robust MIA hit (negative control) |
| litter (cage) and mouse random effects | mouse-level tests are too optimistic; litter-level tests are honest |
| two shared within-region programmes ("myelin", "energy") + sex-linked proteins | MOFA+ factors shared across omics |
| two outlying samples (sample numbers 10 and 45) | flagged by the outlier screens |

**Speed.** `Settings.fast` uses small permutation and bootstrap counts, so the
whole notebook runs in about 15-20 minutes. Set `RERUN = False` to reload
tables that are already in `results/` instead of recomputing them.
`Settings.full` reproduces the thesis-size run.
""")

md("## 0. Setup")
code("""
import sys, time
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

PROJECT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT / "src"))           # not needed after `pip install -e .`

from mia_multiomics import config, inputs
from mia_multiomics.config import Settings
from mia_multiomics.toy_data import write_toy_dataset
from mia_multiomics.stages import (qc, noise_floor, feature_modules, outliers, differential, sensitivity,
                                   region_persistence, asca, effect_sizes, integration, vendor_figures)

settings = Settings.fast(PROJECT)    # Settings.full(PROJECT) for the real, slow run
RERUN = True                         # False: reuse tables already written to results/

pd.set_option("display.width", 160, "display.max_columns", 30, "display.precision", 3)

def show(*parts, width=650):
    \"\"\"Display a figure written by the pipeline: show('lipidomics', '01_qc', 'pca', 'figures', 'x.png').\"\"\"
    display(Image(filename=str(settings.results_dir.joinpath(*parts)), width=width))

def step(name, function, check_file=None):
    \"\"\"Run one pipeline step (or skip it if RERUN is False and its output exists) and time it.\"\"\"
    if not RERUN and check_file is not None and (settings.results_dir / check_file).exists():
        print(f"{name}: using saved results")
        return None
    start = time.time()
    result = function()
    print(f"{name}: {time.time() - start:.0f} s")
    return result

def table(*parts):
    return pd.read_csv(settings.results_dir.joinpath(*parts))

settings
""")

md("""
## 1. The input files

`write_toy_dataset` writes the five deliverables: one Excel report (sample sheet,
lipidomics, metabolomics), the cage/collection-date workbook, one Perseus export
per region for proteomics, and GO/Reactome-style gene-set files.
""")
code("""
write_toy_dataset(settings.data_dir)
sorted(p.relative_to(PROJECT).as_posix() for p in settings.data_dir.rglob("*") if p.is_file())
""")
md("""
`inputs` parses the vendor layouts. For example, the lipidomics sheet has label
rows above its header and repeats the header before the negative-mode lipids.
Every omics comes back in the same shape: raw values, blanks, pooled-QC
(Tech-Mix) injections, a feature annotation table and the sample metadata.
""")
code("""
lipids = inputs.load_lipidomics(settings)
metabolites = inputs.load_metabolomics(settings)
proteins = inputs.load_proteomics(settings)
for data in (lipids, metabolites, proteins):
    print(f"{data.name:13s} values {data.values.shape}  blanks {None if data.blanks is None else data.blanks.shape}"
          f"  techmix {None if data.techmix is None else data.techmix.shape}")
lipids.values.iloc[:5, :6]
""")
md("""
**The study design.** Control mice come from 2 mixed-sex litters and MIA mice
from 6 single-sex litters. Each litter was collected on one date. Pups of one
litter share a mother, a cage and a date, so they are not independent: the
experimental unit for MIA is the **litter**. With 2 vs 6 litters there are only
C(8,2) = 28 ways to assign treatment to whole litters.
""")
code("""
meta = lipids.metadata
mice = meta.drop_duplicates(config.MOUSE_COL)
pd.crosstab([mice[config.CAGE_COL], mice[config.COHORT_COL]], [mice[config.GROUP_COL], mice[config.SEX_COL]])
""")

md("""
## 2. Preprocessing

One function, `preprocessing.preprocess`, applies the same chain to lipidomics
and metabolomics:

1. **prevalence:** drop features missing in more than 20 % of all 60 samples;
2. **blank filter:** drop features whose median signal is below 3 x the mean blank;
3. **detection limit:** set single values below max(3 x blank, floor) to missing;
4. choose the population's samples, then apply **log2** and **median normalisation**
   (per acquisition mode for metabolomics).

Nothing is imputed, and every removed feature is logged with its reason.
""")
code("""
from mia_multiomics.preprocessing import preprocess
prepared = preprocess(lipids)
display(prepared.summary)
prepared.exclusions.groupby(["stage", "reason"]).size().rename("n_features").reset_index()
""")

md("""
## 3. Quality control

The QC stage writes the preprocessing log, pooled-QC reproducibility (Tech-Mix
CV), technical vs biological variance, missingness by group and class, and a
descriptive PCA. Two more QC analyses look for structure among the features:
an abundance-similarity noise floor (noisyR-style), and feature-space PCA with
co-abundance modules.
""")
code("""
headlines = step("qc", lambda: [qc.run(settings, o) for o in config.VIEWS], "lipidomics/01_qc/headline.json")
step("noise floor", lambda: noise_floor.run(settings, "lipidomics"), "lipidomics/01_qc/noise_floor/candidate_thresholds.csv")
step("feature modules", lambda: [feature_modules.run(settings, o) for o in ("lipidomics", "metabolomics")],
     "metabolomics/01_qc/feature_modules/summary.csv")
pd.DataFrame([pd.read_json(settings.results_dir / o / "01_qc" / "headline.json", typ="series") for o in config.VIEWS]).set_index("omics").T
""")
md("""
**The data have strong structure, and it is region.** Pooled PC1 is almost
entirely brain region. MIA explains close to nothing on any of the first five
components.
""")
code("""
show("lipidomics", "01_qc", "pca", "figures", "scores_pooled_sample_type.png", width=520)
show("lipidomics", "01_qc", "pca", "figures", "score_attribution_pooled.png")
""")
code("""
show("lipidomics", "01_qc", "missingness", "figures", "missingness_heatmap_by_region.png", width=700)
table("metabolomics", "01_qc", "feature_modules", "summary.csv")
""")

md("""
## 4. Outliers

Two screens. Both only flag samples; nothing is deleted.

* **Regional PCA screen:** Hotelling T² and robust Mahalanobis distance in 3 PCs,
  plus DModX. The T² 95 % and 99 % flags define the only two permitted
  sensitivity populations, `t2_95` and `t2_99`. Exclusion is always per sample,
  never per mouse.
* **Calibrated ensemble:** six detector families. Every parameter is swept, and
  each setting is calibrated on simulated outlier-free data. A sample is a
  candidate only if at least two detectors flag it stably.
""")
code("""
screen = step("outlier screen", lambda: [outliers.run_screen(settings, o) for o in ("lipidomics", "metabolomics")],
              "lipidomics/02_outliers/outlier_screen.csv")
screen = table("lipidomics", "02_outliers", "outlier_screen.csv")
screen.loc[screen["hotelling_t2_95_flag"], ["Sample", "Sample number", "Region", "hotelling_t2", "hotelling_t2_95_limit", "hotelling_t2_99_flag"]]
""")
code("""
show("lipidomics", "02_outliers", "figures", "pca_hotelling_cortex.png", width=520)
""")
code("""
step("outlier ensemble", lambda: outliers.run_ensemble(settings, "lipidomics"), "lipidomics/02_outliers/ensemble/ensemble_verdict.csv")
outliers.run_concordance(settings)
verdict = table("lipidomics", "02_outliers", "ensemble", "ensemble_verdict.csv")
verdict.loc[verdict["candidate_outlier"], ["scope", "Sample number", "stable_detector_count", "calibration_sensitive", "robust_candidate"]]
""")
code("""
show("lipidomics", "02_outliers", "ensemble", "figures", "stability_pooled.png", width=700)
""")

md("""
## 5. MIA vs control, feature by feature

For every feature and region we fit `log2 value ~ group * sex` (OLS), with
Benjamini-Hochberg FDR over the whole panel. Around this model the stage adds a
label-permutation null, the **effective** number of independent tests, residual
and influence diagnostics, the **minimum detectable effect**, and PERMANOVA and
PLS-DA checks. We run it for all three populations.
""")
code("""
def run_differential():
    for omics in ("lipidomics", "metabolomics"):
        for population in config.POPULATIONS:
            differential.run(settings, omics, population)
        differential.compare_populations(settings, omics)
step("differential", run_differential, "lipidomics/03_differential/population_comparison/hits_by_population.csv")
hits = table("lipidomics", "03_differential", "population_comparison", "hits_by_population.csv")
hits.pivot_table(index=["Region", "Effect"], columns="population", values="n_fdr_hits")
""")
code("""
show("lipidomics", "03_differential", "all60", "figures", "volcano_grid.png", width=800)
""")
md("""
**Power, stated up front.** The minimum effect detectable with 80 % power is
large. Correlated features also mean far fewer *effective* tests than the panel
size.
""")
code("""
display(table("lipidomics", "03_differential", "all60", "detectable_effect_size_summary.csv").head(6))
table("lipidomics", "03_differential", "all60", "effective_number_of_tests.csv")
""")
md("""
**The key failure in the thesis: removing outliers creates hits.** For the same
feature and effect, compare the estimate and its standard error with and
without the T²-flagged samples. In the study, a ceramide's sex effect stayed at
1.17 while its standard error dropped from 0.26 to 0.18, and q fell from 0.29 to
0.01. The toy data repeat this. The flagged cortex sample is an MIA mouse, so
the control-group sex estimate cannot move. Only the pooled residual variance
shrinks, and that alone turns the ceramide into a "hit". This is standard-error
shrinkage, not biology.
""")
code("""
change = table("lipidomics", "03_differential", "population_comparison", "estimate_vs_standard_error_change.csv")
display(change.groupby(["population", "Region"])["se_ratio"].describe()[["min", "25%", "50%"]].round(2))
table("lipidomics", "03_differential", "population_comparison", "exclusion_dependent_hits.csv")[
    ["population", "Feature", "Region", "Effect", "estimate_log2_all60", "estimate_log2_excluded",
     "standard_error_all60", "standard_error_excluded", "p_value_fdr_all60", "p_value_fdr_excluded"]]
""")

md("""
## 6. Does the answer depend on analysis choices?

* **Specification curve:** prevalence filter x Tech-Mix CV filter x imputation methods.
* **RF/scaling grid:** RF imputation x Pareto scaling x population, with an IQR filter.
  (`Settings.fast` skips the slowest RF arms; `Settings.full` runs them all.)
* **Median vs PQN normalisation.**
* **Litter-aware models:** cage means, a cage random intercept, a collection-date
  effect, the date-matched subset, exact whole-cage permutations and cage ICCs.
""")
code("""
def run_sensitivity():
    sensitivity.specification_curve(settings, "lipidomics")
    sensitivity.rf_variants(settings, "lipidomics")
    for omics in ("lipidomics", "metabolomics"):
        sensitivity.normalization_comparison(settings, omics)
        sensitivity.cage_structure(settings, omics)
step("sensitivity", run_sensitivity, "lipidomics/04_sensitivity/cage_structure/cage_aware_effects.csv")
show("lipidomics", "04_sensitivity", "specification_curve", "figures", "specification_curve.png")
""")
md("""
The toy data contain **no MIA effect**, so every MIA "hit" below is a false
positive. The table counts FDR hits (q ≤ 0.05, 282 lipids x 3 regions) for the
naive model and for each litter-aware variant. The last row is the exact
whole-cage permutation, which shuffles the MIA label over the 28 possible
assignments of whole cages and so respects the design exactly.
""")
code("""
import json
cage_dir = ("lipidomics", "04_sensitivity", "cage_structure")
arms = json.loads(settings.results_dir.joinpath(*cage_dir, "cage_structure_manifest.json").read_text())["arms"]

naive = table("lipidomics", "03_differential", "all60", "differential_effects.csv")
naive = naive.loc[naive["Effect"].eq("MIA vs Saline Control")]
cage = table(*cage_dir, "cage_aware_effects.csv")
exact = pd.concat([table(*cage_dir, f"cage_permutation_{r.lower()}.csv") for r in config.REGIONS])

rows = [("naive", "group x sex model per region (samples as the unit)", naive["p_value_fdr"].le(0.05).sum(), naive["p_value_fdr"].min())]
rows += [(arm, arms[arm], g["p_value_fdr"].le(0.05).sum(), g["p_value_fdr"].min()) for arm, g in cage.groupby("Arm")]
rows += [("exact", "whole-cage permutation (28 assignments)", exact["p_value_empirical_fdr"].le(0.05).sum(), exact["p_value_empirical_fdr"].min())]
pd.DataFrame(rows, columns=["arm", "model", "n_fdr_hits", "smallest_q"]).round(3)
""")
md("""
Model-based litter adjustments are **not enough with 8 litters**. A cage random
intercept (B) or a collection-date term (C) estimates its variance from very
few groups, so their tests are too liberal. Even a plain t-test on cage means
(A) is fragile: the cages hold 1 to 6 mice, so their means are not equally
precise. In each of these arms a few lipids pass FDR although nothing was
planted. Only the exact whole-cage permutation is
calibrated by construction, and it finds nothing. Its smallest possible p-value
is 1/28 ≈ 0.036, so with 282 lipids it *cannot* reach FDR significance at all.
That limit is a property of the design, not of the method.
""")

md("""
## 7. A better question: do regional differences persist after MIA?

Region is strong, and every mouse contributes all three regions, so each mouse
is its own control. For each pair of regions we compare the control and MIA
**difference profiles**. The noise-corrected (disattenuated) correlation is 1 if
the regional profile is fully preserved.
""")
code("""
def run_persistence():
    for omics in ("lipidomics", "metabolomics"):
        for population in config.POPULATIONS:
            region_persistence.run(settings, omics, population)
        region_persistence.plot_summary(settings, omics)
step("region persistence", run_persistence, "lipidomics/05_region_persistence/disattenuated_concordance_all.csv")
show("lipidomics", "05_region_persistence", "figures", "disattenuated_concordance.png")
""")

md("""
## 8. ASCA+: one multivariate ANOVA instead of hundreds of tests

The data matrix is split into Region, MIA, Sex, their interactions, Mouse and
residual. Each effect gets a variance share and a permutation p-value, with
labels shuffled at the right unit (region within mouse, whole litters for MIA).
""")
code("""
# the notebook fits the primary population; pipeline.run_all also fits t2_95 and t2_99
step("ASCA+", lambda: [asca.run(settings, omics, "all60") for omics in ("lipidomics", "metabolomics")],
     "lipidomics/06_asca/all60/permutation_tests.csv")
tests = table("lipidomics", "06_asca", "all60", "permutation_tests.csv")
display(tests.loc[tests["Scaling"].eq("autoscaled"), ["Engine", "Effect", "variance_share", "p_value", "n_permutations"]])
show("lipidomics", "06_asca", "all60", "figures", "asca_composite.png", width=800)
""")
md("""
Region explains about half of the variance, and its p-value is at the floor of
the permutation test. For MIA, compare the two *Group* rows. They test the same
variance share with different shuffle units:

* **Group (mouse):** shuffles MIA labels between individual mice. Mice from one
  litter share a mother and a cage, so they are not independent. The small
  litter differences planted in the toy data are then mistaken for a treatment
  effect, which gives a small p-value although MIA does nothing.
* **Group (cage):** shuffles whole litters, the true experimental unit. The
  same variance share is now unremarkable.

Section 10 measures this inflation directly with simulated noise.
""")

md("""
## 9. Beyond p-values: effect sizes

When a litter-level p-value can never go below 1/28, it helps more to ask how
large an effect could be. The stage covers pre-defined lipid composites (myelin
index, DHA share, Cer:SM, ...), protein sets, leave-one-litter-out ranges, and
which fold changes the intervals already rule out.
""")
code("""
step("effect sizes", lambda: effect_sizes.run(settings), "integration/07_effect_sizes/all60/lipid_composites.csv")
display(table("integration", "07_effect_sizes", "all60", "effects_ruled_out.csv"))
show("integration", "07_effect_sizes", "all60", "figures", "composite_effects_forest.png", width=800)
""")

md("""
## 10. Multi-omics integration

The three omics are joined on the same 60 samples. Region is the known positive
control and MIA the characterised negative, so the question for every method
is: **does it find region, and does it correctly find nothing for MIA?**
""")
code("""
blocks, block_meta, _ = integration.build_blocks(settings)
{view: block.shape for view, block in blocks.items()}
""")
code("""
step("MOFA+ (all samples)", lambda: integration.run_mofa(settings), "integration/01_mofa/variance_explained.csv")
show("integration", "01_mofa", "figures", "variance_explained.png")
display(table("integration", "01_mofa", "factor_associations.csv").pivot(index="Factor", columns="Covariate", values="r_squared").round(2))
table("integration", "01_mofa", "mia_tests.csv")[["Score name", "mouse_permutation_p", "cage_permutation_p", "group_r_squared", "cage_r_squared", "treatment_associated"]]
""")
md("""
A factor can separate MIA from control at the mouse level and still fail at
the litter level. If the litter explains more of a score than the treatment
does (cage R² > group R²), the "MIA effect" cannot be told apart from litter or
collection date.
""")
md("""
**Within region.** A multi-group MOFA+ with region means removed asks what mice
share *inside* a region. The toy data hide three such programmes: sex-linked
proteins, a "myelin" module and an "energy" module.
""")
code("""
step("MOFA+ within region", lambda: integration.run_mofa_within_region(settings), "integration/02_mofa_within_region/factor_stability.csv")
display(table("integration", "02_mofa_within_region", "factor_stability.csv"))
show("integration", "02_mofa_within_region", "figures", "variance_stable_factors.png")
""")
code("""
step("benchmark", lambda: integration.run_benchmark(settings), "integration/03_benchmark/region_recovery.csv")
step("cross-omics screen", lambda: integration.run_cross_omics_screen(settings), "integration/04_cross_omics_screen/modules.csv")
step("enrichment", lambda: integration.run_enrichment(settings), "integration/05_enrichment/protein_enrichment.csv")
step("covariate map", lambda: integration.run_covariate_map(settings), "integration/06_covariate_map/axis_covariate_r2.csv")
display(table("integration", "03_benchmark", "region_recovery.csv"))
display(table("integration", "04_cross_omics_screen", "association_counts_vs_null.csv"))
enrich = table("integration", "05_enrichment", "protein_enrichment.csv")
enrich.sort_values("q_value").groupby("unit").head(1)[["unit", "term", "n_overlap", "q_value", "overlap_genes"]]
""")
code("""
show("integration", "06_covariate_map", "figures", "covariate_map.png")
""")
md("""
**Why tests are done at the litter level.** Simulated pure-noise scores, with
and without a shared litter offset, show how often each test falsely rejects.
""")
code("""
step("inference checks", lambda: integration.run_inference_checks(settings), "integration/09_inference_checks/false_positive_rates.csv")
table("integration", "09_inference_checks", "false_positive_rates.csv")
""")

md("""
## 11. Reproducing the facility's report figures

The core facility's report showed PLS-DA score plots in which the groups looked
well separated. A supervised method separates groups even when they do not
differ, which is what the permutation test checks.
""")
code("""
step("vendor figures", lambda: vendor_figures.run(settings), "lipidomics/09_vendor_figures/plsda_permutation_tests.csv")
show("lipidomics", "09_vendor_figures", "plsda_cortex.png", width=480)
""")

md("""
## 12. The whole pipeline in one call

Everything above is what `pipeline.run_all` does, step by step. From a terminal:

```bash
python -m mia_multiomics.pipeline --fast            # this notebook's settings
python -m mia_multiomics.pipeline                   # full-size run
python -m mia_multiomics.pipeline --fast --only qc outliers differential
```
""")
code("""
from mia_multiomics.pipeline import STEPS
pd.DataFrame([(name, description) for name, description, _ in STEPS], columns=["step", "what it does"])
""")

md("""
## Summary

* Region is a large, reproducible signal in every omics and every method: the positive control works.
* No method finds a robust MIA effect. Apparent hits come from sample exclusion
  (standard-error shrinkage), from treating mice as independent, or from model-based
  litter adjustments with too few litters. The exact whole-cage permutation removes all of them.
* The shared within-region programmes planted in the toy data are found by MOFA+
  and the cross-omics screen, and enrichment of their proteins points to the right gene sets.
* With 20 mice in 8 litters, the most useful honest outputs are effect sizes, detectable-effect
  bounds and litter-level permutation tests.
""")


notebook = nbf.v4.new_notebook()
notebook["cells"] = [nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text) for kind, text in cells]
notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
target = Path(__file__).with_name("walkthrough.ipynb")
nbf.write(notebook, target)
print(f"wrote {target}")
