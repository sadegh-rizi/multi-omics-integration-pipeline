"""Generate supplementary_figures.ipynb (keeps the notebook diff-able).

    python notebooks/build_supplementary.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/supplementary_figures.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells: list[tuple[str, str, list[str]]] = []


def md(text: str) -> None:
    cells.append(("md", text.strip("\n"), []))


def code(text: str, tags: list[str] | None = None) -> None:
    cells.append(("code", text.strip("\n"), tags or []))


md("""
# Supplementary figures: explore the data with your own settings

The walkthrough notebook (`walkthrough.ipynb`) tells the story of the pipeline
with a handful of figures. The original thesis repository drew many more:
QC and missingness plots, every PCA loading view, influence plots, the
outlier-ensemble diagnostics, p-value calibration, feature modules, litter
plots and the facility-style figures. This notebook draws them all, and
**recomputes them from the settings cell below**, so you can:

* **remove samples** you consider outliers (or none at all);
* **change the preprocessing thresholds**: prevalence filter, blank filter,
  detection floor, median or PQN normalisation;
* **change analysis choices**: scaling, FDR level, number of PCs for the
  outlier screen, permutation and bootstrap counts;
* switch between **lipidomics, metabolomics and proteomics**.

**How to use it.** Edit the settings cell, then *Kernel → Restart & Run All*.
The defaults reproduce the pipeline's primary analysis: all 60 samples and the
current lipid filtering (features missing in more than 20 % of samples
removed, median signal at least 3 x the blank, no imputation). A full run takes
about one to two minutes, plus two minutes the first time the outlier ensemble
is computed. Every figure states the data it was drawn from in grey
underneath it, so a saved figure always carries its settings.

The figure code lives in `src/mia_multiomics/explore/`. Each figure is one
function that takes a *view* (the preprocessed data plus your settings) and
returns a Matplotlib figure.

| Section | Figures |
|---|---|
| 2 Preprocessing and pooled QC | feature funnel, prevalence curve, signal-to-blank, Tech-Mix CV and drift |
| 3 Missingness | per feature, per sample, by region and group, heatmaps before and after filtering |
| 4 Noise floor | abundance-similarity curves |
| 5 Sample-space PCA | scree (grouped, attributed), scores coloured by every design factor, pooled composite |
| 6 PCA loadings | lollipop, scatter, heatmap, class composition, concentration, top features |
| 7 Outlier screen (your settings) | Hotelling T², DModX, influence plot, T² ellipses, assay concordance |
| 8 Outlier ensemble (all 60) | stability per scope, corroboration, null percentiles |
| 9 Per-feature models | hit counts, volcanoes, p-value calibration, top hits, forest, single feature, lipid classes |
| 10 Feature space | correlation modules, feature view, loading stability, module stability |
| 11 Litters | PCA by cage, cage panels, within-litter spread |
| 12 Regional profiles | region differences in control vs MIA |
| 13 ASCA+ | variance partition and effect scores |
| 14 Facility-style figures | heatmap and PLS-DA with permutation null, every region |
| 15 Lipid vs protein | myelin index against myelin proteins, every region |
""")

md("""
## 1. Settings

Change anything here, then restart and run all cells.

*A note on removing samples.* The thesis analyses allow only three
populations: all 60 samples, or the samples flagged by the regional Hotelling T²
screen at 95 % or 99 % removed, and never a whole mouse. This notebook lets you
remove any sample so that you can see what each choice does. The table in the
next section lists the T²-flagged samples, and a warning is printed if you
remove every region of a mouse.
""")
code("""
# ---- data -------------------------------------------------------------------------
OMICS = "lipidomics"            # "lipidomics", "metabolomics" or "proteomics"
EXCLUDE = []                    # samples to remove: sample numbers, e.g. [10, 45], or IDs, e.g. ["TOY_001-10"]

# ---- preprocessing thresholds (lipidomics and metabolomics) ------------------------
MAX_MISSING_FRACTION = 0.20     # drop features missing in more than this share of all 60 samples (None = off)
BLANK_MULTIPLIER = 3.0          # blank filter and detection limit: multiple of the mean blank signal
DETECTION_FLOOR = None          # None = default (1.0 for lipids, smallest positive value for metabolites)
NORMALIZATION = "median"        # "median" or "pqn"

# ---- analysis choices ---------------------------------------------------------------
SCALING = "autoscaled"          # "autoscaled", "pareto" or "unscaled": PCA, outlier screen, heatmaps, ASCA
FDR_ALPHA = 0.05                # Benjamini-Hochberg level for "hits"
MIN_PER_CELL = 3                # a feature is modelled only with >= this many values in every group x sex cell
OUTLIER_PCS = 3                 # principal components used for Hotelling T2 and DModX
TOP_N = 20                      # features shown in loading and forest plots
N_PERMUTATIONS = 199            # PLS-DA permutation test
N_BOOTSTRAPS = 100              # loading and module stability

# ---- what to show -------------------------------------------------------------------
REGION = "Cortex"               # region of the single-region figures: "Cerebellum", "Cortex" or "Hippocampus"
EFFECT = "MIA vs Saline Control"  # or "Female vs Male", "MIA x Female interaction"
FEATURE = None                  # one feature to profile, e.g. "Cer(d18:0_14:0)"; None = the top hit of EFFECT in REGION
SAVE_FIGURES = False            # True: also write every figure as PNG to results/supplementary/<run name>/
""", tags=["parameters"])

code("""
import io, re, sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

PROJECT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT / "src"))           # not needed after `pip install -e .`

from mia_multiomics import config, explore, plotting
from mia_multiomics.config import Settings
from mia_multiomics.toy_data import write_toy_dataset

settings = Settings.fast(PROJECT)
if not (settings.data_dir / config.REPORT_WORKBOOK).exists():
    write_toy_dataset(settings.data_dir)            # the simulated data, if not written yet
plotting.apply_plot_style()
pd.set_option("display.width", 160, "display.max_columns", 30, "display.precision", 3)

options = explore.Options(
    omics=OMICS, exclude=EXCLUDE, max_missing_fraction=MAX_MISSING_FRACTION, blank_multiplier=BLANK_MULTIPLIER,
    detection_floor=DETECTION_FLOOR, normalization=NORMALIZATION, scaling=SCALING, fdr_alpha=FDR_ALPHA,
    min_per_cell=MIN_PER_CELL, outlier_pcs=OUTLIER_PCS, top_n=TOP_N, n_permutations=N_PERMUTATIONS,
    n_bootstraps=N_BOOTSTRAPS)
view = explore.build_view(settings, options)

FIGURES = {}                                        # every figure drawn, as PNG bytes, for optional saving

def draw(function, *args, **kwargs):
    \"\"\"Draw one figure inline. A figure that does not apply to these settings prints why instead.\"\"\"
    try:
        fig = function(*args, **kwargs)
    except ValueError as error:
        print(f"skipped {function.__name__}: {error}")
        return
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    labels = [str(a) for a in [*args, *kwargs.values()] if a is not None and not isinstance(a, (explore.View, explore.EnsembleResults))]
    name = re.sub(r"[^A-Za-z0-9_.]+", "-", "_".join([function.__name__, *labels])).strip("-")
    FIGURES[name] = buffer.getvalue()
    display(Image(buffer.getvalue()))

print(view.tag())
for mouse in view.removed_whole_mice():
    print(f"WARNING: every region of mouse {mouse} is excluded - the thesis analyses never remove a whole mouse.")
display(view.summary())
display(view.cell_counts())
""")

md("""
Which samples does the regional Hotelling T² screen flag? The screen uses all 60
samples with the default preprocessing, because outliers have to be found
before anything is removed. At 95 %, these are the samples of the `t2_95`
population; those that also exceed the 99 % limit form `t2_99`.
""")
code("""
explore.suggest_exclusions(settings, OMICS, level=0.95)
""")

md("""
## 2. Preprocessing and pooled QC

How many features each filter removes, and how the thresholds were chosen.
Prevalence and the blank filter are judged on all 60 raw samples, so every
population sees the same feature panel. The Tech-Mix is one pooled sample
injected six times; its CV measures technical reproducibility (reported, never
used as a filter).
""")
code("""
draw(explore.feature_funnel, view)
draw(explore.prevalence_curve, view)
draw(explore.signal_to_blank, view)
draw(explore.techmix_cv, view)
draw(explore.techmix_drift, view)
""")

md("""
## 3. Missingness

Missing values are never imputed in the main analysis. Do they cluster in
particular samples, regions, groups or lipid classes? The two heatmaps show the
delivered panel and the data after preprocessing (the detection limit adds
missing values; the prevalence filter removes the emptiest features).
""")
code("""
draw(explore.missingness_per_feature, view)
draw(explore.missingness_per_sample, view)
draw(explore.missingness_by_region_and_group, view)
draw(explore.missingness_heatmap, view, stage="raw")
draw(explore.missingness_heatmap, view, stage="filtered")
""")

md("""
## 4. Noise floor

noisyR-style abundance-similarity curve. Features are sorted by abundance and
cut into sliding windows, and samples are correlated within each window. At
low abundance (noise) samples agree poorly; the dashed Tech-Mix curve is the
purely technical reference. Candidate thresholds are reported, never applied.
""")
code("""
draw(explore.abundance_similarity, view)
""")

md("""
## 5. Sample-space PCA

PCA of complete features, scaled as chosen. *Pooled* uses all kept samples,
where region dominates; the regional PCAs show what remains within a region.
The attribution plot splits each component's variance by the design factor
its scores track (marginal R²; factors overlap).
""")
code("""
draw(explore.scree_grouped, view)
draw(explore.scree_attribution, view, "pooled")
draw(explore.scree_attribution, view, REGION)
""")
code("""
for column in ["Sample type", "Bio group", "Sex", "Cage", "Date collected", "Tissue mass (binned)", "Sample number"]:
    draw(explore.pca_scores, view, "pooled", column)
""")
code("""
for column in ["Bio group", "Sex", "Group x sex", "Cage"]:
    draw(explore.pca_scores, view, REGION, column, label="Sample number" if column == "Bio group" else None)
draw(explore.pca_scores, view, REGION, "Bio group", x="PC 3", y="PC 4")
draw(explore.region_group_composite, view)
""")

md("""
## 6. PCA loadings

Which features drive each component, and are the components carried by a few
features or spread over many? Change `LOADING_SCOPE` to `"pooled"` or a region.
""")
code("""
LOADING_SCOPE = "pooled"
draw(explore.loadings_lollipop, view, LOADING_SCOPE, "PC 1")
draw(explore.loadings_lollipop, view, LOADING_SCOPE, "PC 2")
draw(explore.loadings_scatter, view, LOADING_SCOPE)
draw(explore.loadings_heatmap, view, LOADING_SCOPE)
draw(explore.loadings_class_composition, view, LOADING_SCOPE)
draw(explore.loadings_cumulative_share, view, LOADING_SCOPE)
draw(explore.loadings_top_features, view, LOADING_SCOPE)
""")

md("""
## 7. Outlier screen with your settings

The regional T²/DModX screen recomputed on *your* samples and thresholds.
Excluding one sample can make another stand out, because the limits are
re-estimated. T² says a sample is extreme *within* the PCA model; DModX says the
model describes it badly. The concordance matrix asks whether lipidomics and
metabolomics flag the same samples. They share one tissue extract, so
agreement is not independent evidence.
""")
code("""
screen = explore.outlier_table(view)
display(screen.loc[screen["Region"].eq(REGION), ["Sample number", config.MOUSE_COL, "Bio group", "Sex", "hotelling_t2",
        "hotelling_t2_95_limit", "hotelling_t2_99_limit", "normalized_dmodx", "dmodx_95_limit"]].sort_values("hotelling_t2", ascending=False).head(8))
draw(explore.t2_plot, view, REGION)
draw(explore.dmodx_plot, view, REGION)
draw(explore.pca_hotelling, view, REGION)
""")
code("""
for region in config.REGIONS:
    draw(explore.influence_plot, view, region)
draw(explore.outlier_concordance, view, "hotelling_t2")
draw(explore.outlier_concordance, view, "normalized_dmodx")
""")

md("""
## 8. Calibrated outlier ensemble (all 60 samples)

Six detector families, every parameter swept, each setting calibrated against
simulated outlier-free data. A detector *stably* flags a sample when at least
60 % of its settings flag it. A sample is a candidate when at least two
detectors, one of them not covariance-based, stably flag it. Like the T² table
above, this always uses all 60 samples. The tables come from `results/` if the
pipeline or walkthrough has run; otherwise they are computed here (about two
minutes).
""")
code("""
ensemble = explore.ensemble_results(settings, OMICS)
for scope in ["pooled", *[r.lower() for r in config.REGIONS]]:
    draw(explore.ensemble_stability, ensemble, scope)
draw(explore.ensemble_corroboration, ensemble)
draw(explore.ensemble_percentiles, ensemble)
""")

md("""
## 9. Per-feature models

`log2 abundance ~ group * sex`, fitted per feature within each region, with
Benjamini-Hochberg FDR over the whole panel. Under no effect the p-values are
flat and the small-p counts match chance. Try the ceramide example from the
walkthrough: `EFFECT = "Female vs Male"`, `REGION = "Cortex"`,
`FEATURE = "Cer(d18:0_14:0)"`, with and without `EXCLUDE = [10, 45]`.
""")
code("""
explore.hit_counts(view)
""")
code("""
draw(explore.volcano, view, REGION, EFFECT)
draw(explore.volcano_grid, view)
draw(explore.p_value_histograms, view)
draw(explore.small_p_counts, view)
""")
code("""
display(explore.top_hits(view, EFFECT, n=5))
draw(explore.effect_forest, view, REGION, EFFECT)
feature = FEATURE or explore.top_hits(view, EFFECT).query("Region == @REGION")["Feature"].iloc[0]
draw(explore.feature_profile, view, feature, EFFECT)
draw(explore.class_effects, view, EFFECT)
""")

md("""
## 10. Feature space: co-abundance modules

Within one region (complete features only): how features correlate, whether
they form modules that survive bootstrapping, and whether the PCA loadings are
stable enough to interpret. In the study, structure is real (modules larger
than chance) but not resolvable with 20 samples per region.
""")
code("""
draw(explore.feature_correlation_heatmap, view, REGION)
draw(explore.feature_loadings_view, view, REGION)
draw(explore.feature_loading_stability, view, REGION)
draw(explore.module_stability, view, REGION)
""")

md("""
## 11. Litters

Pups of one litter share a mother, cage and collection date, so they are not
independent. Do they cluster together in the PCA?
""")
code("""
draw(explore.pca_by_cage, view, REGION)
draw(explore.cage_panels, view, REGION)
draw(explore.within_cage_distance, view, REGION)
""")

md("""
## 12. Regional profiles: does MIA change how regions differ?

Every mouse gives all three regions, so each mouse is its own control. For each
region pair, every feature's mean within-mouse difference in controls is
plotted against the same difference in MIA mice. Points on the diagonal mean
the regional profile is unchanged.
""")
code("""
draw(explore.region_differences, view)
""")

md("""
## 13. ASCA+

One multivariate ANOVA instead of hundreds of tests: the share of variance of
each model term, and the first score of selected effects. Permutation tests
for these shares are in the walkthrough (section 8).
""")
code("""
draw(explore.asca_overview, view)
""")

md("""
## 14. Facility-style figures, every region

The core facility's report showed a heatmap of the top lipids and PLS-DA score
plots that seemed to separate MIA from control. Both are expected even with no
difference at all, which the permutation null next to each score plot shows.
""")
code("""
for region in config.REGIONS:
    draw(explore.vendor_heatmap, view, region)
    draw(explore.plsda_scores, view, region)
""")

md("""
## 15. Lipid versus protein myelin signal

Mouse by mouse: the lipid myelin index (myelin lipids relative to other
glycerophospholipids) against the mean z-score of myelin structural proteins.
The same exclusions are applied to both assays.
""")
code("""
for region in config.REGIONS:
    draw(explore.myelin_lipid_vs_protein, view, region)
""")

md("""
## 16. Saving the figures

With `SAVE_FIGURES = True`, every figure drawn above is written as a PNG into a
folder named after the settings.
""")
code("""
run_name = "_".join([OMICS, f"{len(view.kept_samples)}samples",
                     "excl-" + "-".join(map(str, view.excluded_numbers())) if view.excluded else "all",
                     f"prev{MAX_MISSING_FRACTION}", f"blank{BLANK_MULTIPLIER:g}", NORMALIZATION, SCALING])
folder = settings.results_dir / "supplementary" / run_name
if SAVE_FIGURES:
    folder.mkdir(parents=True, exist_ok=True)
    for name, png in FIGURES.items():
        (folder / f"{name}.png").write_bytes(png)
    print(f"{len(FIGURES)} figures written to {folder.relative_to(PROJECT)}")
else:
    print(f"{len(FIGURES)} figures drawn; set SAVE_FIGURES = True to write them to {folder.relative_to(PROJECT)}")
""")


notebook = nbf.v4.new_notebook()
for kind, text, tags in cells:
    cell = nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
    if tags:
        cell.metadata["tags"] = tags
    notebook.cells.append(cell)
notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
target = Path(__file__).with_name("supplementary_figures.ipynb")
nbf.write(notebook, target)
print(f"wrote {target}")
