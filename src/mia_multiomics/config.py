"""Every fixed choice of the analysis in one place.

Change a threshold, a file name or a permutation count here, not inside the
analysis code.  The values are grouped in the order the pipeline uses them:
input files -> study design -> preprocessing -> statistics -> integration.

Run-size settings (how many permutations, bootstraps, MOFA seeds, ...) live in
the :class:`Settings` dataclass, because the same pipeline is run in two sizes:
``Settings.full()`` reproduces the thesis analysis, ``Settings.fast()`` gives a
few-minute run of every stage on the toy dataset (used by the notebook).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

# --------------------------------------------------------------------------
# Input files (all inside <project>/data).  The toy-data generator writes
# files with exactly these names and layouts; point them at the real files to
# run the pipeline on the real study.
# --------------------------------------------------------------------------
REPORT_WORKBOOK = "lipidomics_metabolomics_report.xlsx"   # Sample Sheet + Lipidomics + Metabolomics sheets
CAGE_WORKBOOK = "sample_cage_metadata.xlsx"               # one row per sample, cage/date per mouse
PROTEOMICS_FILES = {                                       # one Perseus export per region
    "Cerebellum": "proteomics_Cerebellum.txt",
    "Cortex": "proteomics_Cortex.txt",
    "Hippocampus": "proteomics_Hippocampus.txt",
}
REFERENCE_DIR = "reference"                                # GMT gene-set files for enrichment

# Biological sample columns are called "<SAMPLE_PREFIX>-<number>".
SAMPLE_PREFIX = "TOY_001"

# --------------------------------------------------------------------------
# Study design
# --------------------------------------------------------------------------
REGIONS = ("Cerebellum", "Cortex", "Hippocampus")
CONTROL = "Saline Control"
MIA = "MIA"
GROUPS = (CONTROL, MIA)
SEXES = ("Male", "Female")
N_MICE = 20
N_SAMPLES = 60                 # 20 mice x 3 regions

# Metadata column names used everywhere (they come from the Sample Sheet).
REGION_COL = "Sample type"
GROUP_COL = "Bio group"
SEX_COL = "Sex"
MOUSE_COL = "Mouse number"
CAGE_COL = "Cage"
DATE_COL = "Date collected"
COHORT_COL = "Collection cohort"

# Analysis populations.  Only these three are permitted: all 60 samples, or a
# *sample-level* removal of the samples flagged by Hotelling T2 at 95 % or 99 %
# in the regional PCA screen.  A whole mouse is never removed.
POPULATIONS = {
    "all60": None,
    "t2_95": 0.95,
    "t2_99": 0.99,
}
PRIMARY_POPULATION = "all60"

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
MAX_MISSING_FRACTION = 0.20    # drop a feature missing in > 20 % of all 60 samples
BLANK_MULTIPLIER = 3.0         # feature removed if median sample signal < 3 x mean blank
LIPID_ABSOLUTE_FLOOR = 1.0     # detection floor for lipids; metabolomics uses its global minimum
QC_CV_THRESHOLD = 30.0         # Tech-Mix CV (%) reported, not used as a filter
QC_CV_MIN_INJECTIONS = 4
MIN_PER_CELL = 3               # a feature is modelled only if every group x sex cell has >= 3 values
IQR_REMOVE_FRACTION = 0.25     # optional unsupervised variance filter (sensitivity only)

# --------------------------------------------------------------------------
# Outlier screen
# --------------------------------------------------------------------------
OUTLIER_SCREEN_PCS = 3         # Hotelling T2 / robust Mahalanobis in the first 3 PCs
ENSEMBLE_PCA_COMPONENTS = (3, 5, 10)
ENSEMBLE_STABILITY_FRACTION = 0.60   # a detector flags a sample if stable in >= 60 % of settings
ENSEMBLE_MIN_DETECTORS = 2           # ... and the ensemble needs >= 2 detectors to agree

# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
FDR_ALPHA = 0.05
POWER_ALPHA = 0.05
POWER_TARGET = 0.80
ILLUSTRATIVE_LITTER_N = 8      # litters, not pups, are the experimental unit
TOP_FDR_HITS = 10
DIAGNOSTIC_FEATURE_COUNT = 12


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------
VIEWS = ("lipidomics", "metabolomics", "proteomics")

# Marker proteins used as a sanity check that region labels are aligned.
REGION_MARKER_GENES = (
    ("Gabra6", "Cerebellum"),
    ("Calb1", "Cerebellum"),
    ("Satb2", "Cortex"),
    ("Tbr1", "Cortex"),
)
# Pre-declared protein sets for the effect-size analysis.
PROTEIN_SETS = {
    "myelin structural": ("Mbp", "Plp1", "Mag", "Mog", "Cnp", "Mobp", "Ugt8"),
    "plasmalogen synthesis": ("Gnpat", "Agps", "Far1"),
    "ganglioside catabolism": ("Hexb", "Gm2a", "Psap"),
    "sphingolipid metabolism": ("Cers2", "Cers6", "Sptlc2", "Smpd3", "Asah1"),
}
OLIGODENDROCYTE_GENES = PROTEIN_SETS["myelin structural"]


# --------------------------------------------------------------------------
# Run settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Where to read and write, and how big the resampling runs are.

    Every stage of the pipeline takes one ``Settings`` object, so a full run
    and a quick demonstration run differ only in this object.
    """

    project_root: Path
    seed: int = 20_261_017

    # resampling sizes
    n_permutations: int = 999          # label permutations for empirical p-values
    n_asca_permutations: int = 9_999   # ASCA+ permutation tests
    n_bootstraps: int = 1_000          # bootstrap CIs (loadings, disattenuation, ICC)
    n_split_halves: int = 1_000        # split-half reliability draws
    n_null_replicates: int = 200       # simulated null data sets for outlier calibration
    n_isolation_forest_seeds: int = 50
    n_isolation_forest_trees: int = 100
    n_power_datasets: int = 200        # simulated data sets for ASCA+ power
    n_power_permutations: int = 199
    n_feature_bootstraps: int = 1_000  # feature-space PCA loading bootstraps
    n_module_bootstraps: int = 100     # co-abundance module stability
    n_module_nulls: int = 100

    # MOFA+ settings
    mofa_factors: int = 15
    mofa_iterations: int = 1_000
    mofa_seeds: int = 10
    mofa_shuffles: int = 20
    n_cross_omics_shuffles: int = 200
    n_cross_omics_bootstraps: int = 100
    n_mouse_permutations: int = 9_999  # mouse-level permutation tests on integration scores

    # optional external tools
    use_r: bool = False                # run the optional R scripts (limma, DIABLO, ALASCA, noisyR)
    # quick=True skips the slowest sensitivity arms (RF imputation of every cell,
    # imputation-recovery RMSE for RF, the RF grid outside the primary population)
    quick: bool = False

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    def out(self, *parts: str) -> Path:
        """Return (and create) a directory under ``results/``."""
        path = self.results_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def full(cls, project_root: Path) -> "Settings":
        """Thesis-size run (hours on the real data)."""
        return cls(project_root=Path(project_root))

    @classmethod
    def fast(cls, project_root: Path) -> "Settings":
        """Small resampling counts so every stage finishes in seconds to minutes.

        Results are noisier (p-values have a coarse floor) but every table and
        figure of the full run is produced.
        """
        return cls(
            project_root=Path(project_root),
            n_permutations=99,
            n_asca_permutations=99,
            n_bootstraps=100,
            n_split_halves=100,
            n_null_replicates=40,
            n_isolation_forest_seeds=5,
            n_isolation_forest_trees=50,
            n_power_datasets=10,
            n_power_permutations=29,
            n_feature_bootstraps=100,
            n_module_bootstraps=20,
            n_module_nulls=20,
            mofa_factors=6,
            mofa_iterations=500,
            mofa_seeds=5,
            mofa_shuffles=3,
            n_cross_omics_shuffles=20,
            n_cross_omics_bootstraps=10,
            n_mouse_permutations=999,
            quick=True,
        )

    def with_(self, **changes) -> "Settings":
        """Return a copy with some fields changed, e.g. ``settings.with_(use_r=True)``."""
        return replace(self, **changes)
