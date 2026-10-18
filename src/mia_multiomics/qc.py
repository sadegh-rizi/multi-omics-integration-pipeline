"""Quality-control summaries: pooled-QC reproducibility, technical vs biological
variance, missingness, and agreement of outlier flags between assays."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ranksums

from . import config


# ---------------------------------------------------------------- pooled QC (Tech-Mix)
def qc_injection_cv(techmix: pd.DataFrame, *, min_injections: int = config.QC_CV_MIN_INJECTIONS) -> pd.DataFrame:
    """Raw and injection-median-normalised CV (%) of the pooled-QC injections per feature.

    Features with fewer than ``min_injections`` positive injections are
    marked not assessable instead of passing silently.
    """
    numeric = techmix.apply(pd.to_numeric, errors="coerce").where(lambda v: v > 0)
    medians = numeric.median(axis=0)
    if medians.isna().any():
        raise ValueError("Each Tech-Mix injection needs at least one positive value.")
    normalized = numeric.divide(medians, axis=1) * medians.median()
    counts = numeric.notna().sum(axis=1)
    assessable = counts.ge(min_injections)
    def cv(table: pd.DataFrame) -> pd.Series:
        return 100 * table.std(axis=1, ddof=1) / table.mean(axis=1)
    return pd.DataFrame({
        "n_techmix_used": counts,
        "raw_cv_percent": cv(numeric).where(assessable),
        "normalized_cv_percent": cv(normalized).where(assessable),
        "cv_status": np.where(assessable, "assessable", "not_assessable"),
    }, index=techmix.index)


def variance_ratio_table(log_techmix: pd.DataFrame, log_samples: pd.DataFrame, metadata: pd.DataFrame,
                         *, min_injections: int = 4, min_per_region: int = 15) -> pd.DataFrame:
    """Technical SD (pooled QC) vs biological SD (per region) on the log2 scale.

    ``variance_ratio`` = technical variance / biological variance; a feature
    with ratio >= 0.5 is called technical-dominated: half or more of its
    spread could be measurement noise.
    """
    technical_sd = log_techmix.std(axis=1, ddof=1).where(log_techmix.notna().sum(axis=1).ge(min_injections))
    regions = metadata.set_index("Sample").loc[log_samples.columns, config.REGION_COL]
    rows = []
    for region in config.REGIONS:
        block = log_samples.loc[:, regions.eq(region).to_numpy()]
        counts = block.notna().sum(axis=1)
        rows.append(pd.DataFrame({
            "Feature": block.index,
            "Region": region,
            "technical_sd_log2": technical_sd.to_numpy(),
            "biological_sd_log2": block.std(axis=1, ddof=1).where(counts.ge(min_per_region)).to_numpy(),
            "n_techmix_used": log_techmix.notna().sum(axis=1).to_numpy(),
            "n_biological_used": counts.to_numpy(),
        }))
    table = pd.concat(rows, ignore_index=True)
    table["variance_ratio"] = table["technical_sd_log2"] ** 2 / table["biological_sd_log2"] ** 2
    table.loc[table["biological_sd_log2"].le(0), "variance_ratio"] = np.nan
    table["technical_dominated"] = table["variance_ratio"].ge(0.5).where(table["variance_ratio"].notna())
    return table


# ---------------------------------------------------------------- missingness
def missingness_by_feature(values: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Number and percentage of missing samples per feature."""
    table = pd.DataFrame({"n_missing": values.isna().sum(axis=1),
                          "pct_missing": 100 * values.isna().mean(axis=1)})
    return features.join(table, how="inner").rename_axis("Feature").sort_values(
        "n_missing", ascending=False).reset_index()


def missingness_by_sample(values: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Number and percentage of missing features per sample, with sample metadata."""
    table = pd.DataFrame({"Sample": values.columns, "n_missing": values.isna().sum(axis=0).to_numpy(),
                          "pct_missing": 100 * values.isna().mean(axis=0).to_numpy()})
    return metadata.merge(table, on="Sample", how="inner")


def missingness_by_group(sample_missingness: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Distribution of per-sample % missing within each level of ``group_col``."""
    grouped = sample_missingness.groupby(group_col, observed=True)["pct_missing"]
    table = grouped.agg(n_samples="count", mean_pct_missing="mean", median_pct_missing="median",
                        q25_pct_missing=lambda x: x.quantile(0.25),
                        q75_pct_missing=lambda x: x.quantile(0.75)).reset_index()
    table["iqr_pct_missing"] = table["q75_pct_missing"] - table["q25_pct_missing"]
    return table


def test_missingness_by_group(sample_missingness: pd.DataFrame, group_col: str, first: str, second: str) -> dict:
    """Wilcoxon rank-sum test of per-sample % missing between two groups."""
    a = sample_missingness.loc[sample_missingness[group_col].eq(first), "pct_missing"]
    b = sample_missingness.loc[sample_missingness[group_col].eq(second), "pct_missing"]
    statistic, p_value = ranksums(a, b)
    return {"comparison": f"{first} vs {second}", "test": "Wilcoxon rank-sum",
            "statistic": statistic, "p_value": p_value}


def missingness_by_class(values: pd.DataFrame, classes: pd.Series, metadata: pd.DataFrame,
                         group_col: str = config.GROUP_COL) -> pd.DataFrame:
    """Missing cells per feature class and group, with a Fisher exact test per class.

    A class that is missing more often in one treatment group would make
    "missing" informative about treatment - a reason not to impute naively.
    """
    long = values.isna().rename_axis(index="Feature", columns="Sample").stack().rename("missing").reset_index()
    long = long.merge(classes.rename("Class").rename_axis("Feature").reset_index(), on="Feature")
    long = long.merge(metadata[["Sample", group_col]], on="Sample")
    counts = long.groupby(["Class", group_col], observed=True)["missing"].agg(n_missing="sum", n_total="count").reset_index()
    counts["n_observed"] = counts["n_total"] - counts["n_missing"]
    counts["pct_missing"] = 100 * counts["n_missing"] / counts["n_total"]
    rows = []
    for lipid_class, sub in counts.groupby("Class"):
        by_group = sub.set_index(group_col)
        if not set(config.GROUPS) <= set(by_group.index):
            continue
        table = [[by_group.loc[g, "n_missing"], by_group.loc[g, "n_observed"]] for g in config.GROUPS]
        result = fisher_exact(table)
        rows.append({"Class": lipid_class,
                     **{f"pct_missing_{g}": by_group.loc[g, "pct_missing"] for g in config.GROUPS},
                     "odds_ratio": result.statistic, "p_value": result.pvalue})
    return pd.DataFrame(rows)


def filtering_stage_missingness(stages: list[tuple[str, pd.DataFrame]], metadata: pd.DataFrame) -> pd.DataFrame:
    """Overall and per-group % missing after each preprocessing stage."""
    groups = {g: metadata.loc[metadata[config.GROUP_COL].eq(g), "Sample"] for g in config.GROUPS}
    rows = []
    for name, table in stages:
        row = {"Stage": name, "n_features": len(table), "pct_missing": 100 * table.isna().to_numpy().mean()}
        for group, samples in groups.items():
            row[f"pct_missing_{group}"] = 100 * table.loc[:, samples].isna().to_numpy().mean()
        row["group_difference"] = abs(row[f"pct_missing_{config.GROUPS[0]}"] - row[f"pct_missing_{config.GROUPS[1]}"])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- cross-assay agreement
SHARED_PROVENANCE_CAVEAT = (
    "Lipidomics and metabolomics may come from one biphasic extraction of the same tissue, "
    "so agreement between them is not independent replication."
)


def outlier_flag_concordance(lipid_flags: pd.DataFrame, metabolite_flags: pd.DataFrame, statistic: str) -> pd.DataFrame:
    """2 x 2 table per region: is a sample flagged by both assays?

    Under independence and 5 % flag rates, about n x 0.05^2 samples would be
    flagged by both by chance.
    """
    flag = f"{statistic}_95_flag"
    merged = lipid_flags[["Sample", "Region", flag]].rename(columns={flag: "lipidomics_flag"}).merge(
        metabolite_flags[["Sample", "Region", flag]].rename(columns={flag: "metabolomics_flag"}),
        on=["Sample", "Region"], validate="one_to_one")
    cells = pd.MultiIndex.from_product([config.REGIONS, [False, True], [False, True]],
                                       names=["Region", "lipidomics_flag", "metabolomics_flag"])
    table = merged.groupby(["Region", "lipidomics_flag", "metabolomics_flag"]).size().reindex(
        cells, fill_value=0).rename("n_samples").reset_index()
    table["statistic"] = statistic
    table["expected_both_under_independence"] = len(merged) / len(config.REGIONS) * 0.05 ** 2
    table["caveat"] = SHARED_PROVENANCE_CAVEAT
    return table
