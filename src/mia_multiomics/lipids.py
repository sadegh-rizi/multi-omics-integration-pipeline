"""Lipid-specific helpers: parse chain length / double bonds from lipid names,
and summarise species into classes, structural indices and compositional shares."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_CHAIN_RE = re.compile(r"(?<!\d)(\d+):(\d+)")
_STRUCTURAL_METRICS = (
    "mean_chain_length",
    "mean_double_bond_count",
    "pufa_fraction",
    "saturated_to_unsaturated_ratio",
)
_MULTI_CHAIN_CLASSES = frozenset({
    "Cer", "CL", "DG", "Hex1Cer", "Hex2Cer", "MGDG",
    "PA", "PC", "PE", "PG", "PI", "PS", "SM", "TG",
})


def parse_lipid_structure(lipid_name: object) -> dict[str, object] | None:
    """Parse chain-length and double-bond totals from a supplied lipid name.

    The report contains both species-level names (for example ``PC(16:0_18:1)``)
    and sum-composition names (for example ``PC(38:4)``).  Both are retained as
    the delivered annotation and are represented by the numeric chain tokens
    in parentheses.  Unparseable names are reported as missing rather than
    inferred.
    """
    if pd.isna(lipid_name):
        return None
    text = str(lipid_name)
    opening = text.find("(")
    closing = text.rfind(")")
    if opening < 0 or closing <= opening:
        return None
    tokens = _CHAIN_RE.findall(text[opening + 1 : closing])
    if not tokens:
        return None
    chain_lengths = tuple(int(length) for length, _ in tokens)
    double_bonds = tuple(int(bonds) for _, bonds in tokens)
    return {
        "chain_lengths": chain_lengths,
        "double_bonds": double_bonds,
        "total_chain_length": sum(chain_lengths),
        "total_double_bond_count": sum(double_bonds),
    }


def annotate_lipid_structures(annotations: pd.DataFrame) -> pd.DataFrame:
    """Return one parsed-structure row per annotation index."""
    required = {"Lipid Name", "Lipid Class"}
    missing = required.difference(annotations.columns)
    if missing:
        raise KeyError(f"Annotations are missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for feature, annotation in annotations.iterrows():
        parsed = parse_lipid_structure(annotation["Lipid Name"])
        row = {
            "Feature": feature,
            "Lipid Name": annotation["Lipid Name"],
            "Lipid Class": annotation["Lipid Class"],
            "parsed": parsed is not None,
        }
        if parsed is None:
            row.update({
                "total_chain_length": np.nan,
                "total_double_bond_count": np.nan,
                "n_chains": np.nan,
            })
        else:
            row.update({
                "total_chain_length": parsed["total_chain_length"],
                "total_double_bond_count": parsed["total_double_bond_count"],
                "n_chains": len(parsed["chain_lengths"]),
            })
        row["structural_ambiguous"] = bool(
            parsed is not None
            and row["n_chains"] == 1
            and str(annotation["Lipid Class"]) in _MULTI_CHAIN_CLASSES
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("Feature")


def aggregate_class_profiles(
    values: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    statistic: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate log-scale feature profiles by supplied lipid class.

    The summaries are unweighted means or medians of the normalized log2
    species values.  This answers a typical species-level abundance question
    without allowing the most abundant species to define the class summary.
    Missing values remain missing when no member of a class is observed for a
    sample.
    """
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'.")
    if not values.index.isin(annotations.index).all():
        raise ValueError("Every value feature must have an annotation row.")
    class_values: list[pd.Series] = []
    class_annotations: list[dict[str, object]] = []
    for lipid_class, class_annotations_frame in annotations.loc[values.index].groupby(
        "Lipid Class", sort=True, dropna=False
    ):
        if pd.isna(lipid_class) or str(lipid_class).strip() == "":
            continue
        features = class_annotations_frame.index
        profile = getattr(values.loc[features], statistic)(axis=0, skipna=True)
        feature_id = f"class_{statistic}::{lipid_class}"
        profile.name = feature_id
        class_values.append(profile)
        class_annotations.append({
            "Feature": feature_id,
            "Lipid Name": f"{lipid_class} class {statistic}",
            "Lipid Class": lipid_class,
            "Aggregation": f"log2_{statistic}",
            "n_species": len(features),
        })
    if not class_values:
        return pd.DataFrame(index=[], columns=values.columns), pd.DataFrame(
            columns=["Lipid Name", "Lipid Class", "Aggregation", "n_species"]
        )
    profiles = pd.DataFrame(class_values)
    profile_annotations = pd.DataFrame(class_annotations).set_index("Feature")
    return profiles, profile_annotations


def structural_index_profiles(
    values: pd.DataFrame,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate unweighted structural indices per sample, class, and overall.

    A sample contributes only observed members of the relevant class.  PUFA is
    defined as total double-bond count >= 2, and the saturated/unsaturated
    ratio is the count of 0-double-bond members divided by the count of
    members with at least one double bond.  The returned long table preserves
    the feature counts used for every value.
    """
    structures = annotate_lipid_structures(annotations.loc[values.index])
    parsed = structures.loc[structures["parsed"]].copy()
    if parsed.empty:
        empty_values = pd.DataFrame(index=[], columns=values.columns)
        empty_annotations = pd.DataFrame(columns=["Lipid Name", "Lipid Class", "Metric", "Scope"])
        return empty_values, empty_annotations, pd.DataFrame()

    scopes: list[tuple[str, pd.Index]] = [("Overall", parsed.index)]
    for lipid_class, frame in parsed.groupby("Lipid Class", sort=True, dropna=False):
        if pd.isna(lipid_class) or str(lipid_class).strip() == "":
            continue
        scopes.append((str(lipid_class), frame.index))

    profiles: list[pd.Series] = []
    profile_annotations: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for scope, feature_index in scopes:
        structure_frame = parsed.loc[feature_index]
        for metric in _STRUCTURAL_METRICS:
            feature_id = f"structural::{scope}::{metric}"
            profile_values: dict[str, float] = {}
            for sample in values.columns:
                available = values.loc[feature_index, sample].notna()
                current = structure_frame.loc[available]
                n_features = len(current)
                if n_features == 0:
                    profile_values[sample] = np.nan
                    continue
                chain_length = current["total_chain_length"].to_numpy(dtype=float)
                double_bonds = current["total_double_bond_count"].to_numpy(dtype=float)
                saturated = np.sum(double_bonds == 0)
                unsaturated = np.sum(double_bonds > 0)
                determinate_pufa = current.loc[~current["structural_ambiguous"]]
                determinate_double_bonds = determinate_pufa["total_double_bond_count"].to_numpy(dtype=float)
                n_ambiguous_pufa_features = int(current["structural_ambiguous"].sum())
                metric_values = {
                    "mean_chain_length": np.mean(chain_length),
                    "mean_double_bond_count": np.mean(double_bonds),
                    "pufa_fraction": np.mean(determinate_double_bonds >= 2) if len(determinate_double_bonds) else np.nan,
                    "saturated_to_unsaturated_ratio": (
                        saturated / unsaturated if unsaturated else np.nan
                    ),
                }
                profile_values[sample] = float(metric_values[metric])
                long_rows.append({
                    "Sample": sample,
                    "Scope": scope,
                    "Lipid Class": scope,
                    "Metric": metric,
                    "value": profile_values[sample],
                    "n_features": n_features,
                    "n_determinate_pufa_features": len(determinate_pufa),
                    "n_ambiguous_pufa_features": n_ambiguous_pufa_features,
                })
            profiles.append(pd.Series(profile_values, name=feature_id, dtype=float))
            profile_annotations.append({
                "Feature": feature_id,
                "Lipid Name": f"{scope} {metric}",
                "Lipid Class": scope,
                "Metric": metric,
                "Scope": scope,
                "Aggregation": "unweighted structural index",
            })

    return (
        pd.DataFrame(profiles),
        pd.DataFrame(profile_annotations).set_index("Feature"),
        pd.DataFrame(long_rows),
    )



# ---------------------------------------------------------------- compositional summaries
_ETHER_RE = re.compile(r"(?:^|[(_])(?:O-|P-)")
_SPHINGOLIPID_CLASSES = frozenset({"Cer", "SM", "Hex1Cer", "Hex2Cer", "GD1a", "GD1b", "GM1", "GM2", "GM3"})


def _mean_profile(values: pd.DataFrame, features: pd.Index, name: str) -> pd.Series:
    if len(features) == 0:
        raise ValueError(f"Cannot build empty lipid summary {name}.")
    return values.loc[:, features].mean(axis=1).rename(name)


def build_lipid_summary_features(
    values: pd.DataFrame,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fixed compositional summaries and return values plus definitions.

    The delivered lipid block is log2-normalized.  We exponentiate and divide
    each sample by its total signal before summarizing, so class and structural
    summaries are within-sample shares rather than absolute-intensity totals.
    """
    values = values.loc[:, annotations.index]
    if annotations.isna().any().any():
        raise ValueError("Lipid annotations do not cover every delivered feature.")

    positive = np.exp2(values.astype(float))
    shares = positive.divide(positive.sum(axis=1), axis=0)
    structures = annotate_lipid_structures(annotations)
    structural_profiles, _, _ = structural_index_profiles(values.T, annotations)
    structural_profiles = structural_profiles.T
    rows: list[pd.Series] = []
    definitions: list[dict[str, object]] = []

    def add(name: str, feature_mask: pd.Series, description: str) -> None:
        selected = annotations.index[feature_mask.to_numpy()]
        if len(selected) == 0:
            return
        rows.append(_mean_profile(shares, selected, name))
        definitions.append({"Feature": name, "description": description, "n_source_lipids": len(selected)})

    class_counts = annotations["Lipid Class"].value_counts()
    for lipid_class in class_counts.index.sort_values():
        mask = annotations["Lipid Class"].eq(lipid_class)
        add(f"class_share::{lipid_class}", mask, f"Mean within-sample share of {lipid_class} features")

    for name, mask, description in (
        ("ether_share", annotations["Lipid Name"].astype(str).str.contains(_ETHER_RE), "Ether lipid share"),
        ("ester_share", ~annotations["Lipid Name"].astype(str).str.contains(_ETHER_RE), "Non-ether lipid share"),
        ("20:4_share", annotations["Lipid Name"].astype(str).str.contains(r"20:4", regex=True), "Features containing a 20:4 chain"),
        ("22:6_share", annotations["Lipid Name"].astype(str).str.contains(r"22:6", regex=True), "Features containing a 22:6 chain"),
        (
            "very_long_chain_sphingolipid_share",
            annotations["Lipid Class"].isin(_SPHINGOLIPID_CLASSES) & structures["total_chain_length"].ge(24),
            "Sphingolipid features with total chain length at least C24",
        ),
    ):
        add(name, mask, description)

    for name, mask in (
        ("chain_length_le_16", structures["total_chain_length"].le(16)),
        ("chain_length_17_18", structures["total_chain_length"].between(17, 18)),
        ("chain_length_19_20", structures["total_chain_length"].between(19, 20)),
        ("chain_length_21_22", structures["total_chain_length"].between(21, 22)),
        ("chain_length_ge_23", structures["total_chain_length"].ge(23)),
        ("double_bonds_0", structures["total_double_bond_count"].eq(0)),
        ("double_bonds_1", structures["total_double_bond_count"].eq(1)),
        ("double_bonds_2", structures["total_double_bond_count"].eq(2)),
        ("double_bonds_ge_3", structures["total_double_bond_count"].ge(3)),
    ):
        add(name, mask.fillna(False), f"Mean within-sample share for {name.replace('_', ' ')}")

    for metric in ("mean_chain_length", "mean_double_bond_count", "pufa_fraction", "saturated_to_unsaturated_ratio"):
        feature = f"structural::Overall::{metric}"
        rows.append(structural_profiles[feature].rename(feature))
        definitions.append({"Feature": feature, "description": f"Overall structural index: {metric}", "n_source_lipids": len(structures)})

    # Add structural summaries for the 10 largest classes to keep the panel
    # interpretable and within the planned approximately 60-80 features.
    for lipid_class in class_counts.nlargest(10).index.sort_values():
        class_index = structures.index[structures["Lipid Class"].eq(lipid_class)]
        for metric in ("mean_chain_length", "mean_double_bond_count"):
            feature = f"structural::{lipid_class}::{metric}"
            rows.append(structural_profiles[feature].rename(feature))
            definitions.append({"Feature": feature, "description": f"Mean {metric} among {lipid_class} annotations", "n_source_lipids": len(class_index)})

    summaries = pd.concat(rows, axis=1)
    definition_table = pd.DataFrame(definitions).drop_duplicates("Feature").set_index("Feature")
    if len(summaries.columns) > 100:
        raise AssertionError(f"Unexpectedly large lipid summary panel: {len(summaries.columns)}")
    return summaries, definition_table
