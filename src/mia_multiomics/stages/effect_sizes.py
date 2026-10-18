"""Stage 7 - beyond p-values: how large could an MIA effect be?

With a litter-level p-value that can never go below 1/28, a yes/no test says
little.  This stage estimates effect sizes with intervals instead:

* sex-balanced MIA effect per lipid species, with leave-one-litter-out range,
  per-sex estimates and the date-matched estimate;
* which fold changes the confidence intervals already rule out;
* lipid subclass shifts (post hoc) and pre-defined lipid composites (C1-C8,
  e.g. a myelin index, DHA share, ceramide:sphingomyelin ratio);
* pre-declared protein sets (myelin, plasmalogen synthesis, ...), compared with
  random protein sets of the same size;
* whether the lipid and protein myelin signals agree mouse by mouse.

All estimates are exploratory; mouse-level intervals ignore litter clustering.
Output: ``results/integration/07_effect_sizes/``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .. import config, plotting
from ..config import Settings
from ..integration.inference import treatment_score_test
from .common import prepare, region_samples, write_csv


def mia_effects(values: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Sex-balanced MIA effect per row: the mean over sexes of (MIA - control).

    The pooled residual variance of the four group x sex cells gives the SE
    (mouse level, ignores litters).  ``meta`` is indexed by sample.
    """
    cells = {}
    for group in config.GROUPS:
        for sex in config.SEXES:
            columns = meta.index[meta[config.GROUP_COL].eq(group) & meta[config.SEX_COL].eq(sex)]
            block = values[columns]
            cells[group, sex] = (block.mean(axis=1), block.var(axis=1, ddof=1), block.notna().sum(axis=1))
    difference = sum(cells[config.MIA, s][0] - cells[config.CONTROL, s][0] for s in config.SEXES) / 2
    df = sum(c[2] - 1 for c in cells.values())
    pooled = sum((c[2] - 1) * c[1] for c in cells.values()) / df
    se = np.sqrt(pooled * sum(1 / c[2] for c in cells.values()) / 4)
    t_crit = stats.t.ppf(0.975, df.clip(lower=1))
    return pd.DataFrame({"estimate_log2": difference, "se_log2": se, "d": difference / np.sqrt(pooled),
                         "ci_lower_log2": difference - t_crit * se, "ci_upper_log2": difference + t_crit * se,
                         "resid_df": df})


def robustness(values: pd.DataFrame, meta: pd.DataFrame, matched_dates: list) -> pd.DataFrame:
    """Leave-one-litter-out range, per-sex estimates and the date-matched estimate."""
    loco = pd.concat([mia_effects(values[meta.index[meta[config.CAGE_COL] != cage]],
                                  meta[meta[config.CAGE_COL] != cage])["estimate_log2"]
                      for cage in meta[config.CAGE_COL].unique()], axis=1)
    out = pd.DataFrame({"loco_min_log2": loco.min(axis=1), "loco_max_log2": loco.max(axis=1)})
    for sex in config.SEXES:
        m = meta[meta[config.SEX_COL] == sex]
        out[f"{sex.lower()}_only_log2"] = (values[m.index[m[config.GROUP_COL] == config.MIA]].mean(axis=1)
                                          - values[m.index[m[config.GROUP_COL] == config.CONTROL]].mean(axis=1))
    matched = meta[meta[config.COHORT_COL].astype(str).isin(matched_dates)]
    out["date_matched_log2"] = mia_effects(values[matched.index], matched)["estimate_log2"]
    return out


def annotate_lipids(features: pd.DataFrame) -> pd.DataFrame:
    """Ether flag, subclass label and resolved sphingolipid acyl length."""
    out = features[["Lipid Name", "Lipid Class"]].copy()
    out["ether"] = out["Lipid Name"].str.contains(r"\((?:O|P)-", regex=True, na=False)
    out["subclass"] = np.where(out["ether"], out["Lipid Class"] + "-ether", out["Lipid Class"])
    acyl = out["Lipid Name"].str.extract(r"^[A-Za-z0-9]+\([dmt]\d+:\d+_(\d+):\d+\)")[0]
    out["sphingo_acyl_c"] = pd.to_numeric(acyl, errors="coerce")
    return out


def composite_members(ann: pd.DataFrame) -> dict[str, tuple[pd.Index, pd.Index]]:
    """(numerator, denominator) features of the pre-defined lipid composites."""
    cls, name = ann["Lipid Class"], ann["Lipid Name"].fillna("")
    glycerophospholipid = cls.isin(["PC", "PE", "PS", "PI", "PG", "PA", "CL", "PIP", "LPC", "LPE", "LPS", "LPG", "LPA", "LPI"])
    long_sphingo = cls.isin(["SM", "Cer", "Hex1Cer"]) & (ann["sphingo_acyl_c"] >= 24)
    myelin = cls.isin(["Hex1Cer", "ST"]) | (cls.eq("PE") & ann["ether"]) | long_sphingo
    pepc = cls.isin(["PE", "PC"])
    idx = ann.index
    return {
        "C1 myelin lipid index": (idx[myelin], idx[glycerophospholipid & ~myelin]),
        "C2 DHA share (PE+PC)": (idx[pepc & name.str.contains("22:6")], idx[pepc]),
        "C3 AA:DHA (PE+PC)": (idx[pepc & name.str.contains("20:4")], idx[pepc & name.str.contains("22:6")]),
        "C4 Cer:SM": (idx[cls.eq("Cer")], idx[cls.eq("SM")]),
        "C5 lyso:parent": (idx[cls.isin(["LPC", "LPE"])], idx[cls.isin(["PC", "PE"])]),
        "C6 ether-PE fraction": (idx[cls.eq("PE") & ann["ether"]], idx[cls.eq("PE")]),
        "C7 cardiolipin": (idx[cls.eq("CL")], idx[cls.isin(["PC", "PE", "PS", "PI"])]),
        "C8 ganglioside maturity": (idx[cls.isin(["GD1a", "GD1b", "GT1a", "GQ1b"])], idx[cls.isin(["GM3", "GD3", "GM2", "GD2", "GT3"])]),
    }


def composite_scores(values: pd.DataFrame, members: dict) -> pd.DataFrame:
    """log2(sum of numerator / sum of denominator) per sample, on linear intensities.

    Within-sample ratios cancel the sample normalisation; only features
    complete in this region are summed, so missingness cannot move a ratio.
    """
    complete = values.dropna()
    linear = np.exp2(complete)
    scores = {}
    for name, (numerator, denominator) in members.items():
        num, den = complete.index.intersection(numerator), complete.index.intersection(denominator)
        if len(num) and len(den):
            scores[name] = np.log2(linear.loc[num].sum() / linear.loc[den].sum())
    return pd.DataFrame(scores).T


def run(settings: Settings, population: str = config.PRIMARY_POPULATION) -> dict[str, pd.DataFrame]:
    """Write all effect-size tables and the composite forest plot."""
    plotting.apply_plot_style()
    out = settings.out("integration", "07_effect_sizes", population)
    lipid = prepare(settings, "lipidomics", population)
    protein = prepare(settings, "proteomics", population)
    meta = lipid.metadata.set_index("Sample")
    both = meta.groupby(config.COHORT_COL, observed=True)[config.GROUP_COL].nunique()
    matched_dates = [str(d) for d in both.index[both.eq(2)]]
    ann = annotate_lipids(lipid.features)
    members = composite_members(ann)
    genes = protein.features["Gene"]
    rng = np.random.default_rng(settings.seed)

    def score_test(score: pd.Series, m: pd.DataFrame, name: str) -> dict:
        result = treatment_score_test(m.loc[score.index].reset_index(), score.reset_index(drop=True), score_name=name,
                                      n_mouse_permutations=settings.n_mouse_permutations)
        return {k: result[k] for k in ("observed_welch_t", "cage_permutation_p", "mouse_permutation_p",
                                       "group_r_squared", "cage_r_squared")}

    species, ruled_out, classes, composites, protein_sets, myelin = [], [], [], [], [], []
    region_effects = {}
    for region in config.REGIONS:
        samples = region_samples(lipid.metadata, region)
        m = meta.loc[samples]
        values = lipid.values[samples]
        effects = mia_effects(values, m).join(robustness(values, m, matched_dates))
        effects = ann[["Lipid Name", "Lipid Class", "subclass"]].join(effects).assign(Region=region)
        species.append(effects)
        region_effects[region] = effects["estimate_log2"]

        ok = effects["se_log2"].notna()
        row = {"Region": region, "n_lipids": int(ok.sum()),
               "median_abs_fold_change": float(2 ** effects.loc[ok, "estimate_log2"].abs().median())}
        for fc in (1.25, 1.5, 2.0):
            inside = (effects["ci_lower_log2"] > -np.log2(fc)) & (effects["ci_upper_log2"] < np.log2(fc))
            row[f"share_ci_excludes_gt_{fc}x"] = float(inside[ok].mean())
        ruled_out.append(row)

        for subclass, index in ann.groupby("subclass").groups.items():
            if len(index) >= 8:
                block = effects.loc[index]
                classes.append({"Region": region, "subclass": subclass, "n_species": len(index),
                                **block[["estimate_log2", "loco_min_log2", "loco_max_log2", "male_only_log2",
                                         "female_only_log2", "date_matched_log2"]].mean().to_dict(),
                                "share_species_down": float((block["estimate_log2"] < 0).mean())})

        scores = composite_scores(values, members)
        composite_effects = mia_effects(scores, m).join(robustness(scores, m, matched_dates))
        sd = scores.std(axis=1, ddof=1)
        for name in scores.index:
            composites.append({"Region": region, "composite": name, **composite_effects.loc[name].to_dict(),
                               "effect_in_sd": composite_effects.loc[name, "estimate_log2"] / sd[name],
                               **score_test(scores.loc[name], m, f"{region}:{name}")})

        # protein sets: mean within-region z of the member proteins
        prot = protein.values[samples]
        z = prot.sub(prot.mean(axis=1), axis=0).div(prot.std(axis=1, ddof=1), axis=0)
        per_protein = mia_effects(prot, m)
        for set_name, set_genes in config.PROTEIN_SETS.items():
            ids = genes.index[genes.isin(set_genes)]
            if not len(ids):
                continue
            set_score = z.loc[ids].mean()
            set_effect = mia_effects(set_score.to_frame().T, m).iloc[0]
            set_d = per_protein.loc[ids, "d"].mean()
            random_d = [per_protein["d"].iloc[rng.choice(len(per_protein), len(ids), replace=False)].mean() for _ in range(1000)]
            protein_sets.append({"Region": region, "set": set_name, "genes": ";".join(genes[ids]),
                                 **set_effect.to_dict(), "mean_protein_d": set_d,
                                 "random_set_percentile": float(np.mean(np.abs(random_d) >= abs(set_d))),
                                 **score_test(set_score, m, f"{region}:{set_name}")})

        # do lipid and protein myelin signals agree mouse by mouse?
        protein_myelin = z.loc[genes.index[genes.isin(config.OLIGODENDROCYTE_GENES)]].mean()
        lipid_myelin = scores.loc["C1 myelin lipid index"]
        control = m[config.GROUP_COL].eq(config.CONTROL)
        myelin.append({"Region": region, "r_lipid_protein_all": lipid_myelin.corr(protein_myelin),
                       "r_lipid_protein_controls": lipid_myelin[control].corr(protein_myelin[control]),
                       "protein_mia_minus_control_sd": protein_myelin[~control].mean() - protein_myelin[control].mean()})

    tables = {
        "lipid_species_effects": pd.concat(species).rename_axis("Feature").reset_index(),
        "effects_ruled_out": pd.DataFrame(ruled_out),
        "cross_region_effect_correlation": pd.DataFrame([
            {"pair": f"{a}-{b}", "pearson_r": float(pd.concat([region_effects[a], region_effects[b]], axis=1).corr().iloc[0, 1])}
            for a, b in (("Cortex", "Cerebellum"), ("Hippocampus", "Cerebellum"), ("Cortex", "Hippocampus"))]),
        "subclass_shifts_posthoc": pd.DataFrame(classes),
        "lipid_composites": pd.DataFrame(composites),
        "protein_set_effects": pd.DataFrame(protein_sets),
        "myelin_lipid_protein_crosscheck": pd.DataFrame(myelin),
    }
    for name, table in tables.items():
        write_csv(table, out / f"{name}.csv")
    _forest_plot(tables["lipid_composites"], list(members), out / "figures" / "composite_effects_forest")
    return tables


def _forest_plot(composites: pd.DataFrame, order: list[str], path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, region in zip(axes, config.REGIONS):
        c = composites[composites["Region"] == region].set_index("composite").reindex(order)
        sd = c["estimate_log2"] / c["effect_in_sd"]
        y = np.arange(len(order))[::-1]
        ax.hlines(y, c["loco_min_log2"] / sd, c["loco_max_log2"] / sd, color="0.8", lw=6, label="leave-one-litter-out range")
        ax.errorbar(c["effect_in_sd"], y, xerr=[(c["estimate_log2"] - c["ci_lower_log2"]) / sd,
                                                (c["ci_upper_log2"] - c["estimate_log2"]) / sd],
                    fmt="o", color="black", ms=5, capsize=2, label="estimate, 95 % CI (mouse level)")
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set(title=region, xlabel="MIA - control (SD units)")
    axes[0].set_yticks(np.arange(len(order))[::-1], order)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Pre-defined lipid composites: effect estimates", fontsize=10)
    plotting.save_figure(fig, path)
