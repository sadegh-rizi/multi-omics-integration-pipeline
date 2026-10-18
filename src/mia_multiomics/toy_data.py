"""Generate a simulated (toy) version of the study data.

The real data cannot be shared, so this module writes fake files with the
*same layout* as the files the core facility delivered:

* ``lipidomics_metabolomics_report.xlsx`` - "Sample Sheet", "Lipidomics" and
  "Metabolomics" sheets, including the vendor's header rows, the blank and
  Tech-Mix (pooled QC) injections and the second, embedded header block of
  the negative-mode lipids;
* ``sample_cage_metadata.xlsx`` - cage, collection date and sex per mouse;
* ``proteomics_<Region>.txt`` - Perseus exports (log2 MaxLFQ, already imputed);
* ``reference/*.gmt`` - small GO- and Reactome-style gene-set files.

Because the readers in :mod:`mia_multiomics.inputs` parse these files exactly
as they parse the real ones, the toy data exercise the whole pipeline.

What the simulated biology contains (the "ground truth"):

* a strong **brain-region** signature in all three omics (the positive control);
* a small **sex** effect on ceramides (large on three of them) and strong male-only
  Y-linked proteins;
* **no MIA effect at all** (the real study is null for MIA);
* litter (**cage**) and **mouse** random effects, so samples are not independent;
* two shared **within-region factors** ("myelin" and "energy") that load on
  lipids, metabolites and proteins together - something for MOFA+ to find;
* realistic nuisance: sample loading differences, blank signal, values below
  the detection limit, a few low-prevalence features and two outlying samples.

Run ``python -m mia_multiomics.toy_data`` or call :func:`write_toy_dataset`.
"""

from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Size of the toy panels (the real ones are ~990 lipids, ~270 metabolites,
# ~4,400 proteins; smaller panels keep the demonstration fast).  The number of
# lipids is the sum of the class counts in LIPID_CLASSES (301).
N_METABOLITES = 120
N_PROTEINS = 700

# Eight litters (cages): two mixed-sex control cages and six single-sex MIA
# cages, every cage collected on one date - the same structure as the study.
# With 2 control vs 6 MIA cages there are only C(8, 2) = 28 ways to assign the
# treatment to whole cages, so an exact cage-level p-value cannot go below 1/28.
CAGES = (
    # cage,    group,           males, females, date collected
    ("C10001", config.CONTROL, 3, 3, "2024-06-11"),
    ("C10002", config.CONTROL, 2, 2, "2024-06-25"),
    ("C20001", config.MIA, 2, 0, "2024-06-11"),
    ("C20002", config.MIA, 2, 0, "2024-06-18"),
    ("C20003", config.MIA, 1, 0, "2024-06-18"),
    ("C20004", config.MIA, 0, 2, "2024-06-11"),
    ("C20005", config.MIA, 0, 2, "2024-06-25"),
    ("C20006", config.MIA, 0, 1, "2024-06-25"),
)
DATE_PROCESSED = "2025-02-17"
REGION_ORDER_IN_MOUSE = ("Cortex", "Cerebellum", "Hippocampus")   # sample numbers 3k-2, 3k-1, 3k

# Lipid classes: (class, count, polarity, name style)
LIPID_CLASSES = (
    ("PC", 70, "+", "diacyl"), ("TG", 40, "+", "triacyl"), ("PE", 28, "+", "diacyl"),
    ("PS", 18, "-", "diacyl"), ("DG", 14, "+", "diacyl"), ("Cer", 14, "+", "sphingo"),
    ("PI", 13, "-", "diacyl"), ("SM", 13, "+", "sphingo"), ("PA", 9, "-", "diacyl"),
    ("LPC", 10, "+", "lyso"), ("FA", 9, "-", "single"), ("AcCa", 8, "+", "single"),
    ("Hex1Cer", 8, "+", "sphingo"), ("PG", 6, "-", "diacyl"), ("CL", 5, "-", "sum"),
    ("LPE", 5, "+", "lyso"), ("GD1a", 5, "-", "sum_sphingo"), ("GM2", 3, "-", "sum_sphingo"),
    ("Hex2Cer", 4, "+", "sphingo"), ("SPH", 3, "+", "sum_sphingo"), ("ChE", 4, "+", "single"),
    ("MGDG", 4, "+", "diacyl"), ("GM3", 2, "-", "sum_sphingo"), ("LPS", 2, "-", "lyso"),
    ("MG", 2, "+", "single"), ("ST", 2, "+", "single"),
)
MYELIN_LIPID_CLASSES = ("Hex1Cer", "Hex2Cer", "SM", "Cer")
METABOLITE_PATHWAYS = (
    "Amino acids", "Nucleotides", "Glycolysis", "TCA cycle", "Acylcarnitines",
    "Fatty acids", "Tryptophan metabolism", "Pentose phosphate pathway",
    "Arginine and proline metabolism", "GSH/Sulfur homeostasis",
)
ENERGY_PATHWAYS = ("Glycolysis", "TCA cycle", "Pentose phosphate pathway")

MYELIN_GENES = ("Mbp", "Plp1", "Mag", "Mog", "Cnp", "Mobp", "Ugt8", "Cldn11", "Mal", "Opalin")
ENERGY_GENES = ("Pkm", "Ldha", "Gapdh", "Eno1", "Pgk1", "Aldoa", "Cs", "Idh3a", "Sdha", "Mdh2")
SPHINGO_GENES = ("Cers2", "Cers6", "Sptlc2", "Smpd3", "Asah1", "Gnpat", "Agps", "Far1", "Hexb", "Gm2a", "Psap")
Y_GENES = ("Ddx3y", "Kdm5d", "Eif2s3y", "Uty")
CEREBELLUM_GENES = ("Gabra6", "Calb1", "Pcp2", "Car8", "Zic1")
CORTEX_GENES = ("Satb2", "Tbr1", "Neurod6", "Bcl11b", "Cux2")
HIPPOCAMPUS_GENES = ("Prox1", "Zbtb20", "Nr3c2")
NAMED_GENES = (MYELIN_GENES + ENERGY_GENES + SPHINGO_GENES + Y_GENES
               + CEREBELLUM_GENES + CORTEX_GENES + HIPPOCAMPUS_GENES)


# ---------------------------------------------------------------- design
def mouse_table() -> pd.DataFrame:
    """One row per mouse with cage, group, sex and collection date."""
    rows = []
    for cage, group, males, females, date in CAGES:
        pups = ["Male"] * males + ["Female"] * females
        for pup, sex in enumerate(pups, start=1):
            mouse = cage if len(pups) == 1 else f"{cage} - {pup}"     # single pups carry the cage ID only
            rows.append({"Cage": cage, "Bio group": group, "Sex": sex,
                         "Date collected": date, "Mouse number": mouse})
    mice = pd.DataFrame(rows)
    # Interleave control and MIA mice so sample numbers are not sorted by group.
    control = mice[mice["Bio group"].eq(config.CONTROL)].reset_index(drop=True)
    mia = mice[mice["Bio group"].eq(config.MIA)].reset_index(drop=True)
    order = [frame.iloc[i] for i in range(len(control)) for frame in (control, mia)]
    return pd.DataFrame(order).reset_index(drop=True)


def sample_table(rng: np.random.Generator) -> pd.DataFrame:
    """One row per sample (60): sample number, region, mouse metadata, tissue mass."""
    rows = []
    for mouse_index, mouse in mouse_table().iterrows():
        for offset, region in enumerate(REGION_ORDER_IN_MOUSE):
            number = 3 * mouse_index + offset + 1
            rows.append({
                "Sample number": number,
                "Sample type": region,
                **mouse.to_dict(),
                "Tissue mass (mg)": round(float(rng.normal(42, 7)), 1),
            })
    return pd.DataFrame(rows)


def sample_id(number: int, padded: bool = True) -> str:
    return f"{config.SAMPLE_PREFIX}-{number:02d}" if padded else f"{config.SAMPLE_PREFIX}-{number}"


# ---------------------------------------------------------------- feature names
def _chain(rng: np.random.Generator, odd: bool = False) -> str:
    length = int(rng.choice([14, 16, 18, 18, 20, 22, 24] + ([15, 17] if odd else [])))
    bonds = int(rng.choice([0, 0, 1, 1, 2, 4, 6] if length >= 18 else [0, 0, 1]))
    return f"{length}:{bonds}"


def lipid_names(rng: np.random.Generator) -> pd.DataFrame:
    """Unique, parseable lipid names in the vendor's notation, e.g. PC(16:0_18:1)."""
    rows, seen = [], set()
    for lipid_class, count, polarity, style in LIPID_CLASSES:
        made = 0
        while made < count:
            if style == "diacyl":
                prefix = rng.choice(["", "", "", "O-", "P-"]) if lipid_class in {"PC", "PE"} else ""
                inner = f"{prefix}{_chain(rng)}_{_chain(rng)}" if rng.random() < 0.8 else f"{int(rng.integers(30, 44))}:{int(rng.integers(0, 8))}"
            elif style == "triacyl":
                inner = f"{_chain(rng)}_{_chain(rng)}_{_chain(rng)}"
            elif style == "sphingo":
                inner = f"d18:{int(rng.choice([0, 1]))}_{_chain(rng)}"
            elif style == "sum_sphingo":
                inner = f"d{int(rng.integers(32, 42))}:{int(rng.integers(0, 3))}"
            elif style == "lyso":
                inner = f"{rng.choice(['', 'O-'])}{_chain(rng)}"
            elif style == "sum":
                inner = f"{int(rng.integers(66, 76))}:{int(rng.integers(2, 10))}"
            else:  # single chain
                inner = _chain(rng, odd=True)
            name = f"{lipid_class}({inner})"
            if name in seen:
                continue
            seen.add(name)
            rows.append({"Lipid Name": name, "Lipid Class": lipid_class, "Polarity": polarity})
            made += 1
    table = pd.DataFrame(rows)
    # The vendor lists positive-mode lipids first, then negative mode.
    return table.sort_values(["Polarity", "Lipid Class", "Lipid Name"], key=lambda s: s.map({"+": 0, "-": 1}) if s.name == "Polarity" else s).reset_index(drop=True)


def metabolite_names(rng: np.random.Generator) -> pd.DataFrame:
    """Invented metabolite panel with pathways and acquisition modes (HILIC +/-)."""
    rows = []
    for i in range(N_METABOLITES):
        rows.append({
            "Metabolite": f"Metabolite {i + 1:03d}",
            "Metabolite ID": f"TOY{i + 1:05d}",
            "Molecular formula": f"C{int(rng.integers(2, 24))}H{int(rng.integers(4, 40))}O{int(rng.integers(1, 8))}",
            "Pathway": str(rng.choice(METABOLITE_PATHWAYS)),
            "m/z": round(float(rng.uniform(70, 900)), 6),
            "r.t.": round(float(rng.uniform(0.4, 12)), 3),
            "Method": "HILIC+" if rng.random() < 0.45 else "HILIC-",
        })
    return pd.DataFrame(rows)


def protein_names(rng: np.random.Generator) -> pd.DataFrame:
    """Protein panel: the named marker / module genes plus anonymous Gene#### entries."""
    letters = np.array(list(string.ascii_uppercase))
    genes = list(NAMED_GENES) + [f"Gene{i:04d}" for i in range(1, N_PROTEINS - len(NAMED_GENES) + 1)]
    ids = set()
    while len(ids) < N_PROTEINS:
        ids.add(f"{rng.choice(['P', 'Q'])}{rng.integers(0, 10)}{''.join(rng.choice(letters, 3))}{rng.integers(0, 10)}")
    ids = sorted(ids)
    return pd.DataFrame({"Protein ID": ids, "Gene": genes,
                         "Entry Name": [f"{g.upper()}_MOUSE" for g in genes],
                         "Description": [f"Toy protein {g}" for g in genes]})


# ---------------------------------------------------------------- abundances
def _simulate_log_abundance(
    rng: np.random.Generator,
    samples: pd.DataFrame,
    n_features: int,
    *,
    base_mean: float,
    base_sd: float,
    region_sd: float,
    noise_sd: float,
    myelin_loadings: np.ndarray,
    energy_loadings: np.ndarray,
    sex_effect: np.ndarray,
    region_markers: dict[int, str] | None = None,
) -> np.ndarray:
    """Return a features x samples log2 abundance matrix (no MIA effect)."""
    n = len(samples)
    regions = samples["Sample type"].to_numpy()
    female = samples["Sex"].eq("Female").to_numpy(dtype=float)

    mean = rng.normal(base_mean, base_sd, n_features)
    region_effect = {r: rng.normal(0, region_sd, n_features) for r in config.REGIONS}
    for feature, region in (region_markers or {}).items():
        region_effect[region][feature] += 3.0
    values = mean[:, None] + np.column_stack([region_effect[r] for r in regions])
    values += sex_effect[:, None] * female[None, :]

    # Litter and mouse random effects: shared by all samples of a cage / mouse.
    for column, sd in (("Cage", 0.05), ("Mouse number", 0.10)):
        for _, index in samples.groupby(column).groups.items():
            positions = samples.index.get_indexer(index)
            values[:, positions] += rng.normal(0, sd, n_features)[:, None]

    # Two shared within-region latent factors; their sample scores are drawn
    # once per study (see write_toy_dataset) and stored on `samples`.
    values += np.outer(myelin_loadings, samples["_myelin"].to_numpy())
    values += np.outer(energy_loadings, samples["_energy"].to_numpy())
    values += rng.normal(0, noise_sd, (n_features, n))
    return values


def _loadings(rng: np.random.Generator, n: int, members: np.ndarray, strength: float) -> np.ndarray:
    loadings = np.zeros(n)
    loadings[members] = rng.normal(strength, strength / 4, members.sum())
    return loadings


def _to_raw_intensity(
    rng: np.random.Generator,
    log_values: np.ndarray,
    loading: np.ndarray,
    *,
    blank_dominated_fraction: float,
    low_prevalence_fraction: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Turn log2 abundances into vendor-like peak areas, blanks and Tech-Mix injections."""
    raw = np.exp2(log_values + loading[None, :])
    n_features, n_samples = raw.shape

    # Blanks: usually a tiny fraction of the sample signal; a few features are
    # dominated by blank signal (they should be removed by the blank filter).
    median = np.median(raw, axis=1)
    ratio = np.exp(rng.normal(np.log(0.01), 1.0, n_features))
    dominated = rng.random(n_features) < blank_dominated_fraction
    ratio[dominated] = rng.uniform(0.6, 1.5, dominated.sum())
    blank_level = median * ratio
    blanks = blank_level[:, None] * np.exp(rng.normal(0, 0.3, (n_features, 6)))
    blanks[rng.random(blanks.shape) < 0.15] = 0.0

    # Low-abundance values: the vendor gap-fills non-detects with small
    # numbers, so some values fall below 3 x blank (detection limit).
    low_tail = rng.random(raw.shape) < 0.01
    raw[low_tail] = (blank_level[:, None] * rng.uniform(0.5, 2.5, raw.shape))[low_tail]

    # Low-prevalence features: missing (zero) in 30-60 % of the samples.
    for feature in np.flatnonzero(rng.random(n_features) < low_prevalence_fraction):
        raw[feature, rng.random(n_samples) < rng.uniform(0.3, 0.6)] = 0.0
    raw[rng.random(raw.shape) < 0.003] = 0.0       # sporadic zeros
    raw[rng.random(raw.shape) < 0.002] = np.nan    # sporadic missing cells

    # Tech-Mix: pooled study sample injected six times (technical noise only).
    pooled = np.nanmean(np.where(raw > 0, raw, np.nan), axis=1)
    techmix = pooled[:, None] * np.exp(rng.normal(0, 0.15, (n_features, 6)))
    return raw, blanks, techmix


def _cv_percent(techmix: np.ndarray) -> np.ndarray:
    return 100 * techmix.std(axis=1, ddof=1) / techmix.mean(axis=1)


# ---------------------------------------------------------------- writers
def _vendor_sheet_rows(
    features: pd.DataFrame,
    feature_columns: list[str],
    samples_in_order: pd.DataFrame,
    sample_columns: list[str],
    values: np.ndarray,
    blanks: np.ndarray,
    blank_names: list[str],
    techmix: np.ndarray,
    techmix_names: list[str],
    label_column: int,
) -> list[list[object]]:
    """Build the block of rows the vendor uses: 4 label rows, a header, data."""
    width = label_column + 1 + len(sample_columns) + 12 + 1
    def blank_row() -> list[object]:
        return [None] * width
    rows = []
    for label, column in (("Sex", "Sex"), ("Bio Group", "Bio group"), ("Sample Type", "Sample type"), ("Sample Number", "Sample number")):
        row = blank_row()
        row[label_column] = label
        row[label_column + 1:label_column + 1 + len(sample_columns)] = samples_in_order[column].tolist()
        rows.append(row)
    header = blank_row()
    header[1:1 + len(feature_columns)] = feature_columns
    header[label_column + 1:] = [*sample_columns, *blank_names, *techmix_names, "%CV"]
    rows.append(header)
    cv = _cv_percent(techmix)
    for i, (_, feature) in enumerate(features.iterrows()):
        row = blank_row()
        row[1:1 + len(feature_columns)] = feature[feature_columns].tolist()
        row[label_column + 1:] = [*values[i].tolist(), *blanks[i].tolist(), *techmix[i].tolist(), round(float(cv[i]), 2)]
        rows.append(row)
    return rows


def _vendor_sample_order(samples: pd.DataFrame) -> pd.DataFrame:
    """The vendor sorts columns by region, then group, then sex."""
    order = samples.assign(
        _r=samples["Sample type"].map({"Cerebellum": 0, "Cortex": 1, "Hippocampus": 2}),
        _g=samples["Bio group"].map({config.CONTROL: 0, config.MIA: 1}),
        _s=samples["Sex"].map({"Male": 0, "Female": 1}),
    )
    return order.sort_values(["_r", "_g", "_s", "Sample number"]).drop(columns=["_r", "_g", "_s"])


def _write_report_workbook(path: Path, samples: pd.DataFrame, lipids: pd.DataFrame, lipid_data: tuple,
                           metabolites: pd.DataFrame, metabolite_data: tuple) -> None:
    """The facility's Excel report: sample sheet, lipidomics and metabolomics sheets, in the vendor layout."""
    ordered = _vendor_sample_order(samples)
    positions = ordered.index.to_numpy()

    # Sample Sheet: a few key/value lines, then the table starting at "Sample ID".
    sheet = [[None, None, "Submitter name", "Toy dataset"], [None, None, "PI name", "Simulated"],
             [None, None, "Date", DATE_PROCESSED], [None, None, "Organism", "Mus musculus (simulated)"],
             [], [], [], [], [], [], [], [], [],
             [None, "Sample ID", "Sample number", "Sample type", "Mouse number", "Bio group",
              "Tissue mass (mg)", "Sex", "Notes"]]
    for _, s in samples.iterrows():
        sheet.append([None, f"{config.SAMPLE_PREFIX.replace('_', '')}-{s['Sample number']}", s["Sample number"], s["Sample type"],
                      s["Mouse number"], s["Bio group"], s["Tissue mass (mg)"], s["Sex"],
                      "For Metabolomics, Lipidomics, and Proteomics"])

    # Lipidomics: positive-mode block, two empty rows, negative-mode block with its own header.
    raw, blanks, techmix = lipid_data
    lipid_samples = [sample_id(n) for n in ordered["Sample number"]]
    blank_names = ["PostRunBlank+.raw", *[f"PreRunBlank_{i}+.raw" for i in range(1, 6)]]
    techmix_names = [f"Tech-Mix_{i}+.raw" for i in range(1, 7)]
    lipid_rows: list[list[object]] = [[None, "*Simulated toy data"]]
    for polarity in ("+", "-"):
        block = lipids.index[lipids["Polarity"].eq(polarity)]
        lipid_rows += _vendor_sheet_rows(
            lipids.loc[block].assign(Polarity=f"QE_{polarity}"), ["Lipid Name", "Lipid Class", "Polarity"],
            ordered, lipid_samples, raw[np.ix_(block, positions)], blanks[block], blank_names,
            techmix[block], techmix_names, label_column=3)
        if polarity == "+":
            lipid_rows += [[], []]

    # Metabolomics: same idea, sample IDs are *not* zero padded in this sheet.
    raw_m, blanks_m, techmix_m = metabolite_data
    metabolite_samples = [sample_id(n, padded=False) for n in ordered["Sample number"]]
    metabolite_rows = [[], [], []] + _vendor_sheet_rows(
        metabolites.assign(**{"N.": np.arange(1, len(metabolites) + 1)}), ["N.", "Metabolite", "Metabolite ID", "Molecular formula", "Pathway", "m/z", "r.t.", "Method"],
        ordered, metabolite_samples, raw_m[:, positions], blanks_m,
        ["PostRunBlank-", *[f"PreRunBlank_{i}-" for i in range(1, 6)]], techmix_m,
        [f"Tech-Mix_{i}-" for i in range(1, 7)], label_column=8)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, rows in (("Sample Sheet", sheet), ("Lipidomics", lipid_rows), ("Metabolomics", metabolite_rows)):
            pd.DataFrame(rows).to_excel(writer, sheet_name=name, header=False, index=False)


def _write_cage_workbook(path: Path, samples: pd.DataFrame) -> None:
    """The cage / collection-date workbook (one row per sample)."""
    rows = []
    for _, s in samples.iterrows():
        first = s["Sample type"] == REGION_ORDER_IN_MOUSE[0]   # mouse-level values only on the first row
        rows.append({
            "Sample no": s["Sample number"], "Tissue type": s["Sample type"], "Weight": s["Tissue mass (mg)"],
            "Date collected": pd.Timestamp(s["Date collected"]) if first else None,
            "Cage no": s["Cage"] if first else None, "Age": "P14" if first else None,
            "Sex": s["Sex"] if first else None, "MIA or Control": s["Bio group"] if first else None,
            "Date processed": pd.Timestamp(DATE_PROCESSED) if first else None,
        })
    pd.DataFrame(rows).to_excel(path, sheet_name="Samples to be shipped", index=False)


def _write_proteomics(path: Path, region: str, samples: pd.DataFrame, proteins: pd.DataFrame, values: np.ndarray) -> None:
    """One Perseus-style text export per region (with the "#!" annotation rows Perseus writes)."""
    region_samples = samples[samples["Sample type"].eq(region)].sort_values(["Bio group", "Sample number"], ascending=[False, True])
    columns = {}
    for position, s in region_samples.iterrows():
        group = "MIA" if s["Bio group"] == config.MIA else "control"
        columns[f"TOY_{region}_{group}_{s['Sample number']:02d} MaxLFQ Unique Intensity"] = values[:, position]
    table = pd.DataFrame(columns)
    n_intensity = table.shape[1]
    table["Protein Probability"] = 0.999
    table["Combined Total Peptides"] = 5
    table["Protein ID"] = proteins["Protein ID"].to_numpy()
    table["Entry Name"] = proteins["Entry Name"].to_numpy()
    table["Gene"] = proteins["Gene"].to_numpy()
    table["Description"] = proteins["Description"].to_numpy()
    type_row = "#!{Type}" + "\t".join(["E"] * n_intensity + ["N", "N", "T", "T", "T", "T"])
    group_row = "#!{C:Condition}" + "\t".join(
        ("MIA" if "_MIA_" in c else "Ctrl") for c in table.columns[:n_intensity])
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(table.columns) + "\n")
        handle.write(type_row + "\n")
        handle.write(group_row + "\n")
        handle.write("#!{C:Region}" + "\t".join([region] * n_intensity) + "\n")
        handle.write("#!{C:Note}simulated\n")
        table.to_csv(handle, sep="\t", header=False, index=False, float_format="%.6f")


def _write_gene_sets(reference_dir: Path, rng: np.random.Generator, proteins: pd.DataFrame) -> None:
    """Toy GO and Reactome GMT files: real-looking terms for the planted modules plus random terms."""
    reference_dir.mkdir(parents=True, exist_ok=True)
    gene_of = dict(zip(proteins["Gene"], proteins["Protein ID"]))
    others = [g for g in proteins["Gene"] if g.startswith("Gene")]
    named_sets = {
        "MYELINATION%GOBP%GO:0042552": MYELIN_GENES + ("Gene0001", "Gene0002"),
        "GLYCOLYTIC PROCESS%GOBP%GO:0006096": ENERGY_GENES[:6],
        "TRICARBOXYLIC ACID CYCLE%GOBP%GO:0006099": ENERGY_GENES[6:],
        "SPHINGOLIPID METABOLIC PROCESS%GOBP%GO:0006665": SPHINGO_GENES,
        "CEREBELLUM DEVELOPMENT%GOBP%GO:0021549": CEREBELLUM_GENES,
        "FOREBRAIN DEVELOPMENT%GOBP%GO:0030900": CORTEX_GENES + HIPPOCAMPUS_GENES,
    }
    random_sets = {f"TOY RANDOM PROCESS {i}%GOBP%GO:99{i:05d}": tuple(rng.choice(others, int(rng.integers(8, 40)), replace=False))
                   for i in range(1, 41)}
    go_sets = {**named_sets, **random_sets}
    with (reference_dir / "toy_go_bp_2026-10-01.gmt").open("w") as handle:
        for name, genes in go_sets.items():
            handle.write("\t".join([name, name.split("%")[0].lower(), *genes]) + "\n")
    # Reactome-style sets use protein accessions instead of gene symbols.
    with (reference_dir / "toy_reactome_v97.gmt").open("w") as handle:
        for name, genes in list(go_sets.items())[:30]:
            title = name.split("%")[0].title()
            members = [gene_of[g] for g in genes if g in gene_of]
            handle.write("\t".join([f"{title} [R-TOY-{len(members)}]", "Toy Reactome", *members]) + "\n")


# ---------------------------------------------------------------- main entry
def write_toy_dataset(data_dir: Path, seed: int = 2026) -> Path:
    """Write the complete simulated dataset into ``data_dir`` and return it."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    samples = sample_table(rng)
    # Shared within-region factor scores (vary between mice, centred per region).
    samples["_myelin"] = rng.normal(0, 1, len(samples)) + samples["Sample type"].map({"Cerebellum": 0.5, "Cortex": -0.3, "Hippocampus": 0.2})
    samples["_energy"] = rng.normal(0, 1, len(samples))
    loading = rng.normal(0, 0.25, len(samples)) + 0.02 * (samples["Tissue mass (mg)"] - 42)   # sample amount
    outliers = samples.index[(samples["Sample number"].isin([10, 45]))]     # two planted outlying samples

    # Lipids
    lipids = lipid_names(rng)
    myelin_lipid = lipids["Lipid Class"].isin(MYELIN_LIPID_CLASSES).to_numpy()
    sex_lipid = np.where(lipids["Lipid Class"].eq("Cer"), rng.normal(0.35, 0.1, len(lipids)), 0.0)
    # Three ceramides carry a clear sex difference (the "positive control" lipids).
    sex_controls = np.flatnonzero(lipids["Lipid Class"].eq("Cer").to_numpy())[:3]
    sex_lipid[sex_controls] = 1.2
    log_lipids = _simulate_log_abundance(
        rng, samples, len(lipids), base_mean=15.5, base_sd=2.0, region_sd=0.6, noise_sd=0.3,
        myelin_loadings=_loadings(rng, len(lipids), myelin_lipid, 0.35),
        energy_loadings=_loadings(rng, len(lipids), lipids["Lipid Class"].isin(["AcCa", "FA"]).to_numpy(), 0.25),
        sex_effect=sex_lipid)
    log_lipids[:, outliers] += rng.normal(0, 1.0, (len(lipids), len(outliers)))
    # The outlying samples are extreme on the positive-control ceramides: they
    # inflate the residual variance, so removing them shrinks the standard error.
    log_lipids[np.ix_(sex_controls, outliers)] += 2.5 * rng.choice([-1, 1], (len(sex_controls), len(outliers)))
    lipid_data = _to_raw_intensity(rng, log_lipids, loading.to_numpy(), blank_dominated_fraction=0.02, low_prevalence_fraction=0.03)

    # Metabolites (two acquisition modes with their own sample loading)
    metabolites = metabolite_names(rng)
    energy_met = metabolites["Pathway"].isin(ENERGY_PATHWAYS).to_numpy()
    log_met = _simulate_log_abundance(
        rng, samples, len(metabolites), base_mean=20.0, base_sd=2.0, region_sd=0.5, noise_sd=0.3,
        myelin_loadings=_loadings(rng, len(metabolites), rng.random(len(metabolites)) < 0.15, 0.3),
        energy_loadings=_loadings(rng, len(metabolites), energy_met, 0.4),
        sex_effect=np.zeros(len(metabolites)))
    log_met[:, outliers] += rng.normal(0, 0.8, (len(metabolites), len(outliers)))
    mode_shift = np.where(metabolites["Method"].eq("HILIC+").to_numpy()[:, None], rng.normal(0, 0.15, len(samples))[None, :], 0.0)
    metabolite_data = _to_raw_intensity(rng, log_met + mode_shift, loading.to_numpy(), blank_dominated_fraction=0.05, low_prevalence_fraction=0.02)

    # Proteins (delivered already log2 and imputed: no missing values)
    proteins = protein_names(rng)
    gene = proteins["Gene"]
    markers = {i: "Cerebellum" for i in np.flatnonzero(gene.isin(CEREBELLUM_GENES))}
    markers |= {i: "Cortex" for i in np.flatnonzero(gene.isin(CORTEX_GENES))}
    markers |= {i: "Hippocampus" for i in np.flatnonzero(gene.isin(HIPPOCAMPUS_GENES))}
    sex_protein = np.where(gene.isin(Y_GENES), -4.0, 0.0)     # Y-linked: low in females
    # Each latent programme drives a co-regulated protein module: its named
    # marker genes plus ~50 anonymous proteins.
    anonymous = np.flatnonzero(gene.str.startswith("Gene").to_numpy())
    myelin_module = gene.isin(MYELIN_GENES + SPHINGO_GENES[:5]).to_numpy().copy()
    myelin_module[rng.choice(anonymous, 50, replace=False)] = True
    energy_module = gene.isin(ENERGY_GENES).to_numpy().copy()
    energy_module[rng.choice(anonymous[~myelin_module[anonymous]], 50, replace=False)] = True
    log_prot = _simulate_log_abundance(
        rng, samples, len(proteins), base_mean=18.0, base_sd=1.4, region_sd=0.35, noise_sd=0.25,
        myelin_loadings=_loadings(rng, len(proteins), myelin_module, 0.35),
        energy_loadings=_loadings(rng, len(proteins), energy_module, 0.3),
        sex_effect=sex_protein, region_markers=markers)

    report = data_dir / config.REPORT_WORKBOOK
    _write_report_workbook(report, samples, lipids, lipid_data, metabolites, metabolite_data)
    _write_cage_workbook(data_dir / config.CAGE_WORKBOOK, samples)
    for region, filename in config.PROTEOMICS_FILES.items():
        _write_proteomics(data_dir / filename, region, samples, proteins, log_prot)
    _write_gene_sets(data_dir / config.REFERENCE_DIR, rng, proteins)
    (data_dir / "TOY_DATA_README.txt").write_text(
        "All files in this folder are SIMULATED by mia_multiomics.toy_data (seed "
        f"{seed}).\nThey copy the layout of the real study files but contain no real measurements.\n"
        "Ground truth: strong region effect, small sex effect (large on 3 ceramides), NO MIA effect,\n"
        "two shared within-region factors (myelin, energy), male-only Y-linked proteins,\n"
        f"outlying samples: {', '.join(sample_id(n) for n in samples.loc[outliers, 'Sample number'])}.\n",
        encoding="utf-8")
    return data_dir


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "data"
    print(f"Toy dataset written to {write_toy_dataset(target)}")
