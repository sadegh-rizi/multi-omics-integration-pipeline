"""Read the delivered files into tidy tables.

Every omics is returned as an :class:`OmicsData` object with the same shape:

* ``values``   - features x 60 biological samples (raw, as delivered)
* ``blanks``   - features x blank injections (lipidomics/metabolomics only)
* ``techmix``  - features x pooled-QC (Tech-Mix) injections (lipidomics/metabolomics only)
* ``features`` - one annotation row per feature (name, class, pathway, gene, ...)
* ``metadata`` - one row per sample (region, group, sex, mouse, cage, date, ...)

Sample columns are always the zero-padded ID ``<SAMPLE_PREFIX>-NN`` so that
the three omics can be joined directly.  Nothing is filtered, normalised or
imputed here - that is :mod:`mia_multiomics.preprocessing`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config
from .config import Settings


@dataclass
class OmicsData:
    """Raw measurements of one omics layer, aligned to the sample metadata."""

    name: str
    values: pd.DataFrame
    features: pd.DataFrame
    metadata: pd.DataFrame
    blanks: pd.DataFrame | None = None
    techmix: pd.DataFrame | None = None

    def region(self, region: str) -> pd.DataFrame:
        """Values of the samples from one brain region."""
        samples = self.metadata.loc[self.metadata[config.REGION_COL].eq(region), "Sample"]
        return self.values.loc[:, samples]


def sample_id(number: int) -> str:
    """Canonical sample identifier, e.g. 7 -> 'TOY_001-07'."""
    return f"{config.SAMPLE_PREFIX}-{int(number):02d}"


_SAMPLE_RE = re.compile(rf"^{re.escape(config.SAMPLE_PREFIX)}-(\d+)$")


def sample_number(column: object) -> int | None:
    """Return the sample number of a biological-sample column, else None."""
    match = _SAMPLE_RE.fullmatch(str(column))
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------- workbook parsing
def read_table_after_header(path: Path, sheet_name: str, header_marker: str) -> pd.DataFrame:
    """Read a vendor sheet whose real header row starts further down.

    The vendor puts a few label rows (sex, group, region, sample number) above
    the table; the header row is the first row containing ``header_marker``.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    matches = raw.eq(header_marker).any(axis=1)
    if not matches.any():
        raise ValueError(f"Header {header_marker!r} was not found in sheet {sheet_name!r}.")
    header_row = int(matches[matches].index[0])
    columns = raw.iloc[header_row]
    keep = columns.notna() & columns.astype(str).str.strip().ne("")
    table = raw.iloc[header_row + 1:, keep.to_numpy()].copy()
    table.columns = columns[keep].astype(str).str.strip()
    return table.dropna(how="all").reset_index(drop=True)


def measurement_columns(table: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Split the columns of a vendor sheet into samples, blanks and Tech-Mix QC."""
    columns = [str(c) for c in table.columns]
    samples = [c for c in columns if sample_number(c) is not None]
    blanks = [c for c in columns if "Blank" in c]
    techmix = [c for c in columns if c.startswith("Tech-Mix")]
    if len(samples) != config.N_SAMPLES:
        raise ValueError(f"Expected {config.N_SAMPLES} biological samples, found {len(samples)}.")
    if not blanks or not techmix:
        raise ValueError("Blank and Tech-Mix injection columns are both required.")
    return samples, blanks, techmix


def _unique_names(names: pd.Series, suffix: pd.Series) -> pd.Index:
    """Use feature names as IDs; disambiguate duplicates with a suffix (e.g. polarity)."""
    names = names.astype(str).str.strip()
    duplicated = names.duplicated(keep=False)
    names = names.where(~duplicated, names + " [" + suffix.astype(str) + "]")
    if names.duplicated().any():
        raise ValueError(f"Feature names are not unique: {names[names.duplicated()].tolist()[:5]}")
    return pd.Index(names, name="Feature")


def _numeric(table: pd.DataFrame, columns: list[str], index: pd.Index) -> pd.DataFrame:
    values = table.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    values.index = index
    return values


# ---------------------------------------------------------------- sample metadata
def read_sample_sheet(path: Path) -> pd.DataFrame:
    """Read the submitter's Sample Sheet: one row per sample."""
    sheet = read_table_after_header(path, "Sample Sheet", "Sample ID")
    sheet = sheet.loc[pd.to_numeric(sheet["Sample number"], errors="coerce").notna()].copy()
    sheet["Sample number"] = sheet["Sample number"].astype(int)
    sheet["Sample"] = sheet["Sample number"].map(sample_id)
    sheet["Tissue mass (mg)"] = pd.to_numeric(sheet["Tissue mass (mg)"], errors="coerce")
    columns = ["Sample", "Sample number", config.REGION_COL, config.MOUSE_COL,
               config.GROUP_COL, config.SEX_COL, "Tissue mass (mg)"]
    return sheet.loc[:, columns].reset_index(drop=True)


def read_cage_workbook(path: Path) -> pd.DataFrame:
    """Read cage / collection-date metadata (mouse values are only on each mouse's first row)."""
    table = pd.read_excel(path, sheet_name="Samples to be shipped")
    mouse_columns = ["Date collected", "Cage no", "Age", "Sex", "MIA or Control", "Date processed"]
    table[mouse_columns] = table[mouse_columns].ffill()
    dates = pd.to_datetime(table["Date collected"])
    cohorts = dates.dt.strftime("%Y-%m-%d")
    return pd.DataFrame({
        "Sample number": table["Sample no"].astype(int),
        config.CAGE_COL: table["Cage no"].astype(str),
        config.DATE_COL: dates,
        config.COHORT_COL: pd.Categorical(cohorts, categories=sorted(cohorts.unique()), ordered=True),
        "Age": table["Age"].astype(str),
        "_sex_check": table["Sex"].astype(str),
        "_group_check": table["MIA or Control"].astype(str),
        "Date processed": pd.to_datetime(table["Date processed"]),
    })


def check_design(metadata: pd.DataFrame) -> None:
    """Fail loudly if the study design is not what every analysis assumes.

    20 mice x 3 regions; each cage holds one treatment group and was collected
    on one date; the cage prefix of the mouse number matches the cage workbook.
    """
    if len(metadata) != config.N_SAMPLES or metadata[config.MOUSE_COL].nunique() != config.N_MICE:
        raise ValueError(f"Expected {config.N_SAMPLES} samples from {config.N_MICE} mice.")
    regions = metadata.groupby(config.MOUSE_COL)[config.REGION_COL].agg(lambda r: sorted(r) == sorted(config.REGIONS))
    if not regions.all():
        raise ValueError("Every mouse must contribute exactly one sample per region.")
    if config.CAGE_COL in metadata:
        nesting = metadata.groupby(config.CAGE_COL)[[config.GROUP_COL, config.DATE_COL]].nunique()
        if nesting.ne(1).any().any():
            raise ValueError("Each cage must map to exactly one treatment group and one collection date.")
        prefixes = metadata[config.MOUSE_COL].astype(str).str.split(" - ").str[0]
        if set(prefixes) != set(metadata[config.CAGE_COL]):
            raise ValueError("Cage IDs do not match the prefixes of the mouse numbers.")


def load_metadata(settings: Settings) -> pd.DataFrame:
    """Sample Sheet joined with cage metadata; one row per sample, sorted by sample number."""
    sheet = read_sample_sheet(settings.data_dir / config.REPORT_WORKBOOK)
    cages = read_cage_workbook(settings.data_dir / config.CAGE_WORKBOOK)
    metadata = sheet.merge(cages, on="Sample number", how="left", validate="one_to_one")
    for column, check in ((config.SEX_COL, "_sex_check"), (config.GROUP_COL, "_group_check")):
        if not metadata[column].astype(str).eq(metadata[check]).all():
            raise ValueError(f"The cage workbook disagrees with the Sample Sheet on {column}.")
    metadata = metadata.drop(columns=["_sex_check", "_group_check"])
    metadata = metadata.sort_values("Sample number").reset_index(drop=True)
    check_design(metadata)
    return metadata


# ---------------------------------------------------------------- the three omics
def _vendor_omics(settings: Settings, name: str, sheet: str, marker: str,
                  annotation_columns: list[str], suffix_column: str) -> OmicsData:
    table = read_table_after_header(settings.data_dir / config.REPORT_WORKBOOK, sheet, marker)
    # The lipidomics sheet repeats its header block before the negative-mode
    # lipids; drop those embedded label/header rows.
    table = table.loc[table[marker].notna() & table[marker].ne(marker)].reset_index(drop=True)
    samples, blanks, techmix = measurement_columns(table)
    index = _unique_names(table[marker], table[suffix_column])
    metadata = load_metadata(settings)
    values = _numeric(table, samples, index)
    values.columns = [sample_id(sample_number(c)) for c in values.columns]
    features = table.loc[:, annotation_columns].set_axis(index)
    return OmicsData(
        name=name,
        values=values.loc[:, metadata["Sample"]],
        features=features,
        metadata=metadata,
        blanks=_numeric(table, blanks, index),
        techmix=_numeric(table, techmix, index),
    )


def load_lipidomics(settings: Settings) -> OmicsData:
    """Lipid peak areas (positive and negative mode) with blanks and Tech-Mix."""
    return _vendor_omics(settings, "lipidomics", "Lipidomics", "Lipid Name",
                         ["Lipid Name", "Lipid Class", "Polarity"], "Polarity")


def load_metabolomics(settings: Settings) -> OmicsData:
    """Metabolite peak areas; ``features['Method']`` is the acquisition mode."""
    return _vendor_omics(settings, "metabolomics", "Metabolomics", "Metabolite",
                         ["Metabolite", "Metabolite ID", "Pathway", "Method"], "Method")


_PROTEOMICS_COLUMN = re.compile(
    r"^[^_]+_(?P<region>[^_]+)_(?P<group>MIA|control)_(?P<number>\d+) MaxLFQ Unique Intensity$",
    re.IGNORECASE,
)


def read_perseus_proteomics(path: Path, region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one Perseus export: log2 MaxLFQ values (already imputed) and gene annotation.

    Perseus adds ``#!`` type/condition rows under the header; they are skipped.
    """
    raw = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    intensity = {c: _PROTEOMICS_COLUMN.fullmatch(c) for c in raw.columns}
    intensity = {c: m for c, m in intensity.items() if m}
    if not intensity:
        raise ValueError(f"No MaxLFQ intensity columns in {path}.")
    wrong = {m.group("region") for m in intensity.values() if m.group("region").casefold() != region.casefold()}
    if wrong:
        raise ValueError(f"{path} should contain {region} samples, found {sorted(wrong)}.")
    ids = raw["Protein ID"].str.strip()
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f"Protein IDs are missing or duplicated in {path}.")
    values = raw.loc[:, list(intensity)].apply(pd.to_numeric, errors="raise")
    values.index = pd.Index(ids, name="Feature")
    values.columns = [sample_id(int(m.group("number"))) for m in intensity.values()]
    annotation = pd.DataFrame({"Protein ID": ids.to_numpy(), "Gene": raw["Gene"].str.strip().to_numpy()},
                              index=values.index)
    return values, annotation


def load_proteomics(settings: Settings) -> OmicsData:
    """Proteins detected in all three regions, as log2 MaxLFQ (already normalised and imputed)."""
    metadata = load_metadata(settings)
    blocks, annotations = [], []
    for region, filename in config.PROTEOMICS_FILES.items():
        values, annotation = read_perseus_proteomics(settings.data_dir / filename, region)
        blocks.append(values)
        annotations.append(annotation)
    shared = blocks[0].index
    for block in blocks[1:]:
        shared = shared.intersection(block.index, sort=False)
    values = pd.concat([block.loc[shared] for block in blocks], axis=1)
    features = annotations[0].loc[shared]
    missing = set(metadata["Sample"]).difference(values.columns)
    if missing:
        raise ValueError(f"Proteomics is missing samples: {sorted(missing)}")
    return OmicsData(name="proteomics", values=values.loc[:, metadata["Sample"]],
                     features=features, metadata=metadata)


def load(settings: Settings, omics: str) -> OmicsData:
    """Load one omics layer by name ('lipidomics', 'metabolomics' or 'proteomics')."""
    loaders = {"lipidomics": load_lipidomics, "metabolomics": load_metabolomics, "proteomics": load_proteomics}
    return loaders[omics](settings)


# ---------------------------------------------------------------- populations
def outlier_flags(settings: Settings, omics: str = "lipidomics") -> pd.DataFrame:
    """Hotelling T2 flags written by the outlier-screen stage (one row per sample)."""
    path = settings.results_dir / omics / "02_outliers" / "outlier_screen.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run stages.outliers.run_screen first.")
    return pd.read_csv(path)


def population_samples(settings: Settings, metadata: pd.DataFrame, population: str,
                       omics: str = "lipidomics") -> list[str]:
    """Samples kept in one of the permitted populations (see ``config.POPULATIONS``).

    Exclusion is per sample: the other regions of the same mouse are kept.
    """
    limit = config.POPULATIONS[population]
    if limit is None:
        return metadata["Sample"].tolist()
    flags = outlier_flags(settings, omics)
    flagged = set(flags.loc[flags[f"hotelling_t2_{int(limit * 100)}_flag"].astype(bool), "Sample"])
    return [s for s in metadata["Sample"] if s not in flagged]
