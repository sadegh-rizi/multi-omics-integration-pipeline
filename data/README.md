# Data folder

Everything here is **simulated**. Recreate it with

```bash
python -m mia_multiomics.toy_data
```

The simulator (`src/mia_multiomics/toy_data.py`) writes files in the same
layout as the real core-facility deliverables. The readers in
`mia_multiomics.inputs` therefore work unchanged on the real files. To use real
data, put the files here (do not commit them) and set the file names and
`SAMPLE_PREFIX` in `src/mia_multiomics/config.py`.

| File | Layout the readers expect |
|---|---|
| `lipidomics_metabolomics_report.xlsx`, sheet **Sample Sheet** | Free-text lines, then a table whose header row contains `Sample ID`. Columns: `Sample number`, `Sample type` (region), `Mouse number` (`<cage> - <pup>`, or just the cage for single pups), `Bio group` (`Saline Control` / `MIA`), `Tissue mass (mg)`, `Sex`. |
| same workbook, sheet **Lipidomics** | 4 label rows (Sex, Bio Group, Sample Type, Sample Number), then a header row with `Lipid Name`, `Lipid Class`, `Polarity`, one column per sample (`<SAMPLE_PREFIX>-NN`), 6 blank injections (`...Blank...`), 6 pooled-QC injections (`Tech-Mix_k...`) and `%CV`. Negative-mode lipids follow after a repeated label and header block. |
| same workbook, sheet **Metabolomics** | Same idea; header row with `N.`, `Metabolite`, `Metabolite ID`, `Molecular formula`, `Pathway`, `m/z`, `r.t.`, `Method` (acquisition mode). Sample IDs are not zero padded here (`<SAMPLE_PREFIX>-2`). |
| `sample_cage_metadata.xlsx`, sheet **Samples to be shipped** | One row per sample: `Sample no`, `Tissue type`, `Weight`; the mouse-level columns `Date collected`, `Cage no`, `Age`, `Sex`, `MIA or Control`, `Date processed` are filled only on each mouse's first row. |
| `proteomics_<Region>.txt` (3 files) | Perseus export: tab-separated, intensity columns `<lab>_<Region>_<MIA or control>_<NN> MaxLFQ Unique Intensity` (log2, already imputed), `Protein ID`, `Gene`; `#!` annotation rows under the header. |
| `reference/*.gmt` | Gene sets, one per line: name, description, members. GO files use gene symbols; Reactome files use protein accessions. |

`TOY_DATA_README.txt` records the simulation seed and the planted ground truth.
