# Cartilage MRI Comparison Tool

Standalone tool for comparing **MRChondralHealth** knee cartilage metrics between two timepoints (e.g. pre/post treatment or longitudinal progression). Changes are shown **neutrally**: green = increase, red = decrease.

Not tied to CartiView or any other app in this repo.

## Requirements

- Python 3.10+
- See `cartilage_comparison/requirements.txt`

## Setup

```bash
cd "/path/to/Thesis Project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r cartilage_comparison/requirements.txt
```

## Run (interactive)

From the **project root** (`Thesis Project`):

```bash
streamlit run run_comparison.py
```

Alternative (same app):

```bash
streamlit run cartilage_comparison/main.py
```

Enter paths to two timepoint folders in the sidebar. Each folder should contain:

- `*_MRCH_NOT_FOR_CLINICAL_USE.csv` (required unless only JSON exists)
- `cartilage.statistics` (optional)

## Run (CLI)

```bash
python scripts/compare_mrch_timepoints.py \
  --timepoint1 /path/to/timepoint1 \
  --timepoint2 /path/to/timepoint2 \
  --output comparison_report \
  --figures \
  --pdf
```

Legacy flags `--pre` / `--post` still work.

In the Streamlit app, open the **Knee imaging** tab after running a comparison:

- Study report PDF diagrams (embedded images)
- NIfTI slice viewer with red/green change overlay on cartilage
- 3D `segmentation_mesh.vtk` surface overlay (pre vs post)
- Optional STL file paths (MRCH folders usually do not include STL; export separately)

## Outputs

| File | Description |
|------|-------------|
| `comparison_table.csv` | Side-by-side metrics, absolute & % change |
| `summary_statistics.txt` | Headline summary |
| `heatmap_*_percentage.png` | Volume / T2 / thickness (with `--figures`) |
| `comparison_report.pdf` | Full report (with `--pdf`) |

## Project layout

```
cartilage_comparison/
├── main.py              # Streamlit app
├── data_loader.py       # Load CSV / JSON from folders
├── analysis.py          # Change calculations & export
├── visualizations.py    # Heatmaps, bar charts, styled table
├── report_generator.py
├── regions.py           # Anatomical grid for heatmaps
├── asset_discovery.py   # Find NIfTI / VTK / PDF / STL in folders
├── knee_imaging.py      # Slice heatmaps, mesh overlay, PDF extract
├── knee_views.py        # Streamlit tab for knee imaging
├── constants.py
└── requirements.txt
```

## Tests

```bash
pytest tests/test_cartilage_comparison.py -q
```

## Privacy

Do not commit real patient data or hospital paths. Use `tests/fixtures/` for development.
