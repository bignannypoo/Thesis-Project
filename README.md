# CartiView (thesis prototype)

Local Streamlit app for **pre vs post** cartilage imaging comparison — demo patients, session selection, image upload, and placeholder analysis tabs.

## Requirements

- Python 3.10+ recommended
- Dependencies in `requirements.txt`

## Setup

```bash
cd "/path/to/Thesis Project"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run cartiview.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Tests

```bash
pytest tests/
```

## Project layout

| Path | Role |
|------|------|
| `cartiview.py` | Thin entry — starts the app |
| `cv/` | Application package (data, UI, images, metrics) |
| `cv/theme.css` | Global UI styling |
| `tests/` | Pytest unit tests for data/metrics helpers |

## What is mock vs real

- **Patient list, sessions, segment numbers in the Segments tab:** demo data for UI development. Replace with your registry and **Chondral Quant** (or CSV) outputs when ready.
- **Images:** loaded in-memory from uploads or from folders on your machine (local use only).

## Privacy

Do not commit real patient data or hospital paths. Use synthetic or de-identified data for demos.
