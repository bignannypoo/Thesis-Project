"""Load MRChondralHealth CSV and cartilage.statistics JSON from a timepoint folder."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cartilage_comparison.constants import (
    CARTILAGE_STATISTICS_FILENAME,
    CSV_COLUMN_ALIASES,
    JSON_METRIC_ALIASES,
    METRIC_COLUMNS,
    MRCH_CSV_GLOB,
    MRCH_DATETIME_FILENAME_PATTERN,
)

NA_VALUES = {"n/a", "na", "", "nan", "none"}


@dataclass(frozen=True)
class TimepointData:
    """Parsed metrics for one MRChondralHealth timepoint folder."""

    folder: Path
    study_datetime: str | None
    source: str  # "csv", "json", or "csv+json"
    metrics: pd.DataFrame


def _normalize_header(name: str) -> str:
    return name.strip().lower()


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert series to float, treating N/A strings as NaN."""
    as_string = series.astype(str).str.strip().str.lower()
    masked = series.mask(as_string.isin(NA_VALUES))
    return pd.to_numeric(masked, errors="coerce")


def _rename_csv_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for column in data_frame.columns:
        key = _normalize_header(str(column))
        renamed[column] = CSV_COLUMN_ALIASES.get(key, key.replace(" ", "_"))
    return data_frame.rename(columns=renamed)


def find_mrch_csv(folder: Path) -> Path:
    """Return the MRCH export CSV inside folder (searches recursively)."""
    matches = sorted(folder.rglob(MRCH_CSV_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No MRCH CSV matching '{MRCH_CSV_GLOB}' found under {folder}"
        )
    if len(matches) > 1:
        # Prefer shallowest path (typically the timepoint root).
        matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def find_cartilage_statistics(folder: Path) -> Path | None:
    """Return cartilage.statistics path if present."""
    direct = folder / CARTILAGE_STATISTICS_FILENAME
    if direct.is_file():
        return direct
    matches = sorted(folder.rglob(CARTILAGE_STATISTICS_FILENAME))
    if not matches:
        return None
    return matches[0]


def extract_study_datetime(folder: Path) -> str | None:
    """Extract YYYY-MM-DD HH:MM:SS from MRCH filenames in the folder."""
    pattern = re.compile(MRCH_DATETIME_FILENAME_PATTERN)
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        match = pattern.search(path.name)
        if match:
            date_part = match.group("date")
            time_part = match.group("time").replace("-", ":")
            return f"{date_part} {time_part}"
    return None


def load_mrch_csv(csv_path: Path) -> pd.DataFrame:
    """Load and normalize the MRCH_NOT_FOR_CLINICAL_USE CSV."""
    data_frame = pd.read_csv(csv_path)
    data_frame = _rename_csv_columns(data_frame)

    if "region" not in data_frame.columns or "layer" not in data_frame.columns:
        raise ValueError(
            f"CSV {csv_path} must include cartilage region and layer columns "
            f"(found: {list(data_frame.columns)})"
        )

    data_frame["region"] = data_frame["region"].astype(str).str.strip()
    data_frame["layer"] = data_frame["layer"].astype(str).str.strip()

    for column in METRIC_COLUMNS:
        if column in data_frame.columns:
            data_frame[column] = _coerce_numeric(data_frame[column])

    keep_columns = ["region", "layer", *[column for column in METRIC_COLUMNS if column in data_frame.columns]]
    return data_frame[keep_columns].copy()


def _normalize_json_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def _extract_metrics_from_leaf(leaf: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_key, raw_value in leaf.items():
        if isinstance(raw_value, dict):
            continue
        normalized_key = JSON_METRIC_ALIASES.get(_normalize_json_key(str(raw_key)))
        if normalized_key is None:
            continue
        try:
            metrics[normalized_key] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return metrics


def _flatten_statistics_node(
    node: Any,
    path_parts: list[str],
    rows: list[dict[str, object]],
) -> None:
    if isinstance(node, dict):
        if any(isinstance(value, (int, float)) for value in node.values()):
            metrics = _extract_metrics_from_leaf(node)
            if metrics:
                rows.append(
                    {
                        "region": " / ".join(path_parts[:-1]) if len(path_parts) > 1 else path_parts[0],
                        "layer": path_parts[-1] if path_parts else "combined",
                        **metrics,
                    }
                )
            return
        for key, child in node.items():
            _flatten_statistics_node(child, [*path_parts, str(key)], rows)
        return

    if isinstance(node, list):
        for item in node:
            _flatten_statistics_node(item, path_parts, rows)


def load_cartilage_statistics(json_path: Path) -> pd.DataFrame:
    """Flatten nested cartilage.statistics JSON into a metrics table."""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    _flatten_statistics_node(payload, [], rows)
    if not rows:
        raise ValueError(f"No metric rows parsed from {json_path}")

    data_frame = pd.DataFrame(rows)
    data_frame["region"] = data_frame["region"].astype(str).str.strip()
    data_frame["layer"] = data_frame["layer"].astype(str).str.strip()
    return data_frame


def _merge_region_layer_keys(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Build stable join keys from region and layer."""
    result = data_frame.copy()
    result["region_key"] = (
        result["region"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    result["layer_key"] = (
        result["layer"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    return result


def load_timepoint_folder(folder: str | Path, prefer_json: bool = False) -> TimepointData:
    """
    Load metrics from an MRChondralHealth timepoint directory.

    Uses CSV by default; falls back to cartilage.statistics when CSV is missing.
    When prefer_json is True, JSON is used when present (CSV still required unless
    only JSON exists).
    """
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Timepoint folder not found: {folder_path}")

    study_datetime = extract_study_datetime(folder_path)
    json_path = find_cartilage_statistics(folder_path)
    csv_path: Path | None = None
    try:
        csv_path = find_mrch_csv(folder_path)
    except FileNotFoundError:
        csv_path = None

    if prefer_json and json_path is not None:
        metrics = _merge_region_layer_keys(load_cartilage_statistics(json_path))
        source = "json"
    elif csv_path is not None:
        metrics = _merge_region_layer_keys(load_mrch_csv(csv_path))
        source = "csv"
        if json_path is not None:
            source = "csv+json"
    elif json_path is not None:
        metrics = _merge_region_layer_keys(load_cartilage_statistics(json_path))
        source = "json"
    else:
        raise FileNotFoundError(
            f"No MRCH CSV or {CARTILAGE_STATISTICS_FILENAME} found under {folder_path}"
        )

    return TimepointData(
        folder=folder_path,
        study_datetime=study_datetime,
        source=source,
        metrics=metrics,
    )


def load_timepoint_data(folder: str | Path, prefer_json: bool = False) -> TimepointData:
    """Alias for spec / minimal examples (same as ``load_timepoint_folder``)."""
    return load_timepoint_folder(folder, prefer_json=prefer_json)
