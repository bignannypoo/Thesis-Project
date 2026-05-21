"""Calculate absolute and percent changes between two timepoints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cartilage_comparison.constants import (
    DEFAULT_T2_MS_THRESHOLD,
    DEFAULT_THICKNESS_MM_THRESHOLD,
    DEFAULT_VOLUME_PERCENT_THRESHOLD,
)
from cartilage_comparison.data_loader import TimepointData

# Metrics that receive absolute and percent change columns in the output table.
PRIMARY_COMPARE_METRICS: tuple[tuple[str, str], ...] = (
    ("volume_ml", "volume"),
    ("bioch_mean", "t2_mean"),
    ("bioch_median", "t2_median"),
    ("thickness_mean", "thickness_mean"),
    ("voxel_count", "voxels"),
)


def _percent_change(pre_value: float, post_value: float) -> float | None:
    if pre_value is None or post_value is None:
        return None
    if np.isnan(pre_value) or np.isnan(post_value):
        return None
    if pre_value == 0:
        return None
    return ((post_value - pre_value) / pre_value) * 100.0


def _absolute_change(pre_value: float, post_value: float) -> float | None:
    if pre_value is None or post_value is None:
        return None
    if np.isnan(pre_value) or np.isnan(post_value):
        return None
    return post_value - pre_value


def _classify_change_direction(delta: float | None, threshold: float) -> str:
    """Neutral increase/decrease label (not clinical improvement)."""
    if delta is None or np.isnan(delta):
        return "n/a"
    if delta <= -threshold:
        return "decreased"
    if delta >= threshold:
        return "increased"
    return "stable"


def _is_significant_abs(delta: float | None, threshold: float) -> bool:
    if delta is None or np.isnan(delta):
        return False
    return abs(delta) >= threshold


def _is_significant_percent(pct: float | None, threshold_percent: float) -> bool:
    if pct is None or np.isnan(pct):
        return False
    return abs(pct) >= threshold_percent


def build_comparison_table(
    pre_data: TimepointData,
    post_data: TimepointData,
    *,
    volume_percent_threshold: float = DEFAULT_VOLUME_PERCENT_THRESHOLD,
    t2_ms_threshold: float = DEFAULT_T2_MS_THRESHOLD,
    thickness_mm_threshold: float = DEFAULT_THICKNESS_MM_THRESHOLD,
) -> pd.DataFrame:
    """
    Merge pre and post metrics and compute absolute/percent changes.

    Returns one row per (region, layer) present in either timepoint.
    """
    pre_metrics = pre_data.metrics
    post_metrics = post_data.metrics

    merged = pre_metrics.merge(
        post_metrics,
        on=["region_key", "layer_key"],
        how="outer",
        suffixes=("_pre", "_post"),
    )

    # Prefer non-null region/layer labels from whichever side exists.
    merged["region"] = merged["region_pre"].combine_first(merged["region_post"])
    merged["layer"] = merged["layer_pre"].combine_first(merged["layer_post"])

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        output_row: dict[str, object] = {
            "region": row["region"],
            "layer": row["layer"],
            "pre_study_datetime": pre_data.study_datetime,
            "post_study_datetime": post_data.study_datetime,
        }

        for metric_key, short_name in PRIMARY_COMPARE_METRICS:
            pre_col = f"{metric_key}_pre"
            post_col = f"{metric_key}_post"
            pre_value = float(row[pre_col]) if pre_col in merged.columns and pd.notna(row.get(pre_col)) else np.nan
            post_value = float(row[post_col]) if post_col in merged.columns and pd.notna(row.get(post_col)) else np.nan

            if np.isnan(pre_value) and np.isnan(post_value):
                continue

            output_row[f"pre_{short_name}"] = None if np.isnan(pre_value) else pre_value
            output_row[f"post_{short_name}"] = None if np.isnan(post_value) else post_value

            abs_delta = _absolute_change(pre_value, post_value)
            pct_delta = _percent_change(pre_value, post_value)
            output_row[f"delta_{short_name}"] = abs_delta
            if short_name in {"volume", "voxels", "t2_mean", "t2_median", "thickness_mean"}:
                output_row[f"pct_change_{short_name}"] = pct_delta

        # Clinical flags on primary endpoints.
        volume_pct = output_row.get("pct_change_volume")
        t2_delta = output_row.get("delta_t2_mean")
        thickness_delta = output_row.get("delta_thickness_mean")

        output_row["significant_volume_change"] = bool(
            _is_significant_percent(
                volume_pct if isinstance(volume_pct, (int, float)) else None,
                volume_percent_threshold,
            )
        )
        output_row["significant_t2_change"] = bool(
            _is_significant_abs(
                t2_delta if isinstance(t2_delta, (int, float)) else None,
                t2_ms_threshold,
            )
        )
        output_row["significant_thickness_change"] = bool(
            _is_significant_abs(
                thickness_delta if isinstance(thickness_delta, (int, float)) else None,
                thickness_mm_threshold,
            )
        )
        output_row["t2_change_direction"] = _classify_change_direction(
            t2_delta if isinstance(t2_delta, (int, float)) else None,
            t2_ms_threshold,
        )

        rows.append(output_row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    sort_columns = ["region", "layer"]
    return result.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def calculate_changes(
    timepoint_1: TimepointData,
    timepoint_2: TimepointData,
    **kwargs: object,
) -> pd.DataFrame:
    """Alias matching the development spec (same as ``build_comparison_table``)."""
    return build_comparison_table(timepoint_1, timepoint_2, **kwargs)


def summarize_overall_changes(comparison_table: pd.DataFrame) -> dict[str, float | int | None]:
    """Compute headline summary stats from a comparison table."""
    if comparison_table.empty:
        return {
            "region_layer_count": 0,
            "mean_volume_pct_change": None,
            "mean_t2_delta_ms": None,
            "decreased_t2_regions": 0,
            "increased_t2_regions": 0,
        }

    volume_pct = comparison_table["pct_change_volume"].dropna()
    t2_delta = comparison_table["delta_t2_mean"].dropna()
    direction = comparison_table.get("t2_change_direction", pd.Series(dtype=str))

    return {
        "region_layer_count": int(len(comparison_table)),
        "mean_volume_pct_change": float(volume_pct.mean()) if not volume_pct.empty else None,
        "mean_t2_delta_ms": float(t2_delta.mean()) if not t2_delta.empty else None,
        "decreased_t2_regions": int((direction == "decreased").sum()),
        "increased_t2_regions": int((direction == "increased").sum()),
    }


def write_comparison_outputs(
    comparison_table: pd.DataFrame,
    output_dir: str | Path,
    *,
    pre_data: TimepointData | None = None,
    post_data: TimepointData | None = None,
) -> Path:
    """Write comparison_table.csv and summary_statistics.txt under output_dir."""
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "comparison_table.csv"
    comparison_table.to_csv(csv_path, index=False)

    summary = summarize_overall_changes(comparison_table)
    lines: list[str] = [
        "MRChondralHealth Pre/Post Comparison Summary",
        "===========================================",
    ]
    if pre_data is not None:
        lines.append(f"Pre folder:  {pre_data.folder}")
        lines.append(f"Pre source:  {pre_data.source}")
        lines.append(f"Pre datetime: {pre_data.study_datetime or 'unknown'}")
    if post_data is not None:
        lines.append(f"Post folder: {post_data.folder}")
        lines.append(f"Post source: {post_data.source}")
        lines.append(f"Post datetime: {post_data.study_datetime or 'unknown'}")
    lines.extend(
        [
            "",
            f"Region/layer rows: {summary['region_layer_count']}",
            f"Mean volume % change: {summary['mean_volume_pct_change']}",
            f"Mean T2 (bioch mean) delta (ms): {summary['mean_t2_delta_ms']}",
            f"T2 decreased regions: {summary['decreased_t2_regions']}",
            f"T2 increased regions: {summary['increased_t2_regions']}",
        ]
    )
    summary_path = output_path / "summary_statistics.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path
