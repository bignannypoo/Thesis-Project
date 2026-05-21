"""Heatmaps, bar charts, and styled tables for MRChondralHealth comparison."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Literal

# Writable cache for CI/sandbox environments before pyplot import.
_mpl_config_dir = Path(__file__).resolve().parent.parent / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
_mpl_config_dir.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from cartilage_comparison.regions import filter_comparison_by_layer, pivot_change_matrix

MetricKind = Literal["volume", "t2", "thickness"]
ChangeKind = Literal["absolute", "percentage"]

METRIC_CONFIG: dict[MetricKind, dict[str, str]] = {
    "volume": {
        "title": "Volume change",
        "delta_col": "delta_volume",
        "pct_col": "pct_change_volume",
        "pre_col": "pre_volume",
        "post_col": "post_volume",
        "unit_abs": "ml",
        "unit_pct": "%",
    },
    "t2": {
        "title": "T2 mapping change (bioch. mean)",
        "delta_col": "delta_t2_mean",
        "pct_col": "pct_change_t2_mean",
        "pre_col": "pre_t2_mean",
        "post_col": "post_t2_mean",
        "unit_abs": "ms",
        "unit_pct": "%",
    },
    "thickness": {
        "title": "Thickness change",
        "delta_col": "delta_thickness_mean",
        "pct_col": "pct_change_thickness_mean",
        "pre_col": "pre_thickness_mean",
        "post_col": "post_thickness_mean",
        "unit_abs": "mm",
        "unit_pct": "%",
    },
}

NEAR_ZERO_PCT = 2.0
COLOR_NEGATIVE = "#ffcccc"
COLOR_POSITIVE = "#ccffcc"
COLOR_NEUTRAL = "#ffffff"
DIVERGING_CMAP = plt.cm.RdYlGn


def _value_column(metric: MetricKind, change_type: ChangeKind) -> str:
    config = METRIC_CONFIG[metric]
    return config["pct_col"] if change_type == "percentage" else config["delta_col"]


def _symmetric_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
    limit = float(np.nanmax(np.abs(finite)))
    if limit == 0:
        limit = 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)


def _format_cell(value: float, change_type: ChangeKind, metric: MetricKind) -> str:
    if not np.isfinite(value):
        return ""
    config = METRIC_CONFIG[metric]
    if change_type == "percentage":
        return f"{value:+.1f}{config['unit_pct']}"
    return f"{value:+.2f}{config['unit_abs']}"


def create_change_heatmap(
    comparison_table: pd.DataFrame,
    *,
    metric: MetricKind = "volume",
    change_type: ChangeKind = "percentage",
    layer_mode: str = "Combined",
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """
    Anatomical heatmap with red (decrease) → white (no change) → green (increase).
    """
    filtered = filter_comparison_by_layer(comparison_table, layer_mode)
    value_column = _value_column(metric, change_type)
    matrix = pivot_change_matrix(filtered, value_column=value_column)

    figure, axis = plt.subplots(figsize=figsize)
    if matrix.empty:
        axis.text(0.5, 0.5, "No mappable regions for heatmap", ha="center", va="center")
        axis.set_axis_off()
        return figure

    values = matrix.to_numpy(dtype=float)
    norm = _symmetric_norm(values)
    image = axis.imshow(values, aspect="auto", cmap=DIVERGING_CMAP, norm=norm)

    axis.set_xticks(range(matrix.shape[1]))
    axis.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(matrix.shape[0]))
    axis.set_yticklabels(matrix.index, fontsize=9)
    axis.set_title(
        f"{METRIC_CONFIG[metric]['title']} — "
        f"{'percent' if change_type == 'percentage' else 'absolute'} change",
    )

    for row_index, row_name in enumerate(matrix.index):
        for col_index, col_name in enumerate(matrix.columns):
            cell_value = matrix.loc[row_name, col_name]
            if pd.isna(cell_value):
                continue
            axis.text(
                col_index,
                row_index,
                _format_cell(float(cell_value), change_type, metric),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02, label="Change")
    figure.tight_layout()
    return figure


def create_side_by_side_bars(
    comparison_table: pd.DataFrame,
    *,
    metric: MetricKind = "volume",
    layer_mode: str = "Combined",
    max_regions: int = 12,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Grouped pre/post bars colored by change direction."""
    config = METRIC_CONFIG[metric]
    filtered = filter_comparison_by_layer(comparison_table, layer_mode)
    required = [config["pre_col"], config["post_col"], config["delta_col"]]
    if filtered.empty or any(column not in filtered.columns for column in required):
        figure, axis = plt.subplots(figsize=figsize)
        axis.text(0.5, 0.5, "No data for bar chart", ha="center", va="center")
        axis.set_axis_off()
        return figure

    working = filtered.dropna(subset=[config["pre_col"], config["post_col"]]).copy()
    pct_col = config["pct_col"]
    if pct_col in working.columns:
        working["_sort_key"] = working[pct_col].abs()
    else:
        working["_sort_key"] = working[config["delta_col"]].abs()

    working = working.sort_values("_sort_key", ascending=False, kind="stable").head(max_regions)
    labels = [f"{row.region}\n({row.layer})" for row in working.itertuples()]

    pre_values = working[config["pre_col"]].astype(float).to_numpy()
    post_values = working[config["post_col"]].astype(float).to_numpy()
    deltas = working[config["delta_col"]].astype(float).to_numpy()

    x_positions = np.arange(len(labels))
    width = 0.35
    figure, axis = plt.subplots(figsize=figsize)
    axis.bar(x_positions - width / 2, pre_values, width, label="Timepoint 1", color="#9ecae1")
    bar_colors = [
        "#1a9850" if delta > 0 else "#d73027" if delta < 0 else "#bdbdbd"
        for delta in deltas
    ]
    axis.bar(x_positions + width / 2, post_values, width, label="Timepoint 2", color=bar_colors)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_title(config["title"])
    axis.set_ylabel(config["unit_abs"])
    axis.legend()
    figure.tight_layout()
    return figure


def _change_background(value: object, *, near_zero_pct: float = NEAR_ZERO_PCT) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if abs(numeric) <= near_zero_pct:
        return f"background-color: {COLOR_NEUTRAL}"
    if numeric > 0:
        return f"background-color: {COLOR_POSITIVE}"
    return f"background-color: {COLOR_NEGATIVE}"


def style_comparison_table(
    comparison_table: pd.DataFrame,
    *,
    near_zero_pct: float = NEAR_ZERO_PCT,
) -> pd.io.formats.style.Styler:
    """Color delta and percent columns (red decrease, green increase)."""
    display_columns = [
        column
        for column in comparison_table.columns
        if column.startswith(("delta_", "pct_change_"))
        or column in {"region", "layer", "pre_volume", "post_volume", "pre_t2_mean", "post_t2_mean", "pre_thickness_mean", "post_thickness_mean"}
    ]
    subset = comparison_table[display_columns] if display_columns else comparison_table

    def _row_style(row: pd.Series) -> list[str]:
        styles: list[str] = []
        for column in row.index:
            if column.startswith("pct_change_"):
                styles.append(_change_background(row[column], near_zero_pct=near_zero_pct))
            elif column.startswith("delta_"):
                styles.append(_change_background(row[column], near_zero_pct=0.0))
            else:
                styles.append("")
        return styles

    return subset.style.apply(_row_style, axis=1)


def build_dashboard_summary(comparison_table: pd.DataFrame) -> dict[str, object]:
    """Headline metrics and top increase/decrease regions for the dashboard."""
    summary: dict[str, object] = {
        "total_pre_volume_ml": None,
        "total_post_volume_ml": None,
        "total_volume_delta_ml": None,
        "total_volume_pct_change": None,
        "mean_pre_t2_ms": None,
        "mean_post_t2_ms": None,
        "mean_t2_delta_ms": None,
        "top_increases": [],
        "top_decreases": [],
        "mean_volume_pct_change": None,
        "std_volume_pct_change": None,
        "volume_pct_range": None,
    }
    if comparison_table.empty:
        return summary

    if "pre_volume" in comparison_table.columns:
        pre_vol = comparison_table["pre_volume"].dropna().astype(float)
        post_vol = comparison_table["post_volume"].dropna().astype(float)
        if not pre_vol.empty and not post_vol.empty:
            total_pre = float(pre_vol.sum())
            total_post = float(post_vol.sum())
            summary["total_pre_volume_ml"] = total_pre
            summary["total_post_volume_ml"] = total_post
            summary["total_volume_delta_ml"] = total_post - total_pre
            if total_pre != 0:
                summary["total_volume_pct_change"] = ((total_post - total_pre) / total_pre) * 100.0

    if "pre_t2_mean" in comparison_table.columns:
        pre_t2 = comparison_table["pre_t2_mean"].dropna().astype(float)
        post_t2 = comparison_table["post_t2_mean"].dropna().astype(float)
        if not pre_t2.empty and not post_t2.empty:
            summary["mean_pre_t2_ms"] = float(pre_t2.mean())
            summary["mean_post_t2_ms"] = float(post_t2.mean())
            summary["mean_t2_delta_ms"] = float(
                (comparison_table["delta_t2_mean"].dropna().astype(float).mean())
            )

    volume_pct = comparison_table.get("pct_change_volume", pd.Series(dtype=float)).dropna()
    if not volume_pct.empty:
        summary["mean_volume_pct_change"] = float(volume_pct.mean())
        summary["std_volume_pct_change"] = float(volume_pct.std())
        summary["volume_pct_range"] = (float(volume_pct.min()), float(volume_pct.max()))

    ranked = comparison_table.dropna(subset=["pct_change_volume"]).copy()
    if not ranked.empty:
        ranked = ranked.sort_values("pct_change_volume", ascending=False, kind="stable")
        top_n = 5

        def _region_label(row: pd.Series) -> str:
            return f"{row['region']} ({row['layer']})"

        increases = ranked.head(top_n)
        decreases = ranked.tail(top_n).sort_values("pct_change_volume", kind="stable")
        summary["top_increases"] = [
            {
                "label": _region_label(row),
                "pct_change_volume": float(row["pct_change_volume"]),
            }
            for _, row in increases.iterrows()
        ]
        summary["top_decreases"] = [
            {
                "label": _region_label(row),
                "pct_change_volume": float(row["pct_change_volume"]),
            }
            for _, row in decreases.iterrows()
        ]

    return summary


def save_figure_png(figure: Figure, path: str) -> None:
    """Persist a matplotlib figure to PNG."""
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def figure_to_png_bytes(figure: Figure) -> bytes:
    """Return PNG bytes for Streamlit ``st.image`` or PDF embedding."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return buffer.read()


def write_heatmap_figures(
    comparison_table: pd.DataFrame,
    output_dir: str,
    *,
    layer_mode: str = "Combined",
    change_type: ChangeKind = "percentage",
) -> list[str]:
    """Write volume, T2, and thickness heatmap PNGs."""
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for metric in ("volume", "t2", "thickness"):
        figure = create_change_heatmap(
            comparison_table,
            metric=metric,
            change_type=change_type,
            layer_mode=layer_mode,
        )
        file_path = output_path / f"heatmap_{metric}_{change_type}.png"
        save_figure_png(figure, str(file_path))
        written.append(str(file_path))

    return written
