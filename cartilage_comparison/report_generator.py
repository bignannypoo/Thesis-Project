"""PDF report export for MRChondralHealth timepoint comparison."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

from cartilage_comparison.analysis import summarize_overall_changes
from cartilage_comparison.data_loader import TimepointData
from cartilage_comparison.visualizations import (
    METRIC_CONFIG,
    build_dashboard_summary,
    create_change_heatmap,
    create_side_by_side_bars,
    figure_to_png_bytes,
)


def create_pdf_report(
    comparison_table,
    output_path: str | Path,
    *,
    pre_data: TimepointData | None = None,
    post_data: TimepointData | None = None,
    layer_mode: str = "Combined",
    change_type: str = "percentage",
) -> Path:
    """
    Build a PDF with summary text, comparison highlights, and heatmap figures.

    Uses reportlab (already a project dependency).
    """
    import pandas as pd

    if not isinstance(comparison_table, pd.DataFrame):
        raise TypeError("comparison_table must be a pandas DataFrame")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    story: list[object] = []

    story.append(Paragraph("MRChondralHealth Cartilage Comparison Report", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))

    if pre_data is not None:
        story.append(
            Paragraph(
                f"Timepoint 1: {pre_data.study_datetime or 'unknown'} — {pre_data.folder}",
                styles["Normal"],
            )
        )
    if post_data is not None:
        story.append(
            Paragraph(
                f"Timepoint 2: {post_data.study_datetime or 'unknown'} — {post_data.folder}",
                styles["Normal"],
            )
        )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Methodology", styles["Heading2"]))
    story.append(
        Paragraph(
            "Changes are shown objectively: green indicates an increase and red a decrease "
            "relative to timepoint 1. Percent changes use ((T2 − T1) / T1) × 100. "
            "Near-zero cells (±2% for percentages) are unshaded in tables.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    dashboard = build_dashboard_summary(comparison_table)
    summary = summarize_overall_changes(comparison_table)

    story.append(Paragraph("Summary", styles["Heading2"]))
    summary_rows = [
        ["Metric", "Value"],
        ["Region/layer rows", str(summary.get("region_layer_count", 0))],
        [
            "Total volume T1 (ml)",
            _format_optional(dashboard.get("total_pre_volume_ml")),
        ],
        [
            "Total volume T2 (ml)",
            _format_optional(dashboard.get("total_post_volume_ml")),
        ],
        [
            "Total volume change",
            _format_delta(
                dashboard.get("total_volume_delta_ml"),
                dashboard.get("total_volume_pct_change"),
                unit="ml",
            ),
        ],
        [
            "Mean T2 T1 (ms)",
            _format_optional(dashboard.get("mean_pre_t2_ms")),
        ],
        [
            "Mean T2 T2 (ms)",
            _format_optional(dashboard.get("mean_post_t2_ms")),
        ],
        [
            "Mean volume % change (regions)",
            _format_optional(summary.get("mean_volume_pct_change")),
        ],
    ]
    table = Table(summary_rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Largest volume increases", styles["Heading3"]))
    story.extend(_bullet_list(dashboard.get("top_increases", []), styles))
    story.append(Paragraph("Largest volume decreases", styles["Heading3"]))
    story.extend(_bullet_list(dashboard.get("top_decreases", []), styles))
    story.append(Spacer(1, 0.2 * inch))

    for metric in ("volume", "t2", "thickness"):
        config = METRIC_CONFIG[metric]
        story.append(Paragraph(config["title"], styles["Heading2"]))
        heatmap_figure = create_change_heatmap(
            comparison_table,
            metric=metric,
            change_type=change_type,
            layer_mode=layer_mode,
        )
        png_bytes = figure_to_png_bytes(heatmap_figure)
        image = Image(BytesIO(png_bytes), width=6.5 * inch, height=3.0 * inch)
        story.append(image)
        story.append(Spacer(1, 0.15 * inch))

        bar_figure = create_side_by_side_bars(
            comparison_table,
            metric=metric,
            layer_mode=layer_mode,
            max_regions=8,
        )
        bar_png = figure_to_png_bytes(bar_figure)
        bar_image = Image(BytesIO(bar_png), width=6.5 * inch, height=3.5 * inch)
        story.append(bar_image)
        story.append(Spacer(1, 0.2 * inch))

    doc = SimpleDocTemplate(str(destination), pagesize=letter)
    doc.build(story)
    return destination


def _format_optional(value: object, *, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_delta(abs_value: object, pct_value: object, *, unit: str) -> str:
    abs_text = _format_optional(abs_value)
    if pct_value is None:
        return f"{abs_text} {unit}"
    return f"{abs_text} {unit} ({_format_optional(pct_value)}%)"


def _bullet_list(entries: list[dict[str, object]], styles) -> list[object]:
    blocks: list[object] = []
    if not entries:
        blocks.append(Paragraph("None", styles["Normal"]))
        return blocks
    for entry in entries:
        label = entry.get("label", "region")
        pct = entry.get("pct_change_volume")
        blocks.append(
            Paragraph(
                f"• {label}: {_format_optional(pct, precision=1)}% volume",
                styles["Normal"],
            )
        )
    return blocks
