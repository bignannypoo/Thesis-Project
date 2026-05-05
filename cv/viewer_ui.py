"""Main viewer: header, session summary, images, overlay strip, analysis tabs."""

import io
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from cv.constants import (
    SCREEN_LOOKUP,
    STATUS_ACTIVE,
    VIEW_MODE_COMPARE,
    VIEW_MODE_POST,
    VIEW_MODE_PRE,
)
from cv.data import load_chondral_quant_csv
from cv.html_util import escape_html
from cv.image_utils import (
    fit_image_for_display,
    open_image_and_caption,
    render_image_load_section,
    render_image_pairs,
)
from cv.metrics import (
    StatisticalSummary,
    build_session_statistics_table,
    build_metric_strip_from_csv_data,
    build_statistical_summary_from_csv_data,
    format_metric_strip_html,
    get_segment_metrics_for_post_session,
)
from cv.models import ImagingSession, PatientRecord
from cv.session_sync import resolve_pre_post_sessions
from cv.u3d_import import convert_u3d_to_stl_with_meshlab, is_meshlab_available, summarize_u3d_file
from cv.lookup_ui import placeholder_panel
from cv.pdf_import import import_pdf_report
from cv.viewer_3d import (
    build_fusion_figure,
    build_mesh_comparison_summary,
    build_overlay_figure,
    load_fusion_points,
    load_stl_mesh,
)

NOT_FOUND_LABEL = "Not found"


def _get_snapshot_store() -> list[dict[str, object]]:
    """Return mutable snapshot list from session state."""
    if "cv_snapshots" not in st.session_state:
        st.session_state.cv_snapshots = []
    return st.session_state.cv_snapshots


def _pil_image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode PIL image as PNG bytes for session storage."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _capture_snapshot(
    patient: PatientRecord,
    pre_session: ImagingSession,
    post_session: ImagingSession,
    pair_index: int,
    map_mode: str,
    note: str,
    map_image: Image.Image,
) -> None:
    """Save current map state as a snapshot in session."""
    snapshot_store = _get_snapshot_store()
    snapshot_store.append(
        {
            "patient_id": patient.patient_id,
            "patient_name": patient.display_name,
            "mrn": patient.mrn,
            "pre_date": pre_session.date,
            "post_date": post_session.date,
            "pair_index": pair_index,
            "map_mode": map_mode,
            "note": note.strip() or "No note provided.",
            "image_bytes": _pil_image_to_png_bytes(map_image),
        }
    )


def _draw_pdf_snapshots(
    pdf: canvas.Canvas,
    snapshots: list[dict[str, object]],
    y_start: float,
    draw_header_callback,
) -> float:
    """Draw snapshot section and return resulting y offset."""
    y_position = y_start
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, y_position, "Snapshots")
    y_position -= 18

    if not snapshots:
        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, y_position, "No snapshots captured yet.")
        return y_position

    for index, snapshot in enumerate(snapshots, start=1):
        if y_position < 140:
            pdf.showPage()
            y_position = draw_header_callback()
            pdf.setFont("Helvetica-Bold", 11)
            pdf.drawString(50, y_position, "Snapshots (continued)")
            y_position -= 18

        image_reader = io.BytesIO(snapshot["image_bytes"])  # type: ignore[index]
        image = Image.open(image_reader).convert("RGB")
        image.thumbnail((220, 140), Image.Resampling.LANCZOS)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(50, y_position, f"Snapshot {index} · Pair {int(snapshot['pair_index']) + 1}")
        pdf.setFont("Helvetica", 9)
        pdf.drawString(50, y_position - 14, f"Mode: {snapshot['map_mode']}")
        note_text = str(snapshot["note"])
        pdf.drawString(50, y_position - 28, f"Note: {note_text[:85]}")
        pdf.drawInlineImage(Image.open(image_bytes), 320, y_position - 95, width=180, height=110)
        y_position -= 125
    return y_position


def _build_report_pdf(
    patient: PatientRecord,
    pre_session: ImagingSession,
    post_session: ImagingSession,
    snapshots: list[dict[str, object]],
    trend_chart_bytes: bytes | None,
    stats_summary: StatisticalSummary | None,
) -> bytes:
    """Generate compact PDF report bytes from current viewer context."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    _, height = letter

    def draw_header() -> float:
        y_top = height - 50
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y_top, "CartiView Comparison Report")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, y_top - 20, f"Patient: {patient.display_name} (MRN {patient.mrn})")
        pdf.drawString(50, y_top - 35, f"Joint: {patient.joint}")
        pdf.drawString(50, y_top - 50, f"Comparison: {pre_session.date} -> {post_session.date}")
        return y_top - 80

    def draw_stats_section(current_y: float, summary: StatisticalSummary) -> float:
        if current_y < 200:
            pdf.showPage()
            current_y = draw_header()
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(50, current_y, "Statistical Summary")
        pdf.setFont("Helvetica", 9)
        pdf.drawString(
            50,
            current_y - 16,
            f"Volume mean delta: {summary.mean_volume_delta_percent:+.2f}% "
            f"(95% CI {summary.volume_ci_low:+.2f} to {summary.volume_ci_high:+.2f}), "
            f"p={summary.volume_p_value:.4f} "
            f"({'significant' if summary.is_volume_significant else 'not significant'})",
        )
        pdf.drawString(
            50,
            current_y - 30,
            f"Thickness mean delta: {summary.mean_thickness_delta_mm:+.3f} mm "
            f"(95% CI {summary.thickness_ci_low:+.3f} to {summary.thickness_ci_high:+.3f}), "
            f"p={summary.thickness_p_value:.4f} "
            f"({'significant' if summary.is_thickness_significant else 'not significant'})",
        )
        return current_y - 48

    def draw_trend_section(current_y: float, chart_bytes: bytes) -> float:
        if current_y < 280:
            pdf.showPage()
            current_y = draw_header()
        trend_image = Image.open(io.BytesIO(chart_bytes)).convert("RGB")
        trend_image.thumbnail((500, 220), Image.Resampling.LANCZOS)
        trend_buffer = io.BytesIO()
        trend_image.save(trend_buffer, format="PNG")
        trend_buffer.seek(0)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(50, current_y, "Volume Trend")
        pdf.drawInlineImage(Image.open(trend_buffer), 50, current_y - 190, width=500, height=170)
        return current_y - 210

    y = draw_header()

    if stats_summary:
        y = draw_stats_section(y, stats_summary)

    if trend_chart_bytes:
        y = draw_trend_section(y, trend_chart_bytes)

    _draw_pdf_snapshots(pdf, snapshots, y, draw_header)

    pdf.save()
    return buffer.getvalue()


def _build_trend_chart_png(trends_data_frame: pd.DataFrame) -> bytes | None:
    """Render a lightweight trend chart image for PDF embedding."""
    if trends_data_frame.empty:
        return None

    working_data_frame = trends_data_frame.copy()
    working_data_frame["parsed_date"] = _parse_display_date(working_data_frame["date"])
    working_data_frame = working_data_frame.dropna(subset=["parsed_date"])
    if working_data_frame.empty:
        return None

    chart_width = 900
    chart_height = 360
    left_margin = 70
    right_margin = 30
    top_margin = 35
    bottom_margin = 50
    plot_width = chart_width - left_margin - right_margin
    plot_height = chart_height - top_margin - bottom_margin

    image = Image.new("RGB", (chart_width, chart_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (left_margin, top_margin, left_margin + plot_width, top_margin + plot_height),
        outline=(180, 180, 180),
        width=1,
    )

    all_dates = sorted(working_data_frame["parsed_date"].unique())
    volume_min = float(working_data_frame["volume"].min())
    volume_max = float(working_data_frame["volume"].max())
    volume_range = max(volume_max - volume_min, 1.0)
    date_span = max(len(all_dates) - 1, 1)

    segment_colors = {
        "medial": (0, 120, 255),
        "lateral": (255, 120, 0),
        "patellar": (20, 150, 90),
    }
    legend_y = top_margin + 5
    legend_x = left_margin + 8

    for segment_name, segment_data_frame in working_data_frame.groupby("segment"):
        ordered_segment = segment_data_frame.sort_values("parsed_date")
        points: list[tuple[float, float]] = []
        for _, row in ordered_segment.iterrows():
            date_idx = all_dates.index(row["parsed_date"])
            x = left_margin + (date_idx / date_span) * plot_width
            y = top_margin + plot_height - ((float(row["volume"]) - volume_min) / volume_range) * plot_height
            points.append((x, y))
        color = segment_colors.get(str(segment_name).lower(), (120, 120, 120))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for point in points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        draw.rectangle((legend_x, legend_y, legend_x + 12, legend_y + 12), fill=color)
        draw.text((legend_x + 16, legend_y - 1), str(segment_name).title(), fill=(40, 40, 40))
        legend_x += 130

    draw.text((left_margin, chart_height - 22), "Session Date", fill=(60, 60, 60))
    draw.text((10, top_margin + 5), "Volume", fill=(60, 60, 60))
    draw.text((left_margin, 8), "Volume Trend by Segment", fill=(20, 20, 20))

    png_buffer = io.BytesIO()
    image.save(png_buffer, format="PNG")
    return png_buffer.getvalue()


def render_viewer_header(patient: PatientRecord) -> None:
    """Patient identity bar."""
    status_badge = (
        '<span class="cv-pr-badge cv-badge-active" style="margin-left:10px;">Active follow-up</span>'
        if patient.status == STATUS_ACTIVE
        else f'<span class="cv-pr-badge cv-badge-done" style="margin-left:10px;">{escape_html(patient.status)}</span>'
    )
    st.markdown(
        f'<div class="cv-viewer-header">'
        f'<div class="cv-vh-patient">'
        f"{escape_html(patient.display_name)}"
        f'<span class="cv-vh-mrn">MRN {escape_html(patient.mrn)} · {escape_html(patient.joint)}</span>'
        f"{status_badge}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def render_viewer_session_summary(
    pre_session: ImagingSession,
    post_session: ImagingSession,
) -> None:
    """Two-card summary of selected baseline vs follow-up."""
    st.markdown(
        f'<div class="cv-session-summary">'
        f"<div>"
        f'<div class="cv-ss-col-title">Baseline · Pre-treatment</div>'
        f'<div class="cv-card">'
        f'<div class="cv-card-date">{escape_html(pre_session.date)}</div>'
        f'<div class="cv-card-mod">{escape_html(pre_session.modality)}</div>'
        f'<span class="cv-card-tag">{escape_html(pre_session.label)}</span>'
        f"</div></div>"
        f"<div>"
        f'<div class="cv-ss-col-title">Comparison · Post-treatment</div>'
        f'<div class="cv-card cv-post-active">'
        f'<div class="cv-card-date">{escape_html(post_session.date)}</div>'
        f'<div class="cv-card-mod">{escape_html(post_session.modality)}</div>'
        f'<span class="cv-card-tag">{escape_html(post_session.label)}</span>'
        f"</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_viewer_overlay_strip() -> None:
    """Overlay chips (non-interactive placeholders)."""
    st.markdown(
        """
<div class="cv-overlay-strip">
  <span class="cv-ov-label">Overlay</span>
  <span class="cv-ov-chip">○ Segmentation</span>
  <span class="cv-ov-chip">○ Heat map</span>
  <span class="cv-ov-chip">○ Anatomy fusion</span>
  <span class="cv-ov-sep"></span>
  <span class="cv-ov-opacity">Opacity 70%</span>
  <span class="cv-ov-badge">Coming soon</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_viewer_metric_strip(post_session: ImagingSession) -> None:
    """Metrics from demo lookup table keyed by post session id."""
    csv_metrics_data_frame = st.session_state.get("cv_metrics_data_frame")
    baseline_session_id = st.session_state.get("cv_pre_session_id")
    stats_summary: StatisticalSummary | None = None

    if isinstance(csv_metrics_data_frame, pd.DataFrame) and isinstance(baseline_session_id, str):
        metrics = build_metric_strip_from_csv_data(
            csv_metrics_data_frame,
            baseline_session_id=baseline_session_id,
            followup_session_id=post_session.session_id,
        )
        stats_summary = build_statistical_summary_from_csv_data(
            csv_metrics_data_frame,
            baseline_session_id=baseline_session_id,
            followup_session_id=post_session.session_id,
        )
    else:
        metrics = get_segment_metrics_for_post_session(post_session)

    st.session_state.cv_stats_summary = stats_summary
    st.markdown(format_metric_strip_html(metrics), unsafe_allow_html=True)
    st.markdown(
        f'<p class="cv-interval-caption">Interval: {escape_html(post_session.label)} vs baseline</p>',
        unsafe_allow_html=True,
    )
    if stats_summary:
        volume_significance = "significant" if stats_summary.is_volume_significant else "not significant"
        thickness_significance = "significant" if stats_summary.is_thickness_significant else "not significant"
        st.caption(
            f"Stats across {stats_summary.sample_size} segment(s): "
            f"volume p={stats_summary.volume_p_value:.4f} ({volume_significance}), "
            f"thickness p={stats_summary.thickness_p_value:.4f} ({thickness_significance})"
        )


def render_viewer_tools_row() -> None:
    """Tool chip row."""
    st.markdown(
        """
<div class="cv-tools-row">
  <span>Segment overlay</span>
  <span>Ruler</span>
  <span>Crop snapshot</span>
  <span>Annotate</span>
  <span>Export report <em style="font-style:normal;color:#94a3b8;">(soon)</em></span>
</div>
""",
        unsafe_allow_html=True,
    )


def _build_demo_trends_data_frame(patient: PatientRecord) -> pd.DataFrame:
    """Create deterministic longitudinal demo data for trend charts."""
    segments = ("medial", "lateral", "patellar")
    rows: list[dict[str, object]] = []
    ordered_sessions = list(patient.sessions)
    for session_index, session in enumerate(ordered_sessions):
        for segment_index, segment in enumerate(segments):
            rows.append(
                {
                    "session_id": session.session_id,
                    "date": session.date,
                    "segment": segment,
                    "volume": 1180 - (session_index * 42) + (segment_index * 15),
                    "thickness": 2.9 - (session_index * 0.08) + (segment_index * 0.05),
                    "t2_relaxation": 43 - (session_index * 0.6) + (segment_index * 0.2),
                }
            )
    return pd.DataFrame(rows)


def _parse_display_date(series: pd.Series) -> pd.Series:
    """Parse human-readable date strings used by demo sessions."""
    return pd.to_datetime(series, format="%d %b %Y", errors="coerce")


def _add_segment_traces(fig: go.Figure, trends_data_frame: pd.DataFrame) -> None:
    """Add per-segment trend lines to a figure."""
    for segment_name, segment_data_frame in trends_data_frame.groupby("segment"):
        fig.add_trace(
            go.Scatter(
                x=segment_data_frame["date"],
                y=segment_data_frame["volume"],
                name=str(segment_name).title(),
                mode="lines+markers",
            )
        )


def _add_session_statistics_traces(fig: go.Figure, session_stats: pd.DataFrame) -> None:
    """Add CI ribbon and mean trend traces to a figure."""
    if session_stats.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=session_stats["date"],
            y=session_stats["volume_ci_high"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=session_stats["date"],
            y=session_stats["volume_ci_low"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(100, 149, 237, 0.15)",
            line={"width": 0},
            name="95% CI (across segments)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=session_stats["date"],
            y=session_stats["volume_mean"],
            mode="lines+markers",
            line={"dash": "dash", "color": "black"},
            name="Mean volume",
        )
    )


def _render_trends_statistics_panel(trends_data_frame: pd.DataFrame, session_stats: pd.DataFrame) -> None:
    """Render table and baseline-vs-followup significance caption."""
    st.markdown("**Session statistics (volume)**")
    st.dataframe(
        session_stats.rename(
            columns={
                "segment_count": "segments",
                "volume_mean": "mean",
                "volume_ci_low": "ci_low",
                "volume_ci_high": "ci_high",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    baseline_session_id = st.session_state.get("cv_pre_session_id")
    followup_session_id = st.session_state.get("cv_post_session_id")
    if not (isinstance(baseline_session_id, str) and isinstance(followup_session_id, str)):
        return
    summary = build_statistical_summary_from_csv_data(
        trends_data_frame,
        baseline_session_id=baseline_session_id,
        followup_session_id=followup_session_id,
    )
    if summary:
        st.caption(
            "Baseline vs selected follow-up significance: "
            f"volume p={summary.volume_p_value:.4f} "
            f"({'significant' if summary.is_volume_significant else 'not significant'}), "
            f"thickness p={summary.thickness_p_value:.4f} "
            f"({'significant' if summary.is_thickness_significant else 'not significant'})."
        )


def render_trends_tab(patient: PatientRecord) -> None:
    """Render longitudinal trends from uploaded CSV or demo fallback."""
    uploaded_file = st.file_uploader(
        "Upload metrics CSV (optional)",
        type=["csv"],
        key="cv_metrics_csv",
        help="Expected columns: session_id, date, segment, volume, thickness, t2_relaxation",
    )

    if uploaded_file is not None:
        try:
            metrics_data_frame = load_chondral_quant_csv(uploaded_file)
            st.session_state.cv_metrics_data_frame = metrics_data_frame
            st.success("Metrics CSV loaded. Trends and segment strip now use uploaded data.")
        except ValueError as error:
            st.error(str(error))
            return

    metrics_data_frame = st.session_state.get("cv_metrics_data_frame")
    if not isinstance(metrics_data_frame, pd.DataFrame):
        metrics_data_frame = _build_demo_trends_data_frame(patient)
        st.caption("Showing demo trend data. Upload CSV above to use real values.")

    trends_data_frame = metrics_data_frame.copy()
    trends_data_frame["parsed_date"] = _parse_display_date(trends_data_frame["date"])
    trends_data_frame = trends_data_frame.sort_values(["parsed_date", "segment"])
    st.session_state.cv_report_trends_data_frame = trends_data_frame[
        ["date", "segment", "volume"]
    ].copy()

    if trends_data_frame["parsed_date"].isna().all():
        st.warning("Could not parse dates for trend chart.")
        return

    fig = go.Figure()
    _add_segment_traces(fig, trends_data_frame)

    session_stats = build_session_statistics_table(trends_data_frame[["session_id", "date", "segment", "volume"]])
    session_stats = session_stats.sort_values("date")
    _add_session_statistics_traces(fig, session_stats)
    fig.update_layout(
        title="Volume Trend by Segment",
        xaxis_title="Session Date",
        yaxis_title="Volume",
        legend_title="Segment",
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
    )
    st.plotly_chart(fig, use_container_width=True)
    _render_trends_statistics_panel(trends_data_frame, session_stats)


def _prepare_grayscale_pair(pre_image: Image.Image, post_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Resize and convert two images into aligned grayscale arrays."""
    pre_rgb = pre_image.convert("RGB")
    post_rgb = post_image.convert("RGB")

    width = min(pre_rgb.width, post_rgb.width)
    height = min(pre_rgb.height, post_rgb.height)
    target_size = (width, height)

    aligned_pre = pre_rgb.resize(target_size, Image.Resampling.LANCZOS).convert("L")
    aligned_post = post_rgb.resize(target_size, Image.Resampling.LANCZOS).convert("L")
    return np.array(aligned_pre, dtype=np.float32), np.array(aligned_post, dtype=np.float32)


def generate_difference_heatmap(
    pre_image: Image.Image,
    post_image: Image.Image,
    threshold: float,
    alpha: float,
) -> Image.Image:
    """Build red/green heatmap where green means improvement and red means degradation."""
    pre_array, post_array = _prepare_grayscale_pair(pre_image, post_image)
    normalized_diff = (post_array - pre_array) / 255.0

    heatmap_rgb = np.zeros((normalized_diff.shape[0], normalized_diff.shape[1], 3), dtype=np.uint8)
    heatmap_rgb[normalized_diff > threshold] = [0, 220, 70]
    heatmap_rgb[normalized_diff < -threshold] = [220, 45, 45]

    base_image = Image.fromarray(post_array.astype(np.uint8), mode="L").convert("RGB")
    heatmap_image = Image.fromarray(heatmap_rgb, mode="RGB")
    blended_image = Image.blend(base_image, heatmap_image, alpha=alpha)
    return fit_image_for_display(blended_image)


def _validate_mask_data_frame(mask_data_frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize mask CSV columns for segment-constrained map mode."""
    normalized_columns = [column.strip().lower() for column in mask_data_frame.columns]
    mask_data_frame.columns = normalized_columns

    required_columns = {"segment", "x", "y"}
    missing_columns = sorted(required_columns - set(mask_data_frame.columns))
    if missing_columns:
        raise ValueError("Mask CSV missing required columns: " + ", ".join(missing_columns))

    mask_data_frame["x"] = pd.to_numeric(mask_data_frame["x"], errors="coerce")
    mask_data_frame["y"] = pd.to_numeric(mask_data_frame["y"], errors="coerce")
    if mask_data_frame[["x", "y"]].isna().any().any():
        raise ValueError("Mask CSV has invalid x/y values.")

    if "pair_index" in mask_data_frame.columns:
        mask_data_frame["pair_index"] = pd.to_numeric(mask_data_frame["pair_index"], errors="coerce")
        if mask_data_frame["pair_index"].isna().any():
            raise ValueError("Mask CSV has invalid pair_index values.")

    return mask_data_frame


def _build_segment_masks(
    mask_data_frame: pd.DataFrame,
    image_width: int,
    image_height: int,
    pair_index: int,
) -> dict[str, np.ndarray]:
    """Build boolean masks keyed by segment name from mask points."""
    working_data_frame = mask_data_frame
    if "pair_index" in working_data_frame.columns:
        working_data_frame = working_data_frame[working_data_frame["pair_index"] == pair_index + 1]

    segment_masks: dict[str, np.ndarray] = {}
    for segment_name, segment_data_frame in working_data_frame.groupby("segment"):
        segment_mask = np.zeros((image_height, image_width), dtype=bool)
        clipped_x = np.clip(segment_data_frame["x"].round().astype(int), 0, image_width - 1)
        clipped_y = np.clip(segment_data_frame["y"].round().astype(int), 0, image_height - 1)
        segment_mask[clipped_y, clipped_x] = True
        segment_masks[str(segment_name).lower()] = segment_mask
    return segment_masks


def _build_heatmap_with_mask(
    pre_image: Image.Image,
    post_image: Image.Image,
    threshold: float,
    alpha: float,
    active_mask: np.ndarray,
) -> Image.Image:
    """Generate heatmap constrained to active mask pixels."""
    pre_array, post_array = _prepare_grayscale_pair(pre_image, post_image)
    normalized_diff = (post_array - pre_array) / 255.0

    heatmap_rgb = np.zeros((normalized_diff.shape[0], normalized_diff.shape[1], 3), dtype=np.uint8)
    improved_mask = (normalized_diff > threshold) & active_mask
    degraded_mask = (normalized_diff < -threshold) & active_mask
    heatmap_rgb[improved_mask] = [0, 220, 70]
    heatmap_rgb[degraded_mask] = [220, 45, 45]

    base_image = Image.fromarray(post_array.astype(np.uint8), mode="L").convert("RGB")
    heatmap_image = Image.fromarray(heatmap_rgb, mode="RGB")
    return fit_image_for_display(Image.blend(base_image, heatmap_image, alpha=alpha))


def _build_segment_statistics(
    pre_image: Image.Image,
    post_image: Image.Image,
    segment_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Calculate per-segment delta summary."""
    pre_array, post_array = _prepare_grayscale_pair(pre_image, post_image)
    normalized_diff = (post_array - pre_array) / 255.0

    rows: list[dict[str, object]] = []
    for segment_name, segment_mask in segment_masks.items():
        active_pixels = int(segment_mask.sum())
        if active_pixels == 0:
            continue
        segment_values = normalized_diff[segment_mask]
        improved_ratio = float((segment_values > 0).sum()) / active_pixels
        degraded_ratio = float((segment_values < 0).sum()) / active_pixels
        rows.append(
            {
                "segment": segment_name.title(),
                "pixels": active_pixels,
                "mean_delta": round(float(segment_values.mean()), 4),
                "improved_ratio": round(improved_ratio, 3),
                "degraded_ratio": round(degraded_ratio, 3),
            }
        )
    return pd.DataFrame(rows)


def _build_segment_constrained_output(
    pre_image: Image.Image,
    post_image: Image.Image,
    threshold: float,
    alpha: float,
    selected_pair: int,
) -> tuple[Image.Image, pd.DataFrame | None]:
    """Return constrained heatmap and optional stats from uploaded mask CSV."""
    default_heatmap = generate_difference_heatmap(pre_image, post_image, threshold, alpha)
    mask_file = st.file_uploader(
        "Upload mask CSV",
        type=["csv"],
        key="cv_mask_csv",
        help="Columns: segment, x, y, optional pair_index (1-based).",
    )
    if mask_file is None:
        st.info("Upload a mask CSV to enable segment-constrained visualization.")
        return default_heatmap, None

    try:
        mask_data_frame = _validate_mask_data_frame(pd.read_csv(mask_file))
        pre_array, _ = _prepare_grayscale_pair(pre_image, post_image)
        image_height, image_width = pre_array.shape
        segment_masks = _build_segment_masks(mask_data_frame, image_width, image_height, selected_pair)
        if not segment_masks:
            st.warning("No mask points found for the selected pair. Falling back to global diff.")
            return default_heatmap, None

        active_mask = np.zeros((image_height, image_width), dtype=bool)
        for mask in segment_masks.values():
            active_mask |= mask
        constrained_heatmap = _build_heatmap_with_mask(pre_image, post_image, threshold, alpha, active_mask)
        segment_stats = _build_segment_statistics(pre_image, post_image, segment_masks)
        return constrained_heatmap, segment_stats
    except ValueError as error:
        st.error(str(error))
        return default_heatmap, None


def render_improvement_map_tab(
    patient: PatientRecord,
    pre_session: ImagingSession,
    post_session: ImagingSession,
    pre_list: list,
    post_list: list,
) -> None:
    """Render map tab with interactive pair selection and heatmap controls."""
    if not pre_list or not post_list:
        placeholder_panel(
            "Improvement map",
            "Upload both pre and post images to generate a difference heat map.",
        )
        return

    pair_count = min(len(pre_list), len(post_list))
    selected_pair = st.selectbox(
        "Image pair",
        options=list(range(pair_count)),
        format_func=lambda index: f"Pair {index + 1}",
        key="cv_map_pair",
    )
    threshold = st.slider(
        "Difference threshold",
        min_value=0.01,
        max_value=0.40,
        value=0.10,
        step=0.01,
        key="cv_map_threshold",
        help="Lower threshold reveals subtle change; higher threshold highlights stronger differences.",
    )
    alpha = st.slider(
        "Heatmap opacity",
        min_value=0.10,
        max_value=1.00,
        value=0.65,
        step=0.05,
        key="cv_map_alpha",
    )
    map_mode = st.radio(
        "Map mode",
        options=["Global diff", "Segment-constrained diff"],
        horizontal=True,
        key="cv_map_mode",
    )

    try:
        pre_image, pre_caption = open_image_and_caption(pre_list[selected_pair])
        post_image, post_caption = open_image_and_caption(post_list[selected_pair])
    except (OSError, ValueError) as error:
        st.error(f"Could not open selected pair: {error}")
        return

    heatmap_image = generate_difference_heatmap(pre_image, post_image, threshold, alpha)
    segment_stats_data_frame: pd.DataFrame | None = None

    if map_mode == "Segment-constrained diff":
        heatmap_image, segment_stats_data_frame = _build_segment_constrained_output(
            pre_image,
            post_image,
            threshold,
            alpha,
            selected_pair,
        )

    pre_col, post_col, map_col = st.columns(3, gap="large")
    with pre_col:
        st.markdown("**Pre-treatment**")
        st.image(fit_image_for_display(pre_image.convert("RGB")), caption=pre_caption, use_container_width=True)
    with post_col:
        st.markdown("**Post-treatment**")
        st.image(fit_image_for_display(post_image.convert("RGB")), caption=post_caption, use_container_width=True)
    with map_col:
        st.markdown("**Improvement / Degradation Map**")
        st.image(heatmap_image, caption="Green = improvement · Red = degradation", use_container_width=True)

    st.caption(
        "Map is grayscale intensity-based and intended as a visual aid. "
        "Use registered segment data for clinically meaningful interpretation."
    )
    if segment_stats_data_frame is not None and not segment_stats_data_frame.empty:
        st.markdown("**Segment stats**")
        st.dataframe(segment_stats_data_frame, use_container_width=True, hide_index=True)

    snapshot_note = st.text_input(
        "Snapshot note",
        key="cv_snapshot_note",
        placeholder="e.g. Medial region shows visible recovery in pair 2",
    )
    if st.button("Capture snapshot", key="cv_capture_snapshot"):
        _capture_snapshot(
            patient=patient,
            pre_session=pre_session,
            post_session=post_session,
            pair_index=selected_pair,
            map_mode=map_mode,
            note=snapshot_note,
            map_image=heatmap_image,
        )
        st.success("Snapshot captured.")


def render_snapshots_tab(patient: PatientRecord) -> None:
    """List and preview captured snapshots for the active patient."""
    snapshot_store = _get_snapshot_store()
    patient_snapshots = [item for item in snapshot_store if item["patient_id"] == patient.patient_id]

    if not patient_snapshots:
        placeholder_panel("Snapshots", "No snapshots yet. Capture from the Map tab.")
        return

    st.caption(f"Saved snapshots: {len(patient_snapshots)}")
    for index, snapshot in enumerate(patient_snapshots, start=1):
        with st.container(border=True):
            st.markdown(
                f"**Snapshot {index}** · Pair {int(snapshot['pair_index']) + 1} · {snapshot['map_mode']}"
            )
            st.caption(f"Comparison: {snapshot['pre_date']} → {snapshot['post_date']}")
            st.write(str(snapshot["note"]))
            st.image(snapshot["image_bytes"], use_container_width=True)


def _render_pdf_import_section(patient: PatientRecord) -> None:
    """Render PDF import tools for extracting report context."""
    st.divider()
    st.markdown("### PDF Import")
    st.caption("Import an existing clinical PDF report to extract key context and text.")
    imported_pdf = st.file_uploader(
        "Clinical report PDF",
        type=["pdf"],
        key="cv_import_pdf",
    )
    if imported_pdf is None:
        return

    try:
        pdf_summary = import_pdf_report(imported_pdf)
        st.success(f"Imported {pdf_summary.page_count} page(s) from PDF.")
        st.markdown("**Extracted fields**")
        fields_col_1, fields_col_2, fields_col_3 = st.columns(3)
        fields_col_1.metric("Patient", pdf_summary.patient_name or NOT_FOUND_LABEL)
        fields_col_2.metric("MRN", pdf_summary.mrn or NOT_FOUND_LABEL)
        fields_col_3.metric("Date", pdf_summary.comparison_date or NOT_FOUND_LABEL)

        preview_text = pdf_summary.extracted_text[:4000] or "No extractable text was found in this PDF."
        st.text_area("Extracted text preview", preview_text, height=220, key="cv_pdf_text_preview")
        st.download_button(
            "Download extracted text",
            data=pdf_summary.extracted_text.encode("utf-8"),
            file_name=f"{patient.mrn}_imported_report.txt",
            mime="text/plain",
            key="cv_download_pdf_text",
        )
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not import PDF: {error}")


def _render_export_section(
    patient: PatientRecord,
    pre_session: ImagingSession,
    post_session: ImagingSession,
    snapshot_store: list[dict[str, object]],
    patient_snapshots: list[dict[str, object]],
) -> None:
    """Render report export controls."""
    st.markdown("### Export")
    st.caption("Generate a compact PDF containing patient context and captured snapshots.")
    trends_for_report = st.session_state.get("cv_report_trends_data_frame")
    trend_chart_bytes = (
        _build_trend_chart_png(trends_for_report)
        if isinstance(trends_for_report, pd.DataFrame)
        else None
    )
    report_bytes = _build_report_pdf(
        patient,
        pre_session,
        post_session,
        patient_snapshots,
        trend_chart_bytes,
        st.session_state.get("cv_stats_summary"),
    )
    file_name = f"cartiview_report_{patient.mrn}_{post_session.date.replace(' ', '_')}.pdf"
    st.download_button(
        "Download PDF report",
        data=report_bytes,
        file_name=file_name,
        mime="application/pdf",
        use_container_width=True,
    )
    if patient_snapshots and st.button("Clear patient snapshots", key="cv_clear_patient_snapshots"):
        st.session_state.cv_snapshots = [
            item for item in snapshot_store if item["patient_id"] != patient.patient_id
        ]
        st.success("Snapshots cleared for this patient.")


def _render_u3d_section() -> None:
    """Render U3D import and optional conversion tools."""
    st.divider()
    st.markdown("### U3D Import")
    st.caption("Upload U3D anatomical models and optionally convert to STL for the 3D workflow.")
    uploaded_u3d_file = st.file_uploader(
        "Anatomical U3D model",
        type=["u3d"],
        key="cv_import_u3d",
    )
    if uploaded_u3d_file is None:
        return

    u3d_summary = summarize_u3d_file(uploaded_u3d_file)
    summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
    summary_col_1.metric("File", u3d_summary.file_name)
    summary_col_2.metric("Size (bytes)", f"{u3d_summary.file_size_bytes:,}")
    summary_col_3.metric("Signature", "Detected" if u3d_summary.has_u3d_signature else "Unknown")

    if is_meshlab_available():
        if st.button("Convert U3D to STL", key="cv_convert_u3d_to_stl"):
            try:
                stl_bytes = convert_u3d_to_stl_with_meshlab(uploaded_u3d_file)
                st.success("Conversion complete.")
                st.download_button(
                    "Download converted STL",
                    data=stl_bytes,
                    file_name="converted_from_u3d.stl",
                    mime="model/stl",
                    key="cv_download_converted_stl",
                )
            except RuntimeError as error:
                st.error(f"Conversion failed: {error}")
    else:
        st.info(
            "Install `meshlabserver` to enable local U3D -> STL conversion. "
            "Without it, the file can still be validated and cataloged."
        )


def _render_stl_section() -> None:
    """Render STL comparison and fusion controls."""
    st.divider()
    st.markdown("### 3D STL Comparison (Prototype)")
    st.caption("Upload pre/post STL meshes for an interactive overlay and coarse mesh-level metrics.")

    stl_pre_col, stl_post_col = st.columns(2, gap="large")
    with stl_pre_col:
        pre_stl_file = st.file_uploader(
            "Pre-treatment STL",
            type=["stl"],
            key="cv_pre_stl",
        )
    with stl_post_col:
        post_stl_file = st.file_uploader(
            "Post-treatment STL",
            type=["stl"],
            key="cv_post_stl",
        )

    if not (pre_stl_file and post_stl_file):
        return

    try:
        pre_mesh = load_stl_mesh(pre_stl_file)
        post_mesh = load_stl_mesh(post_stl_file)
        summary = build_mesh_comparison_summary(pre_mesh, post_mesh)
        st.plotly_chart(
            build_overlay_figure(pre_mesh, post_mesh),
            use_container_width=True,
        )
        metric_columns = st.columns(3)
        metric_columns[0].metric("Pre volume", f"{summary.pre_volume:.2f}")
        metric_columns[1].metric("Post volume", f"{summary.post_volume:.2f}")
        metric_columns[2].metric("Volume delta", f"{summary.volume_delta_percent:+.2f}%")
        st.caption(
            f"Vertices: pre {summary.pre_vertices:,}, post {summary.post_vertices:,} · "
            f"Faces: pre {summary.pre_faces:,}, post {summary.post_faces:,}"
        )

        st.markdown("#### Anatomical fusion overlay")
        st.caption("CSV columns: `vertex_index,segment,change` (change is optional).")
        fusion_template = (
            "vertex_index,segment,change\n"
            "0,medial,0.12\n"
            "15,lateral,-0.08\n"
            "32,patellar,0.03\n"
        )
        st.download_button(
            "Download fusion CSV template",
            data=fusion_template,
            file_name="fusion_template.csv",
            mime="text/csv",
            key="cv_download_fusion_template",
        )
        fusion_file = st.file_uploader(
            "Fusion CSV (vertex_index, segment, optional change)",
            type=["csv"],
            key="cv_fusion_csv",
        )
        fusion_color_mode = st.radio(
            "Fusion color mode",
            options=["Segment labels", "Change magnitude"],
            horizontal=True,
            key="cv_fusion_color_mode",
        )
        if fusion_file is not None:
            fusion_points_data_frame = load_fusion_points(fusion_file, vertex_count=summary.post_vertices)
            st.plotly_chart(
                build_fusion_figure(post_mesh, fusion_points_data_frame, fusion_color_mode),
                use_container_width=True,
            )
            st.caption(
                f"Fusion points loaded: {len(fusion_points_data_frame):,} "
                f"across {fusion_points_data_frame['segment'].nunique()} segment(s)."
            )
            with st.expander("Fusion CSV format guide"):
                st.markdown(
                    "- `vertex_index`: zero-based index into post-treatment mesh vertices.\n"
                    "- `segment`: label such as `medial`, `lateral`, `patellar`.\n"
                    "- `change` (optional): numeric value used in `Change magnitude` mode."
                )
        else:
            st.info("Upload a fusion CSV to project segment points onto the post-treatment mesh.")
    except (ValueError, OSError) as error:
        st.error(f"Could not load STL files: {error}")


def render_tools_tab(
    patient: PatientRecord,
    pre_session: ImagingSession,
    post_session: ImagingSession,
) -> None:
    """Render report export actions and utilities."""
    snapshot_store = _get_snapshot_store()
    patient_snapshots = [item for item in snapshot_store if item["patient_id"] == patient.patient_id]
    _render_export_section(patient, pre_session, post_session, snapshot_store, patient_snapshots)
    _render_pdf_import_section(patient)
    _render_u3d_section()
    _render_stl_section()


def render_image_viewer_area(view_mode: str) -> tuple[list, list]:
    """Upload zones + comparison; returns (pre_list, post_list)."""
    pre_staged = list(st.session_state.get("cv_pre_upload") or [])
    post_staged = list(st.session_state.get("cv_post_upload") or [])
    has_staged = bool(pre_staged or post_staged)

    pre_list, post_list = render_image_load_section()

    if pre_list or post_list:
        st.divider()
        render_image_pairs(view_mode, pre_list, post_list)
    elif not has_staged:
        st.markdown(
            """
<div class="cv-img-placeholder">
  <div class="cv-img-placeholder-icon">🩻</div>
  <div><strong>No scan images loaded yet</strong></div>
  <div style="color:#b0bec5;font-size:0.82rem;margin-top:4px;">
    Upload scans above — then use the overlay controls and analysis tabs below.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    return pre_list, post_list


def render_metrics_dashboard(pre_session: ImagingSession, post_session: ImagingSession) -> None:
    """Prominent metrics dashboard showing key changes."""
    metrics = get_segment_metrics_for_post_session(post_session.session_id)
    if not metrics:
        return

    st.markdown("### Key Changes")
    cols = st.columns(4)

    metric_data = [
        ("Volume Change", metrics.total_volume, "🔵"),
        ("Medial Thickness", metrics.medial_thickness, "🟢"),
        ("Lateral Thickness", metrics.lateral_thickness, "🟡"),
        ("T2 Relaxation", metrics.t2_relaxation, "🔴"),
    ]

    for col, (label, value, icon) in zip(cols, metric_data):
        with col:
            st.metric(label=f"{icon} {label}", value=value)

    st.divider()


def render_viewer_screen(patient: PatientRecord) -> None:
    """Full viewer pipeline."""
    pre_session, post_session = resolve_pre_post_sessions(patient)

    render_viewer_header(patient)
    render_metrics_dashboard(pre_session, post_session)

    mode_col, change_col = st.columns([5, 1])
    with mode_col:
        view_mode: str = (
            st.segmented_control(
                "View mode",
                options=[VIEW_MODE_COMPARE, VIEW_MODE_PRE, VIEW_MODE_POST],
                key="cv_view_mode",
                default=VIEW_MODE_COMPARE,
                label_visibility="collapsed",
            )
            or VIEW_MODE_COMPARE
        )
    with change_col:
        st.button(
            "Change sessions",
            key="vs_change",
            use_container_width=True,
            on_click=lambda: st.session_state.__setitem__("cv_screen", SCREEN_LOOKUP),
            help="Go back to patient lookup to select different sessions.",
        )

    render_viewer_session_summary(pre_session, post_session)

    pre_list, post_list = render_image_viewer_area(view_mode)

    render_viewer_overlay_strip()

    tab_seg, tab_trends, tab_improve, tab_snap, tab_tools = st.tabs(
        ["🔬 Segments", "📈 Trends", "🗺️ Map", "📸 Snapshots", "🔧 Tools"]
    )

    with tab_seg:
        st.caption(
            "Quantitative segment summary · values keyed to the **selected follow-up** (demo — replace with Chondral Quant)."
        )
        render_viewer_metric_strip(post_session)
        render_viewer_tools_row()

    with tab_trends:
        render_trends_tab(patient)

    with tab_improve:
        render_improvement_map_tab(patient, pre_session, post_session, pre_list, post_list)

    with tab_snap:
        render_snapshots_tab(patient)

    with tab_tools:
        render_tools_tab(patient, pre_session, post_session)
