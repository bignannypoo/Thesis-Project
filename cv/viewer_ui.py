"""Main viewer: header, session summary, images, overlay strip, analysis tabs."""

import streamlit as st

from cv.constants import (
    SCREEN_LOOKUP,
    STATUS_ACTIVE,
    VIEW_MODE_COMPARE,
    VIEW_MODE_POST,
    VIEW_MODE_PRE,
)
from cv.html_util import escape_html
from cv.image_utils import render_image_load_section, render_image_pairs
from cv.metrics import format_metric_strip_html, get_segment_metrics_for_post_session
from cv.models import ImagingSession, PatientRecord
from cv.session_sync import resolve_pre_post_sessions
from cv.lookup_ui import placeholder_panel


def render_viewer_header(patient: PatientRecord) -> None:
    """Back button + patient identity bar."""
    btn_col, bar_col = st.columns([1, 5])
    with btn_col:
        st.button(
            "← Back",
            key="vh_back",
            use_container_width=True,
            on_click=lambda: st.session_state.__setitem__("cv_screen", SCREEN_LOOKUP),
        )
    with bar_col:
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
    metrics = get_segment_metrics_for_post_session(post_session)
    st.markdown(format_metric_strip_html(metrics), unsafe_allow_html=True)
    st.markdown(
        f'<p class="cv-interval-caption">Interval: {escape_html(post_session.label)} vs baseline</p>',
        unsafe_allow_html=True,
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


def render_viewer_screen(patient: PatientRecord) -> None:
    """Full viewer pipeline."""
    pre_session, post_session = resolve_pre_post_sessions(patient)

    render_viewer_header(patient)

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

    render_image_viewer_area(view_mode)

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
        placeholder_panel(
            "Trends",
            "Thickness / volume over time. Plan: connect saved metrics and Plotly charts.",
        )

    with tab_improve:
        placeholder_panel(
            "Improvement map",
            "Colour heat map — green = improved, red = decreased. Needs registered segment exports.",
        )

    with tab_snap:
        placeholder_panel(
            "Snapshots",
            "Crop, annotate, and save regions to your report.",
        )

    with tab_tools:
        placeholder_panel(
            "Tools",
            "Measure, ruler, opacity controls. Anatomy fusion will link to a 3D knee view.",
        )
