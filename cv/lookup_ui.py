"""Patient lookup: list view and detail view."""

from typing import Tuple

import streamlit as st

from cv.constants import JOINT_FILTER_OPTIONS, SCREEN_VIEWER, STATUS_ACTIVE
from cv.data import (
    compute_initials,
    find_patients,
    get_patient_by_id,
    session_radio_label,
)
from cv.models import PatientRecord


def navigate_to_patient_detail(patient_id: str) -> None:
    """Select a patient and switch the lookup panel to detail mode."""
    st.session_state.cv_selected_patient = patient_id
    st.session_state.cv_lookup_mode = "detail"


def placeholder_panel(title: str, body: str) -> None:
    """Soft grey panel for 'coming soon' sections."""
    from cv.html_util import escape_html

    st.markdown(
        f'<div class="cv-placeholder">'
        f'<strong style="color:#0f172a;">{escape_html(title)}</strong><br/>'
        f'<span style="color:#475569;font-size:0.95rem;line-height:1.55;">{escape_html(body)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_patient_list(matches: Tuple[PatientRecord, ...]) -> None:
    """Patient list rows with avatar, name button, and scan meta."""
    from cv.html_util import escape_html

    if not matches:
        st.warning("No patients match — try a different search or filter.")
        return

    for patient in matches:
        scan_count = len(patient.sessions)
        plural_suffix = "s" if scan_count != 1 else ""
        last_scan = patient.sessions[-1].date if patient.sessions else "—"
        initials = compute_initials(patient.display_name)

        with st.container(border=True):
            col_avatar, col_info, col_meta = st.columns([1, 7, 3])
            with col_avatar:
                st.markdown(
                    f'<div class="cv-avatar">{escape_html(initials)}</div>',
                    unsafe_allow_html=True,
                )
            with col_info:
                st.button(
                    patient.display_name,
                    key=f"sel_{patient.patient_id}",
                    use_container_width=True,
                    on_click=lambda pid=patient.patient_id: navigate_to_patient_detail(pid),
                )
                st.caption(f"MRN {patient.mrn} · {patient.joint}")
            with col_meta:
                st.markdown(
                    f'<div class="cv-scan-count">{scan_count} scan{plural_suffix}</div>'
                    f'<div class="cv-scan-last">Last: {escape_html(last_scan)}</div>',
                    unsafe_allow_html=True,
                )


def render_patient_info_two_section(patient: PatientRecord) -> None:
    """Two-section patient info card (patient info + treatment)."""
    from cv.html_util import escape_html

    treatment_status = "Active" if patient.status == STATUS_ACTIVE else patient.status

    info_rows = [
        ("MRN", patient.mrn),
        ("DOB", patient.dob),
        ("Age", f"{patient.age} yrs"),
        ("Sex", patient.sex),
        ("Joint", patient.joint),
        ("Treating", patient.treating),
    ]
    treatment_rows = [
        ("Type", patient.treatment_type),
        ("Started", patient.treatment_started),
        ("Status", treatment_status),
    ]

    def _kv_rows_html(rows: list[tuple[str, str]]) -> str:
        return "".join(
            f'<div class="cv-info-kv-row">'
            f'<span class="cv-info-kv-key">{escape_html(k)}</span>'
            f'<span class="cv-info-kv-val">{escape_html(v)}</span>'
            f"</div>"
            for k, v in rows
        )

    st.markdown(
        f'<div class="cv-info-two-section">'
        f'<div class="cv-info-section-hdr">Patient info</div>'
        f"{_kv_rows_html(info_rows)}"
        f'<div class="cv-info-section-hdr" style="margin-top:1.1rem;">Treatment</div>'
        f"{_kv_rows_html(treatment_rows)}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_sessions_detail_panel(patient: PatientRecord) -> None:
    """Pre/post session radios."""
    pre_sessions = [s for s in patient.sessions if s.role == "pre"]
    post_sessions = [s for s in patient.sessions if s.role in ("post", "followup")]

    if pre_sessions and "cv_pre_session_id" not in st.session_state:
        st.session_state.cv_pre_session_id = pre_sessions[0].session_id
    if post_sessions and "cv_post_session_id" not in st.session_state:
        st.session_state.cv_post_session_id = post_sessions[0].session_id

    pre_map = {s.session_id: s for s in pre_sessions}
    post_map = {s.session_id: s for s in post_sessions}

    st.markdown("**Imaging sessions**")

    st.markdown('<div class="cv-ses-subhdr">Pre-treatment</div>', unsafe_allow_html=True)
    if pre_sessions:
        st.radio(
            "Pre-treatment session",
            options=[s.session_id for s in pre_sessions],
            format_func=lambda sid: session_radio_label(pre_map[sid]),
            key="cv_pre_session_id",
            label_visibility="collapsed",
        )
    else:
        st.caption("No pre-treatment sessions recorded.")

    st.markdown('<div class="cv-ses-subhdr">Post-treatment</div>', unsafe_allow_html=True)
    if post_sessions:
        st.radio(
            "Post-treatment session",
            options=[s.session_id for s in post_sessions],
            format_func=lambda sid: session_radio_label(post_map[sid]),
            key="cv_post_session_id",
            label_visibility="collapsed",
        )
    else:
        st.caption("No post-treatment sessions yet — follow-up visits will appear here.")


def _render_lookup_header() -> None:
    """Page title + badge."""
    title_col, badge_col = st.columns([5, 2])
    with title_col:
        st.markdown(
            '<div class="cv-page-title">'
            "<strong>Patient</strong> "
            '<span class="cv-green">lookup &amp; record</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with badge_col:
        st.markdown(
            '<div style="text-align:right;padding-top:6px;">'
            '<span class="cv-page-badge">Interactive — search, filter, select sessions</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<hr style="margin:0.6rem 0 0.75rem 0;border:none;border-top:1px solid #f1f5f9;">',
        unsafe_allow_html=True,
    )


def _render_patient_list_view() -> None:
    """Search + joint filter + patient rows (results update as you type)."""
    _, col_main, _ = st.columns([1, 6, 1])
    with col_main:
        st.markdown(
            '<div class="cv-breadcrumb">'
            '<span class="cv-breadcrumb-link">CartiView</span>'
            '<span class="cv-breadcrumb-sep">|</span>'
            '<span class="cv-breadcrumb-page">Patient lookup</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Find a patient")
        st.caption("Search by name, MRN, or date of birth — results update as you type.")

        st.text_input(
            "Search",
            placeholder="Name or MRN...",
            label_visibility="collapsed",
            key="cv_patient_search",
        )

        st.segmented_control(
            "Joint filter",
            options=list(JOINT_FILTER_OPTIONS),
            key="cv_joint_filter",
            default="All",
            label_visibility="collapsed",
        )
        st.markdown("")

        joint_filter: str = st.session_state.get("cv_joint_filter") or "All"
        search_query: str = st.session_state.get("cv_patient_search") or ""
        matches = find_patients(search_query, joint_filter)
        render_patient_list(matches)


def _render_patient_detail_view(patient: PatientRecord) -> None:
    """Detail layout: back, info card, sessions, open viewer."""
    from cv.html_util import escape_html

    pre_sessions = [s for s in patient.sessions if s.role == "pre"]
    post_sessions = [s for s in patient.sessions if s.role in ("post", "followup")]
    can_open_viewer = bool(pre_sessions and post_sessions)

    back_col, crumb_col = st.columns([1, 8], gap="small")
    with back_col:
        st.button(
            "← Back",
            key="pl_back",
            on_click=lambda: st.session_state.__setitem__("cv_lookup_mode", "list"),
        )
    with crumb_col:
        st.markdown(
            f'<div style="display:flex;align-items:center;height:100%;padding-top:6px;">'
            f'<span class="cv-breadcrumb-sep" style="margin-right:6px;">/</span>'
            f'<strong style="font-size:1.05rem;color:#0f172a;">{escape_html(patient.display_name)}</strong>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    info_col, sessions_col = st.columns([3, 5], gap="large")

    with info_col:
        render_patient_info_two_section(patient)

    with sessions_col:
        render_sessions_detail_panel(patient)
        st.markdown("")
        st.button(
            "Open in viewer →",
            type="primary",
            use_container_width=True,
            disabled=not can_open_viewer,
            on_click=lambda: st.session_state.__setitem__("cv_screen", SCREEN_VIEWER),
            key="pl_open_viewer",
        )
        if not can_open_viewer:
            st.caption("Need at least one pre and one post session to compare.")


def render_patient_lookup_screen() -> None:
    """Lookup screen: list ↔ detail."""
    _render_lookup_header()

    lookup_mode: str = st.session_state.get("cv_lookup_mode", "list")
    if lookup_mode == "detail":
        patient = get_patient_by_id(st.session_state.cv_selected_patient)
        if patient is not None:
            _render_patient_detail_view(patient)
            return

    _render_patient_list_view()
