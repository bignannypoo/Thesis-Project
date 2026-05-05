"""Streamlit entry: unified dashboard with persistent sidebar patient search."""

import streamlit as st

from cv.constants import JOINT_FILTER_OPTIONS, STATUS_ACTIVE
from cv.data import (
    compute_initials,
    find_patients,
    get_patient_by_id,
    list_patients,
)
from cv.html_util import escape_html
from cv.styles import inject_custom_css
from cv.viewer_ui import render_viewer_screen


def render_sidebar_patient_search() -> None:
    """Compact patient search in sidebar."""
    st.markdown("### 🦵 CartiView")
    st.caption("Pre & post cartilage comparison")
    st.divider()

    # Search input
    search_query = st.text_input(
        "Search patients",
        placeholder="Name or MRN...",
        key="sidebar_search",
        label_visibility="collapsed",
    )

    # Joint filter
    joint_filter = st.segmented_control(
        "Joint",
        options=JOINT_FILTER_OPTIONS,
        default="All",
        key="sidebar_joint_filter",
        label_visibility="collapsed",
    )

    st.divider()

    # Find matching patients
    matches = find_patients(search_query, joint_filter)

    if not matches:
        st.warning("No patients found")
        return

    st.caption(f"{len(matches)} patient{'s' if len(matches) != 1 else ''}")

    # Render compact patient list
    for patient in matches:
        initials = compute_initials(patient.display_name)
        is_selected = st.session_state.cv_selected_patient == patient.patient_id

        # Highlight selected patient
        container_class = "🔹 " if is_selected else ""

        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(
                    f'<div class="cv-avatar-small">{escape_html(initials)}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button(
                    f"{container_class}{patient.display_name}",
                    key=f"sb_sel_{patient.patient_id}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    st.session_state.cv_selected_patient = patient.patient_id
                    st.rerun()
                st.caption(f"MRN {patient.mrn} · {patient.joint}")


def run() -> None:
    st.set_page_config(
        page_title="CartiView",
        page_icon="🦵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    patients = list_patients()
    if not patients:
        st.error("No patients available in backend storage.")
        st.stop()

    # Initialize session state
    if "cv_selected_patient" not in st.session_state:
        st.session_state.cv_selected_patient = patients[0].patient_id

    # Sidebar with patient search
    with st.sidebar:
        render_sidebar_patient_search()

        st.divider()
        st.link_button(
            "UI concept deck ↗",
            "https://cartiview-presentation.netlify.app/",
            use_container_width=True,
        )

    # Main area - always show viewer
    active_patient = get_patient_by_id(st.session_state.cv_selected_patient)
    if active_patient is None:
        st.error("Invalid patient selection.")
        st.stop()

    render_viewer_screen(active_patient)
