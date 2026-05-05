"""Streamlit entry: page config, theme, session defaults, sidebar, routing."""

import streamlit as st

from cv.constants import SCREEN_LOOKUP, SCREEN_VIEWER
from cv.data import MOCK_PATIENTS, get_patient_by_id
from cv.lookup_ui import render_patient_lookup_screen
from cv.session_sync import resolve_pre_post_sessions
from cv.styles import inject_custom_css
from cv.viewer_ui import render_viewer_screen


def run() -> None:
    st.set_page_config(
        page_title="CartiView",
        page_icon="🦵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    if "cv_selected_patient" not in st.session_state:
        st.session_state.cv_selected_patient = MOCK_PATIENTS[0].patient_id
    if "cv_screen" not in st.session_state:
        st.session_state.cv_screen = SCREEN_LOOKUP
    if "cv_lookup_mode" not in st.session_state:
        st.session_state.cv_lookup_mode = "list"

    with st.sidebar:
        st.markdown("### CartiView")
        st.caption("Pre & post cartilage comparison · prototype")
        st.divider()

        st.markdown("##### Navigate")
        st.radio(
            "Screen",
            [SCREEN_LOOKUP, SCREEN_VIEWER],
            key="cv_screen",
            label_visibility="collapsed",
        )
        st.divider()

        sidebar_patient = get_patient_by_id(st.session_state.cv_selected_patient)
        if sidebar_patient:
            st.caption("Active patient")
            st.markdown(f"**{sidebar_patient.display_name}**")
            st.caption(f"MRN {sidebar_patient.mrn} · {sidebar_patient.joint}")

            if st.session_state.cv_screen == SCREEN_VIEWER:
                try:
                    pre, post = resolve_pre_post_sessions(sidebar_patient)
                    st.divider()
                    st.caption("Comparing")
                    st.markdown(
                        f"**Pre** {pre.date}  \n{pre.label}  \n"
                        f"**Post** {post.date}  \n{post.label}"
                    )
                    st.button(
                        "Change sessions ↩",
                        use_container_width=True,
                        on_click=lambda: st.session_state.__setitem__("cv_screen", SCREEN_LOOKUP),
                        key="sb_change_sessions",
                    )
                except (KeyError, StopIteration):
                    pass

        st.divider()
        st.link_button(
            "UI concept deck ↗",
            "https://cartiview-presentation.netlify.app/",
            use_container_width=True,
        )

    active = get_patient_by_id(st.session_state.cv_selected_patient)
    if active is None:
        st.error("Invalid patient selection.")
        st.stop()

    if st.session_state.cv_screen == SCREEN_LOOKUP:
        render_patient_lookup_screen()
    else:
        render_viewer_screen(active)
