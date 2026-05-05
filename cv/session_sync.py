"""Session-state helpers that depend on Streamlit (selected pre/post visit ids)."""

import streamlit as st

from cv.models import ImagingSession, PatientRecord


def sync_session_ids_for_patient(patient: PatientRecord) -> None:
    """Keep pre/post session ids valid when the active patient changes."""
    pre_sessions = [s for s in patient.sessions if s.role == "pre"]
    post_sessions = [s for s in patient.sessions if s.role in ("post", "followup")]
    pre_ids = {s.session_id for s in pre_sessions}
    post_ids = {s.session_id for s in post_sessions}
    if not pre_sessions or not post_sessions:
        return
    if "cv_pre_session_id" not in st.session_state or st.session_state.cv_pre_session_id not in pre_ids:
        st.session_state.cv_pre_session_id = pre_sessions[0].session_id
    if "cv_post_session_id" not in st.session_state or st.session_state.cv_post_session_id not in post_ids:
        st.session_state.cv_post_session_id = post_sessions[0].session_id


def resolve_pre_post_sessions(patient: PatientRecord) -> tuple[ImagingSession, ImagingSession]:
    """Resolve selected baseline and follow-up sessions from session state."""
    sync_session_ids_for_patient(patient)
    pre_sessions = [s for s in patient.sessions if s.role == "pre"]
    post_sessions = [s for s in patient.sessions if s.role in ("post", "followup")]
    pre_map = {s.session_id: s for s in pre_sessions}
    post_map = {s.session_id: s for s in post_sessions}
    pre = pre_map[st.session_state.cv_pre_session_id]
    post = post_map[st.session_state.cv_post_session_id]
    return pre, post
