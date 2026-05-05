"""Inject global theme CSS (see theme.css)."""

from pathlib import Path

import streamlit as st

_THEME_PATH = Path(__file__).with_name("theme.css")


def inject_custom_css() -> None:
    st.markdown(_THEME_PATH.read_text(encoding="utf-8"), unsafe_allow_html=True)
