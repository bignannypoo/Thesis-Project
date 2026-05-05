"""Load, resize, and display scan images (browser upload or local folder paths)."""

import io
from pathlib import Path
from typing import Any, List, Optional

import streamlit as st
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

from cv.constants import MAX_DISPLAY_IMAGE_PX, VIEW_MODE_POST, VIEW_MODE_PRE

IMAGE_EXTENSIONS = frozenset(["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"])
ImageSource = UploadedFile | Path


def fit_image_for_display(image: Image.Image, max_dim: int = MAX_DISPLAY_IMAGE_PX) -> Image.Image:
    """Downscale large slices so the UI stays responsive (LANCZOS, keeps aspect ratio)."""
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    if w >= h:
        new_w = max_dim
        new_h = max(1, int(h * (max_dim / w)))
    else:
        new_h = max_dim
        new_w = max(1, int(w * (max_dim / h)))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def load_image_from_upload(uploaded_file: UploadedFile) -> Image.Image:
    """Turn a Streamlit UploadedFile into a PIL Image (in memory)."""
    return Image.open(io.BytesIO(uploaded_file.getvalue()))


def open_image_and_caption(source: ImageSource) -> tuple[Image.Image, str]:
    """Load a PIL Image and caption from an upload or a filesystem path."""
    if isinstance(source, Path):
        return Image.open(source), source.name
    return load_image_from_upload(source), getattr(source, "name", "Image")


def list_images_in_folder(folder: Path) -> List[Path]:
    """Sorted image files directly inside `folder` (not subfolders)."""
    if not folder.is_dir():
        return []
    found = [
        entry
        for entry in folder.iterdir()
        if entry.is_file() and entry.suffix.lower().removeprefix(".") in IMAGE_EXTENSIONS
    ]
    return sorted(found, key=lambda p: p.name.lower())


def show_image_in_column(column: Any, title: str, source: Optional[ImageSource]) -> None:
    """Show one preview in a column, or a placeholder if missing."""
    with column:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if source is None:
                st.caption("No image loaded for this slot.")
                return
            try:
                image, caption = open_image_and_caption(source)
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGBA")
                image = fit_image_for_display(image)
                st.image(image, caption=caption, use_container_width=True)
            except (OSError, ValueError) as exc:
                st.error(f"Could not open this file: {exc}")


def render_image_pairs(
    view_mode: str,
    pre_list: List[ImageSource],
    post_list: List[ImageSource],
) -> None:
    """Render images according to the selected view mode."""
    pair_count = max(len(pre_list), len(post_list))

    if view_mode == VIEW_MODE_PRE:
        if not pre_list:
            st.warning("Add pre-treatment images below to preview them here.")
            return
        for index, item in enumerate(pre_list):
            st.markdown(f"##### Pre {index + 1}")
            show_image_in_column(st.container(), "Pre-treatment", item)
            st.divider()
        return

    if view_mode == VIEW_MODE_POST:
        if not post_list:
            st.warning("Add post-treatment images below to preview them here.")
            return
        for index, item in enumerate(post_list):
            st.markdown(f"##### Post {index + 1}")
            show_image_in_column(st.container(), "Post-treatment", item)
            st.divider()
        return

    if pair_count == 0:
        st.info("Use **Add images** below to load pre and post files. They are paired in order.")
        return

    st.caption("Pairs follow **file / upload order**: row 1 = first pre & first post, …")
    for index in range(pair_count):
        st.markdown(f"##### Pair {index + 1}")
        pre_item = pre_list[index] if index < len(pre_list) else None
        post_item = post_list[index] if index < len(post_list) else None
        row_pre, row_post = st.columns(2, gap="large")
        show_image_in_column(row_pre, "Pre-treatment", pre_item)
        show_image_in_column(row_post, "Post-treatment", post_item)
        st.divider()


def _load_images_from_folder_path(path_str: str, label: str) -> tuple[List[ImageSource], bool]:
    """Resolve a folder path string to a sorted image list. Returns (images, is_valid_dir)."""
    if not path_str.strip():
        return [], False
    folder = Path(path_str.strip()).expanduser().resolve()
    if not folder.is_dir():
        st.error(f"{label} folder not found: `{folder}`")
        return [], False
    images = list_images_in_folder(folder)
    st.caption(f"Found **{len(images)}** image(s) in `{folder.name}/`")
    return images, True


def render_image_load_section() -> tuple[List[ImageSource], List[ImageSource]]:
    """
    Two drop zones (pre + post). Folder path option overrides uploads when a valid path is set.
    """
    st.markdown("#### Add scan images")
    st.caption(
        "Drag & drop files onto a zone, or click **Browse files** to pick them. "
        "Files are paired in order — 1st pre with 1st post, 2nd with 2nd, etc."
    )

    col_pre, col_post = st.columns(2, gap="large")

    with col_pre:
        pre_count = len(st.session_state.get("cv_pre_upload") or [])
        pre_plural = "s" if pre_count != 1 else ""
        count_html = (
            f'<span class="cv-upload-count">✓ {pre_count} file{pre_plural} loaded</span>'
            if pre_count
            else ""
        )
        st.markdown(
            f'<div class="cv-upload-header">'
            f'<span class="cv-upload-icon">🩻</span>'
            f'<div>'
            f'<div class="cv-upload-title">Pre-treatment</div>'
            f'<div class="cv-upload-subtitle">Baseline scans · before treatment</div>'
            f'</div></div>{count_html}',
            unsafe_allow_html=True,
        )
        pre_files = st.file_uploader(
            "Pre-treatment scans",
            type=sorted(IMAGE_EXTENSIONS),
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="cv_pre_upload",
        )

    with col_post:
        post_count = len(st.session_state.get("cv_post_upload") or [])
        post_plural = "s" if post_count != 1 else ""
        count_html = (
            f'<span class="cv-upload-count">✓ {post_count} file{post_plural} loaded</span>'
            if post_count
            else ""
        )
        st.markdown(
            f'<div class="cv-upload-header">'
            f'<span class="cv-upload-icon">🩻</span>'
            f'<div>'
            f'<div class="cv-upload-title">Post-treatment</div>'
            f'<div class="cv-upload-subtitle">Follow-up scans · after treatment</div>'
            f'</div></div>{count_html}',
            unsafe_allow_html=True,
        )
        post_files = st.file_uploader(
            "Post-treatment scans",
            type=sorted(IMAGE_EXTENSIONS),
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="cv_post_upload",
        )

    pre_list: List[ImageSource] = list(pre_files or [])
    post_list: List[ImageSource] = list(post_files or [])

    with st.expander("🗂️  Load from a folder path instead"):
        st.caption(
            "Useful when you have many slices already saved in a folder on this computer. "
            "Paste the full path — `~` expands to your home directory."
        )
        path_col_pre, path_col_post = st.columns(2)
        with path_col_pre:
            pre_path_str = st.text_input(
                "Pre-treatment folder",
                placeholder="~/Documents/scans/pre",
                key="cv_pre_folder",
            )
        with path_col_post:
            post_path_str = st.text_input(
                "Post-treatment folder",
                placeholder="~/Documents/scans/post",
                key="cv_post_folder",
            )
        folder_pre, pre_valid = _load_images_from_folder_path(pre_path_str, "Pre-treatment")
        folder_post, post_valid = _load_images_from_folder_path(post_path_str, "Post-treatment")

        if pre_valid:
            pre_list = folder_pre
        if post_valid:
            post_list = folder_post

    counts_differ = bool(pre_list and post_list and len(pre_list) != len(post_list))
    if counts_differ:
        st.warning(
            f"Counts differ — **{len(pre_list)} pre** vs **{len(post_list)} post**. "
            "Shorter side will have empty slots in compare mode."
        )

    return pre_list, post_list
