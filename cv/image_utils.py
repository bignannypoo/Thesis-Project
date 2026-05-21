"""Load, resize, and display scan images (browser upload or local folder paths)."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pydicom
import streamlit as st
from PIL import Image
from pydicom.dataset import Dataset
from streamlit.runtime.uploaded_file_manager import UploadedFile

from cv.constants import MAX_DISPLAY_IMAGE_PX, VIEW_MODE_POST, VIEW_MODE_PRE

IMAGE_EXTENSIONS = frozenset(["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "dcm"])
ImageSource = UploadedFile | Path
WINDOW_PRESETS: dict[str, tuple[float, float]] = {
    "Custom": (40.0, 400.0),
    "Soft tissue": (50.0, 350.0),
    "Bone": (400.0, 2000.0),
    "Cartilage": (120.0, 500.0),
}


@dataclass(frozen=True)
class DicomSeriesSlice:
    """One DICOM item with derived series metadata."""

    source: ImageSource
    series_uid: str
    series_description: str
    instance_number: int
    slice_label: str


def _is_dicom_source(source: ImageSource) -> bool:
    """Return True when the source is a DICOM file."""
    if isinstance(source, Path):
        return source.suffix.lower() == ".dcm"
    return getattr(source, "name", "").lower().endswith(".dcm")


def _read_dicom_dataset(file_bytes: bytes, *, stop_before_pixels: bool = False) -> Dataset:
    """Read DICOM bytes with permissive parsing."""
    return pydicom.dcmread(io.BytesIO(file_bytes), stop_before_pixels=stop_before_pixels, force=True)


def _normalize_array_to_uint8(pixel_array: np.ndarray) -> np.ndarray:
    """Normalize any numeric image array into 0-255 uint8 for display."""
    array = pixel_array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = (array - min_value) / (max_value - min_value)
    return (normalized * 255).clip(0, 255).astype(np.uint8)


def _apply_window_level(pixel_array: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply DICOM window/level and return uint8 display array."""
    clipped_width = max(width, 1.0)
    lower = center - (clipped_width / 2.0)
    upper = center + (clipped_width / 2.0)
    windowed = np.clip(pixel_array.astype(np.float32), lower, upper)
    return _normalize_array_to_uint8(windowed)


def _get_dicom_display_array(dataset: Dataset) -> np.ndarray:
    """Convert DICOM pixel data to a display-ready uint8 array."""
    pixel_array = dataset.pixel_array
    use_manual_window = bool(st.session_state.get("cv_dicom_use_window", False))
    if not use_manual_window:
        return _normalize_array_to_uint8(pixel_array)

    window_center = float(st.session_state.get("cv_dicom_window_center", 40.0))
    window_width = float(st.session_state.get("cv_dicom_window_width", 400.0))
    return _apply_window_level(pixel_array, window_center, window_width)


def _load_dicom_from_bytes(file_bytes: bytes) -> Image.Image:
    """Decode DICOM bytes into a PIL image."""
    dataset = _read_dicom_dataset(file_bytes)
    if "PixelData" not in dataset:
        raise ValueError("DICOM has no pixel data.")
    display_array = _get_dicom_display_array(dataset)
    if display_array.ndim == 2:
        return Image.fromarray(display_array, mode="L").convert("RGB")
    if display_array.ndim == 3 and display_array.shape[-1] in (3, 4):
        return Image.fromarray(display_array[:, :, :3], mode="RGB")
    raise ValueError("Unsupported DICOM pixel array shape.")


def _extract_dicom_metadata_from_bytes(file_bytes: bytes) -> dict[str, str]:
    """Extract clinically useful DICOM metadata for preview panels."""
    dataset = _read_dicom_dataset(file_bytes, stop_before_pixels=True)
    rows = str(getattr(dataset, "Rows", "n/a"))
    columns = str(getattr(dataset, "Columns", "n/a"))
    pixel_spacing = getattr(dataset, "PixelSpacing", None)
    spacing_text = "n/a"
    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        spacing_text = f"{pixel_spacing[0]} x {pixel_spacing[1]} mm"

    return {
        "Patient Name": str(getattr(dataset, "PatientName", "n/a")),
        "Patient ID": str(getattr(dataset, "PatientID", "n/a")),
        "Modality": str(getattr(dataset, "Modality", "n/a")),
        "Study Date": str(getattr(dataset, "StudyDate", "n/a")),
        "Study Time": str(getattr(dataset, "StudyTime", "n/a")),
        "Series Description": str(getattr(dataset, "SeriesDescription", "n/a")),
        "Dimensions": f"{rows} x {columns}",
        "Pixel Spacing": spacing_text,
    }


def _get_dicom_bytes(source: ImageSource) -> bytes | None:
    """Return bytes for DICOM sources, otherwise None."""
    if isinstance(source, Path):
        if source.suffix.lower() == ".dcm":
            return source.read_bytes()
        return None
    source_name = getattr(source, "name", "").lower()
    if source_name.endswith(".dcm"):
        return source.getvalue()
    return None


def _get_dicom_slice_label(source: ImageSource) -> str:
    """Build a user-friendly label for DICOM slice selectors."""
    dicom_bytes = _get_dicom_bytes(source)
    if dicom_bytes is None:
        return "Non-DICOM"
    dataset = _read_dicom_dataset(dicom_bytes, stop_before_pixels=True)
    series_desc = str(getattr(dataset, "SeriesDescription", "Series"))
    instance_number = str(getattr(dataset, "InstanceNumber", "?"))
    return f"{series_desc} · slice {instance_number}"


def _build_dicom_series_slices(items: List[ImageSource]) -> list[DicomSeriesSlice]:
    """Build sorted DICOM series entries for browsing."""
    slices: list[DicomSeriesSlice] = []
    for source in items:
        dicom_bytes = _get_dicom_bytes(source)
        if dicom_bytes is None:
            continue
        dataset = _read_dicom_dataset(dicom_bytes, stop_before_pixels=True)
        series_uid = str(getattr(dataset, "SeriesInstanceUID", "unknown-series"))
        series_description = str(getattr(dataset, "SeriesDescription", "Series"))
        instance_number = int(getattr(dataset, "InstanceNumber", 0) or 0)
        slices.append(
            DicomSeriesSlice(
                source=source,
                series_uid=series_uid,
                series_description=series_description,
                instance_number=instance_number,
                slice_label=f"{series_description} · slice {instance_number}",
            )
        )
    return sorted(slices, key=lambda item: (item.series_description.lower(), item.instance_number))


def _render_dicom_series_browser(items: List[ImageSource], panel_title: str, state_prefix: str) -> None:
    """Render DICOM browser with series-first and optional cine mode."""
    slices = _build_dicom_series_slices(items)
    if not slices:
        st.warning("No readable DICOM slices found in this panel.")
        return

    unique_series_keys = list(dict.fromkeys(slice_info.series_uid for slice_info in slices))
    selected_series_uid = st.selectbox(
        f"{panel_title} series",
        options=unique_series_keys,
        format_func=lambda uid: next(
            (entry.series_description for entry in slices if entry.series_uid == uid),
            uid,
        ),
        key=f"{state_prefix}_series_uid",
    )
    selected_series_slices = [slice_info for slice_info in slices if slice_info.series_uid == selected_series_uid]

    cine_mode_enabled = st.toggle("Cine mode", key=f"{state_prefix}_cine_mode")
    if cine_mode_enabled:
        selected_index = st.slider(
            f"{panel_title} cine frame",
            min_value=0,
            max_value=len(selected_series_slices) - 1,
            value=min(st.session_state.get(f"{state_prefix}_cine_idx", 0), len(selected_series_slices) - 1),
            step=1,
            key=f"{state_prefix}_cine_idx",
        )
    else:
        options = list(range(len(selected_series_slices)))
        selected_index = st.selectbox(
            f"{panel_title} slice",
            options=options,
            format_func=lambda index: selected_series_slices[index].slice_label,
            key=f"{state_prefix}_slice_idx",
        )
    show_image_in_column(st.container(), panel_title, selected_series_slices[selected_index].source)


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
    file_name = getattr(uploaded_file, "name", "").lower()
    file_bytes = uploaded_file.getvalue()
    if file_name.endswith(".dcm"):
        return _load_dicom_from_bytes(file_bytes)
    return Image.open(io.BytesIO(file_bytes))


def open_image_and_caption(source: ImageSource) -> tuple[Image.Image, str]:
    """Load a PIL Image and caption from an upload or a filesystem path."""
    if isinstance(source, Path):
        if source.suffix.lower() == ".dcm":
            return _load_dicom_from_bytes(source.read_bytes()), source.name
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

                dicom_bytes = _get_dicom_bytes(source)
                if dicom_bytes is not None:
                    metadata = _extract_dicom_metadata_from_bytes(dicom_bytes)
                    with st.expander("DICOM metadata"):
                        for key, value in metadata.items():
                            st.markdown(f"**{key}:** {value}")
            except (OSError, ValueError) as exc:
                st.error(f"Could not open this file: {exc}")


def _render_single_series(items: List[ImageSource], title_prefix: str, panel_title: str) -> None:
    """Render a one-sided list of images for pre-only or post-only mode."""
    for index, item in enumerate(items):
        st.markdown(f"##### {title_prefix} {index + 1}")
        show_image_in_column(st.container(), panel_title, item)
        st.divider()


def _render_single_series_dicom_browser(items: List[ImageSource], panel_title: str, state_key: str) -> None:
    """Render one DICOM slice chosen via selector."""
    _render_dicom_series_browser(items, panel_title, state_key)


def _render_compare_dicom_browser(pre_list: List[ImageSource], post_list: List[ImageSource]) -> None:
    """Render compare mode using explicit DICOM slice selectors."""
    pre_col, post_col = st.columns(2, gap="large")
    with pre_col:
        _render_single_series_dicom_browser(pre_list, "Pre-treatment", "cv_dicom_pre")
    with post_col:
        _render_single_series_dicom_browser(post_list, "Post-treatment", "cv_dicom_post")


def _render_pre_mode(pre_list: List[ImageSource], is_dicom_only: bool) -> None:
    """Render pre-only mode."""
    if not pre_list:
        st.warning("Add pre-treatment images below to preview them here.")
        return
    if is_dicom_only and len(pre_list) > 1:
        _render_single_series_dicom_browser(pre_list, "Pre-treatment", "cv_dicom_pre")
        return
    _render_single_series(pre_list, "Pre", "Pre-treatment")


def _render_post_mode(post_list: List[ImageSource], is_dicom_only: bool) -> None:
    """Render post-only mode."""
    if not post_list:
        st.warning("Add post-treatment images below to preview them here.")
        return
    if is_dicom_only and len(post_list) > 1:
        _render_single_series_dicom_browser(post_list, "Post-treatment", "cv_dicom_post")
        return
    _render_single_series(post_list, "Post", "Post-treatment")


def _render_compare_mode(pre_list: List[ImageSource], post_list: List[ImageSource], is_dicom_only: bool) -> None:
    """Render compare mode."""
    pair_count = max(len(pre_list), len(post_list))
    if pair_count == 0:
        st.info("Use **Add images** below to load pre and post files. They are paired in order.")
        return

    if is_dicom_only and pre_list and post_list and (len(pre_list) > 1 or len(post_list) > 1):
        st.caption("DICOM browser mode: select slices explicitly for side-by-side review.")
        _render_compare_dicom_browser(pre_list, post_list)
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


def render_image_pairs(
    view_mode: str,
    pre_list: List[ImageSource],
    post_list: List[ImageSource],
) -> None:
    """Render images according to the selected view mode."""
    is_dicom_only = bool(pre_list or post_list) and all(_is_dicom_source(item) for item in pre_list + post_list)

    if view_mode == VIEW_MODE_PRE:
        _render_pre_mode(pre_list, is_dicom_only)
        return

    if view_mode == VIEW_MODE_POST:
        _render_post_mode(post_list, is_dicom_only)
        return

    _render_compare_mode(pre_list, post_list, is_dicom_only)


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
        "Supported formats: image files + DICOM (`.dcm`). "
        "Files are paired in order — 1st pre with 1st post, 2nd with 2nd, etc."
    )
    st.info(
        "Only scan images belong here (`png/jpg/jpeg/webp/tif/tiff/bmp/dcm`). "
        "Use **Trends** for metrics `.csv`, and **Tools** for `.pdf`, `.u3d`, and `.stl` files."
    )
    with st.expander("⚙️ DICOM display controls"):
        st.caption("Use manual window/level for better soft-tissue or bone contrast.")
        selected_preset = st.selectbox(
            "Window/level preset",
            options=list(WINDOW_PRESETS.keys()),
            key="cv_dicom_window_preset",
        )
        preset_center, preset_width = WINDOW_PRESETS[selected_preset]
        if selected_preset != "Custom":
            st.session_state.cv_dicom_use_window = True
            st.session_state.cv_dicom_window_center = preset_center
            st.session_state.cv_dicom_window_width = preset_width
        st.toggle("Enable manual window/level", key="cv_dicom_use_window")
        st.slider("Window center", -2048.0, 4096.0, 40.0, 1.0, key="cv_dicom_window_center")
        st.slider("Window width", 1.0, 8192.0, 400.0, 1.0, key="cv_dicom_window_width")

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
