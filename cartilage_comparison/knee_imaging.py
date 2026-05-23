"""2D/3D knee visuals: NIfTI slices, VTK/STL meshes, study report metadata, change heatmaps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
import numpy as np

SliceAxis = Literal["axial", "coronal", "sagittal"]
DifferenceView = Literal["clinical_panel", "overlay", "change_only"]

# MRCH NIfTI axis order: UI "axial" / "coronal" are swapped vs naive array indices.
_SLICE_AXIS_DIMENSION: dict[SliceAxis, int] = {
    "axial": 1,
    "coronal": 2,
    "sagittal": 0,
}
_COUNT_AXES_FOR_DIMENSION: dict[int, tuple[int, int]] = {
    0: (1, 2),
    1: (0, 2),
    2: (0, 1),
}

COLOR_INCREASE = np.array([0.1, 0.75, 0.2])  # green
COLOR_DECREASE = np.array([0.9, 0.15, 0.15])  # red
COLOR_PRE_BORDER = "#2563eb"
COLOR_POST_BORDER = "#ea580c"


@dataclass(frozen=True)
class SliceChangeStats:
    """Summary of pixel-level changes on one slice (cartilage mask only)."""

    tissue_pixels: int
    increased_pixels: int
    decreased_pixels: int
    unchanged_pixels: int
    mean_delta: float
    max_increase: float
    max_decrease: float
    unit_label: str

    @property
    def increased_percent(self) -> float:
        if self.tissue_pixels == 0:
            return 0.0
        return 100.0 * self.increased_pixels / self.tissue_pixels

    @property
    def decreased_percent(self) -> float:
        if self.tissue_pixels == 0:
            return 0.0
        return 100.0 * self.decreased_pixels / self.tissue_pixels


def load_nifti_volume(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume as a float numpy array."""
    import nibabel as nib

    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def slice_axis_length(volume: np.ndarray, axis: SliceAxis) -> int:
    """Number of slices available along the chosen anatomical axis."""
    dimension = _SLICE_AXIS_DIMENSION[axis]
    return int(volume.shape[dimension])


def extract_slice(volume: np.ndarray, axis: SliceAxis, index: int) -> np.ndarray:
    """Return a 2D slice from a 3D volume."""
    dimension = _SLICE_AXIS_DIMENSION[axis]
    if dimension == 0:
        return volume[index, :, :]
    if dimension == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def default_slice_index(volume: np.ndarray, axis: SliceAxis) -> int:
    """Pick the slice with the most non-zero voxels (cartilage present)."""
    dimension = _SLICE_AXIS_DIMENSION[axis]
    count_axes = _COUNT_AXES_FOR_DIMENSION[dimension]
    counts = np.count_nonzero(volume, axis=count_axes)
    if counts.max() == 0:
        return slice_axis_length(volume, axis) // 2
    return int(np.argmax(counts))


def load_vtk_surface_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load VTK POLYDATA (points + triangles), ignoring broken POINT_DATA blocks.

    MRCH ``segmentation_mesh.vtk`` files often fail in pyvista; geometry-only parsing
    is enough for 3D overlay.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    points: list[list[float]] = []
    faces: list[list[int]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("POINTS "):
            count = int(line.split()[1])
            index += 1
            while len(points) < count and index < len(lines):
                chunk = lines[index].strip().split()
                index += 1
                for offset in range(0, len(chunk), 3):
                    if offset + 2 < len(chunk):
                        points.append(
                            [float(chunk[offset]), float(chunk[offset + 1]), float(chunk[offset + 2])]
                        )
            continue
        if line.startswith("POLYGONS "):
            index += 1
            while len(faces) < int(line.split()[1]) and index < len(lines):
                chunk = lines[index].strip().split()
                index += 1
                offset = 0
                while offset < len(chunk):
                    vertex_count = int(chunk[offset])
                    indices = [int(chunk[offset + 1 + vertex]) for vertex in range(vertex_count)]
                    if vertex_count == 3:
                        faces.append(indices)
                    elif vertex_count == 4:
                        faces.append([indices[0], indices[1], indices[2]])
                        faces.append([indices[0], indices[2], indices[3]])
                    offset += vertex_count + 1
            continue
        if line.startswith("POINT_DATA"):
            break
        index += 1

    if not points or not faces:
        raise ValueError(f"Could not parse VTK surface from {path}")

    return np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def load_stl_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load STL via trimesh."""
    import trimesh

    mesh = trimesh.load(str(path), file_type="stl")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected one mesh in {path}")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def build_mesh_overlay_figure(
    pre_vertices: np.ndarray,
    pre_faces: np.ndarray,
    post_vertices: np.ndarray,
    post_faces: np.ndarray,
    *,
    title: str = "3D knee surface overlay",
) -> Figure:
    """Matplotlib 3D overlay of pre (blue) and post (orange) meshes."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")

    pre_tris = pre_vertices[pre_faces]
    post_tris = post_vertices[post_faces]
    axis.add_collection3d(
        Poly3DCollection(pre_tris, facecolor="#4c78a8", edgecolor="none", alpha=0.35)
    )
    axis.add_collection3d(
        Poly3DCollection(post_tris, facecolor="#f58518", edgecolor="none", alpha=0.35)
    )

    all_vertices = np.vstack([pre_vertices, post_vertices])
    axis.set_xlim(all_vertices[:, 0].min(), all_vertices[:, 0].max())
    axis.set_ylim(all_vertices[:, 1].min(), all_vertices[:, 1].max())
    axis.set_zlim(all_vertices[:, 2].min(), all_vertices[:, 2].max())
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.view_init(elev=22, azim=-58)
    figure.tight_layout()
    return figure


def build_plotly_mesh_overlay(
    pre_vertices: np.ndarray,
    pre_faces: np.ndarray,
    post_vertices: np.ndarray,
    post_faces: np.ndarray,
):
    """Interactive Plotly 3D overlay (rotatable in Streamlit)."""
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=pre_vertices[:, 0],
            y=pre_vertices[:, 1],
            z=pre_vertices[:, 2],
            i=pre_faces[:, 0],
            j=pre_faces[:, 1],
            k=pre_faces[:, 2],
            color="royalblue",
            opacity=0.4,
            name="Pre (earlier)",
        )
    )
    figure.add_trace(
        go.Mesh3d(
            x=post_vertices[:, 0],
            y=post_vertices[:, 1],
            z=post_vertices[:, 2],
            i=post_faces[:, 0],
            j=post_faces[:, 1],
            k=post_faces[:, 2],
            color="orangered",
            opacity=0.4,
            name="Post (later)",
        )
    )
    figure.update_layout(
        title="Pre / earlier (blue) vs Post / later (orange)",
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return figure


def _decode_thickness_mm(values: np.ndarray) -> np.ndarray:
    """MRCH thickness volumes often store mm × 100 as label codes (e.g. 202 → 2.02 mm)."""
    decoded = np.zeros_like(values, dtype=np.float32)
    mask = values > 0
    decoded[mask] = values[mask] / 100.0
    return decoded


MAX_REGISTRATION_SHIFT_PX = 24
REGISTRATION_SMOOTH_WINDOW = 5


@dataclass(frozen=True)
class SliceRegistration:
    """2D integer shift that moves pre cartilage toward post on one slice."""

    row_shift: int
    col_shift: int
    overlap_pixels: int


def find_mask_registration_shift(
    pre_slice: np.ndarray,
    post_slice: np.ndarray,
    *,
    max_shift: int = MAX_REGISTRATION_SHIFT_PX,
) -> SliceRegistration:
    """Find the shift that maximizes cartilage mask overlap (search all slices)."""
    from scipy.ndimage import shift as nd_shift

    pre_mask = (pre_slice > 0).astype(np.float32)
    post_mask = (post_slice > 0).astype(np.float32)
    if not pre_mask.any() or not post_mask.any():
        return SliceRegistration(0, 0, 0)

    def _score_shift(row_shift: int, col_shift: int) -> float:
        moved = nd_shift(pre_mask, (row_shift, col_shift), order=0, mode="constant")
        return float((moved * post_mask).sum())

    best_row = 0
    best_col = 0
    best_overlap = _score_shift(0, 0)
    for row_shift in range(-max_shift, max_shift + 1, 2):
        for col_shift in range(-max_shift, max_shift + 1, 2):
            overlap = _score_shift(row_shift, col_shift)
            if overlap > best_overlap:
                best_overlap = overlap
                best_row = row_shift
                best_col = col_shift

    for row_shift in range(best_row - 2, best_row + 3):
        for col_shift in range(best_col - 2, best_col + 3):
            overlap = _score_shift(row_shift, col_shift)
            if overlap > best_overlap:
                best_overlap = overlap
                best_row = row_shift
                best_col = col_shift

    return SliceRegistration(best_row, best_col, int(best_overlap))


def _smooth_shift_series(shifts: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Median-smooth shifts along the slice stack; fill empty slices from neighbors."""
    filled = shifts.astype(np.float64, copy=True)
    if not valid.any():
        return filled

    last_valid = 0.0
    for index, is_valid in enumerate(valid):
        if is_valid:
            last_valid = filled[index]
        else:
            filled[index] = last_valid

    if window <= 1:
        return filled

    from scipy.ndimage import median_filter

    return median_filter(filled, size=window, mode="nearest")


def compute_volume_registration(
    pre_volume: np.ndarray,
    post_volume: np.ndarray,
    axis: SliceAxis,
    *,
    decode_thickness: bool = False,
    max_shift: int = MAX_REGISTRATION_SHIFT_PX,
    smooth_window: int = REGISTRATION_SMOOTH_WINDOW,
) -> list[SliceRegistration]:
    """
    Register pre→post for every slice along *axis*, then smooth shifts for consistency.

    Produces stack-wide alignment similar to a manual “good” overlay on all slices.
    """
    slice_count = slice_axis_length(pre_volume, axis)
    row_shifts = np.zeros(slice_count, dtype=np.float64)
    col_shifts = np.zeros(slice_count, dtype=np.float64)
    overlaps = np.zeros(slice_count, dtype=np.int64)
    valid = np.zeros(slice_count, dtype=bool)

    for slice_index in range(slice_count):
        pre_slice = extract_slice(pre_volume, axis, slice_index).astype(np.float32)
        post_slice = extract_slice(post_volume, axis, slice_index).astype(np.float32)
        if decode_thickness:
            pre_slice = _decode_thickness_mm(pre_slice)
            post_slice = _decode_thickness_mm(post_slice)

        registration = find_mask_registration_shift(
            pre_slice,
            post_slice,
            max_shift=max_shift,
        )
        row_shifts[slice_index] = registration.row_shift
        col_shifts[slice_index] = registration.col_shift
        overlaps[slice_index] = registration.overlap_pixels
        valid[slice_index] = registration.overlap_pixels > 0

    row_shifts = _smooth_shift_series(row_shifts, valid, smooth_window)
    col_shifts = _smooth_shift_series(col_shifts, valid, smooth_window)

    return [
        SliceRegistration(
            int(round(row_shifts[index])),
            int(round(col_shifts[index])),
            int(overlaps[index]),
        )
        for index in range(slice_count)
    ]


def align_pre_slice_to_post(
    pre_slice: np.ndarray,
    post_slice: np.ndarray,
    registration: SliceRegistration | None = None,
) -> np.ndarray:
    """Shift pre toward post using per-slice registration (mask overlap)."""
    from scipy.ndimage import shift

    if registration is None:
        registration = find_mask_registration_shift(pre_slice, post_slice)

    if registration.row_shift == 0 and registration.col_shift == 0:
        return pre_slice

    return shift(
        pre_slice,
        (registration.row_shift, registration.col_shift),
        order=0,
        mode="constant",
        cval=0.0,
    )


def _slice_change_stats(
    pre_slice: np.ndarray,
    post_slice: np.ndarray,
    *,
    threshold: float,
    decode_thickness: bool,
    unit_label: str,
) -> tuple[np.ndarray, np.ndarray, SliceChangeStats]:
    """Build delta, tissue mask, and stats for one pre/post slice pair."""
    tissue_mask = (pre_slice > 0) | (post_slice > 0)
    delta = np.zeros_like(post_slice)
    delta[tissue_mask] = post_slice[tissue_mask] - pre_slice[tissue_mask]

    increased, decreased = _delta_masks(
        delta,
        tissue_mask,
        threshold=threshold,
        decode_thickness=decode_thickness,
        pre_slice=pre_slice,
        post_slice=post_slice,
    )
    unchanged = tissue_mask & ~(increased | decreased)
    tissue_count = int(tissue_mask.sum())

    stats = SliceChangeStats(
        tissue_pixels=tissue_count,
        increased_pixels=int(increased.sum()),
        decreased_pixels=int(decreased.sum()),
        unchanged_pixels=int(unchanged.sum()),
        mean_delta=float(delta[tissue_mask].mean()) if tissue_count else 0.0,
        max_increase=float(delta[increased].max()) if increased.any() else 0.0,
        max_decrease=float(delta[decreased].min()) if decreased.any() else 0.0,
        unit_label=unit_label,
    )
    return delta, tissue_mask, stats


def prepare_slice_delta(
    pre_volume: np.ndarray,
    post_volume: np.ndarray,
    *,
    axis: SliceAxis = "axial",
    slice_index: int,
    decode_thickness: bool = False,
    threshold: float = 0.05,
    align_pre_to_post: bool = False,
    registration: SliceRegistration | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SliceChangeStats]:
    """Return pre/post slices, delta, tissue mask, and change statistics."""
    pre_slice = extract_slice(pre_volume, axis, slice_index).astype(np.float32)
    post_slice = extract_slice(post_volume, axis, slice_index).astype(np.float32)

    if decode_thickness:
        pre_slice = _decode_thickness_mm(pre_slice)
        post_slice = _decode_thickness_mm(post_slice)
        unit_label = "mm"
    else:
        unit_label = "region label"

    if align_pre_to_post:
        pre_for_delta = align_pre_slice_to_post(
            pre_slice,
            post_slice,
            registration=registration,
        )
    else:
        pre_for_delta = pre_slice
    delta, tissue_mask, stats = _slice_change_stats(
        pre_for_delta,
        post_slice,
        threshold=threshold,
        decode_thickness=decode_thickness,
        unit_label=unit_label,
    )
    return pre_slice, post_slice, delta, tissue_mask, stats


def format_timepoint_caption(role: str, study_datetime: str | None, folder_name: str) -> str:
    """Build a multi-line panel title for pre/post."""
    date_line = study_datetime or "Date unknown"
    return f"{role}\n{date_line}\n{folder_name}"


def build_volume_slice_figure(
    volume: np.ndarray,
    *,
    axis: SliceAxis = "axial",
    slice_index: int | None = None,
    title: str = "",
    cmap: str = "tab20",
) -> Figure:
    """Display one anatomical slice (label map or scalar volume)."""
    if slice_index is None:
        slice_index = default_slice_index(volume, axis)
    slice_2d = extract_slice(volume, axis, slice_index)

    figure, axis_plot = plt.subplots(figsize=(5, 5))
    mask = slice_2d > 0
    display = np.zeros_like(slice_2d, dtype=float)
    display[mask] = slice_2d[mask]
    axis_plot.imshow(display, cmap=cmap, origin="lower")
    axis_plot.set_title(title or f"{axis} slice {slice_index}")
    axis_plot.axis("off")
    figure.tight_layout()
    return figure


def _delta_masks(
    delta: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    threshold: float,
    decode_thickness: bool,
    pre_slice: np.ndarray,
    post_slice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if decode_thickness:
        increased = tissue_mask & (delta > threshold)
        decreased = tissue_mask & (delta < -threshold)
    else:
        increased = tissue_mask & (post_slice != pre_slice) & (delta > 0)
        decreased = tissue_mask & (post_slice != pre_slice) & (delta < 0)
    return increased, decreased


def _draw_aligned_cartilage_contours(
    axis_plot,
    post_slice: np.ndarray,
    pre_aligned: np.ndarray,
) -> None:
    """Overlay post (green) and aligned pre (red) outlines on the same slice."""
    post_mask = post_slice > 0
    pre_mask = pre_aligned > 0
    if post_mask.any():
        axis_plot.contour(
            post_mask.astype(float),
            levels=[0.5],
            colors=[COLOR_INCREASE],
            linewidths=1.6,
            origin="lower",
        )
    if pre_mask.any():
        axis_plot.contour(
            pre_mask.astype(float),
            levels=[0.5],
            colors=[COLOR_DECREASE],
            linewidths=1.6,
            origin="lower",
        )


def _render_change_overlay_on_axis(
    axis_plot,
    post_slice: np.ndarray,
    pre_slice: np.ndarray,
    *,
    threshold: float,
    decode_thickness: bool,
    unit_label: str = "mm",
    registration: SliceRegistration | None = None,
) -> SliceChangeStats:
    """Grey post scan with aligned pre (red) and post (green) cartilage contours."""
    pre_aligned = align_pre_slice_to_post(
        pre_slice,
        post_slice,
        registration=registration,
    )
    delta, tissue_mask, stats = _slice_change_stats(
        pre_aligned,
        post_slice,
        threshold=threshold,
        decode_thickness=decode_thickness,
        unit_label=unit_label,
    )

    base = post_slice.astype(np.float32, copy=True)
    vmax = float(np.percentile(base[tissue_mask], 99)) if tissue_mask.any() else 1.0
    if vmax <= 0:
        vmax = 1.0
    axis_plot.imshow(base, cmap="gray", origin="lower", vmin=0, vmax=vmax)
    _draw_aligned_cartilage_contours(axis_plot, post_slice, pre_aligned)
    return stats


def build_change_only_figure(
    pre_slice: np.ndarray,
    post_slice: np.ndarray,
    *,
    threshold: float,
    title: str,
    decode_thickness: bool,
    unit_label: str = "mm",
    registration: SliceRegistration | None = None,
) -> tuple[Figure, SliceChangeStats]:
    """Change map: green/red overlaid together on the post scan (pre aligned to post)."""
    figure, axis_plot = plt.subplots(figsize=(6, 6), facecolor="black")
    axis_plot.set_facecolor("black")
    stats = _render_change_overlay_on_axis(
        axis_plot,
        post_slice,
        pre_slice,
        threshold=threshold,
        decode_thickness=decode_thickness,
        unit_label=unit_label,
        registration=registration,
    )

    axis_plot.set_title(title, color="white", fontsize=11, pad=10)
    axis_plot.axis("off")

    legend_items = [
        mpatches.Patch(color=COLOR_INCREASE, label="Post (later) outline"),
        mpatches.Patch(color=COLOR_DECREASE, label="Pre (earlier) outline, aligned"),
    ]
    axis_plot.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.9,
        fontsize=8,
    )

    summary = (
        f"On this slice: {stats.increased_percent:.0f}% increased · "
        f"{stats.decreased_percent:.0f}% decreased · "
        f"mean Δ {stats.mean_delta:+.2f} {stats.unit_label}"
    )
    axis_plot.text(
        0.5, 1.02, summary, transform=axis_plot.transAxes, ha="center", va="bottom",
        fontsize=8, color="white",
    )

    figure.tight_layout()
    return figure, stats


def build_slice_change_heatmap_figure(
    pre_volume: np.ndarray,
    post_volume: np.ndarray,
    *,
    axis: SliceAxis = "axial",
    slice_index: int | None = None,
    threshold: float = 0.05,
    title: str = "Change map (overlay on post)",
    decode_thickness: bool = False,
    stats: SliceChangeStats | None = None,
    registration: SliceRegistration | None = None,
) -> Figure:
    """Aligned pre/post cartilage outlines on the post (later) scan."""
    if pre_volume.shape != post_volume.shape:
        raise ValueError(
            f"Volume shapes must match for heatmap (got {pre_volume.shape} vs {post_volume.shape})"
        )

    if slice_index is None:
        slice_index = default_slice_index(post_volume, axis)

    pre_slice, post_slice, _, _, computed_stats = prepare_slice_delta(
        pre_volume,
        post_volume,
        axis=axis,
        slice_index=slice_index,
        decode_thickness=decode_thickness,
        threshold=threshold,
        align_pre_to_post=True,
        registration=registration,
    )
    stats = stats or computed_stats

    figure, axis_plot = plt.subplots(figsize=(6, 6))
    _render_change_overlay_on_axis(
        axis_plot,
        post_slice,
        pre_slice,
        threshold=threshold,
        decode_thickness=decode_thickness,
        unit_label=stats.unit_label,
        registration=registration,
    )

    axis_plot.set_title(title, fontsize=11)
    axis_plot.axis("off")
    axis_plot.legend(
        handles=[
            mpatches.Patch(color=COLOR_INCREASE, label="Post outline"),
            mpatches.Patch(color=COLOR_DECREASE, label="Pre outline (aligned)"),
        ],
        loc="lower center",
        fontsize=8,
    )
    figure.tight_layout()
    return figure


def build_clinical_comparison_figure(
    pre_volume: np.ndarray,
    post_volume: np.ndarray,
    *,
    pre_title: str,
    post_title: str,
    axis: SliceAxis = "axial",
    slice_index: int,
    threshold: float = 0.05,
    decode_thickness: bool = False,
    registration: SliceRegistration | None = None,
) -> tuple[Figure, SliceChangeStats]:
    """
    Three-panel layout: Pre | Post | Change map — intended for clinical review.
    """
    pre_slice, post_slice, _, _, stats = prepare_slice_delta(
        pre_volume,
        post_volume,
        axis=axis,
        slice_index=slice_index,
        decode_thickness=decode_thickness,
        threshold=threshold,
        align_pre_to_post=True,
        registration=registration,
    )
    unit_label = stats.unit_label

    figure, axes = plt.subplots(1, 3, figsize=(14, 5))
    figure.patch.set_facecolor("#f8fafc")

    for axis_plot, slice_data, title, border_color, cmap in (
        (axes[0], pre_slice, pre_title, COLOR_PRE_BORDER, "Blues"),
        (axes[1], post_slice, post_title, COLOR_POST_BORDER, "Oranges"),
    ):
        masked = np.ma.masked_where(slice_data <= 0, slice_data)
        axis_plot.imshow(masked, cmap=cmap, origin="lower", vmin=0)
        axis_plot.set_title(title, fontsize=10, fontweight="bold", color=border_color)
        axis_plot.axis("off")
        for spine in axis_plot.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)

    change_axis = axes[2]
    _render_change_overlay_on_axis(
        change_axis,
        post_slice,
        pre_slice,
        threshold=threshold,
        decode_thickness=decode_thickness,
        unit_label=unit_label,
        registration=registration,
    )
    overlap_note = ""
    if registration is not None and registration.overlap_pixels > 0:
        overlap_note = f" · overlap {registration.overlap_pixels:,} px"
    change_axis.set_title(
        "WHAT CHANGED\n(green = post · red = pre, aligned)",
        fontsize=10,
        fontweight="bold",
    )
    change_axis.axis("off")
    change_axis.legend(
        handles=[
            mpatches.Patch(color=COLOR_INCREASE, label="Post (later) outline"),
            mpatches.Patch(color=COLOR_DECREASE, label="Pre (earlier) outline"),
        ],
        loc="lower center",
        fontsize=8,
    )
    if registration is not None:
        change_axis.text(
            0.5,
            -0.08,
            f"Slice shift (row, col): ({registration.row_shift}, {registration.col_shift}){overlap_note}",
            transform=change_axis.transAxes,
            ha="center",
            fontsize=7,
            color="#475569",
        )

    figure.suptitle(
        f"Pre vs Post · slice {slice_index} ({axis}) · "
        f"mean change {stats.mean_delta:+.2f} {stats.unit_label}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    figure.tight_layout()
    return figure, stats


@dataclass(frozen=True)
class StudyReportMetadata:
    """Key fields from the MRChondralHealth study report PDF (page 1)."""

    pdf_path: Path
    page_count: int
    laterality: str | None
    created: str | None
    software_version: str | None
    segmentation_type: str | None
    registration_type: str | None
    field_strength: str | None
    case_comments: str | None


def _parse_study_report_text(text: str) -> dict[str, str | None]:
    """Parse MRCH case-report header fields from extracted PDF text."""

    def _match(pattern: str) -> str | None:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    comments_match = re.search(
        r"Case Comments\s*\n(.+?)(?:\n[A-Z][a-z]|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    case_comments = comments_match.group(1).strip() if comments_match else None
    if case_comments in {None, "", "None."}:
        case_comments = None

    return {
        "laterality": _match(r"Laterality:\s*([A-Za-z]+)"),
        "created": _match(r"Created:\s*([\d-]+(?:\s+[\d:]+)?)"),
        "software_version": _match(r"MR ChondralHealth Version:\s*([^\n]+)"),
        "segmentation_type": _match(r"Segmentation Type:\s*([^\n(]+)"),
        "registration_type": _match(r"Registration Type:\s*([^\n(]+)"),
        "field_strength": _match(r"MagneticFieldStrength:\s*([^\n(]+)"),
        "case_comments": case_comments,
    }


def extract_study_report_metadata(pdf_path: str | Path) -> StudyReportMetadata:
    """
    Extract study metadata from an MRCH ``*_StudyReport.pdf``.

    These PDFs are mostly tables and Siemens branding — not knee diagram images.
    """
    from pypdf import PdfReader

    path = Path(pdf_path)
    reader = PdfReader(str(path))
    header_text = "".join(page.extract_text() or "" for page in reader.pages[:2])
    fields = _parse_study_report_text(header_text)

    return StudyReportMetadata(
        pdf_path=path,
        page_count=len(reader.pages),
        laterality=fields["laterality"],
        created=fields["created"],
        software_version=fields["software_version"],
        segmentation_type=fields["segmentation_type"],
        registration_type=fields["registration_type"],
        field_strength=fields["field_strength"],
        case_comments=fields["case_comments"],
    )
