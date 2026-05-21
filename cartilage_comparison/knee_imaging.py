"""2D/3D knee visuals: NIfTI slices, VTK/STL meshes, PDF diagrams, change heatmaps."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

SliceAxis = Literal["axial", "coronal", "sagittal"]
DifferenceView = Literal["clinical_panel", "overlay", "change_only"]

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


def extract_slice(volume: np.ndarray, axis: SliceAxis, index: int) -> np.ndarray:
    """Return a 2D slice from a 3D volume."""
    if axis == "axial":
        return volume[:, :, index]
    if axis == "coronal":
        return volume[:, index, :]
    return volume[index, :, :]


def default_slice_index(volume: np.ndarray, axis: SliceAxis) -> int:
    """Pick the slice with the most non-zero voxels (cartilage present)."""
    if axis == "axial":
        counts = np.count_nonzero(volume, axis=(0, 1))
    elif axis == "coronal":
        counts = np.count_nonzero(volume, axis=(0, 2))
    else:
        counts = np.count_nonzero(volume, axis=(1, 2))
    if counts.max() == 0:
        return volume.shape[{"axial": 2, "coronal": 1, "sagittal": 0}[axis]] // 2
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


def prepare_slice_delta(
    pre_volume: np.ndarray,
    post_volume: np.ndarray,
    *,
    axis: SliceAxis = "axial",
    slice_index: int,
    decode_thickness: bool = False,
    threshold: float = 0.05,
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


def build_change_only_figure(
    pre_slice: np.ndarray,
    post_slice: np.ndarray,
    delta: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    threshold: float,
    title: str,
    decode_thickness: bool,
    stats: SliceChangeStats | None = None,
) -> Figure:
    """High-contrast change map: only green/red on cartilage (easiest to read)."""
    increased, decreased = _delta_masks(
        delta, tissue_mask, threshold=threshold, decode_thickness=decode_thickness,
        pre_slice=pre_slice, post_slice=post_slice,
    )

    figure, axis_plot = plt.subplots(figsize=(6, 6), facecolor="black")
    axis_plot.set_facecolor("black")
    canvas = np.zeros((*delta.shape, 3), dtype=float)
    canvas[tissue_mask] = [0.15, 0.15, 0.15]
    canvas[increased] = COLOR_INCREASE
    canvas[decreased] = COLOR_DECREASE
    axis_plot.imshow(canvas, origin="lower", interpolation="nearest")

    for mask, color in ((increased, COLOR_INCREASE), (decreased, COLOR_DECREASE)):
        if mask.any():
            axis_plot.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.2)

    axis_plot.set_title(title, color="white", fontsize=11, pad=10)
    axis_plot.axis("off")

    legend_items = [
        mpatches.Patch(color=COLOR_INCREASE, label="Increase (post > pre)"),
        mpatches.Patch(color=COLOR_DECREASE, label="Decrease (post < pre)"),
        mpatches.Patch(color=(0.15, 0.15, 0.15), label="No change on slice"),
    ]
    axis_plot.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.9,
        fontsize=8,
    )

    if stats is not None:
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
    return figure


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
) -> Figure:
    """Overlay red/green change on the post (later) scan."""
    if pre_volume.shape != post_volume.shape:
        raise ValueError(
            f"Volume shapes must match for heatmap (got {pre_volume.shape} vs {post_volume.shape})"
        )

    if slice_index is None:
        slice_index = default_slice_index(post_volume, axis)

    pre_slice, post_slice, delta, tissue_mask, computed_stats = prepare_slice_delta(
        pre_volume, post_volume, axis=axis, slice_index=slice_index,
        decode_thickness=decode_thickness, threshold=threshold,
    )
    stats = stats or computed_stats

    base = post_slice.copy()
    vmax = float(np.percentile(base[tissue_mask], 99)) if tissue_mask.any() else 1.0
    if vmax <= 0:
        vmax = 1.0

    figure, axis_plot = plt.subplots(figsize=(6, 6))
    axis_plot.imshow(base, cmap="gray", origin="lower", vmin=0, vmax=vmax)

    increased, decreased = _delta_masks(
        delta, tissue_mask, threshold=threshold, decode_thickness=decode_thickness,
        pre_slice=pre_slice, post_slice=post_slice,
    )
    overlay = np.zeros((*delta.shape, 4), dtype=float)
    overlay[increased] = [*COLOR_INCREASE, 0.85]
    overlay[decreased] = [*COLOR_DECREASE, 0.85]
    axis_plot.imshow(overlay, origin="lower")

    axis_plot.set_title(title, fontsize=11)
    axis_plot.axis("off")
    axis_plot.legend(
        handles=[
            mpatches.Patch(color=COLOR_INCREASE, label="Increase"),
            mpatches.Patch(color=COLOR_DECREASE, label="Decrease"),
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
) -> tuple[Figure, SliceChangeStats]:
    """
    Three-panel layout: Pre | Post | Change map — intended for clinical review.
    """
    pre_slice, post_slice, delta, tissue_mask, stats = prepare_slice_delta(
        pre_volume, post_volume, axis=axis, slice_index=slice_index,
        decode_thickness=decode_thickness, threshold=threshold,
    )

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
    increased, decreased = _delta_masks(
        delta, tissue_mask, threshold=threshold, decode_thickness=decode_thickness,
        pre_slice=pre_slice, post_slice=post_slice,
    )
    canvas = np.ones((*delta.shape, 3), dtype=float)
    canvas[tissue_mask] = 0.92
    canvas[increased] = COLOR_INCREASE
    canvas[decreased] = COLOR_DECREASE
    change_axis.imshow(canvas, origin="lower", interpolation="nearest")
    change_axis.set_title(
        "WHAT CHANGED\n(green = more · red = less)",
        fontsize=10,
        fontweight="bold",
    )
    change_axis.axis("off")
    change_axis.legend(
        handles=[
            mpatches.Patch(color=COLOR_INCREASE, label=f"Increase ({stats.increased_percent:.0f}% of cartilage)"),
            mpatches.Patch(color=COLOR_DECREASE, label=f"Decrease ({stats.decreased_percent:.0f}% of cartilage)"),
        ],
        loc="lower center",
        fontsize=8,
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


def extract_pdf_diagram_images(pdf_path: str | Path, max_pages: int = 6) -> list[Image.Image]:
    """Extract embedded images from the MRCH study report PDF."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    images: list[Image.Image] = []
    for page_index, page in enumerate(reader.pages[:max_pages]):
        for image_file in getattr(page, "images", {}).values():
            try:
                images.append(Image.open(io.BytesIO(image_file.data)).convert("RGB"))
            except OSError:
                continue
        if page_index >= max_pages:
            break
    return images


def build_side_by_side_pil_figure(
    pre_image: Image.Image,
    post_image: Image.Image,
    *,
    title: str = "",
) -> Figure:
    """Show two PIL images side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pre_image)
    axes[0].set_title("Timepoint 1")
    axes[0].axis("off")
    axes[1].imshow(post_image)
    axes[1].set_title("Timepoint 2")
    axes[1].axis("off")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    return figure
