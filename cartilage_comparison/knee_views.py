"""Streamlit UI blocks for knee imaging (meshes, NIfTI, PDF, STL)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from cartilage_comparison.asset_discovery import TimepointAssets, discover_timepoint_assets
from cartilage_comparison.data_loader import TimepointData
from cartilage_comparison.knee_imaging import (
    SliceAxis,
    SliceRegistration,
    build_change_only_figure,
    build_clinical_comparison_figure,
    build_mesh_overlay_figure,
    build_plotly_mesh_overlay,
    build_slice_change_heatmap_figure,
    compute_volume_registration,
    default_slice_index,
    extract_study_report_metadata,
    format_timepoint_caption,
    load_nifti_volume,
    load_stl_mesh,
    load_vtk_surface_mesh,
    slice_axis_length,
)

RegistrationTable = list[tuple[int, int, int]]


@st.cache_data(show_spinner="Aligning pre→post across all slices (once per volume)…")
def _cached_nifti_volumes_and_registration(
    pre_path: str,
    post_path: str,
    axis: SliceAxis,
    decode_thickness: bool,
) -> tuple[bytes, bytes, tuple[int, int, int], RegistrationTable]:
    """
    Load NIfTI pair and register every slice; cached until paths/axis/decode change.

    Volumes are returned as bytes so Streamlit can hash/cache them without reloading
    from disk on every slice slider movement.
    """
    pre_volume = load_nifti_volume(pre_path)
    post_volume = load_nifti_volume(post_path)
    registrations = compute_volume_registration(
        pre_volume,
        post_volume,
        axis,
        decode_thickness=decode_thickness,
    )
    table = [
        (entry.row_shift, entry.col_shift, entry.overlap_pixels)
        for entry in registrations
    ]
    return (
        pre_volume.tobytes(),
        post_volume.tobytes(),
        pre_volume.shape,
        table,
    )


def _restore_cached_volumes(
    pre_bytes: bytes,
    post_bytes: bytes,
    shape: tuple[int, int, int],
) -> tuple:
    import numpy as np

    pre_volume = np.frombuffer(pre_bytes, dtype=np.float32).reshape(shape)
    post_volume = np.frombuffer(post_bytes, dtype=np.float32).reshape(shape)
    return pre_volume, post_volume


def _registration_at_index(table: RegistrationTable, slice_index: int) -> SliceRegistration:
    row_shift, col_shift, overlap_pixels = table[slice_index]
    return SliceRegistration(row_shift, col_shift, overlap_pixels)


def _discover_pair(tp1: TimepointData, tp2: TimepointData) -> tuple[TimepointAssets, TimepointAssets]:
    return discover_timepoint_assets(tp1.folder), discover_timepoint_assets(tp2.folder)


def _render_timepoint_legend(tp1: TimepointData, tp2: TimepointData) -> None:
    """Explain which sidebar folder is pre vs post."""
    st.info(
        "**Which scan is which?** The first folder you enter in the sidebar is **Pre (earlier)**. "
        "The second folder is **Post (later)**. "
        "This is not MRI “T1/T2 weighting” — it is timepoint 1 vs timepoint 2."
    )
    col_pre, col_post = st.columns(2)
    with col_pre:
        st.markdown(
            f"**Pre (earlier)** — `{tp1.folder.name}`  \n"
            f"Study date: **{tp1.study_datetime or 'unknown'}**"
        )
    with col_post:
        st.markdown(
            f"**Post (later)** — `{tp2.folder.name}`  \n"
            f"Study date: **{tp2.study_datetime or 'unknown'}**"
        )


def render_knee_imaging_section(tp1: TimepointData, tp2: TimepointData) -> None:
    """Render 3D/2D knee visuals discovered from both timepoint folders."""
    st.subheader("Knee imaging (from timepoint folders)")
    _render_timepoint_legend(tp1, tp2)

    try:
        assets_1, assets_2 = _discover_pair(tp1, tp2)
    except NotADirectoryError as error:
        st.error(str(error))
        return

    tab_nifti, tab_mesh, tab_report, tab_stl = st.tabs(
        ["MRI volume slices", "3D surface mesh", "Study report", "STL (optional)"],
    )

    with tab_nifti:
        _render_nifti_tab(assets_1, assets_2, tp1, tp2)

    with tab_mesh:
        _render_vtk_tab(assets_1, assets_2, tp1, tp2)

    with tab_report:
        _render_study_report_tab(assets_1, assets_2, tp1, tp2)

    with tab_stl:
        _render_stl_tab(assets_1, assets_2, tp1, tp2)


def _metadata_rows(metadata) -> list[tuple[str, str]]:
    """Turn study report metadata into label/value rows for display."""
    return [
        ("Created", metadata.created or "—"),
        ("Laterality", metadata.laterality or "—"),
        ("Field strength", metadata.field_strength or "—"),
        ("Segmentation", metadata.segmentation_type or "—"),
        ("Registration", metadata.registration_type or "—"),
        ("Software version", metadata.software_version or "—"),
        ("Pages", str(metadata.page_count)),
        ("Case comments", metadata.case_comments or "—"),
    ]


def _render_study_report_tab(
    assets_1: TimepointAssets,
    assets_2: TimepointAssets,
    tp1: TimepointData,
    tp2: TimepointData,
) -> None:
    if assets_1.study_report_pdf is None and assets_2.study_report_pdf is None:
        st.warning("No *_StudyReport.pdf found in either folder.")
        return

    st.info(
        "MRChondralHealth study reports are **PDF tables and metadata**, not knee images. "
        "Embedded graphics are usually Siemens branding only. "
        "Use **MRI volume slices** for visual pre/post comparison; regional metrics are in the "
        "**Comparison table** tab (from the MRCH CSV)."
    )

    col1, col2 = st.columns(2)
    for column, assets, tp, role in (
        (col1, assets_1, tp1, "Pre (earlier)"),
        (col2, assets_2, tp2, "Post (later)"),
    ):
        with column:
            st.markdown(f"**{role}** — {tp.study_datetime or 'date unknown'}")
            if assets.study_report_pdf is None:
                st.info("No study report PDF.")
                continue
            try:
                metadata = extract_study_report_metadata(assets.study_report_pdf)
            except Exception as error:
                st.error(f"Could not read PDF: {error}")
                continue

            for label, value in _metadata_rows(metadata):
                st.markdown(f"**{label}:** {value}")

            pdf_bytes = assets.study_report_pdf.read_bytes()
            st.download_button(
                label=f"Download {assets.study_report_pdf.name}",
                data=pdf_bytes,
                file_name=assets.study_report_pdf.name,
                mime="application/pdf",
                key=f"download_study_report_{role}",
            )


def _render_nifti_tab(
    assets_1: TimepointAssets,
    assets_2: TimepointAssets,
    tp1: TimepointData,
    tp2: TimepointData,
) -> None:
    shared_names = sorted(set(assets_1.nifti_files) & set(assets_2.nifti_files))
    if not shared_names:
        st.warning("No matching `.nii` files in both folders.")
        with st.expander("Files found per folder"):
            st.write("Pre", list(assets_1.nifti_files.keys()) or "none")
            st.write("Post", list(assets_2.nifti_files.keys()) or "none")
        return

    pre_title = format_timepoint_caption("PRE (earlier)", tp1.study_datetime, tp1.folder.name)
    post_title = format_timepoint_caption("POST (later)", tp2.study_datetime, tp2.folder.name)

    default_volume = "thickness_bci.registered.nii"
    volume_options = shared_names
    default_index = volume_options.index(default_volume) if default_volume in volume_options else 0

    volume_name = st.selectbox(
        "NIfTI volume",
        volume_options,
        index=default_index,
        key="knee_nifti_name",
        help="For thickness change maps, use thickness_bci.registered.nii. "
        "layering_mask / morphological show region labels (different colours = different segments).",
    )
    if "layering_mask" in volume_name or "morphological" in volume_name:
        st.warning(
            "This volume is a **region label map** (colours = anatomy segments), not thickness in mm. "
            "The change panel highlights **where segments changed** between pre and post. "
            "For thickness gain/loss in millimetres, select **thickness_bci.registered.nii**."
        )

    axis = st.selectbox("Slice axis", ["coronal", "axial", "sagittal"], key="knee_nifti_axis")
    view_mode = st.radio(
        "Layout",
        [
            "Clinical 3-panel (Pre | Post | Changes) — recommended",
            "Change only (high contrast)",
            "Overlay on post scan",
        ],
        horizontal=True,
        key="knee_view_mode",
    )
    decode_thickness = "thickness" in volume_name and st.checkbox(
        "Thickness in millimetres (decode ÷100)",
        value=True,
        key="knee_decode_thickness",
    )
    threshold = st.slider(
        "Minimum change to highlight (mm)",
        0.01,
        0.5,
        0.05,
        0.01,
        disabled=not decode_thickness,
        key="knee_change_threshold",
        help="Smaller = more sensitive (more green/red). Only applies to thickness volumes.",
    )

    pre_path = str(assets_1.nifti_files[volume_name])
    post_path = str(assets_2.nifti_files[volume_name])

    try:
        pre_bytes, post_bytes, volume_shape, registration_table = _cached_nifti_volumes_and_registration(
            pre_path,
            post_path,
            axis,
            decode_thickness,
        )
        pre_volume, post_volume = _restore_cached_volumes(pre_bytes, post_bytes, volume_shape)
    except Exception as error:
        st.error(f"Could not load NIfTI: {error}")
        return

    max_index = slice_axis_length(pre_volume, axis) - 1
    default_slice = default_slice_index(post_volume, axis)
    slice_index = st.slider("Slice index", 0, max_index, default_slice, key="knee_slice_index")
    st.caption(
        "Slice alignment is computed once per NIfTI volume and cached — moving the slider should be quick."
    )

    slice_registration = _registration_at_index(registration_table, slice_index)
    if slice_registration.overlap_pixels == 0:
        st.warning("Little or no overlapping cartilage on this slice.")

    if view_mode.startswith("Clinical"):
        figure, stats = build_clinical_comparison_figure(
            pre_volume,
            post_volume,
            pre_title=pre_title,
            post_title=post_title,
            axis=axis,
            slice_index=slice_index,
            threshold=threshold,
            decode_thickness=decode_thickness,
            registration=slice_registration,
        )
        st.pyplot(figure)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cartilage pixels (slice)", f"{stats.tissue_pixels:,}")
        m2.metric("Increased", f"{stats.increased_percent:.1f}%", help="Post thicker / larger than pre")
        m3.metric("Decreased", f"{stats.decreased_percent:.1f}%", help="Post thinner / smaller than pre")
        m4.metric("Mean change", f"{stats.mean_delta:+.2f} {stats.unit_label}")

    elif view_mode.startswith("Change only"):
        from cartilage_comparison.knee_imaging import prepare_slice_delta

        pre_slice, post_slice, _, _, _ = prepare_slice_delta(
            pre_volume,
            post_volume,
            axis=axis,
            slice_index=slice_index,
            decode_thickness=decode_thickness,
            threshold=threshold,
        )
        unit_label = "mm" if decode_thickness else "region label"
        figure, stats = build_change_only_figure(
            pre_slice,
            post_slice,
            threshold=threshold,
            title="CHANGES: green = post · red = pre (aligned)",
            decode_thickness=decode_thickness,
            unit_label=unit_label,
            registration=slice_registration,
        )
        st.pyplot(figure)
        st.caption(
            f"Compared **pre** ({tp1.study_datetime or '?'}) → **post** ({tp2.study_datetime or '?'}). "
            f"On this slice, {stats.increased_percent:.0f}% of cartilage voxels increased and "
            f"{stats.decreased_percent:.0f}% decreased."
        )

    else:
        st.pyplot(
            build_slice_change_heatmap_figure(
                pre_volume,
                post_volume,
                axis=axis,
                slice_index=slice_index,
                title="Aligned outlines on POST (later) scan",
                decode_thickness=decode_thickness,
                threshold=threshold,
                registration=slice_registration,
            )
        )
        st.caption(
            "Grey = post scan. Green = post cartilage outline · Red = pre outline (aligned per slice)."
        )


def _render_vtk_tab(
    assets_1: TimepointAssets,
    assets_2: TimepointAssets,
    tp1: TimepointData,
    tp2: TimepointData,
) -> None:
    if assets_1.vtk_mesh is None or assets_2.vtk_mesh is None:
        st.warning(
            "Both folders need a VTK mesh (e.g. `segmentation_mesh.vtk`). "
            f"Pre: {assets_1.vtk_mesh or 'missing'} · Post: {assets_2.vtk_mesh or 'missing'}"
        )
        return

    st.caption(
        f"**Pre** {tp1.study_datetime or '?'} (blue) vs **Post** {tp2.study_datetime or '?'} (orange)"
    )
    renderer = st.radio("3D viewer", ["Interactive (Plotly)", "Static (matplotlib)"], horizontal=True)

    try:
        pre_vertices, pre_faces = load_vtk_surface_mesh(assets_1.vtk_mesh)
        post_vertices, post_faces = load_vtk_surface_mesh(assets_2.vtk_mesh)
    except Exception as error:
        st.error(f"Could not load VTK meshes: {error}")
        return

    if renderer.startswith("Interactive"):
        figure = build_plotly_mesh_overlay(pre_vertices, pre_faces, post_vertices, post_faces)
        figure.update_layout(title="Pre (blue) vs Post (orange) — 3D cartilage surface")
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.pyplot(
            build_mesh_overlay_figure(
                pre_vertices,
                pre_faces,
                post_vertices,
                post_faces,
                title="Pre (blue) vs Post (orange)",
            )
        )


def _render_stl_tab(
    assets_1: TimepointAssets,
    assets_2: TimepointAssets,
    tp1: TimepointData,
    tp2: TimepointData,
) -> None:
    st.markdown("**Optional STL paths** (leave blank to use `.stl` files found in folders)")

    col1, col2 = st.columns(2)
    with col1:
        stl_path_1 = st.text_input(
            "Pre (earlier) STL",
            value=str(assets_1.stl_files[0]) if assets_1.stl_files else "",
            key="knee_stl_1",
        )
    with col2:
        stl_path_2 = st.text_input(
            "Post (later) STL",
            value=str(assets_2.stl_files[0]) if assets_2.stl_files else "",
            key="knee_stl_2",
        )

    if not stl_path_1.strip() or not stl_path_2.strip():
        st.info(
            "No STL in MRCH folders by default. Export STL from MRChondralHealth or MeshLab, "
            "then paste paths above."
        )
        return

    try:
        pre_vertices, pre_faces = load_stl_mesh(stl_path_1)
        post_vertices, post_faces = load_stl_mesh(stl_path_2)
    except Exception as error:
        st.error(str(error))
        return

    figure = build_plotly_mesh_overlay(pre_vertices, pre_faces, post_vertices, post_faces)
    figure.update_layout(title="Pre (blue) vs Post (orange) — STL")
    st.plotly_chart(figure, use_container_width=True)
