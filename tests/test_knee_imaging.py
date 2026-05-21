"""Tests for knee imaging asset discovery and VTK parsing."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from cartilage_comparison.asset_discovery import discover_timepoint_assets
from cartilage_comparison.knee_imaging import (
    _iter_page_embedded_images,
    align_pre_slice_to_post,
    build_slice_change_heatmap_figure,
    compute_volume_registration,
    default_slice_index,
    extract_slice,
    find_mask_registration_shift,
    load_vtk_surface_mesh,
    slice_axis_length,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
PATIENT_PRE = Path("/Users/raytan/Desktop/MRCH_Comparison/Pre/Patient 3 pre robust")
PATIENT_POST = Path("/Users/raytan/Desktop/MRCH_Comparison/Post/Patient 3 post robust")


class _FakeImageFile:
    def __init__(self, data: bytes) -> None:
        self.data = data


class _FakeVirtualListImages(list):
    """Mimics pypdf 5 ``VirtualListImages`` (sequence without ``.values()``)."""


def test_iter_page_embedded_images_supports_virtual_list() -> None:
    class _Page:
        images = _FakeVirtualListImages([_FakeImageFile(b"\x00")])

    assert len(list(_iter_page_embedded_images(_Page()))) == 1


def test_iter_page_embedded_images_supports_dict() -> None:
    class _Page:
        images = {"img0": _FakeImageFile(b"\x01")}

    assert len(list(_iter_page_embedded_images(_Page()))) == 1


def test_find_mask_registration_shift_moves_pre_to_post() -> None:
    pre_slice = np.zeros((30, 30), dtype=np.float32)
    pre_slice[4:10, 8:20] = 2.0
    post_slice = np.zeros((30, 30), dtype=np.float32)
    post_slice[14:20, 8:20] = 2.0

    registration = find_mask_registration_shift(pre_slice, post_slice, max_shift=12)
    assert registration.row_shift == 10
    assert registration.col_shift == 0
    aligned = align_pre_slice_to_post(pre_slice, post_slice, registration=registration)
    assert aligned[14:20, 8:20].sum() > 0.0
    assert aligned[4:10, 8:20].sum() == 0.0


def test_compute_volume_registration_smooths_across_slices() -> None:
    pre_volume = np.zeros((5, 20, 20), dtype=np.float32)
    post_volume = np.zeros((5, 20, 20), dtype=np.float32)
    for index in range(5):
        row_offset = 6 + index
        pre_volume[index, 4:10, 6:14] = 1.0
        post_volume[index, row_offset : row_offset + 6, 6:14] = 1.0

    registrations = compute_volume_registration(
        pre_volume,
        post_volume,
        "sagittal",
        smooth_window=3,
    )
    assert len(registrations) == 5
    assert registrations[2].overlap_pixels > 0
    assert registrations[2].row_shift > 0


def test_axial_and_coronal_slice_axes_are_swapped_for_mrch() -> None:
    volume = np.zeros((10, 20, 30), dtype=np.float32)
    assert extract_slice(volume, "axial", 5).shape == (10, 30)
    assert extract_slice(volume, "coronal", 7).shape == (10, 20)
    assert extract_slice(volume, "sagittal", 3).shape == (20, 30)
    assert slice_axis_length(volume, "axial") == 20
    assert slice_axis_length(volume, "coronal") == 30
    assert slice_axis_length(volume, "sagittal") == 10


def test_discover_assets_in_fixtures() -> None:
    assets = discover_timepoint_assets(FIXTURES_DIR / "mrch_pre")
    assert assets.nifti_files == {}


@pytest.mark.skipif(not PATIENT_PRE.is_dir(), reason="Patient 3 pre folder not on machine")
def test_load_vtk_mesh_patient3() -> None:
    vtk_path = PATIENT_PRE / "segmentation_mesh.vtk"
    vertices, faces = load_vtk_surface_mesh(vtk_path)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(vertices) > 1000
    assert len(faces) > 1000


@pytest.mark.skipif(not PATIENT_PRE.is_dir(), reason="Patient 3 folders not on machine")
def test_nifti_slice_heatmap_patient3() -> None:
    import nibabel as nib

    pre = nib.load(str(PATIENT_PRE / "thickness_bci.registered.nii")).get_fdata()
    post = nib.load(str(PATIENT_POST / "thickness_bci.registered.nii")).get_fdata()
    index = default_slice_index(pre, "axial")
    figure = build_slice_change_heatmap_figure(
        pre,
        post,
        slice_index=index,
        decode_thickness=True,
    )
    assert figure.axes
