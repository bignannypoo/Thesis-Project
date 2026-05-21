"""Tests for knee imaging asset discovery and VTK parsing."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from cartilage_comparison.asset_discovery import discover_timepoint_assets
from cartilage_comparison.knee_imaging import (
    build_slice_change_heatmap_figure,
    default_slice_index,
    load_vtk_surface_mesh,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
PATIENT_PRE = Path("/Users/raytan/Desktop/MRCH_Comparison/Pre/Patient 3 pre robust")
PATIENT_POST = Path("/Users/raytan/Desktop/MRCH_Comparison/Post/Patient 3 post robust")


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
