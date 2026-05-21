"""Discover visual assets inside an MRChondralHealth timepoint folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

NIFTI_PRIORITY: tuple[str, ...] = (
    "morphological.nii",
    "morphological.registered.nii",
    "thickness_bci.registered.nii",
    "thickness_bci.nii",
    "layering_mask.registered.nii",
    "biochemical.nii",
)

VTK_MESH_NAMES: tuple[str, ...] = (
    "segmentation_mesh.vtk",
    "segmentation_mask.registered.confirmed.wem",
)


@dataclass(frozen=True)
class TimepointAssets:
    """Paths to imaging files found under one timepoint directory."""

    folder: Path
    nifti_files: dict[str, Path]
    vtk_mesh: Path | None
    study_report_pdf: Path | None
    stl_files: tuple[Path, ...]


def discover_timepoint_assets(folder: str | Path) -> TimepointAssets:
    """Scan folder (non-recursive first, then shallow rglob) for known MRCH outputs."""
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Timepoint folder not found: {folder_path}")

    nifti_files: dict[str, Path] = {}
    for name in NIFTI_PRIORITY:
        direct = folder_path / name
        if direct.is_file():
            nifti_files[name] = direct
            continue
        matches = sorted(folder_path.rglob(name))
        if matches:
            nifti_files[name] = matches[0]

    vtk_mesh: Path | None = None
    for name in VTK_MESH_NAMES:
        direct = folder_path / name
        if direct.is_file() and name.endswith(".vtk"):
            vtk_mesh = direct
            break
    if vtk_mesh is None:
        vtk_matches = sorted(folder_path.rglob("*.vtk"))
        if vtk_matches:
            vtk_mesh = vtk_matches[0]

    pdf_matches = sorted(folder_path.rglob("*_StudyReport.pdf"))
    if not pdf_matches:
        pdf_matches = sorted(folder_path.rglob("*StudyReport*.pdf"))
    study_report_pdf = pdf_matches[0] if pdf_matches else None

    stl_files = tuple(sorted({*folder_path.glob("*.stl"), *folder_path.rglob("*.stl")}))

    return TimepointAssets(
        folder=folder_path,
        nifti_files=nifti_files,
        vtk_mesh=vtk_mesh,
        study_report_pdf=study_report_pdf,
        stl_files=stl_files,
    )
