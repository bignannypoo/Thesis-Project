"""U3D import and optional conversion helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from streamlit.runtime.uploaded_file_manager import UploadedFile


@dataclass(frozen=True)
class U3DImportSummary:
    """Basic metadata extracted from an uploaded U3D file."""

    file_name: str
    file_size_bytes: int
    has_u3d_signature: bool


def summarize_u3d_file(uploaded_file: UploadedFile) -> U3DImportSummary:
    """Summarize a U3D upload with a lightweight signature check."""
    file_bytes = uploaded_file.getvalue()
    signature_window = file_bytes[:64]
    has_u3d_signature = b"U3D" in signature_window
    return U3DImportSummary(
        file_name=getattr(uploaded_file, "name", "uploaded.u3d"),
        file_size_bytes=len(file_bytes),
        has_u3d_signature=has_u3d_signature,
    )


def is_meshlab_available() -> bool:
    """Return True when meshlabserver is available for conversion."""
    return shutil.which("meshlabserver") is not None


def convert_u3d_to_stl_with_meshlab(uploaded_file: UploadedFile) -> bytes:
    """Convert U3D to STL using meshlabserver."""
    converter = shutil.which("meshlabserver")
    if converter is None:
        raise RuntimeError("meshlabserver is not installed on this machine.")

    with tempfile.TemporaryDirectory() as temp_directory:
        temp_dir = Path(temp_directory)
        input_path = temp_dir / "input.u3d"
        output_path = temp_dir / "output.stl"
        input_path.write_bytes(uploaded_file.getvalue())

        command = [
            converter,
            "-i",
            str(input_path),
            "-o",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not output_path.exists():
            stderr = (result.stderr or "").strip()
            raise RuntimeError(stderr or "U3D conversion failed.")
        return output_path.read_bytes()
