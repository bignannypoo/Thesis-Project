"""Materialize MRChondralHealth timepoint folders from Streamlit uploads."""

from __future__ import annotations

import io
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

# Streamlit is only required at runtime in the app; tests pass plain stubs.
UploadedFile = Any

_MACOSX_DIR = re.compile(r"^__MACOSX(/|$)")


def _safe_relative_path(name: str) -> Path | None:
    """Return a safe relative path inside the upload root, or None if unsafe."""
    normalized = name.replace("\\", "/").strip().lstrip("/")
    if not normalized or normalized.endswith("/"):
        return None
    parts = Path(normalized).parts
    if any(part in ("..", "") for part in parts):
        return None
    if _MACOSX_DIR.match(normalized):
        return None
    return Path(*parts)


def _unwrap_single_root_folder(root: Path) -> Path:
    """If *root* contains one directory, use it as the timepoint folder."""
    children = [path for path in root.iterdir() if path.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return root


def materialize_zip_upload(zip_bytes: bytes, *, prefix: str = "mrch_tp_") -> Path:
    """Extract a ZIP archive to a new temporary directory."""
    temp_root = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            for info in archive.infolist():
                safe = _safe_relative_path(info.filename)
                if safe is None:
                    continue
                target = temp_root / safe
                if info.is_dir() or str(info.filename).endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
        return _unwrap_single_root_folder(temp_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def materialize_files_upload(
    uploaded_files: Sequence[UploadedFile],
    *,
    prefix: str = "mrch_tp_",
) -> Path:
    """Write multiple uploaded files into a temp folder, preserving relative paths."""
    if not uploaded_files:
        raise ValueError("No files uploaded.")

    temp_root = Path(tempfile.mkdtemp(prefix=prefix))
    wrote_any = False
    try:
        for uploaded in uploaded_files:
            name = getattr(uploaded, "name", None) or "uploaded_file"
            safe = _safe_relative_path(name)
            if safe is None:
                safe = Path(Path(name).name)
            target = temp_root / safe
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(uploaded.getvalue())
            wrote_any = True
        if not wrote_any:
            raise ValueError("No valid files in upload.")
        return temp_root
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def upload_signature(
    zip_file: UploadedFile | None,
    files: Sequence[UploadedFile] | None,
) -> str | None:
    """Stable id for cache invalidation when uploads change."""
    if zip_file is not None:
        name = getattr(zip_file, "name", "zip")
        size = len(zip_file.getvalue())
        return f"zip:{name}:{size}"
    if files:
        parts = []
        for uploaded in files:
            name = getattr(uploaded, "name", "file")
            parts.append(f"{name}:{len(uploaded.getvalue())}")
        return "files:" + "|".join(sorted(parts))
    return None


def resolve_uploaded_timepoint(
    zip_file: UploadedFile | None,
    files: Sequence[UploadedFile] | None,
    *,
    cache_key: str,
    cached: dict[str, Path | str | None],
) -> Path | None:
    """
    Return a folder path for the current upload, reusing a cached temp dir when unchanged.

    *cached* is typically ``st.session_state``; keys used: ``{cache_key}_path``,
    ``{cache_key}_signature``.
    """
    signature = upload_signature(zip_file, files)
    if signature is None:
        old_path = cached.get(f"{cache_key}_path")
        if old_path is not None:
            shutil.rmtree(Path(old_path), ignore_errors=True)
        cached[f"{cache_key}_path"] = None
        cached[f"{cache_key}_signature"] = None
        return None

    if cached.get(f"{cache_key}_signature") == signature:
        existing = cached.get(f"{cache_key}_path")
        if existing is not None and Path(existing).is_dir():
            return Path(existing)

    old_path = cached.get(f"{cache_key}_path")
    if old_path is not None:
        shutil.rmtree(Path(old_path), ignore_errors=True)

    if zip_file is not None:
        folder = materialize_zip_upload(zip_file.getvalue())
    else:
        folder = materialize_files_upload(files or [])

    cached[f"{cache_key}_path"] = str(folder)
    cached[f"{cache_key}_signature"] = signature
    return folder


def count_uploaded_files(files: Iterable[UploadedFile] | None) -> int:
    return len(list(files)) if files else 0
