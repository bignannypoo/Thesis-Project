"""Tests for U3D import helpers."""

from cv.u3d_import import summarize_u3d_file


class _UploadedFileStub:
    """Simple UploadedFile-compatible stub for tests."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_summarize_u3d_file_detects_signature() -> None:
    uploaded = _UploadedFileStub("model.u3d", b"ABCDU3Dxxxx")
    summary = summarize_u3d_file(uploaded)  # type: ignore[arg-type]

    assert summary.file_name == "model.u3d"
    assert summary.file_size_bytes == 11
    assert summary.has_u3d_signature is True


def test_summarize_u3d_file_without_signature() -> None:
    uploaded = _UploadedFileStub("bad.u3d", b"not-a-real-signature")
    summary = summarize_u3d_file(uploaded)  # type: ignore[arg-type]

    assert summary.has_u3d_signature is False
