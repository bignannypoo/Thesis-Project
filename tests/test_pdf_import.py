"""Tests for PDF import helpers."""

from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from cv.pdf_import import import_pdf_report


class _UploadedFileStub:
    """Simple UploadedFile-compatible stub for tests."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _build_sample_pdf_bytes() -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.drawString(100, 750, "Patient: Jane Doe")
    pdf.drawString(100, 730, "MRN: 12345")
    pdf.drawString(100, 710, "Comparison Date: 2026-05-05")
    pdf.save()
    return buffer.getvalue()


def test_import_pdf_report_extracts_fields() -> None:
    uploaded = _UploadedFileStub(_build_sample_pdf_bytes())
    summary = import_pdf_report(uploaded)  # type: ignore[arg-type]

    assert summary.page_count == 1
    assert summary.patient_name == "Jane Doe"
    assert summary.mrn == "12345"
    assert summary.comparison_date == "2026-05-05"
    assert "Patient: Jane Doe" in summary.extracted_text
