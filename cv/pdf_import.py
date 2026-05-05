"""PDF import helpers for extracting report text and key fields."""

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile


@dataclass(frozen=True)
class ImportedPdfSummary:
    """Structured summary extracted from an uploaded PDF report."""

    page_count: int
    extracted_text: str
    patient_name: str | None
    mrn: str | None
    comparison_date: str | None


_PATIENT_PATTERN = re.compile(r"(?:patient|name)\s*[:\-]\s*([^\n\r:]{2,80})", re.IGNORECASE)
_MRN_PATTERN = re.compile(r"\bMRN\s*[:#-]?\s*([A-Z0-9-]{4,20})\b", re.IGNORECASE)
_DATE_SEPARATOR = r"(?: |/|-)"
_DATE_PATTERN = re.compile(
    rf"\b(\d{{1,2}}{_DATE_SEPARATOR}[A-Za-z]{{3}}{_DATE_SEPARATOR}\d{{4}}|\d{{4}}{_DATE_SEPARATOR}\d{{2}}{_DATE_SEPARATOR}\d{{2}})\b"
)


def import_pdf_report(uploaded_file: UploadedFile) -> ImportedPdfSummary:
    """Extract text and common identifiers from a PDF upload."""
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    extracted_chunks: list[str] = []
    for page in reader.pages:
        extracted_chunks.append(page.extract_text() or "")
    full_text = "\n".join(extracted_chunks).strip()

    patient_match = _PATIENT_PATTERN.search(full_text)
    mrn_match = _MRN_PATTERN.search(full_text)
    date_match = _DATE_PATTERN.search(full_text)

    return ImportedPdfSummary(
        page_count=len(reader.pages),
        extracted_text=full_text,
        patient_name=patient_match.group(1).strip() if patient_match else None,
        mrn=mrn_match.group(1).strip() if mrn_match else None,
        comparison_date=date_match.group(1).strip() if date_match else None,
    )
