"""Demo registry and pure lookup helpers (no Streamlit)."""

from typing import Tuple

from cv.constants import DEFAULT_MODALITY, STATUS_ACTIVE
from cv.models import ImagingSession, PatientRecord

MOCK_PATIENTS: Tuple[PatientRecord, ...] = (
    PatientRecord(
        patient_id="pt-001",
        display_name="Tan Wei Ming",
        mrn="048291",
        dob="12 Mar 1978",
        age=47,
        sex="M",
        joint="Right knee",
        treating="Dr Lim Siew Hua",
        treatment_type="MACI",
        treatment_started="Feb 2024",
        status=STATUS_ACTIVE,
        sessions=(
            ImagingSession("s-001-pre", "pre", "14 Jan 2024", DEFAULT_MODALITY, "Baseline"),
            ImagingSession("s-001-post-9m", "post", "18 Oct 2024", DEFAULT_MODALITY, "9 mo follow-up"),
            ImagingSession("s-001-post-14m", "followup", "03 Mar 2025", DEFAULT_MODALITY, "14 mo follow-up"),
        ),
    ),
    PatientRecord(
        patient_id="pt-002",
        display_name="Priya Nair",
        mrn="081482",
        dob="22 Aug 1982",
        age=43,
        sex="F",
        joint="Left hip",
        treating="Dr Ong Boon Kheng",
        treatment_type="Hip resurfacing",
        treatment_started="Jun 2023",
        status=STATUS_ACTIVE,
        sessions=(
            ImagingSession("s-002-pre", "pre", "15 Jun 2023", DEFAULT_MODALITY, "Baseline"),
            ImagingSession("s-002-post-6m", "post", "28 Dec 2023", DEFAULT_MODALITY, "6 mo follow-up"),
            ImagingSession("s-002-post-14m", "followup", "28 Aug 2024", DEFAULT_MODALITY, "14 mo follow-up"),
        ),
    ),
    PatientRecord(
        patient_id="pt-003",
        display_name="Ahmad Razif",
        mrn="073519",
        dob="10 Sep 1990",
        age=35,
        sex="M",
        joint="Right shoulder",
        treating="Dr Tan Soo Kee",
        treatment_type="Rotator cuff repair",
        treatment_started="Mar 2024",
        status=STATUS_ACTIVE,
        sessions=(
            ImagingSession("s-003-pre", "pre", "05 Mar 2024", DEFAULT_MODALITY, "Baseline"),
            ImagingSession("s-003-post-6m", "post", "05 Sep 2024", DEFAULT_MODALITY, "6 mo follow-up"),
        ),
    ),
    PatientRecord(
        patient_id="pt-004",
        display_name="Siti Rahimah",
        mrn="082740",
        dob="03 Dec 1975",
        age=50,
        sex="F",
        joint="Left knee",
        treating="Dr Lim Siew Hua",
        treatment_type="MACI",
        treatment_started="Jan 2024",
        status=STATUS_ACTIVE,
        sessions=(
            ImagingSession("s-004-pre", "pre", "10 Jan 2024", DEFAULT_MODALITY, "Baseline"),
            ImagingSession("s-004-post-6m", "post", "10 Jul 2024", DEFAULT_MODALITY, "6 mo follow-up"),
            ImagingSession("s-004-post-9m", "followup", "10 Oct 2024", DEFAULT_MODALITY, "9 mo follow-up"),
        ),
    ),
    PatientRecord(
        patient_id="pt-005",
        display_name="Lee Chong Wei",
        mrn="094321",
        dob="21 Jan 1998",
        age=28,
        sex="M",
        joint="Right ankle",
        treating="Dr Rajesh Kumar",
        treatment_type="Ankle arthroscopy",
        treatment_started="Nov 2024",
        status=STATUS_ACTIVE,
        sessions=(
            ImagingSession("s-005-pre", "pre", "08 Jan 2025", DEFAULT_MODALITY, "Baseline"),
        ),
    ),
)


def find_patients(query: str, joint_filter: str = "All") -> Tuple[PatientRecord, ...]:
    """Case-insensitive filter on name, MRN, or id, with optional joint type filter."""
    q = query.strip().lower()
    results: list[PatientRecord] = []
    for p in MOCK_PATIENTS:
        if q and q not in p.display_name.lower() and q not in p.mrn.lower() and q not in p.patient_id.lower():
            continue
        if joint_filter != "All" and joint_filter.lower() not in p.joint.lower():
            continue
        results.append(p)
    return tuple(results)


def compute_initials(name: str) -> str:
    """Two-letter avatar initials: first letter of the first two words."""
    parts = name.split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return name[:2].upper() if len(name) >= 2 else name.upper()


def session_choice_label(s: ImagingSession) -> str:
    """Short label for select boxes (date, modality, visit label)."""
    return f"{s.date} · {s.modality} · {s.label}"


def session_radio_label(session: ImagingSession) -> str:
    """Formatted label for session radio rows (date · modality · visit tag)."""
    return f"{session.date}  ·  {session.modality}  ·  {session.label}"


def get_patient_by_id(patient_id: str) -> PatientRecord | None:
    """Return a demo patient by id, or None if unknown."""
    for p in MOCK_PATIENTS:
        if p.patient_id == patient_id:
            return p
    return None
