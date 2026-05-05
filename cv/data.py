"""Patient data repository and lookup helpers."""

import json
from pathlib import Path
from typing import Tuple

import pandas as pd

from cv.constants import DEFAULT_MODALITY, STATUS_ACTIVE
from cv.models import ImagingSession, PatientRecord

REQUIRED_CHONDRAL_QUANT_COLUMNS: tuple[str, ...] = (
    "session_id",
    "date",
    "segment",
    "volume",
    "thickness",
    "t2_relaxation",
)
FOLLOWUP_LABEL_6_MONTHS = "6 mo follow-up"
DEFAULT_PATIENT_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "patients.json"

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
            ImagingSession("s-002-post-6m", "post", "28 Dec 2023", DEFAULT_MODALITY, FOLLOWUP_LABEL_6_MONTHS),
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
            ImagingSession("s-003-post-6m", "post", "05 Sep 2024", DEFAULT_MODALITY, FOLLOWUP_LABEL_6_MONTHS),
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
            ImagingSession("s-004-post-6m", "post", "10 Jul 2024", DEFAULT_MODALITY, FOLLOWUP_LABEL_6_MONTHS),
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


def _session_to_dict(session: ImagingSession) -> dict[str, str]:
    """Serialize one imaging session to a JSON-compatible dict."""
    return {
        "session_id": session.session_id,
        "role": session.role,
        "date": session.date,
        "modality": session.modality,
        "label": session.label,
    }


def _patient_to_dict(patient: PatientRecord) -> dict[str, object]:
    """Serialize one patient record to a JSON-compatible dict."""
    return {
        "patient_id": patient.patient_id,
        "display_name": patient.display_name,
        "mrn": patient.mrn,
        "dob": patient.dob,
        "age": patient.age,
        "sex": patient.sex,
        "joint": patient.joint,
        "treating": patient.treating,
        "treatment_type": patient.treatment_type,
        "treatment_started": patient.treatment_started,
        "status": patient.status,
        "sessions": [_session_to_dict(session) for session in patient.sessions],
    }


def _session_from_dict(raw: dict[str, object]) -> ImagingSession:
    """Parse one imaging session from persisted JSON."""
    return ImagingSession(
        session_id=str(raw["session_id"]),
        role=str(raw["role"]),  # type: ignore[arg-type]
        date=str(raw["date"]),
        modality=str(raw["modality"]),
        label=str(raw["label"]),
    )


def _patient_from_dict(raw: dict[str, object]) -> PatientRecord:
    """Parse one patient record from persisted JSON."""
    sessions_raw = raw.get("sessions", [])
    sessions = tuple(_session_from_dict(item) for item in sessions_raw if isinstance(item, dict))
    return PatientRecord(
        patient_id=str(raw["patient_id"]),
        display_name=str(raw["display_name"]),
        mrn=str(raw["mrn"]),
        dob=str(raw["dob"]),
        age=int(raw["age"]),
        sex=str(raw["sex"]),
        joint=str(raw["joint"]),
        treating=str(raw["treating"]),
        treatment_type=str(raw["treatment_type"]),
        treatment_started=str(raw["treatment_started"]),
        status=str(raw["status"]),
        sessions=sessions,
    )


class PatientRepository:
    """File-backed patient repository with demo seed fallback."""

    def __init__(self, storage_path: Path = DEFAULT_PATIENT_STORE_PATH) -> None:
        self.storage_path = storage_path

    def _ensure_seed_data(self) -> None:
        """Seed storage with demo records when file does not exist."""
        if self.storage_path.exists():
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"patients": [_patient_to_dict(patient) for patient in MOCK_PATIENTS]}
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_patients(self) -> Tuple[PatientRecord, ...]:
        """Load all patients from storage."""
        self._ensure_seed_data()
        raw_payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        patients_raw = raw_payload.get("patients", [])
        patients = [_patient_from_dict(item) for item in patients_raw if isinstance(item, dict)]
        return tuple(patients)

    def save_patients(self, patients: Tuple[PatientRecord, ...]) -> None:
        """Persist patient list to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"patients": [_patient_to_dict(patient) for patient in patients]}
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_patient_repository = PatientRepository()


def list_patients() -> Tuple[PatientRecord, ...]:
    """Return all patients from repository."""
    return _patient_repository.load_patients()


def find_patients(query: str, joint_filter: str = "All") -> Tuple[PatientRecord, ...]:
    """Case-insensitive filter on name, MRN, or id, with optional joint type filter."""
    q = query.strip().lower()
    results: list[PatientRecord] = []
    for p in list_patients():
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
    for p in list_patients():
        if p.patient_id == patient_id:
            return p
    return None


def load_chondral_quant_csv(csv_file: str) -> pd.DataFrame:
    """Load and validate Chondral Quant-style CSV metrics."""
    data_frame = pd.read_csv(csv_file)
    normalized_columns = [column.strip().lower() for column in data_frame.columns]
    data_frame.columns = normalized_columns

    missing_columns = [column for column in REQUIRED_CHONDRAL_QUANT_COLUMNS if column not in data_frame.columns]
    if missing_columns:
        raise ValueError(
            "CSV missing required columns: "
            + ", ".join(missing_columns)
        )

    numeric_columns = ["volume", "thickness", "t2_relaxation"]
    for column_name in numeric_columns:
        data_frame[column_name] = pd.to_numeric(data_frame[column_name], errors="coerce")
    if data_frame[numeric_columns].isna().any().any():
        raise ValueError("CSV contains non-numeric values in volume, thickness, or t2_relaxation.")

    return data_frame
