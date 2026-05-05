"""Immutable data models for patients and imaging sessions."""

from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass(frozen=True)
class ImagingSession:
    """One imaging visit (pre, post, or follow-up)."""

    session_id: str
    role: Literal["pre", "post", "followup"]
    date: str
    modality: str
    label: str


@dataclass(frozen=True)
class PatientRecord:
    """Demographics and session list for lookup and viewer context."""

    patient_id: str
    display_name: str
    mrn: str
    dob: str
    age: int
    sex: str
    joint: str
    treating: str
    treatment_type: str
    treatment_started: str
    status: str
    sessions: Tuple[ImagingSession, ...]
