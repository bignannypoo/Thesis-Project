"""Unit tests for pure data helpers (no Streamlit)."""

import pytest

from cv.data import MOCK_PATIENTS, compute_initials, find_patients, get_patient_by_id


def test_find_patients_empty_query_returns_all_when_filter_all() -> None:
    found = find_patients("", "All")
    assert len(found) == len(MOCK_PATIENTS)


def test_find_patients_by_mrn() -> None:
    found = find_patients("048291", "All")
    assert len(found) == 1
    assert found[0].display_name == "Tan Wei Ming"


def test_find_patients_joint_filter_knee() -> None:
    found = find_patients("", "Knee")
    for p in found:
        assert "knee" in p.joint.lower()


def test_get_patient_by_id_unknown() -> None:
    assert get_patient_by_id("nope") is None


def test_compute_initials_two_words() -> None:
    assert compute_initials("Tan Wei Ming") == "TW"


def test_compute_initials_single_word() -> None:
    assert compute_initials("Madonna") == "MA"


@pytest.mark.parametrize(
    "post_id,expect_medial",
    [
        ("s-001-post-9m", "+0.4 mm"),
        ("s-001-post-14m", "+0.5 mm"),
        ("s-004-post-9m", "+0.42 mm"),
    ],
)
def test_metrics_lookup_by_post_session(post_id: str, expect_medial: str) -> None:
    from cv.metrics import DEMO_METRICS_BY_POST_SESSION_ID

    m = DEMO_METRICS_BY_POST_SESSION_ID[post_id]
    assert m.medial_thickness == expect_medial
