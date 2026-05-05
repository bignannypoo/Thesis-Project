"""Unit tests for pure data helpers (no Streamlit)."""

import pytest

from cv.data import (
    MOCK_PATIENTS,
    REQUIRED_CHONDRAL_QUANT_COLUMNS,
    compute_initials,
    find_patients,
    get_patient_by_id,
    load_chondral_quant_csv,
)
from cv.metrics import build_session_statistics_table, build_statistical_summary_from_csv_data


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


def test_load_chondral_quant_csv_valid(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "session_id,date,segment,volume,thickness,t2_relaxation\n"
        "s-001-pre,14 Jan 2024,medial,1200,2.9,43\n"
        "s-001-post-9m,18 Oct 2024,medial,1160,2.8,41\n",
        encoding="utf-8",
    )
    data_frame = load_chondral_quant_csv(str(csv_path))
    assert tuple(data_frame.columns) == REQUIRED_CHONDRAL_QUANT_COLUMNS


def test_load_chondral_quant_csv_missing_columns(tmp_path) -> None:
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text(
        "session_id,date,segment,volume\n"
        "s-001-pre,14 Jan 2024,medial,1200\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CSV missing required columns"):
        load_chondral_quant_csv(str(csv_path))


def test_build_statistical_summary_from_csv_data_returns_summary(tmp_path) -> None:
    csv_path = tmp_path / "stats_metrics.csv"
    csv_path.write_text(
        "session_id,date,segment,volume,thickness,t2_relaxation\n"
        "s-pre,01 Jan 2024,medial,1200,2.9,43\n"
        "s-pre,01 Jan 2024,lateral,1180,2.8,44\n"
        "s-pre,01 Jan 2024,patellar,1150,2.7,45\n"
        "s-post,01 Jul 2024,medial,1140,2.7,41\n"
        "s-post,01 Jul 2024,lateral,1100,2.6,42\n"
        "s-post,01 Jul 2024,patellar,1090,2.5,43\n",
        encoding="utf-8",
    )
    data_frame = load_chondral_quant_csv(str(csv_path))
    summary = build_statistical_summary_from_csv_data(data_frame, "s-pre", "s-post")

    assert summary is not None
    assert summary.sample_size == 3
    assert summary.volume_p_value >= 0.0
    assert summary.volume_p_value <= 1.0
    assert isinstance(summary.is_volume_significant, bool)
    assert isinstance(summary.is_thickness_significant, bool)


def test_build_session_statistics_table_returns_ci_rows(tmp_path) -> None:
    csv_path = tmp_path / "session_stats_metrics.csv"
    csv_path.write_text(
        "session_id,date,segment,volume,thickness,t2_relaxation\n"
        "s-1,01 Jan 2024,medial,1200,2.9,43\n"
        "s-1,01 Jan 2024,lateral,1180,2.8,44\n"
        "s-2,01 Jul 2024,medial,1140,2.7,41\n"
        "s-2,01 Jul 2024,lateral,1100,2.6,42\n",
        encoding="utf-8",
    )
    data_frame = load_chondral_quant_csv(str(csv_path))
    session_table = build_session_statistics_table(data_frame)

    assert len(session_table) == 2
    assert "volume_ci_low" in session_table.columns
    assert "volume_ci_high" in session_table.columns
