"""Tests for the standalone cartilage_comparison package."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from cartilage_comparison.analysis import (
    build_comparison_table,
    calculate_changes,
    summarize_overall_changes,
    write_comparison_outputs,
)
from cartilage_comparison.data_loader import (
    extract_study_datetime,
    load_mrch_csv,
    load_timepoint_data,
    load_timepoint_folder,
)
from cartilage_comparison.regions import filter_comparison_by_layer, parse_cartilage_region, pivot_change_matrix
from cartilage_comparison.visualizations import build_dashboard_summary, create_change_heatmap, figure_to_png_bytes

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TIMEPOINT_1 = FIXTURES_DIR / "mrch_pre"
TIMEPOINT_2 = FIXTURES_DIR / "mrch_post"


def test_extract_study_datetime_from_filename() -> None:
    assert extract_study_datetime(TIMEPOINT_1) == "2024-01-14 10:30:00"
    assert extract_study_datetime(TIMEPOINT_2) == "2024-10-18 14:00:00"


def test_load_mrch_csv_normalizes_columns() -> None:
    csv_path = next(TIMEPOINT_1.glob("*_MRCH_NOT_FOR_CLINICAL_USE.csv"))
    data_frame = load_mrch_csv(csv_path)
    assert "region" in data_frame.columns
    assert "volume_ml" in data_frame.columns
    assert len(data_frame) == 3


def test_load_timepoint_data_alias() -> None:
    data = load_timepoint_data(TIMEPOINT_1)
    assert data.source == "csv"
    assert data.study_datetime == "2024-01-14 10:30:00"


def test_calculate_changes_matches_build_table() -> None:
    tp1 = load_timepoint_folder(TIMEPOINT_1)
    tp2 = load_timepoint_folder(TIMEPOINT_2)
    assert len(calculate_changes(tp1, tp2)) == len(build_comparison_table(tp1, tp2))


def test_build_comparison_table_volume_and_t2_changes() -> None:
    tp1 = load_timepoint_folder(TIMEPOINT_1)
    tp2 = load_timepoint_folder(TIMEPOINT_2)
    table = build_comparison_table(tp1, tp2)

    assert len(table) == 4
    femur_deep = table[
        (table["region"] == "Femur - Medial anterior") & (table["layer"] == "Deep")
    ].iloc[0]

    assert femur_deep["pct_change_volume"] == pytest.approx(10.0)
    assert femur_deep["delta_t2_mean"] == pytest.approx(-7.0)
    assert femur_deep["t2_change_direction"] == "decreased"


def test_parse_region_and_heatmap_matrix() -> None:
    parsed = parse_cartilage_region("Femur - Medial anterior")
    assert parsed is not None
    assert parsed.column_key == "Femur\nMedial"

    tp1 = load_timepoint_folder(TIMEPOINT_1)
    tp2 = load_timepoint_folder(TIMEPOINT_2)
    table = build_comparison_table(tp1, tp2)
    filtered = filter_comparison_by_layer(table, "Deep only")
    matrix = pivot_change_matrix(filtered, value_column="pct_change_volume")
    assert matrix.loc["Anterior", "Femur\nMedial"] == pytest.approx(10.0)


def test_create_change_heatmap_returns_png() -> None:
    tp1 = load_timepoint_folder(TIMEPOINT_1)
    tp2 = load_timepoint_folder(TIMEPOINT_2)
    table = build_comparison_table(tp1, tp2)
    figure = create_change_heatmap(table, metric="volume")
    assert len(figure_to_png_bytes(figure)) > 500


def test_write_comparison_outputs(tmp_path: Path) -> None:
    tp1 = load_timepoint_folder(TIMEPOINT_1)
    tp2 = load_timepoint_folder(TIMEPOINT_2)
    table = build_comparison_table(tp1, tp2)
    csv_path = write_comparison_outputs(table, tmp_path, pre_data=tp1, post_data=tp2)
    assert csv_path.exists()
    summary = summarize_overall_changes(table)
    assert summary["decreased_t2_regions"] >= 1


def test_dashboard_summary() -> None:
    tp1 = load_timepoint_folder(TIMEPOINT_1)
    tp2 = load_timepoint_folder(TIMEPOINT_2)
    table = build_comparison_table(tp1, tp2)
    summary = build_dashboard_summary(table)
    assert summary["top_increases"]
