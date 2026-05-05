"""Metrics utilities for segment summaries and trend charts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from cv.models import ImagingSession

DEFAULT_T2_RANGE = "42 → 38 ms"


@dataclass(frozen=True)
class SegmentDemoMetrics:
    """Four headline numbers shown in the Segments tab (mock / placeholder)."""

    medial_thickness: str
    lateral_thickness: str
    total_volume: str
    t2_relaxation: str


@dataclass(frozen=True)
class SegmentComputedMetrics:
    """Computed values for one segment compared against baseline."""

    segment: str
    thickness_delta_mm: float
    volume_delta_percent: float
    t2_baseline_ms: float
    t2_followup_ms: float


@dataclass(frozen=True)
class StatisticalSummary:
    """Statistical summary computed across available segments."""

    sample_size: int
    mean_volume_delta_percent: float
    volume_ci_low: float
    volume_ci_high: float
    volume_p_value: float
    mean_thickness_delta_mm: float
    thickness_ci_low: float
    thickness_ci_high: float
    thickness_p_value: float
    is_volume_significant: bool
    is_thickness_significant: bool


# Default when a session has no specific row (still demo data).
DEFAULT_SEGMENT_METRICS = SegmentDemoMetrics(
    medial_thickness="+0.4 mm",
    lateral_thickness="−0.1 mm",
    total_volume="+3.2%",
    t2_relaxation=DEFAULT_T2_RANGE,
)

# Per follow-up session — varies slightly so switching sessions visibly updates the strip.
DEMO_METRICS_BY_POST_SESSION_ID: dict[str, SegmentDemoMetrics] = {
    "s-001-post-9m": SegmentDemoMetrics("+0.4 mm", "−0.1 mm", "+3.2%", DEFAULT_T2_RANGE),
    "s-001-post-14m": SegmentDemoMetrics("+0.5 mm", "−0.2 mm", "+4.1%", "41 → 37 ms"),
    "s-002-post-6m": SegmentDemoMetrics("+0.2 mm", "+0.05 mm", "+1.8%", "44 → 41 ms"),
    "s-002-post-14m": SegmentDemoMetrics("+0.3 mm", "0.0 mm", "+2.4%", "43 → 40 ms"),
    "s-003-post-6m": SegmentDemoMetrics("+0.1 mm", "+0.2 mm", "+0.9%", "45 → 43 ms"),
    "s-004-post-6m": SegmentDemoMetrics("+0.35 mm", "−0.05 mm", "+2.9%", "43 → 39 ms"),
    "s-004-post-9m": SegmentDemoMetrics("+0.42 mm", "−0.12 mm", "+3.5%", DEFAULT_T2_RANGE),
}


def get_segment_metrics_for_post_session(post_session: ImagingSession) -> SegmentDemoMetrics:
    """Return demo metrics for the selected follow-up visit."""
    return DEMO_METRICS_BY_POST_SESSION_ID.get(post_session.session_id, DEFAULT_SEGMENT_METRICS)


def format_metric_strip_html(metrics: SegmentDemoMetrics) -> str:
    """Build the Segments tab metric pill row."""
    from cv.html_util import escape_html

    return f"""
<div class="cv-metric-strip">
  <div class="cv-metric-pill">
    <div class="cv-m-label">Medial thickness</div>
    <div class="cv-m-val">{escape_html(metrics.medial_thickness)}</div>
  </div>
  <div class="cv-metric-pill">
    <div class="cv-m-label">Lateral thickness</div>
    <div class="cv-m-val">{escape_html(metrics.lateral_thickness)}</div>
  </div>
  <div class="cv-metric-pill">
    <div class="cv-m-label">Total volume</div>
    <div class="cv-m-val">{escape_html(metrics.total_volume)}</div>
  </div>
  <div class="cv-metric-pill">
    <div class="cv-m-label">T2 relaxation</div>
    <div class="cv-m-val">{escape_html(metrics.t2_relaxation)}</div>
  </div>
</div>
"""


def _format_signed(value: float, unit: str, precision: int = 2) -> str:
    """Format signed values with unit suffix."""
    return f"{value:+.{precision}f} {unit}"


def _format_percent(value: float, precision: int = 2) -> str:
    """Format signed percentages."""
    return f"{value:+.{precision}f}%"


def _build_demo_metrics_from_computed(metrics_rows: list[SegmentComputedMetrics]) -> SegmentDemoMetrics:
    """Convert computed metrics rows into the 4-pill UI structure."""
    row_map = {row.segment.lower(): row for row in metrics_rows}
    medial = row_map.get("medial")
    lateral = row_map.get("lateral")

    if not medial or not lateral:
        return DEFAULT_SEGMENT_METRICS

    mean_volume_delta_percent = sum(row.volume_delta_percent for row in metrics_rows) / len(metrics_rows)
    mean_t2_baseline = sum(row.t2_baseline_ms for row in metrics_rows) / len(metrics_rows)
    mean_t2_followup = sum(row.t2_followup_ms for row in metrics_rows) / len(metrics_rows)

    return SegmentDemoMetrics(
        medial_thickness=_format_signed(medial.thickness_delta_mm, "mm"),
        lateral_thickness=_format_signed(lateral.thickness_delta_mm, "mm"),
        total_volume=_format_percent(mean_volume_delta_percent),
        t2_relaxation=f"{mean_t2_baseline:.1f} → {mean_t2_followup:.1f} ms",
    )


def calculate_segment_changes(
    baseline_data_frame: pd.DataFrame,
    followup_data_frame: pd.DataFrame,
) -> list[SegmentComputedMetrics]:
    """Calculate per-segment changes between baseline and follow-up data."""
    merged_data_frame = baseline_data_frame.merge(
        followup_data_frame,
        on="segment",
        how="inner",
        suffixes=("_baseline", "_followup"),
    )
    if merged_data_frame.empty:
        return []

    computed_rows: list[SegmentComputedMetrics] = []
    for _, row in merged_data_frame.iterrows():
        baseline_volume = float(row["volume_baseline"])
        followup_volume = float(row["volume_followup"])
        if baseline_volume == 0:
            volume_delta_percent = 0.0
        else:
            volume_delta_percent = ((followup_volume - baseline_volume) / baseline_volume) * 100

        computed_rows.append(
            SegmentComputedMetrics(
                segment=str(row["segment"]),
                thickness_delta_mm=float(row["thickness_followup"] - row["thickness_baseline"]),
                volume_delta_percent=volume_delta_percent,
                t2_baseline_ms=float(row["t2_relaxation_baseline"]),
                t2_followup_ms=float(row["t2_relaxation_followup"]),
            )
        )
    return computed_rows


def build_metric_strip_from_csv_data(
    metrics_data_frame: pd.DataFrame,
    baseline_session_id: str,
    followup_session_id: str,
) -> SegmentDemoMetrics:
    """Build Segments pill metrics from CSV data for selected sessions."""
    baseline_data_frame = metrics_data_frame[metrics_data_frame["session_id"] == baseline_session_id]
    followup_data_frame = metrics_data_frame[metrics_data_frame["session_id"] == followup_session_id]
    if baseline_data_frame.empty or followup_data_frame.empty:
        return DEFAULT_SEGMENT_METRICS

    computed_rows = calculate_segment_changes(baseline_data_frame, followup_data_frame)
    if not computed_rows:
        return DEFAULT_SEGMENT_METRICS
    return _build_demo_metrics_from_computed(computed_rows)


def _calculate_confidence_interval(values: np.ndarray) -> tuple[float, float]:
    """Compute 95% CI for a 1D numeric array."""
    if values.size == 0:
        return 0.0, 0.0
    mean_value = float(np.mean(values))
    if values.size == 1:
        return mean_value, mean_value
    if np.allclose(values, values[0]):
        return mean_value, mean_value

    standard_error = stats.sem(values)
    if np.isnan(standard_error):
        return mean_value, mean_value
    interval = stats.t.interval(0.95, df=values.size - 1, loc=mean_value, scale=standard_error)
    return float(interval[0]), float(interval[1])


def _calculate_p_value(values: np.ndarray) -> float:
    """Run one-sample t-test against zero."""
    if values.size < 2:
        return 1.0
    if np.allclose(values, values[0]):
        return 1.0
    test_result = stats.ttest_1samp(values, popmean=0.0)
    p_value = float(test_result.pvalue)
    if np.isnan(p_value):
        return 1.0
    return p_value


def calculate_statistical_summary(metrics_rows: list[SegmentComputedMetrics]) -> StatisticalSummary | None:
    """Calculate confidence intervals and p-values for segment changes."""
    if not metrics_rows:
        return None

    volume_deltas = np.array([row.volume_delta_percent for row in metrics_rows], dtype=np.float64)
    thickness_deltas = np.array([row.thickness_delta_mm for row in metrics_rows], dtype=np.float64)

    volume_ci_low, volume_ci_high = _calculate_confidence_interval(volume_deltas)
    thickness_ci_low, thickness_ci_high = _calculate_confidence_interval(thickness_deltas)

    return StatisticalSummary(
        sample_size=len(metrics_rows),
        mean_volume_delta_percent=float(np.mean(volume_deltas)),
        volume_ci_low=volume_ci_low,
        volume_ci_high=volume_ci_high,
        volume_p_value=_calculate_p_value(volume_deltas),
        mean_thickness_delta_mm=float(np.mean(thickness_deltas)),
        thickness_ci_low=thickness_ci_low,
        thickness_ci_high=thickness_ci_high,
        thickness_p_value=_calculate_p_value(thickness_deltas),
        is_volume_significant=_calculate_p_value(volume_deltas) < 0.05,
        is_thickness_significant=_calculate_p_value(thickness_deltas) < 0.05,
    )


def build_statistical_summary_from_csv_data(
    metrics_data_frame: pd.DataFrame,
    baseline_session_id: str,
    followup_session_id: str,
) -> StatisticalSummary | None:
    """Build a statistical summary from selected baseline/follow-up sessions."""
    baseline_data_frame = metrics_data_frame[metrics_data_frame["session_id"] == baseline_session_id]
    followup_data_frame = metrics_data_frame[metrics_data_frame["session_id"] == followup_session_id]
    if baseline_data_frame.empty or followup_data_frame.empty:
        return None

    computed_rows = calculate_segment_changes(baseline_data_frame, followup_data_frame)
    return calculate_statistical_summary(computed_rows)


def build_session_statistics_table(metrics_data_frame: pd.DataFrame) -> pd.DataFrame:
    """Build per-session volume mean and 95% CI table for trend interpretation."""
    grouped = metrics_data_frame.groupby("session_id", as_index=False)
    rows: list[dict[str, object]] = []
    for _, session_frame in grouped:
        volume_values = session_frame["volume"].to_numpy(dtype=np.float64)
        ci_low, ci_high = _calculate_confidence_interval(volume_values)
        rows.append(
            {
                "session_id": str(session_frame["session_id"].iloc[0]),
                "date": str(session_frame["date"].iloc[0]),
                "segment_count": int(len(session_frame)),
                "volume_mean": float(np.mean(volume_values)),
                "volume_ci_low": ci_low,
                "volume_ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)
