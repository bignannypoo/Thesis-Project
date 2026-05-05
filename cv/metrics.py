"""
Demo segment metrics keyed by **post-treatment** session id.

Replace with Chondral Quant (or CSV) loading later; structure is stable for the UI.
"""

from dataclasses import dataclass

from cv.models import ImagingSession


@dataclass(frozen=True)
class SegmentDemoMetrics:
    """Four headline numbers shown in the Segments tab (mock / placeholder)."""

    medial_thickness: str
    lateral_thickness: str
    total_volume: str
    t2_relaxation: str


# Default when a session has no specific row (still demo data).
DEFAULT_SEGMENT_METRICS = SegmentDemoMetrics(
    medial_thickness="+0.4 mm",
    lateral_thickness="−0.1 mm",
    total_volume="+3.2%",
    t2_relaxation="42 → 38 ms",
)

# Per follow-up session — varies slightly so switching sessions visibly updates the strip.
DEMO_METRICS_BY_POST_SESSION_ID: dict[str, SegmentDemoMetrics] = {
    "s-001-post-9m": SegmentDemoMetrics("+0.4 mm", "−0.1 mm", "+3.2%", "42 → 38 ms"),
    "s-001-post-14m": SegmentDemoMetrics("+0.5 mm", "−0.2 mm", "+4.1%", "41 → 37 ms"),
    "s-002-post-6m": SegmentDemoMetrics("+0.2 mm", "+0.05 mm", "+1.8%", "44 → 41 ms"),
    "s-002-post-14m": SegmentDemoMetrics("+0.3 mm", "0.0 mm", "+2.4%", "43 → 40 ms"),
    "s-003-post-6m": SegmentDemoMetrics("+0.1 mm", "+0.2 mm", "+0.9%", "45 → 43 ms"),
    "s-004-post-6m": SegmentDemoMetrics("+0.35 mm", "−0.05 mm", "+2.9%", "43 → 39 ms"),
    "s-004-post-9m": SegmentDemoMetrics("+0.42 mm", "−0.12 mm", "+3.5%", "42 → 38 ms"),
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
