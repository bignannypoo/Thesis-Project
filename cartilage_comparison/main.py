"""
Streamlit entry point for the cartilage comparison tool.

Run from the project root:

    streamlit run cartilage_comparison/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit adds this file's directory to sys.path, not the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from cartilage_comparison.analysis import build_comparison_table, summarize_overall_changes
from cartilage_comparison.data_loader import load_timepoint_folder
from cartilage_comparison.report_generator import create_pdf_report
from cartilage_comparison.knee_views import render_knee_imaging_section
from cartilage_comparison.visualizations import (
    build_dashboard_summary,
    create_change_heatmap,
    create_side_by_side_bars,
    style_comparison_table,
    write_heatmap_figures,
)


def _render_dashboard(summary: dict[str, object], headline: dict[str, float | int | None]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total volume Pre (ml)",
        f"{summary.get('total_pre_volume_ml', 0):.2f}" if summary.get("total_pre_volume_ml") else "n/a",
    )
    col2.metric(
        "Total volume Post (ml)",
        f"{summary.get('total_post_volume_ml', 0):.2f}" if summary.get("total_post_volume_ml") else "n/a",
    )
    vol_pct = summary.get("total_volume_pct_change")
    col3.metric(
        "Total volume change",
        f"{vol_pct:+.1f}%" if vol_pct is not None else "n/a",
        delta=summary.get("total_volume_delta_ml"),
    )
    col4.metric(
        "Mean T2 change (ms)",
        f"{headline.get('mean_t2_delta_ms', 0):.2f}" if headline.get("mean_t2_delta_ms") is not None else "n/a",
    )

    st.caption("Green = increase · Red = decrease (neutral; not clinical improvement)")

    left, right = st.columns(2)
    with left:
        st.subheader("Top 5 volume increases")
        for entry in summary.get("top_increases", []):
            st.write(f"**{entry['label']}**: {entry['pct_change_volume']:+.1f}%")
    with right:
        st.subheader("Top 5 volume decreases")
        for entry in summary.get("top_decreases", []):
            st.write(f"**{entry['label']}**: {entry['pct_change_volume']:+.1f}%")


def run() -> None:
    st.set_page_config(page_title="Cartilage Comparison", page_icon="🦵", layout="wide")
    st.title("Cartilage MRI Comparison Tool")
    st.caption("MRChondralHealth — compare any two timepoints (pre/post treatment or progression)")

    with st.sidebar:
        st.header("Timepoints")
        timepoint_1 = st.text_input(
            "Timepoint 1 folder",
            placeholder="/Users/.../MRCH_Comparison/Pre/Patient 3 pre robust",
        )
        timepoint_2 = st.text_input(
            "Timepoint 2 folder",
            placeholder="/Users/.../MRCH_Comparison/Post/Patient 3 post robust",
        )
        st.caption("Pre scans → `Pre/... pre robust` · Post scans → `Post/... post robust`")
        prefer_json = st.checkbox("Prefer cartilage.statistics JSON", value=False)
        st.divider()
        st.header("Display")
        metric_label = st.selectbox("Metric", ["Volume", "T2 Mapping", "Thickness"])
        metric_key = {"Volume": "volume", "T2 Mapping": "t2", "Thickness": "thickness"}[metric_label]
        layer_mode = st.radio("Layer", ["Combined", "Deep only", "Superficial only"])
        change_label = st.radio("Display", ["Percentage change", "Absolute change"])
        change_type = "percentage" if change_label.startswith("Percentage") else "absolute"
        sort_by_magnitude = st.checkbox("Sort table by largest |volume %|", value=False)
        st.divider()
        output_dir = st.text_input("Export folder", value="comparison_report")

    if st.button("Run comparison", type="primary"):
        if not timepoint_1.strip() or not timepoint_2.strip():
            st.error("Enter both timepoint folder paths.")
            st.stop()
        try:
            tp1 = load_timepoint_folder(timepoint_1, prefer_json=prefer_json)
            tp2 = load_timepoint_folder(timepoint_2, prefer_json=prefer_json)
            table = build_comparison_table(tp1, tp2)
        except (FileNotFoundError, NotADirectoryError, ValueError) as error:
            st.error(str(error))
            st.stop()

        if table.empty:
            st.warning("No region/layer rows to compare.")
            st.stop()

        st.session_state["comparison_table"] = table
        st.session_state["timepoint_1"] = tp1
        st.session_state["timepoint_2"] = tp2

    table = st.session_state.get("comparison_table")
    if table is None:
        st.info("Enter folder paths and click **Run comparison**.")
        return

    tp1 = st.session_state.get("timepoint_1")
    tp2 = st.session_state.get("timepoint_2")
    if tp1 and tp2:
        st.success(
            f"Pre: {tp1.study_datetime or 'unknown'} · "
            f"Post: {tp2.study_datetime or 'unknown'} · {len(table)} rows"
        )

    dashboard = build_dashboard_summary(table)
    headline = summarize_overall_changes(table)
    _render_dashboard(dashboard, headline)

    st.subheader(f"{metric_label} change heatmap")
    st.pyplot(
        create_change_heatmap(
            table,
            metric=metric_key,
            change_type=change_type,
            layer_mode=layer_mode,
        )
    )

    tab_heatmaps, tab_table, tab_bars, tab_knee = st.tabs(
        ["All heatmaps", "Comparison table", "Bar charts", "Knee imaging"],
    )

    with tab_heatmaps:
        for heatmap_metric in ("volume", "t2", "thickness"):
            st.markdown(f"**{heatmap_metric.upper()}**")
            st.pyplot(
                create_change_heatmap(
                    table,
                    metric=heatmap_metric,
                    change_type=change_type,
                    layer_mode=layer_mode,
                )
            )

    with tab_table:
        display_table = table.copy()
        if sort_by_magnitude and "pct_change_volume" in display_table.columns:
            display_table = display_table.sort_values(
                "pct_change_volume",
                key=lambda series: series.abs(),
                ascending=False,
                kind="stable",
            )
        st.dataframe(style_comparison_table(display_table), use_container_width=True, height=480)

    with tab_bars:
        for bar_metric in ("volume", "t2", "thickness"):
            st.markdown(f"**{bar_metric.upper()}**")
            st.pyplot(
                create_side_by_side_bars(table, metric=bar_metric, layer_mode=layer_mode)
            )

    with tab_knee:
        if tp1 and tp2:
            render_knee_imaging_section(tp1, tp2)

    st.subheader("Export")
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        if st.button("Write CSV + PNG heatmaps"):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            table.to_csv(out / "comparison_table.csv", index=False)
            for path in write_heatmap_figures(
                table, out, layer_mode=layer_mode, change_type=change_type
            ):
                st.caption(path)
            st.success(f"Saved to `{out}`")
    with export_col2:
        if st.button("Generate PDF report"):
            pdf_path = Path(output_dir) / "comparison_report.pdf"
            create_pdf_report(
                table,
                pdf_path,
                pre_data=tp1,
                post_data=tp2,
                layer_mode=layer_mode,
                change_type=change_type,
            )
            st.success(f"Wrote {pdf_path}")


if __name__ == "__main__":
    run()
