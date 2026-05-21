#!/usr/bin/env python3
"""
Compare two MRChondralHealth timepoint folders (CLI).

Example:
    python scripts/compare_mrch_timepoints.py \\
        --timepoint1 /path/to/folder1 \\
        --timepoint2 /path/to/folder2 \\
        --output comparison_report \\
        --figures --pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cartilage_comparison.analysis import build_comparison_table, write_comparison_outputs
from cartilage_comparison.constants import (
    DEFAULT_T2_MS_THRESHOLD,
    DEFAULT_THICKNESS_MM_THRESHOLD,
    DEFAULT_VOLUME_PERCENT_THRESHOLD,
)
from cartilage_comparison.data_loader import load_timepoint_folder
from cartilage_comparison.report_generator import create_pdf_report
from cartilage_comparison.visualizations import write_heatmap_figures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare MRChondralHealth cartilage metrics between two timepoints.",
    )
    parser.add_argument("--timepoint1", "--pre", dest="timepoint1", required=True, type=Path)
    parser.add_argument("--timepoint2", "--post", dest="timepoint2", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("comparison_report"))
    parser.add_argument("--prefer-json", action="store_true")
    parser.add_argument("--figures", action="store_true", help="Write heatmap PNGs")
    parser.add_argument("--pdf", action="store_true", help="Write comparison_report.pdf")
    parser.add_argument("--layer", default="Combined")
    parser.add_argument("--volume-threshold-pct", type=float, default=DEFAULT_VOLUME_PERCENT_THRESHOLD)
    parser.add_argument("--t2-threshold-ms", type=float, default=DEFAULT_T2_MS_THRESHOLD)
    parser.add_argument("--thickness-threshold-mm", type=float, default=DEFAULT_THICKNESS_MM_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    tp1 = load_timepoint_folder(args.timepoint1, prefer_json=args.prefer_json)
    tp2 = load_timepoint_folder(args.timepoint2, prefer_json=args.prefer_json)

    table = build_comparison_table(
        tp1,
        tp2,
        volume_percent_threshold=args.volume_threshold_pct,
        t2_ms_threshold=args.t2_threshold_ms,
        thickness_mm_threshold=args.thickness_threshold_mm,
    )
    if table.empty:
        print("No region/layer rows to compare.", file=sys.stderr)
        return 1

    csv_path = write_comparison_outputs(table, args.output, pre_data=tp1, post_data=tp2)

    if args.figures:
        for png_path in write_heatmap_figures(table, args.output, layer_mode=args.layer):
            print(f"Wrote: {png_path}")

    if args.pdf:
        pdf_path = create_pdf_report(
            table,
            args.output / "comparison_report.pdf",
            pre_data=tp1,
            post_data=tp2,
            layer_mode=args.layer,
        )
        print(f"Wrote: {pdf_path}")

    print(f"Compared {len(table)} region/layer rows.")
    print(f"T1: {tp1.folder} ({tp1.source}, {tp1.study_datetime or 'no date'})")
    print(f"T2: {tp2.folder} ({tp2.source}, {tp2.study_datetime or 'no date'})")
    print(f"Wrote: {csv_path}")
    print(f"Summary: {args.output / 'summary_statistics.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
