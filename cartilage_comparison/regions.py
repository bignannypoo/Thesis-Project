"""Parse MRChondralHealth region labels for anatomical heatmap layouts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BONES: tuple[str, ...] = ("Femur", "Tibia", "Patella", "Trochlea")
COMPARTMENTS: tuple[str, ...] = ("Medial", "Lateral", "Central")
SUBREGIONS_FEMUR_TIBIA: tuple[str, ...] = ("Anterior", "Central", "Posterior")
SUBREGIONS_PATELLA: tuple[str, ...] = ("Superior", "Central", "Inferior")
SUBREGIONS_TROCHLEA: tuple[str, ...] = ("Medial", "Central", "Lateral")

LAYER_COMBINED_ALIASES: frozenset[str] = frozenset({"all", "combined", "total"})
LAYER_DEEP_ALIASES: frozenset[str] = frozenset({"deep"})
LAYER_SUPERFICIAL_ALIASES: frozenset[str] = frozenset({"superficial", "sup"})


@dataclass(frozen=True)
class ParsedRegion:
    """Structured anatomy parsed from a free-text region label."""

    bone: str
    compartment: str
    subregion: str
    column_key: str
    row_key: str


def _title_token(value: str) -> str:
    return value.strip().capitalize()


def parse_cartilage_region(region: str) -> ParsedRegion | None:
    """
    Parse labels such as ``Femur - Medial anterior`` or ``Tibia lateral central``.

    Returns None when the label cannot be mapped to the heatmap grid.
    """
    text = str(region).strip().replace("_", " ")
    if not text:
        return None

    if " - " in text:
        bone_raw, remainder = text.split(" - ", 1)
        bone = _title_token(bone_raw)
        tokens = remainder.split()
    else:
        tokens = text.split()
        if not tokens:
            return None
        bone = _title_token(tokens[0])
        tokens = tokens[1:]

    if bone not in BONES or len(tokens) < 2:
        return None

    if bone == "Trochlea":
        # Trochlea medial / central / lateral — compartment is Central, subregion varies.
        subregion = _title_token(tokens[-1])
        if subregion not in SUBREGIONS_TROCHLEA:
            return None
        compartment = "Central"
        column_key = f"Trochlea\n{subregion}"
        row_key = "Central"
        return ParsedRegion(
            bone=bone,
            compartment=compartment,
            subregion=subregion,
            column_key=column_key,
            row_key=row_key,
        )

    compartment = _title_token(tokens[0])
    if compartment not in ("Medial", "Lateral"):
        return None

    subregion = _title_token(tokens[-1])
    if bone == "Patella" and subregion not in SUBREGIONS_PATELLA:
        return None
    if bone in ("Femur", "Tibia") and subregion not in SUBREGIONS_FEMUR_TIBIA:
        return None

    column_key = f"{bone}\n{compartment}"
    return ParsedRegion(
        bone=bone,
        compartment=compartment,
        subregion=subregion,
        column_key=column_key,
        row_key=subregion,
    )


def heatmap_row_order() -> list[str]:
    """Y-axis order shared across heatmaps (anterior→posterior, patella superior→inferior)."""
    return [
        "Anterior",
        "Central",
        "Posterior",
        "Superior",
        "Inferior",
    ]


def heatmap_column_order() -> list[str]:
    """Default X-axis column order for anatomical heatmaps."""
    columns: list[str] = []
    for bone in ("Femur", "Tibia", "Patella"):
        for compartment in ("Medial", "Lateral"):
            columns.append(f"{bone}\n{compartment}")
    for subregion in SUBREGIONS_TROCHLEA:
        columns.append(f"Trochlea\n{subregion}")
    return columns


def filter_comparison_by_layer(
    comparison_table: pd.DataFrame,
    layer_mode: str,
) -> pd.DataFrame:
    """
    Restrict rows to a layer selection.

    ``Combined`` prefers ``all``/``combined`` layers; otherwise averages metrics
    per region across deep and superficial rows.
    """
    if comparison_table.empty:
        return comparison_table.copy()

    mode = layer_mode.strip().lower()
    working = comparison_table.copy()
    working["_layer_norm"] = working["layer"].astype(str).str.strip().str.lower()

    if mode in {"deep", "deep only", "deep_only"}:
        return working[working["_layer_norm"].isin(LAYER_DEEP_ALIASES)].drop(
            columns="_layer_norm",
        )

    if mode in {"superficial", "superficial only", "superficial_only"}:
        return working[working["_layer_norm"].isin(LAYER_SUPERFICIAL_ALIASES)].drop(
            columns="_layer_norm",
        )

    combined_rows = working[working["_layer_norm"].isin(LAYER_COMBINED_ALIASES)]
    if not combined_rows.empty:
        return combined_rows.drop(columns="_layer_norm")

    numeric_cols = [
        column
        for column in working.columns
        if column.startswith(("pre_", "post_", "delta_", "pct_change_"))
        and pd.api.types.is_numeric_dtype(working[column])
    ]
    meta_cols = [
        column
        for column in ("pre_study_datetime", "post_study_datetime")
        if column in working.columns
    ]
    grouped = working.groupby("region", as_index=False)[numeric_cols].mean(numeric_only=True)
    if meta_cols:
        meta = working.groupby("region", as_index=False)[meta_cols].first()
        grouped = grouped.merge(meta, on="region", how="left")
    grouped["layer"] = "combined"
    return grouped


def pivot_change_matrix(
    comparison_table: pd.DataFrame,
    *,
    value_column: str,
) -> pd.DataFrame:
    """
    Build a heatmap matrix (rows=subregions, columns=anatomical sites).

    Unmapped regions are omitted. Duplicate cells use the mean when multiple
    rows collide (e.g. combined layer averaging).
    """
    if comparison_table.empty or value_column not in comparison_table.columns:
        return pd.DataFrame()

    cells: list[dict[str, object]] = []
    for _, row in comparison_table.iterrows():
        parsed = parse_cartilage_region(str(row["region"]))
        if parsed is None:
            continue
        value = row.get(value_column)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        cells.append(
            {
                "row_key": parsed.row_key,
                "column_key": parsed.column_key,
                "value": float(value),
            }
        )

    if not cells:
        return pd.DataFrame()

    frame = pd.DataFrame(cells)
    aggregated = frame.groupby(["row_key", "column_key"], as_index=False)["value"].mean()
    matrix = aggregated.pivot(index="row_key", columns="column_key", values="value")

    row_order = [row for row in heatmap_row_order() if row in matrix.index]
    col_order = [column for column in heatmap_column_order() if column in matrix.columns]
    extra_rows = [row for row in matrix.index if row not in row_order]
    extra_cols = [column for column in matrix.columns if column not in col_order]
    return matrix.reindex(index=[*row_order, *extra_rows], columns=[*col_order, *extra_cols])
