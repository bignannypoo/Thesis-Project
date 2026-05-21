"""Thresholds and column mappings for MRChondralHealth exports."""

from typing import Final

# Default clinical significance thresholds (user-adjustable in later phases).
DEFAULT_VOLUME_PERCENT_THRESHOLD: Final[float] = 10.0
DEFAULT_T2_MS_THRESHOLD: Final[float] = 5.0
DEFAULT_THICKNESS_MM_THRESHOLD: Final[float] = 0.3

MRCH_CSV_GLOB: Final[str] = "*_MRCH_NOT_FOR_CLINICAL_USE.csv"
CARTILAGE_STATISTICS_FILENAME: Final[str] = "cartilage.statistics"
MRCH_DATETIME_FILENAME_PATTERN: Final[str] = (
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_MRCH"
)

# Normalized metric column names used internally after CSV load.
METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "voxel_count",
    "volume_ml",
    "bioch_mean",
    "bioch_stddev",
    "bioch_median",
    "bioch_min",
    "bioch_max",
    "thickness_mean",
    "thickness_stddev",
    "thickness_median",
    "thickness_min",
    "thickness_max",
)

# CSV header aliases (lowercased before lookup).
CSV_COLUMN_ALIASES: Final[dict[str, str]] = {
    "cartilage region": "region",
    "layer": "layer",
    "#voxels": "voxel_count",
    "volume (ml)": "volume_ml",
    "bioch. mean": "bioch_mean",
    "bioch. stddev": "bioch_stddev",
    "bioch. median": "bioch_median",
    "bioch. min": "bioch_min",
    "bioch. max": "bioch_max",
    "thickness mean": "thickness_mean",
    "thickness stddev": "thickness_stddev",
    "thickness median": "thickness_median",
    "thickness min": "thickness_min",
    "thickness max": "thickness_max",
}

# JSON leaf keys mapped to the same normalized names where possible.
JSON_METRIC_ALIASES: Final[dict[str, str]] = {
    "volume": "volume_ml",
    "volume_ml": "volume_ml",
    "voxels": "voxel_count",
    "voxel_count": "voxel_count",
    "#voxels": "voxel_count",
    "bioch_mean": "bioch_mean",
    "biochemical_mean": "bioch_mean",
    "t2_mean": "bioch_mean",
    "mean_t2": "bioch_mean",
    "bioch_stddev": "bioch_stddev",
    "bioch_median": "bioch_median",
    "bioch_min": "bioch_min",
    "bioch_max": "bioch_max",
    "thickness_mean": "thickness_mean",
    "thickness_stddev": "thickness_stddev",
    "thickness_median": "thickness_median",
    "thickness_min": "thickness_min",
    "thickness_max": "thickness_max",
}
