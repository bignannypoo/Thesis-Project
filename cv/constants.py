"""Shared string constants for screens, view modes, and filters."""

STATUS_ACTIVE = "Active follow-up"
STATUS_COMPLETED = "Completed"
DEFAULT_MODALITY = "MRI 3T DESS"

SCREEN_LOOKUP = "Patient lookup"
SCREEN_VIEWER = "Viewer"

VIEW_MODE_COMPARE = "Pre + post"
VIEW_MODE_PRE = "Pre only"
VIEW_MODE_POST = "Post only"

JOINT_FILTER_OPTIONS = ("All", "Knee", "Hip", "Shoulder", "Ankle")

# Downscale very large slices before sending to the browser (width/height cap).
MAX_DISPLAY_IMAGE_PX = 1920
