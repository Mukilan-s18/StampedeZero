"""
config.py — StampedeZero Vision Tracker
Branch: mukil  |  Engineer 1 (Vision & Tracking Lead)

Central configuration file. Tune these values to match your demo environment
without touching any business logic in crowd_tracker.py.
"""

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH: str = "yolov8n.pt"        # YOLOv8 nano weights (auto-downloads on first run)
TRACKER_CFG: str = "bytetrack.yaml"   # ByteTrack — lowest latency built-in tracker

# ─── Detection ────────────────────────────────────────────────────────────────
PERSON_CLASS: int = 0                  # COCO class 0 = "person" (ignore everything else)
CONFIDENCE_THRESHOLD: float = 0.4     # Minimum detection confidence (0.0–1.0)

# ─── Virtual Counting Line ────────────────────────────────────────────────────
#
#   LINE_Y is expressed as a FRACTION of frame height (0.0 → 1.0).
#   This makes the line position resolution-independent.
#
#   Example:  LINE_Y_FRACTION = 0.5  →  always the horizontal midpoint
#
#   INFLOW  convention: person moves DOWNWARD past the line (prev_cy < line_y, curr_cy >= line_y)
#   OUTFLOW convention: person moves UPWARD  past the line (prev_cy > line_y, curr_cy <= line_y)
#
#   Adjust to match physical camera placement:
#     • Camera above a doorway facing down → midpoint works well
#     • Camera at an angle → move the line closer to the entry side

LINE_Y_FRACTION: float = 0.55          # ~55% from top — slightly below midpoint
LINE_COLOR: tuple = (0, 255, 255)       # Cyan line (BGR)
LINE_THICKNESS: int = 2

# ─── Frame Processing ─────────────────────────────────────────────────────────
FRAME_W: int = 640                     # Width to resize input before YOLO inference
FRAME_H: int = 480                     # Height to resize input before YOLO inference
SKIP_FRAMES: int = 2                   # Run YOLO inference every Nth frame (~50% CPU saving)

# ─── Memory Management ────────────────────────────────────────────────────────
MAX_STALE_AGE: int = 30                # Frames before an unseen track ID is evicted

# ─── Drawing / Annotation ─────────────────────────────────────────────────────
BOX_COLOR_IN: tuple = (0, 255, 0)       # Green  → person crossed inward
BOX_COLOR_OUT: tuple = (0, 0, 255)      # Red    → person crossed outward
BOX_COLOR_DEFAULT: tuple = (255, 165, 0)# Orange → person not yet crossed
CENTROID_COLOR: tuple = (255, 255, 255) # White centroid dot
CENTROID_RADIUS: int = 5
FONT = 0                                # cv2.FONT_HERSHEY_SIMPLEX (int avoids cv2 import here)
FONT_SCALE: float = 0.6
FONT_THICKNESS: int = 2
