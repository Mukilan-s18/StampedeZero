"""
config.py — StampedeZero Shared Configuration
==============================================
Central configuration for ALL engineers. Tune these values to match your
demo environment without touching any business logic.

  Engineer 1 (Vision):   MODEL_PATH, LINE_Y_FRACTION, SKIP_FRAMES …
  Engineer 2 (Heatmap):  CSRNET_WEIGHTS, HEATMAP_INFER_SIZE …
  Engineer 3 (Alerts):   VENUE_CAPACITY, SMS_COOLDOWN, BUFFER_SIZE …
  Engineer 4 (UI):       DEMO_MODE, MAX_HIST …
"""

# ─── DEMO MODE ────────────────────────────────────────────────────────────────
# Set True  → app starts without hardware (webcam/GPU/weights) — uses mocks
# Set False → full real inference (requires webcam + CSRNet weights + Twilio .env)
DEMO_MODE: bool = True

# ─── Engineer 1 — Vision Tracker ──────────────────────────────────────────────
MODEL_PATH: str = "yolov8n.pt"        # YOLOv8 nano weights (auto-downloads on first run)
TRACKER_CFG: str = "bytetrack.yaml"   # ByteTrack — lowest latency built-in tracker

PERSON_CLASS: int = 0                  # COCO class 0 = "person"
CONFIDENCE_THRESHOLD: float = 0.4     # Minimum detection confidence (0.0–1.0)

#   LINE_Y_FRACTION: counting line as a FRACTION of frame height (0.0→1.0)
#   INFLOW  = person moves DOWN  past line (prev_cy < line_y, curr_cy >= line_y)
#   OUTFLOW = person moves UP    past line (prev_cy > line_y, curr_cy <= line_y)
LINE_Y_FRACTION: float = 0.55         # ~55% from top — slightly below midpoint
LINE_COLOR: tuple = (0, 255, 255)      # Cyan (BGR)
LINE_THICKNESS: int = 2

FRAME_W: int = 640                    # Resize width before YOLO inference
FRAME_H: int = 480                    # Resize height before YOLO inference
SKIP_FRAMES: int = 2                  # Run YOLO every Nth frame (~50% CPU saving)
MAX_STALE_AGE: int = 30               # Frames before unseen track ID is evicted

BOX_COLOR_IN: tuple = (0, 255, 0)      # Green  → crossed inward
BOX_COLOR_OUT: tuple = (0, 0, 255)     # Red    → crossed outward
BOX_COLOR_DEFAULT: tuple = (255, 165, 0)  # Orange → not yet crossed
CENTROID_COLOR: tuple = (255, 255, 255)
CENTROID_RADIUS: int = 5
FONT = 0                               # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float = 0.6
FONT_THICKNESS: int = 2

# ─── Engineer 2 — CSRNet Heatmap ──────────────────────────────────────────────
CSRNET_WEIGHTS: str = "weights/csrnet_weights.pth"
HEATMAP_INFER_SIZE: tuple = (320, 240)  # Pre-downscale before CNN (PERF-04)

# ─── Engineer 3 — Alert Engine ────────────────────────────────────────────────
VENUE_CAPACITY: int = 50              # Max safe crowd count before alerts fire
WARN_PCT: float = 0.75                # Fraction of capacity that triggers warning
BUFFER_SIZE: int = 60                 # Rolling window size (seconds at 1 FPS)
SMS_COOLDOWN_SECONDS: int = 120       # Minimum seconds between consecutive SMS
WARNING_HORIZON_SECONDS: int = 60     # ETA threshold for PREDICTIVE_WARNING
VELOCITY_FLOOR: float = 0.5           # Min inflow rate (ppl/s) to consider "rising"

# ─── Engineer 4 — Streamlit UI ────────────────────────────────────────────────
MAX_HIST: int = 120                   # Max data points in history chart
CHART_UPDATE_INTERVAL: int = 30       # Update Plotly charts every N frames (PERF-02)
VIDEO_SOURCE: int = 0                 # Default webcam index (0 = built-in)
CNN_VIDEO_PATH: str = "data/crowd_concert.mp4"
