"""
crowd_tracker.py  —  Engineer 1 Module (YOLO Footfall Tracker)
==============================================================
OWNER: Engineer 1
CONTRACT: This file MUST export a class called VisionTracker with:
    - __init__(self, line_y: int)
    - process_frame(self, frame: np.ndarray) -> dict

The dict returned by process_frame() MUST contain these exact keys:
    {
        "annotated_frame":   np.ndarray,  # BGR frame with bounding boxes
        "current_on_screen": int,         # people visible right now
        "inflow_rate":       float,       # people entering per second
        "outflow_rate":      float,       # people leaving per second
        "in_count":          int,         # cumulative entry count
        "out_count":         int,         # cumulative exit count
    }

Engineer 1: Replace the MOCK SECTION below with your real YOLOv8 inference.
"""

import time
import numpy as np
import cv2


class VisionTracker:
    """
    Real-time pedestrian tracker using YOLOv8 + ByteTrack / DeepSORT.
    Engineer 1 replaces the mock body with actual inference.
    """

    def __init__(self, line_y: int = 300):
        self.line_y = line_y
        self.in_count  = 0
        self.out_count = 0
        self._prev_count = 0
        self._last_ts    = time.time()

        # ── ENGINEER 1: load your model here ────────────────────────────────
        # from ultralytics import YOLO
        # self.model = YOLO("yolov8n.pt")
        # ────────────────────────────────────────────────────────────────────
        print("[VisionTracker] Running in MOCK mode — replace with YOLO.")

    def process_frame(self, frame: np.ndarray) -> dict:
        # ── MOCK SECTION — Engineer 1 replaces below ────────────────────────
        out = frame.copy()
        h, w = out.shape[:2]

        mock_count = int(30 + 20 * np.sin(time.time() * 0.4) + np.random.randint(-3, 4))
        mock_count = max(0, mock_count)

        # Virtual line
        cv2.line(out, (0, self.line_y), (w, self.line_y), (0, 255, 255), 2)
        cv2.putText(out, f"LINE  |  Count: {mock_count}", (10, self.line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Mock boxes
        rng = np.random.default_rng(int(time.time()))
        for _ in range(min(mock_count, 10)):
            x = int(rng.integers(0, max(w - 60, 1)))
            y = int(rng.integers(0, max(h - 110, 1)))
            cv2.rectangle(out, (x, y), (x + 50, y + 100), (0, 80, 255), 2)

        cv2.putText(out, "YOLO ENGINE  [MOCK]", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 120, 255), 2)

        now  = time.time()
        dt   = max(now - self._last_ts, 1e-6)
        diff = mock_count - self._prev_count
        inflow_rate  = max(0.0, diff / dt)
        outflow_rate = max(0.0, -diff / dt)
        self._last_ts    = now
        self._prev_count = mock_count
        # ────────────────────────────────────────────────────────────────────

        return {
            "annotated_frame":   out,
            "current_on_screen": mock_count,
            "inflow_rate":       inflow_rate,
            "outflow_rate":      outflow_rate,
            "in_count":          self.in_count,
            "out_count":         self.out_count,
        }
