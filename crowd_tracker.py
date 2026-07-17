"""
crowd_tracker.py — StampedeZero Vision Tracker
Branch: mukil  |  Engineer 1 (Vision & Tracking Lead)

PRIMARY DELIVERABLE — import this file from Engineer 4's Streamlit app:

    from crowd_tracker import VisionTracker

    # New API (preferred)
    tracker = VisionTracker(line_y_fraction=0.55)

    # Legacy API (Engineer 4 contract — also works)
    tracker = VisionTracker(line_y=300)

    payload = tracker.process_frame(raw_numpy_frame)

Payload keys (superset of Engineer 4's contract):
    annotated_frame   np.ndarray  BGR image with overlays
    current_on_screen int         people visible right now
    total_in          int         cumulative inflow (preferred)
    total_out         int         cumulative outflow (preferred)
    net_flow          int         total_in - total_out
    track_ids_on_screen list[int] active track IDs
    in_count          int         alias for total_in  (Engineer 4 compat)
    out_count         int         alias for total_out (Engineer 4 compat)
    inflow_rate       float       people entering per second
    outflow_rate      float       people leaving per second

The class is intentionally self-contained. No Streamlit, no Twilio, no database.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Optional, Union

import cv2
import numpy as np
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    _YOLO_AVAILABLE = False
    logging.getLogger("VisionTracker").warning(
        "ultralytics not installed — VisionTracker running in DEMO/MOCK mode."
    )

import config as cfg

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VisionTracker")


# ─── Data types ───────────────────────────────────────────────────────────────
class TrackState:
    """Tracks the per-person state needed for line-crossing detection."""

    __slots__ = ("prev_cy", "last_seen_frame", "crossed_direction")

    def __init__(self, cy: int, frame_idx: int) -> None:
        self.prev_cy: int = cy
        self.last_seen_frame: int = frame_idx
        self.crossed_direction: Optional[str] = None  # "in" | "out" | None


# ─── Main Class ───────────────────────────────────────────────────────────────
class VisionTracker:
    """
    Person tracking and bidirectional line-crossing counter.

    Uses YOLOv8 + ByteTrack to assign stable integer IDs to each person,
    then uses a virtual horizontal line to count inflow vs outflow.

    Backward-compatible with Engineer 4's contract:
        VisionTracker(line_y=300)  — legacy pixel-based API
        VisionTracker(line_y_fraction=0.55)  — preferred fraction-based API

    Args:
        line_y_fraction: Vertical position of the counting line as a fraction
            of frame height (default from config.LINE_Y_FRACTION).
        line_y: Absolute pixel position of counting line (Engineer 4 compat).
            If provided, overrides line_y_fraction.
        model_path: Path / URL to YOLO weights file.
        tracker_cfg: YOLO tracker config name (bytetrack.yaml or botsort.yaml).
        skip_frames: Run YOLO inference every Nth frame to reduce CPU load.
        max_stale_age: Delete a track ID after this many frames of absence.
        confidence: Minimum detection confidence to accept.
    """

    def __init__(
        self,
        line_y_fraction: float = cfg.LINE_Y_FRACTION,
        line_y: Optional[int] = None,          # Engineer 4 legacy compat
        model_path: str = cfg.MODEL_PATH,
        tracker_cfg: str = cfg.TRACKER_CFG,
        skip_frames: int = cfg.SKIP_FRAMES,
        max_stale_age: int = cfg.MAX_STALE_AGE,
        confidence: float = cfg.CONFIDENCE_THRESHOLD,
    ) -> None:

        self._demo_mode = not _YOLO_AVAILABLE
        if _YOLO_AVAILABLE:
            logger.info("Loading YOLO model: %s", model_path)
            self._model = YOLO(model_path)
        else:
            logger.warning("DEMO MODE: skipping YOLO model load — returning mock frames.")
            self._model = None
        self._tracker_cfg = tracker_cfg
        self._confidence = confidence

        # Virtual line — support both pixel (legacy) and fraction (preferred) APIs
        if line_y is not None:
            # Legacy: Engineer 4 passes line_y=300; store as-is and set fraction=None
            self._line_y_fixed: Optional[int] = line_y
            self._line_y_fraction = None
            self._line_y: int = line_y
            logger.info("VisionTracker: using fixed line_y=%d px (legacy mode)", line_y)
        else:
            self._line_y_fixed = None
            self._line_y_fraction = line_y_fraction
            self._line_y = 300  # Recalculated on first frame

        # Frame bookkeeping
        self._frame_count: int = 0
        self._skip_frames: int = max(1, skip_frames)
        self._max_stale_age: int = max_stale_age

        # State
        self._track_states: dict[int, TrackState] = {}
        self._last_results = None  # Cache for frame-skip reuse

        # Counters (Engineer 4 contract uses in_count / out_count directly)
        self.in_count: int = 0
        self.out_count: int = 0

        # Rate tracking for Engineer 4's inflow_rate / outflow_rate keys
        self._prev_on_screen: int = 0
        self._last_rate_ts: float = time.time()
        self._inflow_rate: float = 0.0
        self._outflow_rate: float = 0.0

        logger.info(
            "VisionTracker ready | line_y=%.0f%% | skip=%d | stale_age=%d",
            (line_y_fraction or 0) * 100,
            skip_frames,
            max_stale_age,
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_id: Optional[int] = None) -> dict:
        """
        Full pipeline: resize → track → extract → crossing logic → annotate.

        Args:
            frame: Raw BGR numpy image array from cv2.VideoCapture or any source.

        Returns:
            dict with keys:
                annotated_frame    (np.ndarray) BGR image with overlays drawn
                current_on_screen  (int)        people visible right now
                total_in           (int)        cumulative inflow count
                total_out          (int)        cumulative outflow count
                net_flow           (int)        total_in − total_out
                track_ids_on_screen(list[int])  active track IDs this frame
        """
        # ── DEMO MODE: return mock data when ultralytics is not installed ──
        if self._demo_mode:
            frame = cv2.resize(frame, (cfg.FRAME_W, cfg.FRAME_H))
            mock_count = int(30 + 20 * np.sin(time.time() * 0.4) + np.random.randint(-3, 4))
            mock_count = max(0, mock_count)
            out = frame.copy()
            h, w = out.shape[:2]
            line_y = int(h * (self._line_y_fraction or 0.55))
            cv2.line(out, (0, line_y), (w, line_y), (0, 255, 255), 2)
            cv2.putText(out, f"DEMO MODE | Count: {mock_count}", (10, line_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(out, "YOLO ENGINE [MOCK - install ultralytics for real]",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 120, 255), 2)
            return {
                "annotated_frame": out,
                "current_on_screen": mock_count,
                "total_in": self.in_count,
                "total_out": self.out_count,
                "net_flow": self.in_count - self.out_count,
                "track_ids_on_screen": [],
                "in_count": self.in_count,
                "out_count": self.out_count,
                "inflow_rate": 0.0,
                "outflow_rate": 0.0,
            }

        # ── Step 1: Resize input to safe resolution ────────────────────────
        frame = cv2.resize(frame, (cfg.FRAME_W, cfg.FRAME_H))
        h, w = frame.shape[:2]

        # Recalculate line_y based on actual frame height (fraction mode only)
        if self._line_y_fraction is not None:
            self._line_y = int(h * self._line_y_fraction)

        # ── Step 2: Run YOLO (or reuse cached result on skipped frames) ───
        self._frame_count += 1
        run_inference = (self._frame_count % self._skip_frames == 0)

        if run_inference:
            results = self._model.track(
                frame,
                classes=[cfg.PERSON_CLASS],
                persist=True,
                tracker=self._tracker_cfg,
                conf=self._confidence,
                verbose=False,
            )
            self._last_results = results
        else:
            results = self._last_results

        # ── Step 3: Extract detections ────────────────────────────────────
        detections = self._extract_detections(results)

        # ── Step 4: Line-crossing logic ───────────────────────────────────
        active_ids: list[int] = []
        for track_id, cx, cy, x1, y1, x2, y2 in detections:
            active_ids.append(track_id)
            if track_id in self._track_states:
                state = self._track_states[track_id]
                self._check_line_crossing(track_id, state.prev_cy, cy)
                state.prev_cy = cy
                state.last_seen_frame = self._frame_count
            else:
                # First appearance — record position, skip crossing check
                self._track_states[track_id] = TrackState(
                    cy=cy, frame_idx=self._frame_count
                )

        # ── Step 5: Evict stale tracks ────────────────────────────────────
        self._cleanup_stale_tracks()

        # ── Step 6: Draw annotations ──────────────────────────────────────
        annotated = self._draw_annotations(frame.copy(), detections)

        # ── Step 7: Compute per-second rates (Engineer 4 compat) ─────────
        now = time.time()
        dt  = max(now - self._last_rate_ts, 1e-6)
        curr_on = len(active_ids)
        diff = curr_on - self._prev_on_screen
        self._inflow_rate  = max(0.0, float(diff) / dt)
        self._outflow_rate = max(0.0, float(-diff) / dt)
        self._prev_on_screen = curr_on
        self._last_rate_ts   = now

        # ── Step 8: Return clean payload ──────────────────────────────────
        return {
            # ── Preferred (new) keys ──────────────────────────────────────
            "annotated_frame":     annotated,
            "current_on_screen":   curr_on,
            "total_in":            self.in_count,
            "total_out":           self.out_count,
            "net_flow":            self.in_count - self.out_count,
            "track_ids_on_screen": active_ids,
            # ── Engineer 4 backward-compat keys ──────────────────────────
            "in_count":            self.in_count,
            "out_count":           self.out_count,
            "inflow_rate":         self._inflow_rate,
            "outflow_rate":        self._outflow_rate,
        }

    def reset(self) -> None:
        """
        Reset all counters and tracking state.
        Call this from Engineer 4's Streamlit 'Reset Counts' button.
        """
        self.in_count = 0
        self.out_count = 0
        self._track_states.clear()
        self._frame_count = 0
        self._last_results = None
        self._prev_on_screen = 0
        self._inflow_rate = 0.0
        self._outflow_rate = 0.0
        logger.info("VisionTracker state reset.")

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _extract_detections(
        self, results
    ) -> list[tuple[int, int, int, int, int, int, int]]:
        """
        Parse raw YOLO results into a clean list of tuples.

        Returns:
            List of (track_id, cx, cy, x1, y1, x2, y2) — all ints.
            Empty list if YOLO found nothing or tracking IDs are absent.
        """
        if results is None:
            return []

        boxes_result = results[0].boxes
        if boxes_result is None or boxes_result.id is None:
            return []

        raw_boxes = boxes_result.xyxy.cpu().numpy()   # shape: (N, 4)
        raw_ids   = boxes_result.id.int().cpu().numpy()  # shape: (N,)

        detections: list[tuple] = []
        for box, tid in zip(raw_boxes, raw_ids):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append((int(tid), cx, cy, x1, y1, x2, y2))

        return detections

    def _check_line_crossing(
        self, track_id: int, prev_cy: int, curr_cy: int
    ) -> None:
        """
        Core crossing logic — direction-aware signed Y-axis comparison.

        Coordinate system (standard OpenCV):
            Y increases downward (0 = top of frame)

        INFLOW  = person moves DOWN  past line (prev_cy < line_y AND curr_cy >= line_y)
        OUTFLOW = person moves UP    past line (prev_cy > line_y AND curr_cy <= line_y)
        """
        state = self._track_states.get(track_id)
        if state is None:
            return

        line = self._line_y

        # Inflow: crossed downward
        if prev_cy < line <= curr_cy:
            if state.crossed_direction != "in":
                self.in_count += 1
                state.crossed_direction = "in"
                logger.debug("Track %d → INFLOW  (prev_cy=%d, curr_cy=%d)", track_id, prev_cy, curr_cy)

        # Outflow: crossed upward
        elif prev_cy > line >= curr_cy:
            if state.crossed_direction != "out":
                self.out_count += 1
                state.crossed_direction = "out"
                logger.debug("Track %d → OUTFLOW (prev_cy=%d, curr_cy=%d)", track_id, prev_cy, curr_cy)

    def _draw_annotations(
        self,
        frame: np.ndarray,
        detections: list[tuple[int, int, int, int, int, int, int]],
    ) -> np.ndarray:
        """
        Draw all visual overlays onto the frame:
          - Virtual counting line
          - Bounding boxes (colour-coded by crossing state)
          - Track ID labels
          - Centroid dots
          - Corner HUD with in/out/net counters
          - FPS indicator
        """
        h, w = frame.shape[:2]
        line_y = self._line_y

        # ── Virtual line ──────────────────────────────────────────────────
        cv2.line(frame, (0, line_y), (w, line_y), cfg.LINE_COLOR, cfg.LINE_THICKNESS)

        # Line label
        cv2.putText(
            frame, "COUNTING LINE",
            (10, line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cfg.LINE_COLOR, 1,
        )

        # ── Per-person overlays ───────────────────────────────────────────
        for track_id, cx, cy, x1, y1, x2, y2 in detections:
            state = self._track_states.get(track_id)
            direction = state.crossed_direction if state else None

            # Box colour by crossing state
            if direction == "in":
                box_color = cfg.BOX_COLOR_IN
            elif direction == "out":
                box_color = cfg.BOX_COLOR_OUT
            else:
                box_color = cfg.BOX_COLOR_DEFAULT

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # ID label (above box)
            label = f"ID:{track_id}"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE, cfg.FONT_THICKNESS
            )
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), box_color, -1)
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE,
                (0, 0, 0), cfg.FONT_THICKNESS,
            )

            # Centroid dot
            cv2.circle(frame, (cx, cy), cfg.CENTROID_RADIUS, cfg.CENTROID_COLOR, -1)

        # ── Corner HUD ────────────────────────────────────────────────────
        hud_lines = [
            (f"IN:  {self.in_count:>4}", cfg.BOX_COLOR_IN),
            (f"OUT: {self.out_count:>4}", cfg.BOX_COLOR_OUT),
            (f"NET: {self.in_count - self.out_count:>+4}", cfg.LINE_COLOR),
            (f"ON SCREEN: {len(detections)}", (200, 200, 200)),
        ]
        # Semi-transparent dark background for HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 160, 10), (w - 5, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, (text, color) in enumerate(hud_lines):
            cv2.putText(
                frame, text,
                (w - 150, 32 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
            )

        return frame

    def _cleanup_stale_tracks(self) -> None:
        """
        Evict track IDs that haven't been seen for `max_stale_age` frames.

        Without this, every person who ever walked past leaves their
        dictionary entry permanently — causing a RAM leak after hours of use.
        """
        stale = [
            tid
            for tid, state in self._track_states.items()
            if (self._frame_count - state.last_seen_frame) > self._max_stale_age
        ]
        for tid in stale:
            del self._track_states[tid]
            logger.debug("Evicted stale track ID %d", tid)

    # ─── Dunder helpers ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"VisionTracker("
            f"in={self.in_count}, out={self.out_count}, "
            f"tracked={len(self._track_states)}, "
            f"frame={self._frame_count})"
        )
