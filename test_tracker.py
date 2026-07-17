"""
test_tracker.py — Unit tests for VisionTracker (crowd_tracker.py)
=================================================================
Run: python -m pytest test_tracker.py -v
     python test_tracker.py           (no pytest needed)
"""

import sys
import unittest
import numpy as np


class TestVisionTrackerPayload(unittest.TestCase):
    """Verify that process_frame() returns the correct payload contract."""

    @classmethod
    def setUpClass(cls):
        """Load tracker once for all tests — YOLO downloads on first run."""
        from crowd_tracker import VisionTracker
        cls.tracker = VisionTracker(line_y=240)
        cls.dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    def _payload(self):
        return self.tracker.process_frame(self.dummy)

    # ── Required payload keys ─────────────────────────────────────────────────

    def test_all_required_keys_present(self):
        """All documented payload keys must exist."""
        required = [
            "annotated_frame", "current_on_screen",
            "total_in", "total_out", "net_flow",
            "track_ids_on_screen",
            # Engineer 4 compat keys:
            "in_count", "out_count", "inflow_rate", "outflow_rate",
        ]
        out = self._payload()
        for key in required:
            self.assertIn(key, out, f"Missing payload key: '{key}'")

    def test_annotated_frame_is_numpy_array(self):
        out = self._payload()
        self.assertIsInstance(out["annotated_frame"], np.ndarray)

    def test_annotated_frame_correct_shape(self):
        out = self._payload()
        h, w, c = out["annotated_frame"].shape
        self.assertEqual(c, 3, "annotated_frame must be 3-channel BGR")
        self.assertGreater(h, 0)
        self.assertGreater(w, 0)

    def test_counts_are_non_negative_ints(self):
        out = self._payload()
        for key in ("current_on_screen", "total_in", "total_out", "in_count", "out_count"):
            self.assertIsInstance(out[key], int, f"{key} must be int")
            self.assertGreaterEqual(out[key], 0, f"{key} must be >= 0")

    def test_net_flow_is_consistent(self):
        out = self._payload()
        self.assertEqual(out["net_flow"], out["total_in"] - out["total_out"])

    def test_in_count_equals_total_in(self):
        """in_count and total_in must always be in sync."""
        out = self._payload()
        self.assertEqual(out["in_count"], out["total_in"])
        self.assertEqual(out["out_count"], out["total_out"])

    def test_track_ids_is_list(self):
        out = self._payload()
        self.assertIsInstance(out["track_ids_on_screen"], list)

    def test_rates_are_floats(self):
        out = self._payload()
        self.assertIsInstance(out["inflow_rate"], float)
        self.assertIsInstance(out["outflow_rate"], float)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def test_reset_zeroes_counters(self):
        self.tracker.in_count  = 5
        self.tracker.out_count = 3
        self.tracker.reset()
        self.assertEqual(self.tracker.in_count,  0)
        self.assertEqual(self.tracker.out_count, 0)
        out = self._payload()
        self.assertEqual(out["total_in"],  0)
        self.assertEqual(out["total_out"], 0)

    # ── Legacy API ────────────────────────────────────────────────────────────

    def test_legacy_line_y_api(self):
        """VisionTracker(line_y=300) must work without error."""
        from crowd_tracker import VisionTracker
        t = VisionTracker(line_y=300)
        out = t.process_frame(self.dummy)
        self.assertIn("annotated_frame", out)

    # ── Line-crossing direction ───────────────────────────────────────────────

    def test_inflow_crossing_logic(self):
        """Simulate a downward crossing — in_count must increment."""
        from crowd_tracker import VisionTracker, TrackState
        t = VisionTracker(line_y=240)
        # Manually insert a track state above the line
        t._track_states[99] = TrackState(cy=100, frame_idx=0)
        t._check_line_crossing(track_id=99, prev_cy=100, curr_cy=300)
        self.assertEqual(t.in_count, 1, "Downward crossing should increment in_count")

    def test_outflow_crossing_logic(self):
        """Simulate an upward crossing — out_count must increment."""
        from crowd_tracker import VisionTracker, TrackState
        t = VisionTracker(line_y=240)
        t._track_states[88] = TrackState(cy=300, frame_idx=0)
        t._check_line_crossing(track_id=88, prev_cy=300, curr_cy=100)
        self.assertEqual(t.out_count, 1, "Upward crossing should increment out_count")

    def test_no_double_count_same_direction(self):
        """Crossing the line twice in the same direction should count only once."""
        from crowd_tracker import VisionTracker, TrackState
        t = VisionTracker(line_y=240)
        t._track_states[77] = TrackState(cy=100, frame_idx=0)
        t._check_line_crossing(77, 100, 300)   # first crossing → in_count = 1
        t._track_states[77].prev_cy = 200
        t._check_line_crossing(77, 200, 350)   # same direction → no increment
        self.assertEqual(t.in_count, 1)

    # ── Stale track eviction ──────────────────────────────────────────────────

    def test_stale_tracks_evicted(self):
        """Tracks older than MAX_STALE_AGE frames must be removed."""
        from crowd_tracker import VisionTracker, TrackState
        t = VisionTracker(line_y=240)
        t._frame_count = 1000
        t._track_states[55] = TrackState(cy=200, frame_idx=0)  # stale
        t._cleanup_stale_tracks()
        self.assertNotIn(55, t._track_states, "Stale track should have been evicted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
