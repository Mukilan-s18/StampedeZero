"""
dense_crowd_ai.py  —  Engineer 2 Module (CSRNet Density Estimator)
==================================================================
OWNER: Engineer 2
CONTRACT: This file MUST export a class called DensityEstimator with:
    - __init__(self, weights_path: str)
    - process_frame(self, frame: np.ndarray) -> dict

The dict returned by process_frame() MUST contain these exact keys:
    {
        "heatmap_frame":   np.ndarray,  # BGR frame with heatmap overlay
        "estimated_count": int,         # total estimated crowd count
        "density_map":     np.ndarray,  # raw float32 density map (H x W)
        "peak_density":    float,       # max value in density map
    }

Engineer 2: Replace the MOCK SECTION below with your real CSRNet inference.
NOTE: .pth weights are git-ignored — share via Google Drive.
"""

import time
import numpy as np
import cv2


class DensityEstimator:
    """
    Dense crowd counter using CSRNet / DM-Count.
    Engineer 2 replaces the mock body with actual inference.
    """

    def __init__(self, weights_path: str = "weights/csrnet_weights.pth"):
        self.weights_path = weights_path

        # ── ENGINEER 2: load your model here ────────────────────────────────
        # import torch
        # from models.csrnet import CSRNet
        # self.model = CSRNet()
        # checkpoint = torch.load(weights_path, map_location="cpu")
        # self.model.load_state_dict(checkpoint)
        # self.model.eval()
        # ────────────────────────────────────────────────────────────────────
        print("[DensityEstimator] Running in MOCK mode — replace with CSRNet.")

    def process_frame(self, frame: np.ndarray) -> dict:
        # ── MOCK SECTION — Engineer 2 replaces below ────────────────────────
        h, w = frame.shape[:2]
        density_map = np.zeros((h, w), dtype=np.float32)

        rng = np.random.default_rng(int(time.time() * 2))
        num_blobs = int(rng.integers(4, 9))
        for _ in range(num_blobs):
            cx    = int(rng.integers(0, w))
            cy    = int(rng.integers(0, h))
            sigma = int(rng.integers(30, 100))
            amp   = float(rng.uniform(0.5, 2.5))
            yy, xx = np.ogrid[:h, :w]
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            density_map += blob.astype(np.float32) * amp

        norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        heat = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.45, heat, 0.55, 0)

        estimated_count = int(density_map.sum() * float(rng.uniform(0.9, 1.1)))
        estimated_count = max(50, min(estimated_count, 3000))

        cv2.putText(blended, f"CSRNet  |  Est: {estimated_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(blended, "DENSITY ENGINE  [MOCK]", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 255), 2)
        # ────────────────────────────────────────────────────────────────────

        return {
            "heatmap_frame":   blended,
            "estimated_count": estimated_count,
            "density_map":     density_map,
            "peak_density":    float(density_map.max()),
        }
