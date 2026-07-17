"""
heatmap_engine/dense_crowd_ai.py — StampedeZero Density Estimator (CSRNet)
===========================================================================
OWNER: Engineer 2 (Heatmap / Density Lead)

CONTRACT: exports class DensityEstimator with:
    - __init__(self, weight_path: str)            ← canonical kwarg name
    - __init__(self, weights_path: str)           ← compat alias (Engineer 4)
    - process_frame(self, frame: np.ndarray) -> dict

Return dict keys:
    heatmap_frame   np.ndarray  BGR frame with jet heatmap overlay
    estimated_count int         total estimated crowd count
    density_map     np.ndarray  raw float32 density map (H x W)
    peak_density    float       max value in density map
"""

import logging
import os
import sys
import time

import cv2
import numpy as np
try:
    import torch
    from torchvision import transforms
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    transforms = None
    _TORCH_AVAILABLE = False
    logging.getLogger("DensityEstimator").warning(
        "torch/torchvision not installed — DensityEstimator running in DEMO/MOCK mode."
    )

# Allow importing model.py regardless of CWD
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    if _TORCH_AVAILABLE:
        from model import CSRNet
    else:
        CSRNet = None
except ImportError:
    CSRNet = None

logger = logging.getLogger("DensityEstimator")


class DensityEstimator:
    """
    Dense crowd counter using CSRNet.

    Falls back to a Gaussian-blob mock if the weights file is absent so the app
    always starts successfully on a clean machine without downloaded weights.

    Args:
        weight_path:  Path to .pth CSRNet checkpoint.
        weights_path: Alias accepted for Engineer 4 compatibility.
        infer_size:   Input size for CNN inference. Smaller = faster on CPU.
    """

    def __init__(
        self,
        weight_path: str = "weights/csrnet_weights.pth",
        weights_path: str = None,       # Engineer 4 compat alias
        infer_size: tuple = (320, 240), # PERF-04: pre-downscale before CNN
    ):
        # Resolve alias
        resolved_path = weights_path if weights_path is not None else weight_path
        self._infer_size = infer_size
        
        if not _TORCH_AVAILABLE:
            logger.warning("DEMO MODE: PyTorch not available — forcing mock mode.")
            self._mock_mode = True
            self.device = None
            self.transform = None
            return

        self._mock_mode = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # QC-03: Graceful fallback — never crash on missing weights
        if not os.path.exists(resolved_path):
            logger.warning(
                "CSRNet weights not found at '%s' — running in MOCK mode. "
                "Download weights and place them at that path for real inference.",
                resolved_path,
            )
            self._mock_mode = True
            return

        try:
            self.model = CSRNet(load_weights=True)
            checkpoint = torch.load(resolved_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            logger.info("CSRNet loaded from '%s' on %s", resolved_path, self.device)
        except Exception as exc:
            logger.error("Failed to load CSRNet weights: %s — falling back to mock", exc)
            self._mock_mode = True

    # ─── Public API ───────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Run density estimation on one BGR frame.

        Returns:
            heatmap_frame   (np.ndarray) BGR frame with heatmap overlay
            estimated_count (int)        crowd count estimate
            density_map     (np.ndarray) raw float32 density map (H × W)
            peak_density    (float)      max density value
        """
        if self._mock_mode:
            return self._mock_frame(frame)
        return self._real_frame(frame)

    # ─── Real Inference ───────────────────────────────────────────────────────

    def _real_frame(self, frame: np.ndarray) -> dict:
        original_h, original_w = frame.shape[:2]

        # PERF-04: Downscale before feeding CNN — faster on CPU
        small = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            self._infer_size,
        )
        img_tensor = self.transform(small).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        count = int(torch.sum(output).item())

        # Upscale density map back to original frame size
        density_map = output.squeeze().cpu().numpy()
        density_map_full = cv2.resize(density_map, (original_w, original_h))

        heatmap_frame = self._apply_heatmap(frame, density_map_full)

        return {
            "heatmap_frame":   heatmap_frame,
            "estimated_count": count,
            "density_map":     density_map_full,
            "peak_density":    float(density_map_full.max()),
        }

    # ─── Mock Fallback ────────────────────────────────────────────────────────

    def _mock_frame(self, frame: np.ndarray) -> dict:
        """Gaussian-blob simulation when weights are unavailable."""
        h, w = frame.shape[:2]
        density_map = np.zeros((h, w), dtype=np.float32)

        rng = np.random.default_rng(int(time.time() * 2))
        for _ in range(int(rng.integers(4, 9))):
            cx    = int(rng.integers(0, w))
            cy    = int(rng.integers(0, h))
            sigma = int(rng.integers(30, 100))
            amp   = float(rng.uniform(0.5, 2.5))
            yy, xx = np.ogrid[:h, :w]
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            density_map += blob.astype(np.float32) * amp

        estimated_count = max(10, min(int(density_map.sum() * rng.uniform(0.9, 1.1)), 3000))
        heatmap_frame   = self._apply_heatmap(frame, density_map)

        cv2.putText(heatmap_frame, f"CSRNet  |  Est: {estimated_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(heatmap_frame, "DENSITY ENGINE  [MOCK — no weights]", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 255), 1)

        return {
            "heatmap_frame":   heatmap_frame,
            "estimated_count": estimated_count,
            "density_map":     density_map,
            "peak_density":    float(density_map.max()),
        }

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_heatmap(frame: np.ndarray, density_map: np.ndarray) -> np.ndarray:
        """Normalize density map and blend as a JET colormap overlay."""
        d_min, d_max = density_map.min(), density_map.max()
        if d_max - d_min > 0:
            norm = ((density_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(density_map, dtype=np.uint8)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
