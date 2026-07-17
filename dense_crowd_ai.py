"""
dense_crowd_ai.py — StampedeZero (root level)
==============================================
⚠️  This file is a COMPATIBILITY SHIM only.
    The real CSRNet implementation lives in heatmap_engine/dense_crowd_ai.py

    Importing DensityEstimator from here is identical to:
        from heatmap_engine.dense_crowd_ai import DensityEstimator

    This file exists so legacy code that does
        from dense_crowd_ai import DensityEstimator
    continues to work without modification.
"""
import sys
import os

# Inject heatmap_engine into path so the real module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "heatmap_engine"))

from dense_crowd_ai import DensityEstimator  # noqa: F401 — real implementation

__all__ = ["DensityEstimator"]
