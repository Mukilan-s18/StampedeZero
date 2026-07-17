"""
StampedeZero Alert Engine
==========================
Predictive crowd stampede prevention system with automated SMS alerts.

Public API:
    from alert_engine.risk_engine import ThreatPredictor
"""

from .risk_engine import ThreatPredictor
from .data_buffer import CrowdDataBuffer
from . import prediction_engine

__all__ = ["ThreatPredictor", "CrowdDataBuffer", "prediction_engine"]
__version__ = "1.0.0"
