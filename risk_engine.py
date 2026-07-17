"""
risk_engine.py — StampedeZero (root level)
==========================================
⚠️  This file is a COMPATIBILITY SHIM only.
    The real implementation lives in alert_engine/risk_engine.py

    Importing ThreatPredictor from here is identical to:
        from alert_engine import ThreatPredictor

    This file exists so legacy code that does
        from risk_engine import ThreatPredictor
    continues to work without modification.
"""
# Re-export the real implementation — do not add business logic here.
from alert_engine import ThreatPredictor  # noqa: F401

__all__ = ["ThreatPredictor"]
