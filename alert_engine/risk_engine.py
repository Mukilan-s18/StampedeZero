"""
StampedeZero - Risk Engine (ThreatPredictor)
=============================================
The main orchestration class that ties together:

    1. CrowdDataBuffer   — time-series memory
    2. prediction_engine  — mathematical threat analysis
    3. Twilio SMS         — automated emergency alerts with cooldown

This is the SINGLE CLASS that Engineer 4 (UI) imports and interacts with.

Usage:
    from risk_engine import ThreatPredictor

    predictor = ThreatPredictor(danger_threshold=50)

    # Inside the UI loop (called once per frame/second):
    analytics = predictor.update_and_predict(latest_crowd_count)

    # analytics = {
    #     "status": "PREDICTIVE_WARNING",
    #     "inflow_rate": 2.3,
    #     "eta_seconds": 42.1,
    #     "current_count": 37,
    #     "threshold": 50,
    #     "message": "⚠️ PREDICTIVE WARNING: ...",
    #     "sms_sent": False,
    #     "buffer_fill": "34/60"
    # }

Architecture Notes:
    - The ThreatPredictor is stateful (holds buffer + last alert time)
    - SMS is fire-and-forget with a configurable cooldown (default 120s)
    - If Twilio credentials are missing, the engine still works — it just
      logs warnings instead of sending SMS (graceful degradation)
"""

import os
import time
import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from .data_buffer import CrowdDataBuffer
from . import prediction_engine as predictor

# ─── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("StampedeZero.RiskEngine")


# ─── ThreatPredictor Class ───────────────────────────────────────────────────

class ThreatPredictor:
    """
    Unified prediction + alerting engine for crowd stampede prevention.

    Accepts crowd count updates, maintains a rolling time-series buffer,
    runs linear regression to predict threshold breaches, and fires
    spam-proof SMS alerts when danger is detected.

    Args:
        danger_threshold:  Crowd count at which density becomes dangerous.
        buffer_size:       Number of data points to retain in the rolling window.
        cooldown_seconds:  Minimum seconds between consecutive SMS alerts.
        warning_horizon:   Seconds of advance warning for PREDICTIVE_WARNING.
        velocity_floor:    Minimum inflow rate (ppl/sec) to consider "rising."
        sms_enabled:       Set to False to disable SMS entirely (dry-run mode).
    """

    def __init__(
        self,
        danger_threshold: int = 100,
        buffer_size: int = 60,
        cooldown_seconds: int = 120,
        warning_horizon: int = 60,
        velocity_floor: float = 0.5,
        sms_enabled: bool = True,
    ):
        # Load environment variables
        load_dotenv()

        # Configuration
        self.threshold = danger_threshold
        self.cooldown_seconds = cooldown_seconds
        self.warning_horizon = warning_horizon
        self.velocity_floor = velocity_floor
        self.sms_enabled = sms_enabled

        # State
        self.buffer = CrowdDataBuffer(buffer_size=buffer_size)
        self.last_alert_time: float = 0.0
        self.alerts_sent: int = 0
        self.alerts_suppressed: int = 0

        # Twilio client (lazy-initialized)
        self._twilio_client = None
        self._twilio_ready = False
        self._init_twilio()

        logger.info(
            f"ThreatPredictor initialized — threshold={danger_threshold}, "
            f"buffer={buffer_size}s, cooldown={cooldown_seconds}s, "
            f"sms={'ON' if self._twilio_ready else 'OFF (no credentials)'}"
        )

    # ─── Public API ──────────────────────────────────────────────────────

    def update_and_predict(self, current_count: int) -> Dict[str, Any]:
        """
        Primary method: push a new crowd reading and get a full threat assessment.

        This should be called once per frame/second from the UI loop.

        Args:
            current_count: Latest crowd count from the detection pipeline.

        Returns:
            Dictionary with threat status, inflow rate, ETA, message,
            and metadata about SMS delivery.
        """
        # Step 1: Push data into buffer
        self.buffer.push(current_count)

        # Step 2: Run prediction engine
        analysis = predictor.analyze(
            timestamps=self.buffer.get_timestamps(),
            counts=self.buffer.get_counts(),
            current_count=current_count,
            threshold=self.threshold,
            warning_horizon=self.warning_horizon,
            velocity_floor=self.velocity_floor,
        )

        # Step 3: Fire SMS if threat level warrants it
        sms_sent = False
        status = analysis["status"]

        if status in (predictor.CRITICAL_CAPACITY, predictor.PREDICTIVE_WARNING):
            sms_sent = self._fire_sms(analysis["message"])

        # Step 4: Enrich response with metadata
        analysis["sms_sent"] = sms_sent
        analysis["buffer_fill"] = f"{self.buffer.size}/{self.buffer.capacity}"
        analysis["alerts_sent"] = self.alerts_sent
        analysis["alerts_suppressed"] = self.alerts_suppressed

        # Step 5: Log status
        self._log_status(analysis)

        return analysis

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Return the current engine state without pushing new data.
        Useful for health checks and dashboard polling.
        """
        latest = self.buffer.get_latest()
        return {
            "buffer_fill": f"{self.buffer.size}/{self.buffer.capacity}",
            "buffer_ready": self.buffer.is_ready,
            "last_count": latest[1] if latest else None,
            "total_updates": self.buffer.total_updates,
            "alerts_sent": self.alerts_sent,
            "alerts_suppressed": self.alerts_suppressed,
            "sms_enabled": self._twilio_ready and self.sms_enabled,
            "cooldown_remaining": max(
                0,
                self.cooldown_seconds - (time.time() - self.last_alert_time),
            ),
        }

    def reset(self) -> None:
        """Clear the buffer and reset alert counters. Use between test runs."""
        self.buffer.clear()
        self.last_alert_time = 0.0
        self.alerts_sent = 0
        self.alerts_suppressed = 0
        logger.info("Engine state reset.")

    # ─── SMS Subsystem ───────────────────────────────────────────────────

    def _init_twilio(self) -> None:
        """
        Lazily initialize the Twilio client. If credentials are missing,
        SMS is silently disabled (engine still works for prediction only).
        """
        sid = os.getenv("TWILIO_SID")
        token = os.getenv("TWILIO_TOKEN")
        from_number = os.getenv("TWILIO_FROM")
        target = os.getenv("TARGET_PHONE")

        if not all([sid, token, from_number, target]):
            logger.warning(
                "Twilio credentials not found in .env — SMS alerts disabled. "
                "Prediction engine will continue to operate in dry-run mode."
            )
            self._twilio_ready = False
            return

        try:
            from twilio.rest import Client
            self._twilio_client = Client(sid, token)
            self._twilio_from = from_number
            self._twilio_to = target
            self._twilio_ready = True
            logger.info("Twilio SMS gateway initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {e}")
            self._twilio_ready = False

    def _fire_sms(self, message_body: str) -> bool:
        """
        Send an SMS alert with cooldown-based spam prevention.

        Returns True if an SMS was actually sent, False if suppressed
        (due to cooldown) or unavailable (no Twilio credentials).
        """
        # Guard: SMS disabled
        if not self.sms_enabled or not self._twilio_ready:
            return False

        # Guard: Cooldown period
        now = time.time()
        elapsed_since_last = now - self.last_alert_time
        if elapsed_since_last < self.cooldown_seconds:
            self.alerts_suppressed += 1
            logger.debug(
                f"SMS suppressed (cooldown: {self.cooldown_seconds - elapsed_since_last:.0f}s remaining)"
            )
            return False

        # Send SMS
        try:
            sent = self._twilio_client.messages.create(
                body=message_body,
                from_=self._twilio_from,
                to=self._twilio_to,
            )
            self.last_alert_time = now
            self.alerts_sent += 1
            logger.critical(f"🚨 SMS ALERT SENT (SID: {sent.sid}): {message_body}")
            return True
        except Exception as e:
            logger.error(f"SMS delivery failed: {e}")
            return False

    # ─── Internal Helpers ────────────────────────────────────────────────

    def _log_status(self, analysis: Dict[str, Any]) -> None:
        """Log the current prediction status at the appropriate severity."""
        status = analysis["status"]
        msg = (
            f"[{status}] count={analysis['current_count']}/{self.threshold} "
            f"rate={analysis['inflow_rate']:.2f} ppl/s "
            f"eta={analysis['eta_seconds']}s "
            f"buffer={analysis['buffer_fill']}"
        )
        if status == predictor.CRITICAL_CAPACITY:
            logger.critical(msg)
        elif status == predictor.PREDICTIVE_WARNING:
            logger.warning(msg)
        elif status == predictor.ELEVATED:
            logger.info(msg)
        else:
            logger.debug(msg)

    def __repr__(self) -> str:
        return (
            f"ThreatPredictor(threshold={self.threshold}, "
            f"buffer={self.buffer.size}/{self.buffer.capacity}, "
            f"alerts_sent={self.alerts_sent}, "
            f"sms={'ON' if self._twilio_ready else 'OFF'})"
        )
