"""
risk_engine.py  —  Engineer 3 Module (Threat Predictor + Twilio SMS)
====================================================================
OWNER: Engineer 3
CONTRACT: This file MUST export a class called ThreatPredictor with:
    - __init__(self, threshold: int, warn_pct: float, window_size: int)
    - update_and_predict(self, current_count: int) -> dict

The dict returned by update_and_predict() MUST contain these exact keys:
    {
        "status":      str,           # "SAFE" | "PREDICTIVE_WARNING" | "CRITICAL_CAPACITY"
        "eta_seconds": float | None,  # seconds to capacity breach, or None if safe
        "inflow_rate": float,         # people/sec (positive = inflow)
        "risk_score":  float,         # 0.0 – 1.0
        "sms_sent":    bool,          # True if Twilio fired this tick
    }

Engineer 3: Replace MOCK SECTION with real LSTM + Twilio code.
Store Twilio credentials in a .env file — NEVER hard-code them.
"""

import os
import time
import numpy as np
from collections import deque


class ThreatPredictor:
    """
    Real-time stampede risk predictor.
    Engineer 3 replaces the mock body with LSTM inference + Twilio.
    """

    def __init__(
        self,
        threshold:   int   = 50,
        warn_pct:    float = 0.75,
        window_size: int   = 30,
    ):
        self.threshold      = threshold
        self.warn_threshold = int(threshold * warn_pct)
        self.window_size    = window_size

        self._history:       deque = deque(maxlen=window_size)
        self._last_sms_time: float = 0.0
        self._sms_cooldown:  float = 60.0   # seconds between SMS bursts

        # ── ENGINEER 3: load LSTM model + Twilio client here ────────────────
        # from twilio.rest import Client
        # self._twilio = Client(os.environ["TWILIO_SID"], os.environ["TWILIO_TOKEN"])
        # ────────────────────────────────────────────────────────────────────
        print("[ThreatPredictor] Running in MOCK mode — replace with LSTM + Twilio.")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _inflow_rate(self) -> float:
        """Linear regression slope over recent history (people / tick)."""
        if len(self._history) < 2:
            return 0.0
        ys = np.array(self._history, dtype=np.float32)
        xs = np.arange(len(ys), dtype=np.float32)
        if xs.std() == 0:
            return 0.0
        return float(np.polyfit(xs, ys, 1)[0])

    def _eta(self, count: int, rate: float) -> "float | None":
        if rate <= 0:
            return None
        remaining = self.threshold - count
        if remaining <= 0:
            return 0.0
        return remaining / rate

    def _try_sms(self, body: str) -> bool:
        now = time.time()
        if now - self._last_sms_time < self._sms_cooldown:
            return False
        # ── ENGINEER 3: replace print() with real Twilio call ───────────────
        # self._twilio.messages.create(
        #     body=body,
        #     from_=os.environ["TWILIO_FROM"],
        #     to=os.environ["TWILIO_TO"],
        # )
        print(f"[ThreatPredictor] [MOCK SMS] {body}")
        # ────────────────────────────────────────────────────────────────────
        self._last_sms_time = now
        return True

    # ── Public API ───────────────────────────────────────────────────────────

    def update_and_predict(self, current_count: int) -> dict:
        self._history.append(current_count)

        rate       = self._inflow_rate()
        eta        = self._eta(current_count, rate)
        risk_score = min(1.0, current_count / max(self.threshold, 1))
        sms_sent   = False

        if current_count >= self.threshold:
            status   = "CRITICAL_CAPACITY"
            sms_sent = self._try_sms(
                f"STAMPEDEZERO CRITICAL: count={current_count} exceeded "
                f"threshold={self.threshold}. Immediate action required."
            )
        elif current_count >= self.warn_threshold and rate > 0:
            status   = "PREDICTIVE_WARNING"
            eta_str  = f"{int(eta)}s" if eta is not None else "unknown"
            sms_sent = self._try_sms(
                f"StampedeZero WARNING: count={current_count}, "
                f"rate={rate:.1f} ppl/tick, ETA to critical={eta_str}."
            )
        else:
            status = "SAFE"
            eta    = None   # No ETA when crowd is safe

        return {
            "status":      status,
            "eta_seconds": eta,
            "inflow_rate": rate,
            "risk_score":  risk_score,
            "sms_sent":    sms_sent,
        }
