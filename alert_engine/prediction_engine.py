"""
StampedeZero - Predictive Math Engine
=======================================
Core mathematical module that performs linear regression on time-series
crowd density data to calculate:

    1. Inflow Rate (Velocity) — people entering per second
    2. ETA to Critical — seconds until crowd count hits the danger threshold
    3. Trend Classification — rising, stable, or decreasing

The engine operates on two threat detection models:
    - VELOCITY ALERT:  Crowd is below threshold but growing fast enough
                       to breach it within a configurable warning horizon.
    - CAPACITY ALERT:  Current crowd count already exceeds the threshold
                       (immediate danger — regardless of trend direction).

Mathematical Basis:
    Linear regression via numpy.polyfit(degree=1) computes the line of
    best fit y = mx + c over the rolling buffer window, where:
        m = slope = inflow rate (people/second)
        c = y-intercept
    ETA is solved algebraically: x_critical = (threshold - c) / m
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List


# ─── Threat Severity Levels ──────────────────────────────────────────────────

SAFE = "SAFE"
ELEVATED = "ELEVATED"
PREDICTIVE_WARNING = "PREDICTIVE_WARNING"
CRITICAL_CAPACITY = "CRITICAL_CAPACITY"


# ─── Core Prediction Functions ───────────────────────────────────────────────

def compute_trend(
    timestamps: List[float],
    counts: List[int],
) -> Tuple[float, float]:
    """
    Fit a linear regression to the time-series data and return the slope
    (inflow rate) and intercept.

    Args:
        timestamps: List of Unix timestamps (x-axis).
        counts:     Corresponding crowd counts (y-axis).

    Returns:
        (slope, intercept) — slope in people/second.
    """
    # Normalize timestamps to seconds elapsed from the first reading
    # to avoid floating-point precision issues with large Unix timestamps
    t0 = timestamps[0]
    x = np.array([t - t0 for t in timestamps], dtype=np.float64)
    y = np.array(counts, dtype=np.float64)

    # Degree 1 = linear fit → y = mx + c
    m, c = np.polyfit(x, y, 1)
    return float(m), float(c)


def predict_eta(
    slope: float,
    intercept: float,
    current_elapsed: float,
    threshold: int,
) -> Optional[float]:
    """
    Calculate the estimated time (in seconds from now) until the crowd
    count reaches the danger threshold.

    Args:
        slope:           Inflow rate (people/second) from linear regression.
        intercept:       Y-intercept of the regression line.
        current_elapsed: Current time offset (seconds since buffer start).
        threshold:       The critical crowd count limit.

    Returns:
        Seconds remaining until threshold breach, or None if the crowd is
        not trending toward the threshold (slope <= 0 or already passed).
    """
    if slope <= 0:
        return None  # Crowd is stable or decreasing — no convergence

    # Solve for x when y = threshold:  x = (threshold - c) / m
    x_critical = (threshold - intercept) / slope
    time_remaining = x_critical - current_elapsed

    if time_remaining <= 0:
        return 0.0  # Threshold already breached per regression estimate

    return float(time_remaining)


def classify_threat(
    current_count: int,
    threshold: int,
    slope: float,
    eta: Optional[float],
    warning_horizon: int = 60,
    velocity_floor: float = 0.5,
) -> str:
    """
    Classify the current threat level based on capacity and velocity.

    Priority order:
        1. CRITICAL_CAPACITY — current count >= threshold (immediate danger)
        2. PREDICTIVE_WARNING — count is below threshold but ETA < warning_horizon
                                and inflow rate exceeds velocity_floor
        3. ELEVATED — crowd is growing but ETA is beyond warning_horizon
        4. SAFE — crowd is stable, decreasing, or well below threshold

    Args:
        current_count:    Latest crowd count reading.
        threshold:        Danger threshold for crowd density.
        slope:            Current inflow rate (people/second).
        eta:              Seconds until threshold breach (None if not converging).
        warning_horizon:  Seconds of advance warning to trigger PREDICTIVE_WARNING.
        velocity_floor:   Minimum slope (ppl/sec) to consider "rising" vs noise.

    Returns:
        One of: SAFE, ELEVATED, PREDICTIVE_WARNING, CRITICAL_CAPACITY.
    """
    # Priority 1: Already over capacity
    if current_count >= threshold:
        return CRITICAL_CAPACITY

    # Priority 2: Fast convergence toward threshold
    if slope > velocity_floor and eta is not None and eta < warning_horizon:
        return PREDICTIVE_WARNING

    # Priority 3: Growing but not immediately dangerous
    if slope > velocity_floor and eta is not None:
        return ELEVATED

    # Default: Safe
    return SAFE


def analyze(
    timestamps: List[float],
    counts: List[int],
    current_count: int,
    threshold: int = 100,
    warning_horizon: int = 60,
    velocity_floor: float = 0.5,
) -> Dict[str, Any]:
    """
    Full prediction pipeline: trend → ETA → classification.

    This is the primary entry point for the prediction engine. Feed it
    the contents of a CrowdDataBuffer and get back a complete threat
    assessment.

    Args:
        timestamps:       List of Unix timestamps from the data buffer.
        counts:           Corresponding crowd count values.
        current_count:    The most recent crowd count (may differ from counts[-1]
                          if a new reading just arrived).
        threshold:        Critical crowd count limit.
        warning_horizon:  Seconds of advance warning for PREDICTIVE_WARNING.
        velocity_floor:   Minimum slope to consider as "rising."

    Returns:
        Dictionary with keys:
            - status:       Threat level string (SAFE / ELEVATED / etc.)
            - inflow_rate:  Slope in people/second
            - eta_seconds:  Seconds until threshold (None if not converging)
            - current_count: Echo of the latest reading
            - threshold:    Echo of the configured threshold
            - message:      Human-readable summary for UI/SMS
    """
    # Guard: need at least 5 data points for a meaningful regression
    if len(timestamps) < 5:
        return {
            "status": SAFE,
            "inflow_rate": 0.0,
            "eta_seconds": None,
            "current_count": current_count,
            "threshold": threshold,
            "message": "Collecting data — need at least 10 readings for prediction.",
        }

    # Step 1: Compute trend
    slope, intercept = compute_trend(timestamps, counts)

    # Step 2: Compute ETA
    t0 = timestamps[0]
    current_elapsed = timestamps[-1] - t0
    eta = predict_eta(slope, intercept, current_elapsed, threshold)

    # Step 3: Classify
    status = classify_threat(
        current_count=current_count,
        threshold=threshold,
        slope=slope,
        eta=eta,
        warning_horizon=warning_horizon,
        velocity_floor=velocity_floor,
    )

    # Step 4: Build human-readable message
    message = _build_message(status, current_count, threshold, slope, eta)

    return {
        "status": status,
        "inflow_rate": round(slope, 3),
        "eta_seconds": round(eta, 1) if eta is not None else None,
        "current_count": current_count,
        "threshold": threshold,
        "message": message,
    }


def _build_message(
    status: str,
    current_count: int,
    threshold: int,
    slope: float,
    eta: Optional[float],
) -> str:
    """Generate a concise, human-readable summary for dashboard display or SMS."""
    if status == CRITICAL_CAPACITY:
        return (
            f"🚨 IMMEDIATE DANGER: Crowd count ({current_count}) has exceeded "
            f"safe capacity ({threshold}). Deploy crowd control immediately."
        )
    elif status == PREDICTIVE_WARNING:
        return (
            f"⚠️ PREDICTIVE WARNING: At current inflow rate ({slope:.1f} ppl/sec), "
            f"crowd will reach critical density in ~{int(eta)}s. "
            f"Current: {current_count}/{threshold}."
        )
    elif status == ELEVATED:
        eta_min = int(eta / 60) if eta else "?"
        return (
            f"📈 ELEVATED: Crowd is growing at {slope:.1f} ppl/sec. "
            f"ETA to threshold: ~{eta_min} min. Current: {current_count}/{threshold}."
        )
    else:
        direction = "rising" if slope > 0 else "stable/falling"
        return (
            f"✅ SAFE: Crowd density is {direction}. "
            f"Current: {current_count}/{threshold}."
        )
