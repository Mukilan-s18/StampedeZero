"""
StampedeZero - Mock Data Simulator
====================================
Simulates realistic crowd density data streams for testing the prediction
engine WITHOUT requiring a live camera feed from Engineer 1.

Provides three simulation profiles:
    1. steady_growth   — Linear crowd build-up (tests velocity prediction)
    2. sudden_surge    — Stable crowd followed by sharp spike (tests reaction time)
    3. safe_oscillation — Normal fluctuation that should NOT trigger alerts

Usage:
    python mock_data_feed.py [profile]
    python mock_data_feed.py steady_growth

This script feeds data into CrowdDataBuffer at 1Hz (one reading per second)
and prints the buffer state to console for visual verification.
"""

import time
import sys
import random
import math
from data_buffer import CrowdDataBuffer


def steady_growth(elapsed: float) -> int:
    """
    Simulates a gate that is letting in ~2 people per second with some noise.
    Starts at 20 people, will cross 100 in about 40 seconds.
    """
    base = 20 + (2.0 * elapsed)
    noise = random.gauss(0, 1.5)
    return max(0, int(base + noise))


def sudden_surge(elapsed: float) -> int:
    """
    Stable crowd of ~40 for 30 seconds, then a sudden rush (e.g., gate opens).
    The surge adds 5 people/sec, simulating a dangerous inrush.
    """
    if elapsed < 30:
        return max(0, int(40 + random.gauss(0, 2)))
    else:
        surge_time = elapsed - 30
        return max(0, int(40 + (5.0 * surge_time) + random.gauss(0, 2)))


def safe_oscillation(elapsed: float) -> int:
    """
    Crowd oscillates between 30-50 using a sine wave. This should NEVER
    trigger a stampede alert — useful for testing false-positive suppression.
    """
    base = 40 + 10 * math.sin(elapsed * 0.2)
    noise = random.gauss(0, 1.5)
    return max(0, int(base + noise))


PROFILES = {
    "steady_growth": steady_growth,
    "sudden_surge": sudden_surge,
    "safe_oscillation": safe_oscillation,
}


def run_simulation(profile_name: str = "steady_growth", duration: int = 90):
    """
    Run a mock data feed for the specified duration (seconds).

    Args:
        profile_name: One of 'steady_growth', 'sudden_surge', 'safe_oscillation'.
        duration:     How many seconds to run the simulation.
    """
    if profile_name not in PROFILES:
        print(f"❌ Unknown profile: '{profile_name}'")
        print(f"   Available: {', '.join(PROFILES.keys())}")
        sys.exit(1)

    generator = PROFILES[profile_name]
    buffer = CrowdDataBuffer(buffer_size=60)

    print(f"🔬 Starting mock simulation: '{profile_name}' for {duration}s")
    print(f"{'─' * 55}")
    print(f"{'Elapsed':>8}  {'Count':>6}  {'Buffer':>8}  {'Status'}")
    print(f"{'─' * 55}")

    start = time.time()
    for tick in range(duration):
        elapsed = float(tick)
        count = generator(elapsed)
        buffer.push(count)

        status = "🟢 OK"
        if count >= 80:
            status = "🟡 ELEVATED"
        if count >= 100:
            status = "🔴 CRITICAL"

        print(f"{tick:>7}s  {count:>6}  {buffer.size:>6}/{buffer.capacity}  {status}")
        time.sleep(0.1)  # 0.1s for fast demo; change to 1.0 for real-time

    print(f"{'─' * 55}")
    print(f"✅ Simulation complete. Total updates: {buffer.total_updates}")
    latest = buffer.get_latest()
    if latest:
        print(f"   Last reading: count={latest[1]} at t={latest[0]:.2f}")


if __name__ == "__main__":
    profile = sys.argv[1] if len(sys.argv) > 1 else "steady_growth"
    run_simulation(profile)
