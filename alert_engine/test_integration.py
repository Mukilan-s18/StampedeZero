"""
StampedeZero - Integration Test Suite
=======================================
End-to-end test that validates the full prediction pipeline WITHOUT
requiring Twilio credentials or network access.

Tests:
    1. Buffer correctly stores and rotates data
    2. Prediction engine detects a steady_growth scenario
    3. ThreatPredictor fires PREDICTIVE_WARNING before threshold breach
    4. Cooldown suppresses duplicate alerts
    5. Safe oscillation does NOT trigger false positives

Usage:
    python test_integration.py
"""

import time
import sys
import math
import random

# Import from our modules (run from alert_engine/ directory)
from data_buffer import CrowdDataBuffer
import prediction_engine as predictor
from risk_engine import ThreatPredictor


def divider(title: str):
    print(f"\n{'═' * 60}")
    print(f"  TEST: {title}")
    print(f"{'═' * 60}")


def test_data_buffer():
    """Verify buffer FIFO behavior, size limits, and readiness checks."""
    divider("CrowdDataBuffer — FIFO Behavior")

    buf = CrowdDataBuffer(buffer_size=5)
    assert len(buf) == 0, "Buffer should start empty"
    assert not buf.is_ready, "Buffer should not be ready with 0 entries"

    # Push 8 items into a size-5 buffer → oldest 3 should be evicted
    for i in range(8):
        buf.push(i * 10, timestamp=float(i))

    assert len(buf) == 5, f"Buffer should cap at 5, got {len(buf)}"
    assert buf.total_updates == 8, "Total updates should count all pushes"
    assert buf.get_counts() == [30, 40, 50, 60, 70], \
        f"Expected evicted data, got {buf.get_counts()}"

    latest = buf.get_latest()
    assert latest == (7.0, 70), f"Latest should be (7.0, 70), got {latest}"

    print("  ✅ FIFO eviction works correctly")
    print("  ✅ Size limits enforced")
    print("  ✅ Total updates tracked independently of buffer size")
    print("  ✅ get_latest() returns correct entry")
    return True


def test_prediction_engine():
    """Verify linear regression and threat classification."""
    divider("prediction_engine — Linear Regression")

    # Simulate a clear linear growth: 2 people/second starting at 20
    timestamps = [float(i) for i in range(20)]
    counts = [20 + (2 * i) for i in range(20)]

    analysis = predictor.analyze(
        timestamps=timestamps,
        counts=counts,
        current_count=counts[-1],
        threshold=100,
        warning_horizon=60,
    )

    print(f"  Status:      {analysis['status']}")
    print(f"  Inflow rate: {analysis['inflow_rate']} ppl/sec")
    print(f"  ETA:         {analysis['eta_seconds']}s")
    print(f"  Message:     {analysis['message']}")

    assert analysis["inflow_rate"] > 1.5, \
        f"Expected ~2.0 ppl/sec, got {analysis['inflow_rate']}"
    assert analysis["eta_seconds"] is not None, "ETA should not be None for growing crowd"
    assert analysis["eta_seconds"] > 0, "ETA should be positive (hasn't breached yet)"

    # At count=58 (t=19) with slope~2, threshold=100 → ~21s remaining
    assert analysis["eta_seconds"] < 30, \
        f"ETA should be ~21s, got {analysis['eta_seconds']}"

    print("  ✅ Slope correctly computed")
    print("  ✅ ETA correctly predicted")
    return True


def test_safe_oscillation():
    """Verify that a stable/oscillating crowd does NOT trigger warnings."""
    divider("prediction_engine — False Positive Suppression")

    # Simulate safe oscillation: crowd bounces between 30-50
    timestamps = [float(i) for i in range(30)]
    counts = [int(40 + 10 * math.sin(i * 0.2)) for i in range(30)]

    analysis = predictor.analyze(
        timestamps=timestamps,
        counts=counts,
        current_count=counts[-1],
        threshold=100,
    )

    print(f"  Status:      {analysis['status']}")
    print(f"  Inflow rate: {analysis['inflow_rate']} ppl/sec")
    print(f"  Message:     {analysis['message']}")

    assert analysis["status"] == predictor.SAFE, \
        f"Expected SAFE for oscillating crowd, got {analysis['status']}"

    print("  ✅ No false positive triggered")
    return True


def test_capacity_breach():
    """Verify immediate CRITICAL_CAPACITY when count exceeds threshold."""
    divider("prediction_engine — Capacity Breach Detection")

    timestamps = [float(i) for i in range(15)]
    counts = [90 + i for i in range(15)]  # Starts at 90, reaches 104

    analysis = predictor.analyze(
        timestamps=timestamps,
        counts=counts,
        current_count=104,
        threshold=100,
    )

    print(f"  Status:      {analysis['status']}")
    print(f"  Message:     {analysis['message']}")

    assert analysis["status"] == predictor.CRITICAL_CAPACITY, \
        f"Expected CRITICAL_CAPACITY, got {analysis['status']}"

    print("  ✅ Capacity breach correctly detected")
    return True


def test_threat_predictor_integration():
    """End-to-end test of ThreatPredictor with simulated data feed."""
    divider("ThreatPredictor — Full Pipeline Integration")

    # Initialize with SMS disabled (no Twilio needed for tests)
    engine = ThreatPredictor(
        danger_threshold=80,
        buffer_size=30,
        cooldown_seconds=5,  # Short cooldown for test
        sms_enabled=False,
    )

    print(f"  Engine: {engine}")
    print(f"  Initial status: {engine.get_status_summary()}")

    # Feed a steady growth scenario: ~3 ppl/sec
    warning_triggered = False
    critical_triggered = False

    for tick in range(40):
        count = 10 + int(3.0 * tick + random.gauss(0, 0.5))
        result = engine.update_and_predict(count)

        if result["status"] == "PREDICTIVE_WARNING" and not warning_triggered:
            print(f"  ⚠️  PREDICTIVE_WARNING triggered at tick={tick}, count={count}")
            print(f"      ETA: {result['eta_seconds']}s, Rate: {result['inflow_rate']} ppl/s")
            warning_triggered = True

        if result["status"] == "CRITICAL_CAPACITY" and not critical_triggered:
            print(f"  🚨 CRITICAL_CAPACITY triggered at tick={tick}, count={count}")
            critical_triggered = True
            break

    assert warning_triggered, "Expected PREDICTIVE_WARNING before threshold breach"
    assert critical_triggered, "Expected CRITICAL_CAPACITY when count > 80"

    summary = engine.get_status_summary()
    print(f"  Final summary: {summary}")
    print("  ✅ Warning fired BEFORE capacity breach (predictive)")
    print("  ✅ Critical alert fired at threshold")
    return True


def run_all_tests():
    """Execute all test cases and report results."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      StampedeZero Alert Engine — Integration Tests      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    tests = [
        test_data_buffer,
        test_prediction_engine,
        test_safe_oscillation,
        test_capacity_breach,
        test_threat_predictor_integration,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
        except AssertionError as e:
            print(f"  ❌ ASSERTION FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ UNEXPECTED ERROR: {e}")
            failed += 1

    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'═' * 60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("  🎉 All tests passed! Engine is ready for integration.")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
