# 🛡️ StampedeZero — Alert Engine

> **Predictive Crowd Stampede Prevention System with Automated SMS Alerts**
>
> Engineer 3: The Predictor & Alerter Module

---

## 🧠 What This Module Does

The Alert Engine is the **brain** of StampedeZero. It takes raw crowd count data from the detection pipeline (Engineers 1 & 2) and:

1. **Remembers** — Maintains a rolling time-series buffer of the last 60 seconds of crowd density readings.
2. **Predicts** — Uses linear regression to calculate the crowd inflow rate and estimate when the crowd will breach the danger threshold.
3. **Alerts** — Automatically sends SMS alerts to authorities via Twilio when a stampede is predicted, with spam-prevention cooldowns.

### Threat Detection Models

| Alert Type | Trigger Condition | Meaning |
|---|---|---|
| `SAFE` | Count < threshold, stable/falling trend | No danger |
| `ELEVATED` | Count < threshold, rising trend, ETA > 60s | Growing crowd — monitor |
| `PREDICTIVE_WARNING` | Count < threshold, rising trend, **ETA < 60s** | ⚠️ Stampede predicted — act NOW |
| `CRITICAL_CAPACITY` | **Count ≥ threshold** | 🚨 Immediate danger — deploy crowd control |

---

## 📁 File Structure

```
alert_engine/
├── __init__.py            # Package exports
├── .env.example           # Twilio credential template
├── .gitignore             # Excludes .env, venv, __pycache__
├── requirements.txt       # Python dependencies
├── data_buffer.py         # CrowdDataBuffer — rolling window (deque)
├── prediction_engine.py   # Linear regression + threat classification
├── risk_engine.py         # ThreatPredictor — main orchestration class
├── sms_tester.py          # Standalone SMS gateway verification
├── mock_data_feed.py      # Simulated crowd data for testing
├── test_integration.py    # End-to-end test suite
└── README.md              # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd alert_engine
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure SMS (Optional)

```bash
cp .env.example .env
# Edit .env with your Twilio credentials
```

> **Note:** The engine works in **dry-run mode** without Twilio credentials. Predictions still work — only SMS is disabled.

### 3. Run Tests

```bash
python test_integration.py
```

### 4. Test SMS Gateway

```bash
python sms_tester.py
```

---

## 🔌 Integration Guide for Engineer 4 (UI / Streamlit)

### Basic Usage

```python
from risk_engine import ThreatPredictor

# Initialize once at app startup
predictor = ThreatPredictor(
    danger_threshold=50,    # Max safe crowd count
    buffer_size=60,         # Remember last 60 readings
    cooldown_seconds=120,   # Min 2 minutes between SMS alerts
)

# Inside your Streamlit loop — call once per frame/second:
analytics = predictor.update_and_predict(latest_crowd_count)

# Use the returned dict to render your dashboard:
# analytics["status"]        → "SAFE" / "ELEVATED" / "PREDICTIVE_WARNING" / "CRITICAL_CAPACITY"
# analytics["inflow_rate"]   → float, people per second
# analytics["eta_seconds"]   → float or None, seconds until threshold breach
# analytics["message"]       → Human-readable status string
# analytics["sms_sent"]      → bool, whether an SMS was just sent
# analytics["buffer_fill"]   → "34/60" string showing buffer utilization
```

### Streamlit Countdown Timer Example

```python
import streamlit as st

if analytics["eta_seconds"] is not None and analytics["eta_seconds"] > 0:
    st.metric(
        label="⏱️ Time to Critical",
        value=f"{int(analytics['eta_seconds'])}s",
        delta=f"{analytics['inflow_rate']:.1f} ppl/sec"
    )
```

---

## 🧪 Mock Data Profiles

Test without a live camera feed:

```bash
python mock_data_feed.py steady_growth     # Linear ramp → triggers WARNING then CRITICAL
python mock_data_feed.py sudden_surge      # Stable then spike → tests reaction time
python mock_data_feed.py safe_oscillation  # Should NOT trigger any alerts
```

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Camera Feed │────▶│  CrowdDataBuffer │────▶│  Prediction   │
│  (Eng 1 & 2) │     │  (Rolling deque) │     │  Engine       │
└──────────────┘     └──────────────────┘     │  (numpy.      │
                                               │   polyfit)    │
                                               └───────┬───────┘
                                                       │
                                                       ▼
                                               ┌───────────────┐
                                               │  Threat       │
                                               │  Classifier   │
                                               └───────┬───────┘
                                                       │
                                           ┌───────────┴───────────┐
                                           ▼                       ▼
                                   ┌──────────────┐       ┌──────────────┐
                                   │  Dashboard   │       │  Twilio SMS  │
                                   │  (Eng 4 UI)  │       │  (Cooldown)  │
                                   └──────────────┘       └──────────────┘
```

---

## 👥 Team

| Role | Engineer | Branch |
|---|---|---|
| Vision Pipeline | Mukilan | `mukil` |
| Object Detection | Muthesh | `muthesh` |
| **Predictor & Alerter** | **Mukesh Kumar** | **`mukeshkumar`** |
| Dashboard UI | Fadil | `fadil` |

---

## 📄 License

Part of the StampedeZero project. See root [LICENSE](../LICENSE).
