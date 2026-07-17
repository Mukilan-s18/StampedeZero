# 🚨 StampedeZero — Proactive Crowd Flow Engine

> **Real-time AI crowd-stampede prevention · SIMATS Hackathon 2026**

---

## Architecture

```
                 ┌──────────────────────────┐
                 │  app.py  (Engineer 4)    │  ← Streamlit UI + Integrator
                 └────────────┬─────────────┘
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
  │crowd_tracker │  │ dense_crowd_ai   │  │ risk_engine  │
  │  (Eng 1)     │  │  (Eng 2)         │  │  (Eng 3)     │
  │  YOLOv8      │  │  CSRNet / CNN    │  │  LSTM+Twilio │
  └──────────────┘  └──────────────────┘  └──────────────┘
```

## Modes

| Mode | Engine | Input | Output |
|------|--------|-------|--------|
| **1 · Live Footfall** | YOLO (Eng 1) | Webcam | Bounding boxes, in/out count |
| **2 · Extreme Crowd Simulator** | CSRNet (Eng 2) | Recorded MP4 | Density heatmap, estimated count |

Both modes feed **ThreatPredictor (Eng 3)** → risk score + Twilio SMS.

---

## Quick Start

```bash
git clone https://github.com/Mukilan-s18/StampedeZero.git
cd StampedeZero && git checkout fadil

python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

streamlit run app.py
```

---

## Repo Layout

```
StampedeZero/
├── app.py                ← Eng 4 · main dashboard
├── crowd_tracker.py      ← Eng 1 · YOLO tracker   (replace stub)
├── dense_crowd_ai.py     ← Eng 2 · CSRNet counter  (replace stub)
├── risk_engine.py        ← Eng 3 · LSTM predictor  (replace stub)
├── requirements.txt
├── logo.png
├── .streamlit/
│   └── config.toml       ← dark theme
├── data/
│   └── README.md         ← put crowd_concert.mp4 here (git-ignored)
└── weights/
    └── README.md         ← put .pth weights here   (git-ignored)
```

---

## Environment Variables (Engineer 3 · Twilio)

Create `.env` in project root — **never commit this file**:

```env
TWILIO_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_TOKEN=your_auth_token
TWILIO_FROM=+1XXXXXXXXXX
TWILIO_TO=+91XXXXXXXXXX
```

---

## ⚠️ GitHub Rules

- **Never push `.pt` / `.pth`** weight files — git-ignored, share via Google Drive
- **Never push `.mp4`** video files — share via Drive link
- **Never push `.env`** — keep secrets local only
- **Always `git pull origin fadil` before pushing**

---

## Team

| Role | Owner | File |
|------|-------|------|
| YOLO Footfall Tracker | Engineer 1 | `crowd_tracker.py` |
| CSRNet Density Estimator | Engineer 2 | `dense_crowd_ai.py` |
| Risk Engine + Twilio SMS | Engineer 3 | `risk_engine.py` |
| UI / Integration Lead | **Engineer 4 (Fadil)** | `app.py` |
