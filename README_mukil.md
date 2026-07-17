# StampedeZero — Vision Tracker Module (`mukil` branch)

**Engineer 1 · Vision & Tracking Lead**

---

## What This Module Does

Accepts a raw webcam frame (NumPy array) and returns a clean dictionary of crowd-counting statistics — with all AI inference, person tracking, and line-crossing mathematics handled internally.

---

## Files

| File | Purpose |
|---|---|
| `crowd_tracker.py` | **Primary deliverable** — import `VisionTracker` from here |
| `config.py` | All tunable constants (line position, model, resolution, etc.) |
| `demo.py` | Standalone webcam test — run to validate before Streamlit integration |
| `requirements.txt` | Pinned deps (install these, **not** Streamlit) |

---

## Installation

```bash
# Clone and switch to this branch
git checkout mukil

# Create a virtual environment (keep deps isolated)
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
# yolov8n.pt will auto-download on first run (~6 MB)
```

---

## Quick Test (Engineer 1)

```bash
python demo.py            # Default webcam
python demo.py --source 1 # Second webcam
```

**Demo keyboard controls:**

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `R` | Reset in/out counters |
| `S` | Save screenshot (PNG) |
| `+` / `-` | Adjust counting line up/down |

---

## Engineer 4 Integration (Streamlit)

```python
# In your Streamlit app
from crowd_tracker import VisionTracker
import config as cfg
import cv2

tracker = VisionTracker(line_y_fraction=cfg.LINE_Y_FRACTION)

cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

while True:
    ret, raw_frame = cap.read()
    if not ret:
        break

    payload = tracker.process_frame(raw_frame)

    # Display annotated video
    frame_placeholder.image(
        payload["annotated_frame"],
        channels="BGR",
        use_container_width=True,
    )

    # Map counts to your UI widgets
    col1, col2, col3 = st.columns(3)
    col1.metric("Total In",  payload["total_in"])
    col2.metric("Total Out", payload["total_out"])
    col3.metric("Net Flow",  payload["net_flow"])

    # Reset button
    if st.button("Reset Counts"):
        tracker.reset()
```

---

## Payload Reference

`tracker.process_frame(frame)` returns:

```python
{
    "annotated_frame":     np.ndarray,  # BGR image — pass directly to st.image()
    "current_on_screen":  int,          # People visible RIGHT NOW
    "total_in":           int,          # Cumulative inflow count
    "total_out":          int,          # Cumulative outflow count
    "net_flow":           int,          # total_in − total_out
    "track_ids_on_screen": list[int],   # Active track IDs (for Engineer 3 alert thresholds)
}
```

---

## Engineer 3 Integration (Twilio Alerts)

```python
# Trigger an SMS when net_flow exceeds threshold
payload = tracker.process_frame(frame)
if payload["net_flow"] > CROWD_THRESHOLD:
    send_twilio_alert(count=payload["net_flow"])
```

---

## Configuration

Edit `config.py` to tune the tracker without touching business logic:

| Constant | Default | Description |
|---|---|---|
| `LINE_Y_FRACTION` | `0.55` | Counting line height (fraction of frame, 0–1) |
| `SKIP_FRAMES` | `2` | Run YOLO every Nth frame (2 = ~50% CPU saving) |
| `MAX_STALE_AGE` | `30` | Frames before old track IDs are deleted |
| `CONFIDENCE_THRESHOLD` | `0.4` | Min detection confidence |
| `FRAME_W / FRAME_H` | `640 / 480` | Resolution cap before YOLO inference |

---

## Line Direction Convention

```
Y=0 ──────────────────── TOP of frame
         (people walk in ↓)
Y=line_y ──── VIRTUAL COUNTING LINE ◄── default: 55% from top
         (people walk out ↑)
Y=frame_h ──────────────── BOTTOM of frame

INFLOW  = person moves DOWN  past line  →  total_in  += 1
OUTFLOW = person moves UP    past line  →  total_out += 1
```

> **Camera placement tip:** Mount the webcam on a bookshelf or table edge angled **downward**. Eye-level cameras cause occlusion (people block each other), making the tracker unreliable. A 30–45° downward angle gives a near-top-down view that ByteTrack handles very well.

---

## Performance Notes

- **Frame skip**: YOLO inference runs every 2nd frame. On skipped frames, the last detection result is reused — bounding boxes still appear, just no new inference. Halves CPU usage.
- **Resolution cap**: Input is resized to 640×480 before inference. YOLOv8 internally uses 640px anyway; pre-shrinking saves memory bandwidth.
- **Model**: `yolov8n.pt` (nano) — fastest CPU model. Upgrade to `yolov8s.pt` if GPU is available.
- **Tracker**: ByteTrack (`bytetrack.yaml`) — lower latency than BoT-SORT.

---

## Git Push (after testing)

```bash
git add crowd_tracker.py config.py demo.py requirements.txt README_mukil.md
git commit -m "feat(mukil): complete VisionTracker with line-crossing counter"
git push origin mukil
```
