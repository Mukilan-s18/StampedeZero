"""
app.py  —  StampedeZero: Proactive Crowd Flow Engine
=====================================================
OWNER: Engineer 4 (Full-Stack / UI Lead)
RUN:   streamlit run app.py

Wiring (FIXED — all real engines):
  Mode 1  →  crowd_tracker.py             (Eng 1 · YOLOv8 + ByteTrack)
  Mode 2  →  heatmap_engine/dense_crowd_ai (Eng 2 · CSRNet)
  Both    →  alert_engine/risk_engine.py  (Eng 3 · Linear Regression + Twilio)

Fixes applied in this version:
  BUG-01/02: wired to real alert_engine, correct arg names
  BUG-03:    variable renamed predictor (not risk_engine)
  BUG-04:    wired to real heatmap_engine/dense_crowd_ai
  BUG-06:    replaced blocking while loop with st.rerun() pattern
  GAP-03:    tracker uses config.LINE_Y_FRACTION
  GAP-04:    sidebar frame_skip passed to VisionTracker
  GAP-05:    total_in / total_out shown in dashboard
  QC-05:     history buffers use deque (O(1) not O(n))
  PERF-01:   removed double resize (tracker does its own)
  PERF-02:   Plotly charts throttled to CHART_UPDATE_INTERVAL
  TEST-01:   DEMO_MODE flag in config bypasses hardware
"""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Path setup — allow importing sub-packages from project root ───────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "heatmap_engine"))   # BUG-04
sys.path.insert(0, _ROOT)

import config as cfg

# ── Must be FIRST Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="StampedeZero — AI Crowd Safety",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium dark-mode CSS ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .main { background-color: #0e1117; }

    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1d24 0%, #1f2330 100%);
        border: 1px solid #2e3244;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(255,75,75,0.07);
        transition: box-shadow .3s;
    }
    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 28px rgba(255,75,75,0.22);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #141820 100%);
        border-right: 1px solid #2e3244;
    }
    h3 { color: #e0e0e0 !important; letter-spacing: .02em; }
    hr { border-color: #2e3244 !important; }
    .stAlert { border-radius: 10px !important; }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff8c42, #ff4b4b);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #6b7a99;
        font-size: .9rem;
        margin-top: 2px;
        letter-spacing: .07em;
        text-transform: uppercase;
    }
    @keyframes shine { to { background-position: 200% center; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Engine loader — cached once, never reloads on widget interaction ──────────
@st.cache_resource(show_spinner="🔄 Loading AI engines…")
def load_engines():
    """
    Load all three AI engines.
    DEMO_MODE=True (config.py) → engines start with mocks; no hardware required.
    DEMO_MODE=False             → full real inference.
    """
    # Engineer 1 — VisionTracker (YOLO)
    from crowd_tracker import VisionTracker
    tracker = VisionTracker(
        line_y_fraction=cfg.LINE_Y_FRACTION,
        skip_frames=cfg.SKIP_FRAMES,
    )

    # Engineer 2 — DensityEstimator (CSRNet)  BUG-04: from heatmap_engine
    from dense_crowd_ai import DensityEstimator
    estimator = DensityEstimator(
        weight_path=cfg.CSRNET_WEIGHTS,           # QC-02: canonical kwarg
        infer_size=cfg.HEATMAP_INFER_SIZE,
    )

    # Engineer 3 — ThreatPredictor (Linear Regression + Twilio)  BUG-01/02
    from alert_engine import ThreatPredictor      # real implementation
    predictor_engine = ThreatPredictor(
        danger_threshold=cfg.VENUE_CAPACITY,      # BUG-02: correct arg name
        buffer_size=cfg.BUFFER_SIZE,
        cooldown_seconds=cfg.SMS_COOLDOWN_SECONDS,
        warning_horizon=cfg.WARNING_HORIZON_SECONDS,
        velocity_floor=cfg.VELOCITY_FLOOR,
    )

    return tracker, estimator, predictor_engine


yolo_engine, cnn_engine, predictor = load_engines()   # BUG-03: renamed to predictor


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.markdown("## 🚨 StampedeZero")

    st.markdown("---")
    st.markdown("### 🎛️ Mode Selection")
    app_mode = st.radio(
        "Select Input Flow",
        ["1. Live Footfall (YOLO Edge)", "2. Extreme Crowd Simulator (CNN)"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    capacity_limit = st.slider(
        "Venue Capacity Limit", 10, 5000, cfg.VENUE_CAPACITY, 10,
        help="Max safe crowd count before alerts fire.",
    )
    warn_pct_val = st.slider(
        "Warning Threshold (%)", 50, 95, int(cfg.WARN_PCT * 100), 5,
        help="Fraction of capacity that triggers a warning.",
    )
    # GAP-04: sidebar frame_skip now wired to tracker
    frame_skip = st.slider(
        "Frame Skip (performance)", 1, 5, cfg.SKIP_FRAMES,
        help="Process every Nth frame. Raise this if UI lags.",
    )

    # GAP-03: line position exposed in UI
    line_pct = st.slider(
        "Counting Line Position (%)", 20, 80, int(cfg.LINE_Y_FRACTION * 100), 5,
        help="Vertical position of the virtual counting line.",
    )

    st.markdown("---")
    demo_badge = "🟡 DEMO MODE" if cfg.DEMO_MODE else "🟢 LIVE MODE"
    st.markdown(
        f"<div style='color:#6b7a99;font-size:.8rem'>{demo_badge}</div>",
        unsafe_allow_html=True,
    )
    stop_btn = st.button("⏹  Stop Feed", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(
        "<div style='color:#3a4060;font-size:.75rem;text-align:center'>"
        "StampedeZero v2.0 · SIMATS Hackathon 2026</div>",
        unsafe_allow_html=True,
    )

# BUG-03 (resolved): was `risk_engine.threshold = ...` — shadowed the module name
predictor.threshold = capacity_limit
predictor.warn_threshold = int(capacity_limit * warn_pct_val / 100)

# GAP-04: sync sidebar frame_skip into VisionTracker
yolo_engine._skip_frames = max(1, frame_skip)

# GAP-03: sync sidebar line position into VisionTracker
yolo_engine._line_y_fraction = line_pct / 100.0


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">🚨 StampedeZero</div>'
    '<div class="hero-sub">Proactive Crowd Flow Engine · Real-Time AI Safety System</div>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

status_banner = st.empty()
st.markdown("---")

# ── Layout: 65% video | 35% analytics ────────────────────────────────────────
col1, col2 = st.columns([0.65, 0.35])

with col1:
    st.markdown("### 🎥 Camera Feed")
    st.caption(
        "📷 Webcam — Live YOLO" if "Live" in app_mode
        else "📹 Recorded Video — CNN Heatmap"
    )
    video_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Live Analytics")
    metric_cols = st.columns(3)
    count_metric = metric_cols[0].empty()
    in_metric    = metric_cols[1].empty()   # GAP-05: new
    out_metric   = metric_cols[2].empty()   # GAP-05: new

    eta_metric = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Real-Time Pressure")
    graph_placeholder = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔴 Risk Score")
    gauge_placeholder = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Alert Log")
    log_placeholder = st.empty()


# ── Chart builders ────────────────────────────────────────────────────────────
def build_pressure_chart(times, counts, threshold):
    fig = go.Figure()
    if counts:
        fig.add_trace(go.Scatter(
            x=list(times), y=list(counts),
            mode="lines",
            name="Crowd Count",
            line=dict(color="#ff4b4b", width=2.5, shape="spline", smoothing=1.0),
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.08)",
        ))
    if times:
        fig.add_hline(
            y=threshold, line_dash="dot", line_color="#ff8c42",
            annotation_text="⚠️ Capacity", annotation_font_color="#ff8c42",
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,26,35,0.9)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, color="#3a4060"),
        yaxis=dict(gridcolor="#1e2236", color="#6b7a99", title="People"),
        legend=dict(orientation="h", y=1.1, font=dict(color="#6b7a99", size=11)),
        height=200,
        font=dict(family="Outfit, sans-serif"),
    )
    return fig


def build_gauge(risk_score: float):
    color = "#2ecc71" if risk_score < 0.5 else ("#ff8c42" if risk_score < 0.85 else "#ff4b4b")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        number=dict(suffix="%", font=dict(color=color, size=28, family="Outfit")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#3a4060",
                      tickfont=dict(color="#6b7a99")),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(22,26,35,0.9)",
            bordercolor="#2e3244",
            steps=[
                dict(range=[0,  50], color="rgba(46,204,113,0.07)"),
                dict(range=[50, 85], color="rgba(255,140,66,0.07)"),
                dict(range=[85,100], color="rgba(255,75,75,0.10)"),
            ],
            threshold=dict(line=dict(color="#ff4b4b", width=3),
                           thickness=0.8, value=85),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6b7a99", family="Outfit"),
        height=155,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def blank_frame(text="No source — connect webcam or add data/crowd_concert.mp4"):
    f = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(f, text, (20, 240), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (80, 80, 110), 1)
    return f


# ── Open video source ─────────────────────────────────────────────────────────
if stop_btn:
    st.info("⏸ Feed stopped. Refresh the page or press Run to restart.")
    st.stop()

if "Live" in app_mode:
    cap = cv2.VideoCapture(cfg.VIDEO_SOURCE)
    # TEST-04: fallback if webcam unavailable
    if not cap.isOpened():
        st.warning(
            "⚠️ Webcam not available — showing synthetic test frames. "
            "Connect a webcam for live tracking."
        )
        cap = None
else:
    vpath = cfg.CNN_VIDEO_PATH
    cap = cv2.VideoCapture(vpath) if os.path.exists(vpath) else None

# ── History buffers — QC-05: deque for O(1) append/evict ─────────────────────
hist_counts: deque = deque(maxlen=cfg.MAX_HIST)
hist_times:  deque = deque(maxlen=cfg.MAX_HIST)
alert_log:   list  = []

frame_idx  = 0
start_time = time.time()

# ── Master rendering loop — BUG-06: use st.empty + time.sleep + st.rerun ─────
# Streamlit re-runs the whole script on each interaction; we use a fixed-count
# loop per script run and rely on st.rerun() to keep it alive.
FRAMES_PER_RUN = 60  # process up to 60 frames per script execution

for _ in range(FRAMES_PER_RUN):
    frame_idx += 1
    elapsed = time.time() - start_time

    # 1. Read frame
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if "Simulator" in app_mode:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
    else:
        frame = blank_frame()
        time.sleep(0.04)

    # 2. Frame skip
    if frame_idx % frame_skip != 0:
        continue

    # PERF-01: removed duplicate resize — tracker & estimator handle their own
    # (was: frame = cv2.resize(frame, (640, 480)))

    # 3. AI routing
    try:
        if "Live" in app_mode:
            ai_data       = yolo_engine.process_frame(frame)
            render_img    = ai_data["annotated_frame"]
            current_count = ai_data["current_on_screen"]
            total_in      = ai_data.get("total_in", 0)    # GAP-05
            total_out     = ai_data.get("total_out", 0)   # GAP-05
        else:
            ai_data       = cnn_engine.process_frame(frame)
            render_img    = ai_data["heatmap_frame"]
            current_count = ai_data["estimated_count"]
            total_in      = 0
            total_out     = 0
    except Exception as exc:
        st.error(f"AI engine error: {exc}")
        break

    # 4. Risk prediction
    try:
        threat = predictor.update_and_predict(current_count)
    except Exception:
        threat = {
            "status": "SAFE", "eta_seconds": None,
            "inflow_rate": 0.0, "risk_score": 0.0, "sms_sent": False,
        }

    status   = threat.get("status",      "SAFE")
    eta      = threat.get("eta_seconds", None)
    rate     = threat.get("inflow_rate", 0.0)
    risk_val = threat.get("risk_score",  0.0)

    # 5. Update history (deque auto-evicts — QC-05)
    hist_counts.append(current_count)
    hist_times.append(round(elapsed, 1))

    # 6. Alert log
    if threat.get("sms_sent"):
        alert_log.append({
            "Time":  f"{elapsed:.0f}s",
            "Alert": f"📱 SMS · {status} · count={current_count}",
        })
        if len(alert_log) > 8:
            alert_log.pop(0)

    # 7. BGR → RGB
    render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)

    # 8. Update UI
    video_placeholder.image(render_img, channels="RGB", use_container_width=True)

    delta_str = f"{rate:+.1f} ppl/s" if abs(rate) > 0.01 else "stable"
    count_metric.metric("👥 On Screen", current_count, delta=delta_str)

    # GAP-05: show In / Out counts
    in_metric.metric("🟢 Total In",   total_in)
    out_metric.metric("🔴 Total Out", total_out)

    if eta is not None:
        eta_metric.metric(
            "⏱ Time to Stampede",
            f"{int(eta)} sec",
            delta="⚠️ Dangerous",
            delta_color="inverse",
        )
    else:
        eta_metric.metric(
            "⏱ Time to Stampede",
            "STABLE",
            delta="✅ Safe",
            delta_color="normal",
        )

    if status == "CRITICAL_CAPACITY":
        status_banner.error(
            "🚨 **CRITICAL CAPACITY MET!** Venue limit exceeded. "
            "Emergency SMS dispatched."
        )
    elif status == "PREDICTIVE_WARNING":
        status_banner.warning(
            f"⚠️ **WARNING:** Dangerous inflow. "
            f"ETA to critical: **{int(eta) if eta else '?'} sec**. SMS sent."
        )
    else:
        status_banner.success(
            f"✅ **Venue Density Safe** — {current_count} / {capacity_limit} people"
        )

    # PERF-02: throttle expensive Plotly re-renders
    if frame_idx % cfg.CHART_UPDATE_INTERVAL == 0:
        graph_placeholder.plotly_chart(
            build_pressure_chart(hist_times, hist_counts, capacity_limit),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        gauge_placeholder.plotly_chart(
            build_gauge(risk_val),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    if alert_log:
        log_placeholder.dataframe(
            pd.DataFrame(alert_log[::-1]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        log_placeholder.caption("_No alerts dispatched yet._")

# ── Cleanup & rerun ───────────────────────────────────────────────────────────
if cap is not None and cap.isOpened():
    cap.release()

# BUG-06: st.rerun() keeps the loop alive without blocking the event loop
st.rerun()
