"""
app.py  —  StampedeZero: Proactive Crowd Flow Engine
=====================================================
OWNER: Engineer 4 (Full-Stack / UI Lead)
RUN:   streamlit run app.py

Wiring:
  Mode 1  →  crowd_tracker.py   (Eng 1 · YOLO)
  Mode 2  →  dense_crowd_ai.py  (Eng 2 · CSRNet)
  Both    →  risk_engine.py     (Eng 3 · LSTM + Twilio)
"""

import os
import time

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

    /* Metric cards */
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

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #141820 100%);
        border-right: 1px solid #2e3244;
    }

    h3 { color: #e0e0e0 !important; letter-spacing: .02em; }
    hr { border-color: #2e3244 !important; }
    .stAlert { border-radius: 10px !important; }

    /* Hero title */
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

# ── Model cache — loads once, never reloads on UI interaction ─────────────────
@st.cache_resource(show_spinner="🔄 Loading AI engines…")
def load_engines():
    from crowd_tracker  import VisionTracker
    from dense_crowd_ai import DensityEstimator
    from risk_engine    import ThreatPredictor

    tracker   = VisionTracker(line_y=300)
    estimator = DensityEstimator(weights_path="weights/csrnet_weights.pth")
    predictor = ThreatPredictor(threshold=50, warn_pct=0.75, window_size=30)
    return tracker, estimator, predictor


yolo_engine, cnn_engine, risk_engine = load_engines()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.markdown("## 🚨 StampedeZero")

    st.markdown("---")
    st.markdown("### 🎛️ Mode Selection")
    app_mode = st.sidebar.radio(
        "Select Input Flow",
        ["1. Live Footfall (YOLO Edge)", "2. Extreme Crowd Simulator (CNN)"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    capacity_limit = st.slider("Venue Capacity Limit", 10, 5000, 50, 10,
                                help="Max safe crowd count before alerts fire.")
    warn_pct_val   = st.slider("Warning Threshold (%)", 50, 95, 75, 5,
                                help="Fraction of capacity that triggers a warning.")
    frame_skip     = st.slider("Frame Skip (performance)", 1, 5, 1,
                                help="Process every Nth frame. Raise this if UI lags.")

    st.markdown("---")
    stop_btn = st.button("⏹  Stop Feed", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(
        "<div style='color:#3a4060;font-size:.75rem;text-align:center'>"
        "StampedeZero v1.0 · SIMATS Hackathon 2026</div>",
        unsafe_allow_html=True,
    )

# Push live slider values into the risk engine
risk_engine.threshold      = capacity_limit
risk_engine.warn_threshold = int(capacity_limit * warn_pct_val / 100)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">🚨 StampedeZero</div>'
    '<div class="hero-sub">Proactive Crowd Flow Engine · Real-Time AI Safety System</div>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

status_banner = st.empty()   # fullwidth banner
st.markdown("---")

# ── Two-column layout: 65 % video | 35 % analytics ───────────────────────────
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
    metric_col_A, metric_col_B = st.columns(2)
    count_text = metric_col_A.empty()
    eta_text   = metric_col_B.empty()

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
            x=times, y=counts,
            mode="lines",
            name="Crowd Count",
            line=dict(color="#ff4b4b", width=2.5, shape="spline", smoothing=1.0),
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.08)",
        ))
    if times:
        fig.add_hline(y=threshold, line_dash="dot", line_color="#ff8c42",
                      annotation_text="⚠️ Capacity", annotation_font_color="#ff8c42")
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
if "Live" in app_mode:
    cap = cv2.VideoCapture(0)
else:
    vpath = "data/crowd_concert.mp4"
    cap   = cv2.VideoCapture(vpath) if os.path.exists(vpath) else None

# ── History buffers ───────────────────────────────────────────────────────────
MAX_HIST = 120
hist_counts: list = []
hist_times:  list = []
alert_log:   list = []

frame_idx  = 0
start_time = time.time()

# ── Master rendering loop ─────────────────────────────────────────────────────
while not stop_btn:
    frame_idx += 1
    elapsed = time.time() - start_time

    # 1. Read frame
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if "Simulator" in app_mode:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video
                continue
            else:
                break
    else:
        frame = blank_frame()
        time.sleep(0.04)

    # 2. Frame skip (performance)
    if frame_idx % frame_skip != 0:
        continue

    # 3. Downscale
    frame = cv2.resize(frame, (640, 480))

    # 4. AI routing
    try:
        if "Live" in app_mode:
            ai_data       = yolo_engine.process_frame(frame)
            render_img    = ai_data["annotated_frame"]
            current_count = ai_data["current_on_screen"]
        else:
            ai_data       = cnn_engine.process_frame(frame)
            render_img    = ai_data["heatmap_frame"]
            current_count = ai_data["estimated_count"]
    except Exception as exc:
        st.error(f"AI engine error: {exc}")
        break

    # 5. Risk prediction (Eng 3)
    try:
        threat = risk_engine.update_and_predict(current_count)
    except Exception:
        threat = {"status": "SAFE", "eta_seconds": None,
                  "inflow_rate": 0.0, "risk_score": 0.0, "sms_sent": False}

    status    = threat.get("status",      "SAFE")
    eta       = threat.get("eta_seconds", None)
    rate      = threat.get("inflow_rate", 0.0)
    risk_val  = threat.get("risk_score",  0.0)

    # 6. Update history
    hist_counts.append(current_count)
    hist_times.append(round(elapsed, 1))
    if len(hist_counts) > MAX_HIST:
        hist_counts.pop(0)
        hist_times.pop(0)

    # 7. Alert log
    if threat.get("sms_sent"):
        alert_log.append({"Time": f"{elapsed:.0f}s",
                           "Alert": f"📱 SMS · {status} · count={current_count}"})
    if len(alert_log) > 8:
        alert_log.pop(0)

    # 8. BGR → RGB
    render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)

    # 9. Update UI placeholders
    video_placeholder.image(render_img, channels="RGB", use_container_width=True)

    delta_str = f"{rate:+.1f} ppl/s" if abs(rate) > 0.01 else "stable"
    count_text.metric("👥 Total Count", current_count, delta=delta_str)

    if eta is not None:
        eta_text.metric("⏱ Time to Stampede",
                        f"{int(eta)} sec", delta="⚠️ Dangerous",
                        delta_color="inverse")
    else:
        eta_text.metric("⏱ Time to Stampede",
                        "STABLE", delta="✅ Safe",
                        delta_color="normal")

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

# ── Cleanup ───────────────────────────────────────────────────────────────────
if cap is not None and cap.isOpened():
    cap.release()

status_banner.info("⏸ Feed stopped. Refresh the page or press Run to restart.")
