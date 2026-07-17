import os
import cv2
import json
import time
import asyncio
import threading
from typing import Dict, Any, Optional

from fastapi import FastAPI, BackgroundTasks, Request, Depends
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import config as cfg
from crowd_tracker import VisionTracker
from heatmap_engine.dense_crowd_ai import DensityEstimator
from alert_engine.risk_engine import ThreatPredictor

# ─── Global State ───────────────────────────────────────────────────────────
class GlobalState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame_bytes: Optional[bytes] = None
        self.latest_analytics: Dict[str, Any] = {
            "status": "SAFE",
            "inflow_rate": 0.0,
            "eta_seconds": None,
            "current_count": 0,
            "threshold": cfg.VENUE_CAPACITY,
            "message": "Initializing...",
        }
        self.running: bool = False
        self.mode: str = "LIVE (YOLOv8)" # Or "STRESS TEST (CSRNet)"
        self.line_y_fraction: float = cfg.LINE_Y_FRACTION
        self.frame_skip: int = cfg.SKIP_FRAMES
        self.capacity: int = cfg.VENUE_CAPACITY
        self.emergency_override: bool = False
        self.pdf_generated_path: Optional[str] = None
        self.pdf_session_active: bool = False
        self.last_critical_time: float = 0.0

STATE = GlobalState()

# ─── Engines ───────────────────────────────────────────────────────────────
yolo_engine = VisionTracker(line_y_fraction=STATE.line_y_fraction, skip_frames=STATE.frame_skip)
cnn_engine = DensityEstimator(weight_path=cfg.CSRNET_WEIGHTS, infer_size=cfg.HEATMAP_INFER_SIZE)
predictor = ThreatPredictor(
    danger_threshold=STATE.capacity,
    buffer_size=cfg.BUFFER_SIZE,
    cooldown_seconds=cfg.SMS_COOLDOWN_SECONDS,
    warning_horizon=cfg.WARNING_HORIZON_SECONDS,
    velocity_floor=cfg.VELOCITY_FLOOR,
)

def play_buzzer():
    import platform
    if platform.system() == "Windows":
        try:
            import winsound
            winsound.Beep(2200, 150)
            winsound.Beep(1800, 150)
        except Exception:
            pass

# ─── Background AI Loop ─────────────────────────────────────────────────────
def ai_processing_loop():
    import logging
    logger = logging.getLogger("FastAPI.AILoop")
    logger.setLevel(logging.INFO)
    logger.info("Starting background AI processing loop...")
    
    # Video source
    video_source = cfg.VIDEO_SOURCE
    if cfg.DEMO_MODE:
        # Use CNN_VIDEO_PATH if it exists, otherwise fall back to webcam 0
        if os.path.exists(cfg.CNN_VIDEO_PATH):
            video_source = cfg.CNN_VIDEO_PATH
        elif os.path.exists("data/stress_test.mp4"):
            video_source = "data/stress_test.mp4"
        else:
            logger.warning(
                f"Demo video files not found. Falling back to webcam source {cfg.VIDEO_SOURCE}."
            )
            video_source = cfg.VIDEO_SOURCE

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_source}")
        
    frame_count = 0
    current_count = 0
    total_in = 0
    total_out = 0
    current_on_screen = 0
    
    while STATE.running:
        ret, frame = cap.read()
        if not ret:
            if not cap.isOpened() and cfg.DEMO_MODE:
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ret = True
            elif cfg.DEMO_MODE or isinstance(video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                yolo_engine.in_count = 0
                yolo_engine.out_count = 0
                yolo_engine._track_states.clear()
                continue
            else:
                break
                
        frame_count += 1
        render_img = frame.copy()
        
        # Apply Logic based on mode
        with STATE.lock:
            mode = STATE.mode
            # Sync tracker settings
            yolo_engine.line_y_fraction = STATE.line_y_fraction
            yolo_engine.skip_frames = STATE.frame_skip
            # Sync predictor settings
            predictor.threshold = STATE.capacity
        if mode == "LIVE (YOLOv8)":
            result = yolo_engine.process_frame(frame, frame_id=frame_count)
            render_img = result["annotated_frame"]
            current_count = result["current_on_screen"]
            total_in = result["in_count"]
            total_out = result["out_count"]
            current_on_screen = result["current_on_screen"]
        else:
            # CSRNet
            # Run every N frames for CSRNet if we want, or every frame
            if frame_count % max(1, STATE.frame_skip) == 0:
                result = cnn_engine.process_frame(frame)
                STATE.last_csr_result = result
            else:
                result = getattr(STATE, "last_csr_result", None)
                if result is None:
                    result = cnn_engine.process_frame(frame)
                    STATE.last_csr_result = result
            
            render_img = result["heatmap_frame"]
            current_count = result["estimated_count"]
            current_on_screen = result["estimated_count"]
                
        # Risk Predictor
        if frame_count % 5 == 0 or mode == "STRESS TEST (CSRNet)":
            analytics = predictor.update_and_predict(current_count)
            with STATE.lock:
                analytics["total_in"] = total_in
                analytics["total_out"] = total_out
                analytics["current_on_screen"] = current_on_screen
                if STATE.emergency_override:
                    analytics["status"] = "CRITICAL_CAPACITY"
                    analytics["sms_sent"] = True
                    analytics["risk_score"] = 1.0
                    analytics["message"] = "🚨 EMERGENCY OVERRIDE TRIGGERED"
                
                status = analytics.get("status")
                if status == "CRITICAL_CAPACITY":
                    STATE.last_critical_time = time.time()
                    # Play buzzer sound
                    import threading
                    threading.Thread(target=play_buzzer, daemon=True).start()
                    
                    # Generate PDF once per session
                    if not STATE.pdf_session_active:
                        from alert_engine.report_generator import generate_incident_report
                        import datetime
                        heatmap_snapshot_path = "heatmap_snapshot.jpg"
                        cv2.imwrite(heatmap_snapshot_path, render_img)
                        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        rate = analytics.get("inflow_rate", 0.0)
                        try:
                            pdf_path = generate_incident_report(ts, rate, current_count, heatmap_snapshot_path)
                            STATE.pdf_generated_path = pdf_path
                            STATE.pdf_session_active = True
                            logger.info(f"Incident report generated successfully: {pdf_path}")
                        except Exception as e:
                            logger.error(f"Failed to generate PDF: {e}")
                else:
                    # Debounce: Only clear the PDF session after 15 seconds of clean/safe status
                    if STATE.pdf_session_active and (time.time() - STATE.last_critical_time > 15.0):
                        STATE.pdf_session_active = False
                        STATE.pdf_generated_path = None
                
                analytics["pdf_generated"] = STATE.pdf_generated_path
                STATE.latest_analytics = analytics
                
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', render_img)
        if ret:
            with STATE.lock:
                STATE.latest_frame_bytes = buffer.tobytes()
                
        time.sleep(0.01) # Small sleep to yield CPU
        
    cap.release()
    logger.info("Background AI loop stopped.")

# ─── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(title="StampedeZero Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    STATE.running = True
    threading.Thread(target=ai_processing_loop, daemon=True).start()

@app.on_event("shutdown")
async def shutdown_event():
    STATE.running = False

# ─── Endpoints ──────────────────────────────────────────────────────────────

def generate_mjpeg():
    while True:
        with STATE.lock:
            frame_bytes = STATE.latest_frame_bytes
            
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

async def generate_sse():
    while True:
        with STATE.lock:
            data = STATE.latest_analytics
            
        yield {
            "event": "message",
            "id": "message_id",
            "retry": 15000,
            "data": json.dumps(data)
        }
        await asyncio.sleep(0.5)

@app.get("/analytics")
async def analytics_feed(request: Request):
    return EventSourceResponse(generate_sse())

class ConfigUpdate(BaseModel):
    mode: Optional[str] = None
    line_y_fraction: Optional[float] = None
    frame_skip: Optional[int] = None
    capacity: Optional[int] = None
    emergency_override: Optional[bool] = None

@app.post("/config")
def update_config(cfg_update: ConfigUpdate):
    with STATE.lock:
        if cfg_update.mode is not None:
            STATE.mode = cfg_update.mode
            predictor.reset() # Reset stats on mode switch
            yolo_engine._track_states.clear()
            yolo_engine.in_count = 0
            yolo_engine.out_count = 0
        if cfg_update.line_y_fraction is not None:
            STATE.line_y_fraction = cfg_update.line_y_fraction
        if cfg_update.frame_skip is not None:
            STATE.frame_skip = cfg_update.frame_skip
        if cfg_update.capacity is not None:
            STATE.capacity = cfg_update.capacity
        if cfg_update.emergency_override is not None:
            STATE.emergency_override = cfg_update.emergency_override
            
    return {"status": "success", "current_config": {
        "mode": STATE.mode,
        "line_y_fraction": STATE.line_y_fraction,
        "frame_skip": STATE.frame_skip,
        "capacity": STATE.capacity,
        "emergency_override": STATE.emergency_override
    }}

@app.get("/config")
def get_config():
    with STATE.lock:
        return {
            "mode": STATE.mode,
            "line_y_fraction": STATE.line_y_fraction,
            "frame_skip": STATE.frame_skip,
            "capacity": STATE.capacity,
            "emergency_override": STATE.emergency_override
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
