<div align="center">
  <img src="frontend/public/vite.svg" alt="StampedeZero Logo" width="120" />
  <h1>🚨 StampedeZero</h1>
  <p>A Proactive AI Defense System for Predictive Crowd Safety & Disaster Management.</p>

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React" />
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  </p>
</div>

<hr />

## 🌟 Overview

**StampedeZero** is a next-generation predictive safety platform designed to prevent crowd crushes before they happen. While traditional CCTV systems rely on human operators to notice when a crowd becomes dangerously dense, StampedeZero uses dual-engine Computer Vision to mathematically forecast crowd buildup and alert authorities *minutes before* capacity thresholds are breached.

Whether monitoring concerts, festivals, stadiums, or religious gatherings, StampedeZero provides event organizers with real-time analytics, automated Twilio SMS emergency warnings, and beautifully formatted incident reports.

## ✨ Key Features

- 👁️ **Real-Time Vision Tracker:** Utilizes YOLOv8 + ByteTrack to monitor sparse crowds. Features a strict temporal filter to reject false positives and precisely counts inflow/outflow through intelligent line-crossing algorithms.
- 🗺️ **Dense Crowd Heatmaps (CSRNet):** Seamlessly switches to a VGG16-backed CSRNet engine to estimate pixel-level density when the crowd becomes too dense to count individuals, displaying a dynamic JET colormap overlay.
- 📈 **Predictive Threat Engine:** Leverages linear regression over a rolling data buffer to calculate the mathematical trend of crowd growth, solving for the exact ETA to critical capacity.
- 📱 **Automated SMS Alerts:** Fully integrated with the Twilio API to dispatch immediate SMS warnings to security personnel with built-in cooldowns to prevent alert fatigue.
- 🛡️ **Graceful Degradation:** Resilient fallback architecture ensures the system continues to run in "Mock Mode" for flawless demos even if hardware, weights, or API credentials are missing.
- 📊 **Live Operations Dashboard:** A modern React SPA streaming real-time video via MJPEG and live telemetry via Server-Sent Events (SSE).

## 🛠️ Tech Stack

**Backend Architecture:**
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3)
- **Computer Vision:** [OpenCV](https://opencv.org/)
- **Deep Learning:** [PyTorch](https://pytorch.org/) (CSRNet) & [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Alerting:** [Twilio API](https://www.twilio.com/)

**Frontend Architecture:**
- **Framework:** [React 19](https://react.dev/)
- **Build Tool:** [Vite](https://vitejs.dev/)
- **Data Visualization:** [Recharts](https://recharts.org/)
- **Styling:** Vanilla CSS (Dark Mode Optimized)

## 🚀 Getting Started

Follow these instructions to get a local copy of StampedeZero up and running.

### Prerequisites

- Python 3.9+
- Node.js (v18 or higher recommended)
- Optional: CUDA-compatible GPU for accelerated PyTorch inference

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mukilan-s18/StampedeZero.git
   cd StampedeZero
   ```

2. **Setup the Python Backend**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the root directory based on `.env.example`:
   ```env
   TWILIO_SID=your_twilio_sid
   TWILIO_TOKEN=your_twilio_token
   TWILIO_FROM=+1234567890
   TARGET_PHONE=+0987654321
   ```

4. **Start the FastAPI Server**
   ```bash
   python server.py
   ```

5. **Start the React Frontend**
   Open a new terminal window:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

6. **Access the Dashboard:**<br/>
   Open [http://localhost:5173](http://localhost:5173) in your browser.

## 📁 Project Structure

```
StampedeZero/
├── alert_engine/         # ThreatPredictor, DataBuffer, & Twilio SMS integrations
├── frontend/             # React SPA (Live dashboard, Recharts, Video Stream)
├── heatmap_engine/       # CSRNet density estimation & heatmap generation
├── weights/              # Pre-trained models (csrnet_weights.pth, yolov8n.pt)
├── server.py             # FastAPI backend, MJPEG streams, SSE analytics
├── crowd_tracker.py      # YOLOv8 + ByteTrack object tracking & line counting
└── config.py             # Global constants & tunable application parameters
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
