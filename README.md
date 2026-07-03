# 🌐 Aura View — AI-Powered Crowd Intelligence Platform

> **Real-time crowd monitoring, density analysis, and security alerting powered by YOLOv5 and WebSockets.**

Aura View is a full-stack crowd management system designed for public safety and event security. It combines a **FastAPI backend** running live YOLOv5 computer vision with a **web dashboard** and a companion **React Native mobile app (Aura Guard)** — giving security teams real-time situational awareness from anywhere.

---

## ✨ Features

### 🧠 AI & Computer Vision
- **YOLOv5 Person Detection** — Detects and counts people in real-time from any video source
- **DBSCAN Cluster Analysis** — Identifies dangerous crowd groupings and calculates the largest cluster
- **Risk Score Engine** — Composite score (0–100) based on crowd density, cluster size, and distribution
- **ML Trend Prediction** — Linear Regression auto-trainer predicts crowd growth over time
- **Heatmap Overlay** — Toggleable persistent heatmap showing crowd accumulation zones

### 📡 Live Streaming & Data
- **Multi-Source Input** — Supports webcam, YouTube live streams (via `yt-dlp`), and RTSP/IP camera feeds
- **MJPEG Video Stream** — Annotated live feed accessible at `/video_feed`
- **WebSocket Broadcasting** — Real-time stats pushed to all connected clients via Socket.IO

### 📱 Aura Guard Mobile App
- Companion React Native app for security personnel
- Live crowd status and person count display
- Embedded live camera feed via WebView
- **Emergency push notifications** when crowd exceeds danger threshold
- Alert acknowledgement workflow
- Guard authentication with PIN-based login via Supabase

### ☁️ Cloud & Backend
- **FastAPI** REST API with CORS support
- **Supabase (PostgreSQL)** for multi-organization, multi-site data management
- Organization/site setup and guard device management
- Configurable danger threshold, frame skip, and auto-training interval
- SMS cooldown logic to prevent alert spam (2-minute cooldown)

---

## 🏗️ Architecture

```
┌────────────────────────────────────────┐
│          Web Browser (Admin)           │
│         index.html Dashboard           │
└────────────┬───────────────────────────┘
             │  HTTP + WebSocket (Socket.IO)
┌────────────▼───────────────────────────┐
│        FastAPI Backend (server.py)     │
│  ┌─────────────┐  ┌─────────────────┐  │
│  │ YOLOv5 (CV) │  │  Socket.IO (WS) │  │
│  └─────────────┘  └─────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │   DBSCAN + LinearRegression AI   │  │
│  └──────────────────────────────────┘  │
└────────────┬───────────────────────────┘
             │
   ┌─────────┴──────────┐
   │                    │
┌──▼──────────┐  ┌──────▼────────────┐
│   Supabase  │  │  Aura Guard App   │
│ (PostgreSQL)│  │ (React Native /   │
│             │  │  Expo)            │
└─────────────┘  └───────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ & npm
- [Expo CLI](https://docs.expo.dev/get-started/installation/) (`npm install -g expo-cli`)
- A [Supabase](https://supabase.com/) project (free tier works)
- (Optional) An Android/iOS device or emulator

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/aura-view.git
cd aura-view
```

---

### 2. Backend Setup

#### Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

#### Configure environment variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Optional overrides
YOLO_MODEL_NAME=yolov5s
FRAME_SKIP=5
RESIZE_WIDTH=640
```

#### Run the backend server

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The **web dashboard** will be available at: http://localhost:8000  
The **live video feed** endpoint: http://localhost:8000/video_feed

---

### 3. Aura Guard Mobile App Setup

```bash
cd aura-guard-app
npm install
npm start
```

- Press **`a`** to launch on an Android emulator
- Scan the **QR code** with the Expo Go app on a physical device

> **Note:** On the login screen, set the **Backend IP** to your machine's local IPv4 address, e.g. `http://192.168.1.x:8000`. Both your phone and PC must be on the same Wi-Fi network.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the web dashboard |
| `GET` | `/video_feed` | MJPEG annotated live video stream |
| `GET` | `/config` | Get current threshold and heatmap config |
| `POST` | `/set_source` | Start a new camera/stream source |
| `POST` | `/set_threshold` | Update the crowd danger threshold |
| `POST` | `/set_heatmap` | Enable or disable the heatmap overlay |
| `POST` | `/guard_login` | Authenticate a security guard |
| `POST` | `/setup_organization` | Register a new company and site |
| `POST` | `/test_alert` | Send a test emergency alert via Socket.IO |

### WebSocket Events (Socket.IO at `/ws`)

| Event | Direction | Payload |
|-------|-----------|---------|
| `join_site` | Client → Server | `{ site_id }` |
| `broadcast_data` | Server → Client | Count, risk score, cluster data, trend |
| `security_alert` | Server → Client | Emergency alert message and crowd count |
| `acknowledge_alert` | Client → Server | Guard ID and alert acknowledgement |

---

## 📊 Analytics & Scoring

### Risk Score (0–100)
Calculated as a weighted composite:

```
Risk Score = (0.45 × crowd_density) + (0.35 × cluster_density) + (0.20 × cluster_count_factor)
```

### Alert Statuses

| Status | Condition |
|--------|-----------|
| 🟢 `SAFE` | Count < 60% of threshold |
| 🟡 `MODERATE` | Count >= 60% of threshold |
| 🔴 `DANGER` | Count >= threshold |

### Auto-Trainer
A background Linear Regression model trains periodically on collected frame data (default: every 1 minute with >5 samples) to predict the **5-minute future crowd count**.

---

## 🗄️ Database Schema (Supabase)

| Table | Description |
|-------|-------------|
| `companies` | Organization records |
| `sites` | Monitoring sites per company |
| `site_admins` | User ↔ Site admin mappings |
| `guard_devices` | Guard login credentials and FCM tokens |

---

## 📦 Tech Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.119.0 | REST API framework |
| Uvicorn | 0.37.0 | ASGI server |
| PyTorch | 2.8.0 | Deep learning runtime |
| Ultralytics YOLOv5 | 8.3.211 | Person detection model |
| OpenCV | 4.11.0 | Frame capture & processing |
| scikit-learn | latest | DBSCAN & Linear Regression |
| python-socketio | 5.14.1 | Real-time WebSocket events |
| Supabase Python | latest | Cloud database client |
| yt-dlp | 2026.6.9 | YouTube stream extraction |

### Mobile App (Aura Guard)
| Technology | Version | Purpose |
|------------|---------|---------|
| React Native | 0.81.5 | Mobile UI framework |
| Expo | ^54.0 | Development & build tooling |
| expo-notifications | ~0.32.17 | Push alert system |
| socket.io-client | ^4.8.3 | Real-time event listener |
| react-native-webview | 13.15.0 | Embedded live camera feed |

---

## 📁 Project Structure

```
aura-view/
├── server.py                  # FastAPI backend with YOLOv5 + Socket.IO
├── index.html                 # Web dashboard (single-file, self-contained)
├── requirements.txt           # Python dependencies
├── yolov5s.pt                 # Pre-trained YOLOv5s weights
├── .env                       # Environment variables (not committed)
│
└── aura-guard-app/            # React Native companion app
    ├── App.js                 # Main application component
    ├── app.json               # Expo configuration
    ├── package.json           # Node.js dependencies
    └── index.js               # App entry point
```

---

## ⚙️ Configuration

All configurable values can be set via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPABASE_URL` | — | Your Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | — | Supabase service role secret key |
| `YOLO_MODEL_NAME` | `yolov5s` | YOLOv5 model variant (`yolov5s`, `yolov5m`, etc.) |
| `FRAME_SKIP` | `5` | Process every Nth frame (higher = faster, less accurate) |
| `RESIZE_WIDTH` | `640` | Frame width before inference (pixels) |

---

## 🛡️ Security Notes

- The `.env` file is **not committed** — never expose your `SUPABASE_SERVICE_ROLE_KEY`
- CORS is open (`*`) for development; **restrict origins in production**
- Guard authentication uses PIN-based login stored in Supabase
- Emergency alerts have a **120-second cooldown** to prevent notification flooding

---

## 📄 License

This project is licensed under the terms found in the [LICENSE](./aura-guard-app/LICENSE) file.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the object detection model
- [Supabase](https://supabase.com/) for the open-source Firebase alternative
- [Expo](https://expo.dev/) for making React Native development seamless
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for stream URL extraction

---

*Built for real-world public safety applications — from concerts to transit hubs.*
