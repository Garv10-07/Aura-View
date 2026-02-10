import os
import asyncio
import fastapi
import socketio
import uvicorn
import cv2
import torch
import yt_dlp
import numpy as np
from datetime import datetime
from fastapi import Request, HTTPException, Body
from fastapi.responses import StreamingResponse, HTMLResponse
from dotenv import load_dotenv
from supabase import create_client
import joblib
from pathlib import Path
from sklearn.cluster import DBSCAN

from prediction_manager import PredictionManager
prediction_model = None


# =========================
# ENV
# =========================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

print("SUPABASE_URL:", SUPABASE_URL)
print("SUPABASE_SERVICE_ROLE_KEY:", "SET" if SUPABASE_SERVICE_ROLE_KEY else "MISSING")
print("SUPABASE_JWT_SECRET:", "SET" if SUPABASE_JWT_SECRET else "MISSING")

# =========================
# CONFIG
# =========================
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov5m")
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "10"))
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", "640"))
NO_MODEL = os.getenv("NO_MODEL", "0").lower() in ("1", "true", "yes")

DEFAULT_THRESHOLD = 10

MODEL_PATH = "prophet_model.pkl"

# =========================
# APP
# =========================
app = fastapi.FastAPI()
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, socketio_path="/ws")
app.mount("/ws", socket_app)

# =========================
# SUPABASE
# =========================
supabase_admin = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def require_supabase_user(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = auth.replace("Bearer ", "").strip()

    if not supabase_admin:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    try:
        res = supabase_admin.auth.get_user(token)
        user = res.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        return user.id
    except Exception as e:
        print("❌ Token verify failed:", e)
        raise HTTPException(status_code=401, detail="Invalid token")


# =========================
# GLOBALS
# =========================
latest_frame_for_streaming = None

current_user_id = None

current_source_type = None
current_source_value = None

requested_source_type = None
requested_source_value = None
source_change_requested = False
stop_requested = False

ALERT_THRESHOLD = DEFAULT_THRESHOLD

# ✅ NEW: heatmap enabled toggle
HEATMAP_ENABLED = False

last_person_count = 0
last_prediction = 0

# =========================
# MODEL LOAD
# =========================
prediction_manager = PredictionManager(model_path=MODEL_PATH)

if NO_MODEL:
    print("⚠️ NO_MODEL enabled; skipping model load")
    model = None
else:
    print("🧠 Loading YOLO model...")
    model = torch.hub.load("ultralytics/yolov5", YOLO_MODEL_NAME, pretrained=True, trust_repo=True)
    model.classes = [0]  # person only
    model.conf = 0.5
    print("✅ Model loaded")

# =========================
# HELPERS
# =========================
def resolve_status(count: int, threshold: int) -> str:
    if count >= threshold:
        return "DANGER"
    if count >= threshold * 0.6:
        return "MODERATE"
    return "SAFE"

def get_cluster_threshold(alert_threshold: int) -> int:
    # ✅ Option A logic: cluster_threshold = 0.6 × alert_threshold
    return max(5, int(alert_threshold * 0.6))

def open_capture(source_type: str, source_value):
    if source_type == "webcam":
        idx = int(source_value)
        cap = cv2.VideoCapture(idx)
        return cap

    if source_type == "youtube":
        url = str(source_value)
        print(f"📡 Extracting stream from YouTube: {url}")
        ydl_opts = {"format": "best", "noplaylist": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info.get("url")
        print("✅ YouTube stream extracted:", stream_url[:90], "...")
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        return cap

    cap = cv2.VideoCapture(str(source_value), cv2.CAP_FFMPEG)
    return cap

async def save_to_supabase(user_id: str, payload: dict):
    if not supabase_admin:
        return
    try:
        supabase_admin.table("crowd_data").insert(payload).execute()
    except Exception as e:
        print("❌ Supabase insert error:", e)

async def upsert_user_threshold(user_id: str, threshold: int):
    if not supabase_admin:
        return
    try:
        supabase_admin.table("user_settings").upsert(
            {"user_id": user_id, "threshold": int(threshold), "updated_at": datetime.utcnow().isoformat()}
        ).execute()
    except Exception as e:
        print("❌ Supabase settings upsert error:", e)

async def get_user_threshold(user_id: str) -> int:
    if not supabase_admin:
        return DEFAULT_THRESHOLD
    try:
        res = (
            supabase_admin.table("user_settings")
            .select("threshold")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if res.data and len(res.data) > 0:
            return int(res.data[0].get("threshold", DEFAULT_THRESHOLD))
    except Exception as e:
        print("❌ Supabase settings fetch error:", e)
    return DEFAULT_THRESHOLD

# =========================
# CLUSTER + HEATMAP
# =========================
def compute_clusters(centers, eps=60, min_samples=2):
    """
    centers: list[(x,y)] in resized frame coordinates
    returns:
      cluster_count, largest_cluster, congestion_score(0..1), labels(list)
    """
    if len(centers) == 0:
        return 0, 0, 0.0, []

    X = np.array(centers, dtype=np.float32)
    if len(X) < 2:
        return 1, 1, 0.15, [0]

    # DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    # label -1 = noise
    unique = [l for l in set(labels) if l != -1]
    if len(unique) == 0:
        # all noise -> treat as separate singles
        return len(centers), 1, min(1.0, len(centers) / 20.0), labels.tolist()

    cluster_sizes = []
    for k in unique:
        cluster_sizes.append(int(np.sum(labels == k)))

    cluster_count = len(unique)
    largest_cluster = max(cluster_sizes) if cluster_sizes else 1

    # congestion score: normalized
    congestion_score = min(1.0, (largest_cluster / max(1, len(centers))) * 1.8)

    return cluster_count, largest_cluster, congestion_score, labels.tolist()

def compute_risk_score(count, clusters, largest_cluster, threshold):
    """
    returns risk_score 0-100
    """
    if threshold <= 0:
        threshold = 10

    cluster_threshold = get_cluster_threshold(threshold)

    score = 0.0
    score += 0.45 * min(1.0, count / threshold)
    score += 0.35 * min(1.0, largest_cluster / cluster_threshold)
    score += 0.20 * min(1.0, clusters / 8.0)

    risk = int(max(0, min(100, score * 100)))
    return risk

def make_heatmap_overlay(frame, centers):
    """
    frame: BGR image
    centers: list[(x,y)] (in frame coords)
    returns overlay frame
    """
    if frame is None:
        return frame
    if len(centers) == 0:
        return frame

    h, w = frame.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for (x, y) in centers:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(heat, (int(x), int(y)), 35, 1.0, -1)

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=25, sigmaY=25)
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    # overlay
    alpha = 0.35
    overlay = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
    return overlay

# =========================
# STREAM
# =========================
async def frame_generator():
    global latest_frame_for_streaming
    while True:
        if latest_frame_for_streaming is not None:
            ok, encoded = cv2.imencode(".jpg", latest_frame_for_streaming)
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded) + b"\r\n"
                )
        await asyncio.sleep(0.03)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/")
async def home():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return {"message": "AuraView backend running"}

@app.get("/config")
async def config(request: Request):
    user_id = require_supabase_user(request)
    th = await get_user_threshold(user_id)
    return {
        "threshold": th,
        "source_type": current_source_type,
        "source_value": str(current_source_value) if current_source_value else None,
        "heatmap": HEATMAP_ENABLED,
        "cluster_threshold": get_cluster_threshold(th)
    }

@app.post("/set_threshold")
async def set_threshold(request: Request, payload: dict = Body(...)):
    print("✅ /set_threshold called")
    print("AUTH HEADER:", request.headers.get("authorization"))
    global ALERT_THRESHOLD, current_user_id
    user_id = require_supabase_user(request)
    current_user_id = user_id

    threshold = int(payload.get("threshold", DEFAULT_THRESHOLD))
    if threshold < 1:
        threshold = DEFAULT_THRESHOLD

    ALERT_THRESHOLD = threshold
    await upsert_user_threshold(user_id, threshold)

    return {"success": True, "threshold": threshold, "cluster_threshold": get_cluster_threshold(threshold)}

@app.post("/set_source")
async def set_source(request: Request, payload: dict = Body(...)):
    global requested_source_type, requested_source_value, source_change_requested, stop_requested, current_user_id

    user_id = require_supabase_user(request)
    current_user_id = user_id

    t = (payload.get("type") or "").strip()
    v = payload.get("value")

    if t not in ("webcam", "youtube", "rtsp", "file"):
        return {"success": False, "error": "Invalid type"}

    requested_source_type = t
    requested_source_value = v
    source_change_requested = True
    stop_requested = False

    print(f"✅ Source request received: {t} -> {v}")
    return {"success": True}

@app.post("/stop")
async def stop(request: Request):
    global stop_requested, source_change_requested
    _ = require_supabase_user(request)
    stop_requested = True
    source_change_requested = False
    return {"success": True}

# ✅ NEW: heatmap toggle
@app.post("/set_heatmap")
async def set_heatmap(request: Request, payload: dict = Body(...)):
    global HEATMAP_ENABLED
    _ = require_supabase_user(request)

    enabled = bool(payload.get("enabled", False))
    HEATMAP_ENABLED = enabled
    return {"success": True, "heatmap": HEATMAP_ENABLED}

# ✅ NEW: train model from frontend
@app.post("/train_model")
async def train_model(request: Request):
    user_id = require_supabase_user(request)

    if not supabase_admin:
        return {"success": False, "error": "Supabase not configured"}

    try:
        # =========================
        # PAGINATION FETCH (ALL DATA)
        # =========================
        all_rows = []
        page_size = 1000
        start = 0

        while True:
            res = (
                supabase_admin.table("crowd_data")
                .select("created_at,count")
                .eq("user_id", user_id)
                .order("created_at", desc=False)
                .range(start, start + page_size - 1)
                .execute()
            )

            batch = res.data or []
            all_rows.extend(batch)

            print(f"📦 Batch fetched: {len(batch)} (from {start})")

            if len(batch) < page_size:
                break

            start += page_size

        rows = all_rows

        print("📊 TOTAL Rows fetched:", len(rows))
        if rows:
            print("🕒 Latest DB timestamp:", rows[-1]["created_at"])

        # =========================
        # DATA CHECK
        # =========================
        if len(rows) < 50:
            return {
                "success": False,
                "error": "Not enough data to train (need >= 50 rows)",
                "points": len(rows),
            }

        # =========================
        # TRAIN MODEL
        # =========================
        import pandas as pd
        from prophet import Prophet

        df = pd.DataFrame(rows)

        df["created_at"] = pd.to_datetime(df["created_at"])
        df["created_at"] = df["created_at"].dt.tz_localize(None)
        df["count"] = df["count"].astype(int)

        df_prophet = df.rename(
            columns={"created_at": "ds", "count": "y"}
        )[["ds", "y"]]

        df_prophet = df_prophet.drop_duplicates(subset=["ds"])
        df_prophet = (
            df_prophet.set_index("ds")
            .resample("1min")
            .mean()
            .interpolate()
            .reset_index()
        )

        df_prophet["y"] = df_prophet["y"].clip(lower=0)

        # =========================
        # PROPHET TRAIN
        # =========================
        model_p = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode="additive",
        )

        model_p.fit(df_prophet)

        # =========================
        # SAVE MODEL
        # =========================
        joblib.dump(model_p, MODEL_PATH)

        # =========================
        # LIVE RELOAD MODEL
        # =========================
        prediction_manager.reload_model()

        prediction_manager.last_prediction = None
        prediction_manager.last_prediction_time = None

        if hasattr(prediction_manager, "hard_reload"):
            print("🔥 HARD reload prediction model")
            prediction_manager.hard_reload()

        print("✅ New model trained & live reloaded")

        return {
            "success": True,
            "points": len(df_prophet),
            "raw_points": len(rows),
        }

    except Exception as e:
        print("❌ Train model error:", e)
        return {"success": False, "error": str(e)}

# =========================
# MAIN LOOP
# =========================
async def main_processing_loop():
    global latest_frame_for_streaming
    global current_source_type, current_source_value
    global requested_source_type, requested_source_value, source_change_requested, stop_requested
    global last_person_count, last_prediction, current_user_id

    cap = None
    frame_counter = 0

    while True:
        try:
            if current_source_type is None and not source_change_requested:
                latest_frame_for_streaming = None
                await asyncio.sleep(0.25)
                continue

            if stop_requested and not source_change_requested:
                latest_frame_for_streaming = None
                await asyncio.sleep(0.25)
                continue

            # apply source change
            if source_change_requested:
                print("🔁 Changing source...")

                try:
                    if cap:
                        cap.release()
                except:
                    pass
                cap = None

                current_source_type = requested_source_type
                current_source_value = requested_source_value

                try:
                    cap = open_capture(current_source_type, current_source_value)
                    if not cap or not cap.isOpened():
                        print("❌ Failed to open capture")
                        current_source_type = None
                        current_source_value = None
                        source_change_requested = False
                        await asyncio.sleep(0.5)
                        continue

                    print(f"✅ New source opened: {current_source_type}")
                    source_change_requested = False
                    frame_counter = 0
                except Exception as e:
                    print("❌ Source open exception:", e)
                    current_source_type = None
                    current_source_value = None
                    source_change_requested = False
                    await asyncio.sleep(0.5)
                    continue

            if not cap:
                await asyncio.sleep(0.05)
                continue

            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.15)
                continue

            h, w = frame.shape[:2]
            scale = RESIZE_WIDTH / w
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

            frame_counter += 1

            if frame_counter % FRAME_SKIP == 0:
                centers = []
                processed_frame = resized.copy()

                if model is None:
                    person_count = 0
                    pred = 0
                else:
                    results = model(resized)
                    dets = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else np.empty((0, 6))

                    person_count = len(dets)

                    # get centers
                    for d in dets:
                        x1, y1, x2, y2 = d[0], d[1], d[2], d[3]
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        centers.append((cx, cy))

                    pred = prediction_manager.get_future_prediction()

                    # render boxes
                    results.render()
                    processed_frame = results.ims[0].copy()

                last_person_count = person_count
                last_prediction = pred

                # cluster compute
                cluster_count, largest_cluster, congestion_score, labels = compute_clusters(centers)

                # risk + alerts
                status = resolve_status(person_count, ALERT_THRESHOLD)

                cluster_threshold = get_cluster_threshold(ALERT_THRESHOLD)
                cluster_alert = largest_cluster >= cluster_threshold

                risk_score = compute_risk_score(person_count, cluster_count, largest_cluster, ALERT_THRESHOLD)

                # overlay heatmap if enabled
                if HEATMAP_ENABLED and len(centers) > 0:
                    processed_frame = make_heatmap_overlay(processed_frame, centers)

                latest_frame_for_streaming = processed_frame

                payload = {
                    "count": int(person_count),
                    "prediction": int(pred),
                    "alert_status": status,
                    "alert_threshold": int(ALERT_THRESHOLD),

                    # ✅ NEW
                    "cluster_count": int(cluster_count),
                    "largest_cluster": int(largest_cluster),
                    "cluster_threshold": int(cluster_threshold),
                    "cluster_alert": bool(cluster_alert),
                    "congestion_score": float(congestion_score),
                    "risk_score": int(risk_score),
                    "heatmap_enabled": bool(HEATMAP_ENABLED),
                    "source_type": current_source_type,
                }

                await sio.emit("broadcast_data", payload)

                # save to supabase
                if current_user_id:
                    supa_row = {
                        "user_id": current_user_id,
                        "source_type": current_source_type,
                        "source_value": str(current_source_value),
                        "count": int(person_count),
                        "prediction": int(pred),
                        "alert_status": status,
                        "threshold": int(ALERT_THRESHOLD),

                        # ✅ NEW cluster info
                        "cluster_count": int(cluster_count),
                        "largest_cluster": int(largest_cluster),
                        "cluster_threshold": int(cluster_threshold),
                        "cluster_alert": bool(cluster_alert),
                        "congestion_score": float(congestion_score),
                        "risk_score": int(risk_score),
                    }
                    await save_to_supabase(current_user_id, supa_row)

            else:
                latest_frame_for_streaming = resized

            await asyncio.sleep(0.01)

        except Exception as e:
            print("❌ Loop error:", e)
            await asyncio.sleep(0.5)

@app.on_event("startup")
async def startup_event():
    print("🚀 Server started in IDLE mode (waiting for user input)")
    asyncio.create_task(main_processing_loop())

@sio.event
async def connect(sid, environ):
    print("✅ Client connected:", sid)

@sio.event
async def disconnect(sid):
    print("❌ Client disconnected:", sid)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
