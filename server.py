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

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN

# =========================
# ENV
# =========================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

print("SUPABASE_URL:", SUPABASE_URL)
print("SUPABASE_SERVICE_ROLE_KEY:", "SET" if SUPABASE_SERVICE_ROLE_KEY else "MISSING")

# =========================
# CONFIG
# =========================
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov5m")
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "10"))
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", "640"))
NO_MODEL = os.getenv("NO_MODEL", "0").lower() in ("1", "true", "yes")

DEFAULT_THRESHOLD = 10

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
HEATMAP_ENABLED = False  # ✅ Heatmap Global Variable is back
last_person_count = 0
current_trend_per_min = 0.0 

# =========================
# YOLO MODEL LOAD
# =========================
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
    return max(5, int(alert_threshold * 0.6))

def open_capture(source_type: str, source_value):
    print(f"🔄 Attempting to open source: [{source_type}] Value: [{source_value}]")
    
    if source_type == "webcam":
        try:
            idx = int(source_value) if str(source_value).strip() else 0
        except ValueError:
            idx = 0
        print(f"📹 Starting Local Webcam (Index: {idx})...")
        return cv2.VideoCapture(idx)

    if source_type == "youtube":
        val = str(source_value).strip()
        if val == "0" or len(val) < 5 or ("youtube.com" not in val and "youtu.be" not in val):
            print("⚠️ UI Mistake Detected: YouTube selected but invalid URL/'0' entered. Falling back to Webcam 0.")
            return cv2.VideoCapture(0)
            
        print(f"📡 Extracting stream from YouTube: {val}")
        try:
            ydl_opts = {"format": "best", "noplaylist": True, "quiet": True} 
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(val, download=False)
                stream_url = info.get("url")
            return cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        except Exception as e:
            print(f"❌ YouTube extraction failed: {e}")
            return None

    print(f"📡 Connecting to Network Stream/File: {source_value}")
    return cv2.VideoCapture(str(source_value), cv2.CAP_FFMPEG)

async def save_to_supabase(user_id: str, payload: dict):
    if not supabase_admin: return
    try: supabase_admin.table("crowd_data").insert(payload).execute()
    except Exception as e: print("❌ Supabase insert error:", e)

async def upsert_user_threshold(user_id: str, threshold: int):
    if not supabase_admin: return
    try: supabase_admin.table("user_settings").upsert({"user_id": user_id, "threshold": int(threshold), "updated_at": datetime.utcnow().isoformat()}).execute()
    except Exception as e: pass

async def get_user_threshold(user_id: str) -> int:
    if not supabase_admin: return DEFAULT_THRESHOLD
    try:
        res = supabase_admin.table("user_settings").select("threshold").eq("user_id", user_id).limit(1).execute()
        if res.data: return int(res.data[0].get("threshold", DEFAULT_THRESHOLD))
    except Exception as e: pass
    return DEFAULT_THRESHOLD

# =========================
# CLUSTER & HEATMAP LOGIC
# =========================
def compute_clusters(centers, eps=60, min_samples=2):
    if len(centers) == 0: return 0, 0, 0.0, []
    X = np.array(centers, dtype=np.float32)
    if len(X) < 2: return 1, 1, 0.15, [0]
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    unique = [l for l in set(labels) if l != -1]
    if len(unique) == 0: return len(centers), 1, min(1.0, len(centers) / 20.0), labels.tolist()
    cluster_sizes = [int(np.sum(labels == k)) for k in unique]
    cluster_count = len(unique)
    largest_cluster = max(cluster_sizes) if cluster_sizes else 1
    congestion_score = min(1.0, (largest_cluster / max(1, len(centers))) * 1.8)
    return cluster_count, largest_cluster, congestion_score, labels.tolist()

def compute_risk_score(count, clusters, largest_cluster, threshold):
    if threshold <= 0: threshold = 10
    cluster_threshold = get_cluster_threshold(threshold)
    score = (0.45 * min(1.0, count / threshold)) + (0.35 * min(1.0, largest_cluster / cluster_threshold)) + (0.20 * min(1.0, clusters / 8.0))
    return int(max(0, min(100, score * 100)))

# ✅ CLASSIC HEATMAP FUNCTION IS BACK
def make_heatmap_overlay(frame, centers):
    if frame is None or len(centers) == 0: return frame
    h, w = frame.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for (x, y) in centers:
        if 0 <= x < w and 0 <= y < h: cv2.circle(heat, (int(x), int(y)), 35, 1.0, -1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=25, sigmaY=25)
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - 0.35, colored, 0.35, 0)

# =========================
# STREAM & API
# =========================
@app.get("/")
async def home():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return {"message": "AuraView backend running. index.html not found."}

async def frame_generator():
    global latest_frame_for_streaming
    while True:
        if latest_frame_for_streaming is not None:
            ok, encoded = cv2.imencode(".jpg", latest_frame_for_streaming)
            if ok: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(encoded) + b"\r\n")
        await asyncio.sleep(0.03)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/config")
async def config(request: Request):
    user_id = require_supabase_user(request)
    th = await get_user_threshold(user_id)
    return { "threshold": th, "source_type": current_source_type, "source_value": str(current_source_value) if current_source_value else None, "heatmap": HEATMAP_ENABLED, "cluster_threshold": get_cluster_threshold(th) }

@app.post("/set_threshold")
async def set_threshold(request: Request, payload: dict = Body(...)):
    global ALERT_THRESHOLD, current_user_id
    current_user_id = require_supabase_user(request)
    threshold = int(payload.get("threshold", DEFAULT_THRESHOLD))
    ALERT_THRESHOLD = threshold if threshold > 0 else DEFAULT_THRESHOLD
    await upsert_user_threshold(current_user_id, ALERT_THRESHOLD)
    return {"success": True, "threshold": ALERT_THRESHOLD, "cluster_threshold": get_cluster_threshold(ALERT_THRESHOLD)}

@app.post("/set_source")
async def set_source(request: Request, payload: dict = Body(...)):
    global requested_source_type, requested_source_value, source_change_requested, stop_requested, current_user_id
    current_user_id = require_supabase_user(request)
    requested_source_type = payload.get("type", "").strip()
    requested_source_value = payload.get("value")
    source_change_requested = True
    stop_requested = False
    return {"success": True}

@app.post("/set_heatmap")
async def set_heatmap(request: Request, payload: dict = Body(...)):
    global HEATMAP_ENABLED
    _ = require_supabase_user(request)
    HEATMAP_ENABLED = bool(payload.get("enabled", False))
    return {"success": True}

@app.post("/train_model")
async def train_model(request: Request):
    global current_trend_per_min, current_source_type, current_source_value
    user_id = require_supabase_user(request)

    if not supabase_admin:
        return {"success": False, "error": "Supabase not configured"}

    if not current_source_type:
        return {"success": False, "error": "No active source to train on"}

    print("\n" + "="*50)
    print(f"🚀 [AUTO-TRAIN] Fetching data for Source: {current_source_type} ({current_source_value})...")

    try:
        res = (supabase_admin.table("crowd_data")
               .select("created_at,count")
               .eq("user_id", user_id)
               .eq("source_type", current_source_type)
               .eq("source_value", str(current_source_value))
               .order("created_at", desc=True) 
               .limit(300).execute())

        rows = res.data or []
        
        if len(rows) < 10:
            current_trend_per_min = 0.0
            print("⚠️ Not enough data for this specific camera yet. Trend set to 0.")
            print("="*50 + "\n")
            return {"success": False, "error": "Need more data points for this feed"}

        print(f"✅ Extracted {len(rows)} latest records for current feed.")
        print("🧠 Running Linear Regression Model...")

        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at")
        
        df["minutes"] = (df["created_at"] - df["created_at"].min()).dt.total_seconds() / 60.0
        
        X = df[["minutes"]].values
        y = df["count"].values

        model_lr = LinearRegression()
        model_lr.fit(X, y)
        
        slope = float(model_lr.coef_[0])
        current_trend_per_min = slope

        print(f"📈 [RESULT] Trend Calculated: {slope:.3f} persons/min")
        print("="*50 + "\n")

        return {"success": True, "trend_per_min": slope, "points": len(rows)}

    except Exception as e:
        print("❌ Train model error:", e)
        print("="*50 + "\n")
        return {"success": False, "error": str(e)}

# =========================
# MAIN LOOP
# =========================
async def main_processing_loop():
    global latest_frame_for_streaming, current_source_type, current_source_value
    global requested_source_type, requested_source_value, source_change_requested, stop_requested
    global last_person_count, current_user_id, current_trend_per_min

    cap = None
    frame_counter = 0

    while True:
        try:
            if current_source_type is None and not source_change_requested:
                await asyncio.sleep(0.25)
                continue

            if source_change_requested:
                if cap: cap.release()
                current_source_type = requested_source_type
                current_source_value = requested_source_value
                current_trend_per_min = 0.0 
                try:
                    cap = open_capture(current_source_type, current_source_value)
                    source_change_requested = False
                    frame_counter = 0
                except:
                    current_source_type = None
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
                else:
                    results = model(resized)
                    dets = results.xyxy[0].cpu().numpy() if len(results.xyxy) > 0 else np.empty((0, 6))
                    person_count = len(dets)
                    
                    for d in dets:
                        centers.append((int((d[0]+d[2])/2), int((d[1]+d[3])/2)))
                    
                    # ✅ YOLO Default Render is Back!
                    results.render()
                    processed_frame = results.ims[0].copy()
                    
                last_person_count = person_count
                cluster_count, largest_cluster, congestion_score, labels = compute_clusters(centers)
                status = resolve_status(person_count, ALERT_THRESHOLD)
                cluster_threshold = get_cluster_threshold(ALERT_THRESHOLD)
                cluster_alert = largest_cluster >= cluster_threshold
                risk_score = compute_risk_score(person_count, cluster_count, largest_cluster, ALERT_THRESHOLD)

                # ✅ CLASSIC HEATMAP CALL IS BACK
                if HEATMAP_ENABLED and len(centers) > 0:
                    processed_frame = make_heatmap_overlay(processed_frame, centers)

                latest_frame_for_streaming = processed_frame

                payload = {
                    "count": int(person_count),
                    "trend_per_min": float(current_trend_per_min),
                    "alert_status": status,
                    "alert_threshold": int(ALERT_THRESHOLD),
                    "cluster_count": int(cluster_count),
                    "largest_cluster": int(largest_cluster),
                    "cluster_threshold": int(cluster_threshold),
                    "cluster_alert": bool(cluster_alert),
                    "congestion_score": float(congestion_score),
                    "risk_score": int(risk_score),
                    "heatmap_enabled": bool(HEATMAP_ENABLED)
                }
                await sio.emit("broadcast_data", payload)

                if current_user_id:
                    db_pred = max(0, int(person_count + (current_trend_per_min * 5)))
                    supa_row = {
                        "user_id": current_user_id,
                        "source_type": current_source_type,
                        "source_value": str(current_source_value),
                        "count": int(person_count),
                        "prediction": db_pred,
                        "alert_status": status,
                        "threshold": int(ALERT_THRESHOLD),
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
            await asyncio.sleep(0.5)

@app.on_event("startup")
async def startup_event():
    print("🚀 Server started. Open http://127.0.0.1:8000 in your browser.")
    asyncio.create_task(main_processing_loop())

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)