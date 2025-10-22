# Force pafy to use the modern, working yt-dlp backend
import os
# os.environ['PAFY_BACKEND'] = 'yt-dlp' # No longer needed as pafy is removed

import fastapi
import uvicorn
import socketio
import cv2
import torch
from fastapi.responses import FileResponse, StreamingResponse
import asyncio
import yt_dlp # Use yt-dlp directly
import time

from database_manager import DatabaseManager
from prediction_manager import PredictionManager

# --- CONFIGURATION ---
YOLO_MODEL_NAME = 'yolov5s'  # YOLOv5s model
# Use 0 for webcam or your YouTube link
INPUT_SOURCE = "https://www.youtube.com/live/FyFAqPHBKiQ?si=TD0W1gxHEwGa1xYL" # Example: Webcam
RESIZE_WIDTH = 640 # Width for processing and streaming

# --- GLOBAL SETUP ---
app = fastapi.FastAPI()
# Correct Socket.IO setup for default path
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_app) # Mount at the default path

latest_frame_for_streaming = None # Global variable to hold the latest frame

print(f"🧠 AI Model ({YOLO_MODEL_NAME}) Load Ho Raha Hai...")
try:
    model = torch.hub.load('ultralytics/yolov5', YOLO_MODEL_NAME, pretrained=True, trust_repo=True)
    model.classes = [0]
    model.conf = 0.5
    print("✅ Model Load Ho Gaya!")
except Exception as e:
    print(f"❌ Model load karne mein error: {e}. Program band ho raha hai.")
    exit()

db_manager = DatabaseManager()
prediction_manager = PredictionManager()
print("✅ Database & Prediction Engine Ready!")

# --- HTML PAGE ROUTE ---
@app.get("/")
async def read_root():
    return FileResponse('index.html')

# --- VIDEO STREAMING LOGIC ---
async def frame_generator():
    global latest_frame_for_streaming
    while True:
        if latest_frame_for_streaming is not None:
            flag, encodedImage = cv2.imencode(".jpg", latest_frame_for_streaming)
            if flag:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')
        await asyncio.sleep(0.04) # Slightly adjusted delay

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(frame_generator(),
                             media_type="multipart/x-mixed-replace;boundary=frame")

# --- BACKGROUND PROCESSING LOOP ---
async def video_processing_loop():
    global latest_frame_for_streaming
    cap = None

    if isinstance(INPUT_SOURCE, str) and ('youtube.com' in INPUT_SOURCE or 'youtu.be' in INPUT_SOURCE):
        try:
            print(f"📡 yt-dlp se YouTube stream ({INPUT_SOURCE}) ka URL nikaal raha hoon...")
            ydl_opts = {'format': 'best', 'noplaylist': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(INPUT_SOURCE, download=False)
                stream_url = info['url']
            print("✅ Direct stream link mil gaya!")
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        except Exception as e:
            print(f"❌ YouTube stream se link nikaalne mein error: {e}")
            return
    else: # Webcam input
        print("🎥 Input source shuru kiya jaa raha hai...")
        cap = cv2.VideoCapture(INPUT_SOURCE)

    if not cap or not cap.isOpened():
        print("❌ Error: Input source start nahin ho paya.")
        return

    print("\n🚀 Feed process ho raha hai... Video dashboard par stream hoga.")
    while True:
        success, frame = cap.read()
        if not success:
            await asyncio.sleep(1)
            continue

        try:
            h, w, _ = frame.shape
            scale = RESIZE_WIDTH / w
            resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        except Exception:
            continue

        # AI Processing
        results = model(resized_frame)
        person_count = len(results.xyxy[0])

        # Data Save & Prediction
        db_manager.save_crowd_data(person_count)
        predicted_count = prediction_manager.get_future_prediction()

        # Broadcast data via Socket.IO
        await sio.emit('broadcast_data', {'count': person_count, 'prediction': predicted_count})
        print(f"📡 Data Broadcast -> Count: {person_count}, Prediction: {predicted_count}", end='\r')

        # Prepare frame for streaming (draw boxes)
        results.render()
        annotated_frame = results.ims[0].copy()
        cv2.putText(annotated_frame, f'Live: {person_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        latest_frame_for_streaming = annotated_frame

        await asyncio.sleep(0.05) # Adjust based on processing speed

# --- SERVER STARTUP ---
@app.on_event("startup")
async def startup_event():
    print("🚀 Background processing shuru ho rahi hai...")
    asyncio.create_task(video_processing_loop())

# --- SOCKET.IO EVENTS ---
@sio.event
async def connect(sid, environ):
    print(f"✅ Frontend client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"❌ Frontend client disconnected: {sid}")