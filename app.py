import os
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
YOLO_MODEL_NAME = os.getenv('YOLO_MODEL_NAME', 'yolov5m')
# ✏️ YAHAN APNI YOUTUBE LINK KA CLEAN VERSION DAALO ya WEBCAM KE LIYE 0 RAKHO
INPUT_SOURCE = os.getenv('INPUT_SOURCE', "https://www.youtube.com/live/FyFAqPHBKiQ?si=sOhbuzRCpz7imZp_")
FRAME_SKIP = int(os.getenv('FRAME_SKIP', '5')) # Process 1 out of 5 frames
RESIZE_WIDTH = int(os.getenv('RESIZE_WIDTH', '1280')) # Keeping the larger size for quality
NO_MODEL = os.getenv('NO_MODEL', '0').lower() in ('1', 'true', 'yes')
# --- Path to your cookies file ---
COOKIE_FILE = 'cookies.txt'

# --- GLOBAL SETUP ---
app = fastapi.FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)
app.mount("/socket.io", socket_app) # Use default socket.io path

model = None
latest_frame_for_streaming = None

# --- MODEL LOADING ---
if not NO_MODEL:
    print(f"🧠 AI Model ({YOLO_MODEL_NAME}) Load Ho Raha Hai...")
    try:
        model = torch.hub.load('ultralytics/yolov5', YOLO_MODEL_NAME, pretrained=True, trust_repo=True)
        model.classes = [0] # Sirf 'person' detect karega
        model.conf = 0.5    # Confidence threshold
        print("✅ Model Load Ho Gaya!")
    except Exception as e:
        print(f"⚠️ Warning: Model load karne mein error aaya: {e}. Bina model ke chal raha hai.")
        model = None
else:
    print("⚠️ NO_MODEL mode on hai. AI model load nahin kiya jaayega.")

db_manager = DatabaseManager()
prediction_manager = PredictionManager()
print("✅ Database & Prediction Engine Ready!")

# --- HTML PAGE ROUTE ---
@app.get("/")
async def read_root():
    """Root URL (/) par index.html file serve karega."""
    return FileResponse('index.html')

# --- VIDEO STREAMING ENDPOINT ---
async def frame_generator():
    """Continuously yields the latest processed frame as JPEG bytes."""
    global latest_frame_for_streaming
    while True:
        if latest_frame_for_streaming is not None:
            flag, encodedImage = cv2.imencode(".jpg", latest_frame_for_streaming)
            if flag:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')
        # Adjust delay slightly based on performance
        await asyncio.sleep(0.04)

@app.get("/video_feed")
async def video_feed():
    """Endpoint for the MJPEG video stream."""
    return StreamingResponse(frame_generator(),
                             media_type="multipart/x-mixed-replace;boundary=frame")

# --- BACKGROUND PROCESSING LOOP ---
async def video_processing_loop():
    """Yeh background mein chalta hai, video process karta hai, aur data broadcast karta hai."""
    global latest_frame_for_streaming
    cap = None

    if isinstance(INPUT_SOURCE, str) and ('youtube.com' in INPUT_SOURCE or 'youtu.be' in INPUT_SOURCE):
        try:
            print(f"📡 yt-dlp se YouTube stream ({INPUT_SOURCE}) ka URL nikaal raha hoon...")
            # --- Use cookie file option ---
            ydl_opts = {'format': 'best', 'noplaylist': True, 'cookiefile': COOKIE_FILE}
            # Check if cookie file exists IN THE CURRENT DIRECTORY
            if not os.path.exists(COOKIE_FILE):
                 print(f"⚠️ Warning: Cookie file '{COOKIE_FILE}' nahin mili in the current folder. Authentication fail ho sakti hai.")
                 # Remove cookie option if file not found
                 if 'cookiefile' in ydl_opts: ydl_opts.pop('cookiefile')
            else:
                 print(f"🍪 Using cookie file: {COOKIE_FILE}")
            # ---------------------------
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(INPUT_SOURCE, download=False)
                # Find the best available video format URL
                stream_url = None
                for fmt in info.get('formats', []):
                    # Prefer mp4 if available, otherwise take the first good one with video
                    if fmt.get('vcodec') != 'none': # Ensure it has video
                         if fmt.get('ext') == 'mp4':
                            stream_url = fmt['url']
                            break
                         elif stream_url is None: # Fallback to first available video stream
                            stream_url = fmt['url']
                if not stream_url:
                    raise ValueError("No suitable video stream format found by yt-dlp.")

            print("✅ Direct stream link mil gaya!")
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        except Exception as e:
            print(f"❌ YouTube stream se link nikaalne mein error: {e}")
            return # Stop processing if link extraction fails
    else: # Webcam input
        print("🎥 Input source shuru kiya jaa raha hai...")
        cap = cv2.VideoCapture(INPUT_SOURCE)

    if not cap or not cap.isOpened():
        print("❌ Error: Input source start nahin ho paya.")
        return

    frame_counter = 0
    print("\n🚀 Feed process ho raha hai... Video dashboard par stream hoga.")
    while True:
        try:
            success, frame = cap.read()
            if not success:
                print("⚠️ Frame nahin mil paya, dobara try kar raha hoon...")
                await asyncio.sleep(1)
                # Attempt to reopen stream if possible (simplified retry)
                cap.release()
                await asyncio.sleep(2) # Give some time before retrying
                if isinstance(INPUT_SOURCE, str):
                     try: # Try fetching the URL again
                         ydl_opts = {'format': 'best', 'noplaylist': True, 'cookiefile': COOKIE_FILE}
                         if not os.path.exists(COOKIE_FILE):
                             if 'cookiefile' in ydl_opts: ydl_opts.pop('cookiefile')
                         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                             info = ydl.extract_info(INPUT_SOURCE, download=False)
                             stream_url = info['url']
                         cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                         if not cap or not cap.isOpened(): raise Exception("Failed to reopen")
                         print("✅ Stream reopened successfully.")
                     except Exception as e:
                         print(f"❌ Stream dobara connect nahin ho paya: {e}. Wait kar raha hoon...")
                         await asyncio.sleep(5)
                else:
                    cap = cv2.VideoCapture(INPUT_SOURCE)
                    if not cap or not cap.isOpened():
                        print("❌ Input source dobara connect nahin ho paya. Wait kar raha hoon...")
                        await asyncio.sleep(5)
                continue # Skip rest of loop

            # Resize frame
            h, w, _ = frame.shape
            if w == 0: continue
            scale = RESIZE_WIDTH / w
            resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            frame_counter += 1

            if frame_counter % FRAME_SKIP == 0:
                person_count = 0
                full_prediction = {}
                annotated_frame = resized_frame

                if model is not None:
                    results = model(resized_frame)
                    person_count = len(results.xyxy[0])

                    try:
                        db_manager.save_crowd_data(person_count)
                    except Exception as e:
                        print(f"❌ DB save error: {e}")

                    full_prediction = prediction_manager.get_future_prediction() # Call without args

                    results.render()
                    annotated_frame = results.ims[0].copy()
                    cv2.putText(annotated_frame, f'Live: {person_count}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if not isinstance(full_prediction, dict): full_prediction = {}

                await sio.emit('broadcast_data', {'count': person_count, 'prediction': full_prediction})
                print(f"📡 Broadcast -> Count: {person_count}, Pred(5min): {full_prediction.get('5min', 'N/A')}", end='\r')
                latest_frame_for_streaming = annotated_frame
            else:
                latest_frame_for_streaming = resized_frame

            await asyncio.sleep(0.01) # Small delay to yield control

        except asyncio.CancelledError:
             print("Processing loop cancelled.")
             break # Exit loop if task is cancelled
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            await asyncio.sleep(1) # Wait before retrying after an error

    # Cleanup when loop exits
    if cap:
        cap.release()
    print("\nVideo processing loop finished.")

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

# --- MAIN RUNNER ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)