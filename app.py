import os
import fastapi
import uvicorn
import socketio
import cv2
import torch
from fastapi.responses import FileResponse, StreamingResponse
import asyncio
import yt_dlp # Ab hum sirf ise use karenge
import time

from database_manager import DatabaseManager
from prediction_manager import PredictionManager

# --- CONFIGURATION ---
YOLO_MODEL_NAME = os.getenv('YOLO_MODEL_NAME', 'yolov5m')
# ✏ YAHAN APNI YOUTUBE LINK KA CLEAN VERSION DAALO ya WEBCAM KE LIYE 0 RAKHO
INPUT_SOURCE = os.getenv('INPUT_SOURCE', "https://www.youtube.com/watch?v=FyFAqPHBKiQ")
FRAME_SKIP = int(os.getenv('FRAME_SKIP', '5')) # Process 1 out of 5 frames
RESIZE_WIDTH = int(os.getenv('RESIZE_WIDTH', '1280')) # Keeping the larger size for quality
NO_MODEL = os.getenv('NO_MODEL', '0').lower() in ('1', 'true', 'yes')
# --- Cookie file ka path (issi folder mein honi chahiye) ---
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
        print(f"⚠ Warning: Model load karne mein error aaya: {e}. Bina model ke chal raha hai.")
        model = None
else:
    print("⚠ NO_MODEL mode on hai. AI model load nahin kiya jaayega.")

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
            (flag, encodedImage) = cv2.imencode(".jpg", latest_frame_for_streaming)
            if flag:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')
        # Delay ko performance ke hisab se adjust karein
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
            
            # --- Cookie file ka logic ---
            ydl_opts = {'format': 'best', 'noplaylist': True, 'cookiefile': COOKIE_FILE}
            if not os.path.exists(COOKIE_FILE):
                 print(f"⚠ Warning: Cookie file '{COOKIE_FILE}' nahin mili. Authentication fail ho sakti hai.")
                 if 'cookiefile' in ydl_opts: ydl_opts.pop('cookiefile') # Agar file nahin hai to option hata do
            else:
                 print(f"🍪 Cookie file '{COOKIE_FILE}' ka istemal kiya jaa raha hai.")
            # ---------------------------

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(INPUT_SOURCE, download=False)
                # Best available video format URL dhoondho
                stream_url = None
                for fmt in info.get('formats', []):
                    if fmt.get('vcodec') != 'none': # Sunishchit karo ki video hai
                         if fmt.get('ext') == 'mp4':
                            stream_url = fmt['url']
                            break
                         elif stream_url is None: # Pehla available video stream use karo
                            stream_url = fmt['url']
                if not stream_url:
                    raise ValueError("No suitable video stream format found by yt-dlp.")

            print("✅ Direct stream link mil gaya!")
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        except Exception as e:
            print(f"❌ YouTube stream se link nikaalne mein error: {e}")
            return # Agar link nikaalne mein error aaye to loop shuru hi mat karo
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
                print("⚠ Frame nahin mil paya, dobara try kar raha hoon...")
                await asyncio.sleep(1)
                # Stream tootne par dobara connect karne ki koshish (simplified retry)
                cap.release()
                await asyncio.sleep(2) # Thoda wait karo
                if isinstance(INPUT_SOURCE, str): # Agar YouTube link hai
                     try: 
                         ydl_opts = {'format': 'best', 'noplaylist': True, 'cookiefile': COOKIE_FILE}
                         if not os.path.exists(COOKIE_FILE):
                             if 'cookiefile' in ydl_opts: ydl_opts.pop('cookiefile')
                         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                             info = ydl.extract_info(INPUT_SOURCE, download=False)
                             stream_url = info['url']
                         cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                         if not cap or not cap.isOpened(): raise Exception("Failed to reopen")
                         print("✅ Stream dobara open ho gaya.")
                     except Exception as e:
                         print(f"❌ Stream dobara connect nahin ho paya: {e}. Wait kar raha hoon...")
                         await asyncio.sleep(5)
                else: # Agar webcam hai
                    cap = cv2.VideoCapture(INPUT_SOURCE)
                    if not cap or not cap.isOpened():
                        print("❌ Input source dobara connect nahin ho paya. Wait kar raha hoon...")
                        await asyncio.sleep(5)
                continue # Loop ka agla iteration try karo

            # Frame ko resize karo
            h, w, _ = frame.shape
            if w == 0: continue
            scale = RESIZE_WIDTH / w
            resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            frame_counter += 1

            # --- FRAME SKIPPING LOGIC ---
            if frame_counter % FRAME_SKIP == 0:
                # Is frame ko process karo
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

                    full_prediction = prediction_manager.get_future_prediction() # Bina argument ke call karo

                    results.render()
                    annotated_frame = results.ims[0].copy()
                    cv2.putText(annotated_frame, f'Live: {person_count}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if not isinstance(full_prediction, dict): full_prediction = {}

                # Data broadcast karo
                await sio.emit('broadcast_data', {'count': person_count, 'prediction': full_prediction})
                print(f"📡 Broadcast -> Count: {person_count}, Pred(5min): {full_prediction.get('5min', 'N/A')}", end='\r')
                latest_frame_for_streaming = annotated_frame
            else:
                # Jin frames ko skip kiya, unhe bhi display ke liye update karo
                latest_frame_for_streaming = resized_frame
            # ------------------------------------

            await asyncio.sleep(0.01) # Loop ko thoda sa aaram do

        except asyncio.CancelledError:
             print("Processing loop cancelled.")
             break # Agar task cancel ho to loop se bahar
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            await asyncio.sleep(1) # Error ke baad thoda wait karo

    # Loop khatam hone par cleanup
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
if _name_ == "_main_":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)