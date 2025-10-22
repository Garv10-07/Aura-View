import torch
import cv2
import socketio
import time
import yt_dlp # Ab hum sirf ise use karenge

from database_manager import DatabaseManager
from prediction_manager import PredictionManager

# --- CONFIGURATION ---
YOLO_MODEL_NAME = 'yolov5s'  # YOLOv5s model
# ✏️ YAHAN APNI YOUTUBE LINK KA CLEAN VERSION DAALO ya WEBCAM KE LIYE 0 RAKHO
INPUT_SOURCE = "https://www.youtube.com/live/FyFAqPHBKiQ?si=TD0W1gxHEwGa1xYL" # Example: Webcam
# INPUT_SOURCE = "https://www.youtube.com/watch?v=F3t_PIC-s_k" # Example: YouTube Link
RESIZE_WIDTH = 640

# --- SETUP ---
print(f"🧠 AI Model ({YOLO_MODEL_NAME}) Load Ho Raha Hai...")
model = torch.hub.load('ultralytics/yolov5', YOLO_MODEL_NAME, pretrained=True, trust_repo=True)
model.classes = [0]
model.conf = 0.5
print("✅ Model Load Ho Gaya!")

db_manager = DatabaseManager()
prediction_manager = PredictionManager()
sio_client = socketio.Client(reconnection_attempts=5, reconnection_delay=3)

def initial_connect():
    try:
        print("⏳ Server se connect karne ki koshish...")
        # Make sure to include the socketio_path if app.py expects it
        sio_client.connect('http://localhost:8000', socketio_path='/socket.io')
        return True
    except socketio.exceptions.ConnectionError as e:
        print(f"❌ Server se connect nahin ho paya: {e}")
        return False

if not initial_connect():
    print("Script band ho raha hai kyunki server on nahin hai.")
    exit()
print("✅ Server se connect ho gaya!")

# --- INPUT SOURCE SETUP ---
cap = None
if isinstance(INPUT_SOURCE, str) and ('youtube.com' in INPUT_SOURCE or 'youtu.be' in INPUT_SOURCE):
    try:
        print(f"📡 yt-dlp se YouTube stream ({INPUT_SOURCE}) ka URL nikaal raha hoon...")
        # Use yt-dlp directly
        ydl_opts = {'format': 'best', 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(INPUT_SOURCE, download=False)
            stream_url = info['url']
        print("✅ Direct stream link mil gaya!")
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG) # FFmpeg use karna zaroori hai
    except Exception as e:
        print(f"❌ YouTube stream se link nikaalne mein error: {e}")
        exit()
else:
    print("🎥 Input source shuru kiya jaa raha hai...")
    cap = cv2.VideoCapture(INPUT_SOURCE)

# --- MAIN LOOP ---
if not cap or not cap.isOpened():
    print("❌ Error: Input source start nahin ho paya.")
else:
    print("\n🚀 Feed process ho raha hai... (OpenCV window khulegi). 'q' dabakar band karein.")
    while True:
        if not sio_client.connected:
            print("⚠️ Server se connection toot gaya! Dobara connect karne ki koshish...")
            initial_connect()
            time.sleep(5)
            continue

        success, frame = cap.read()
        if not success:
            print("⚠️ Frame nahin mil paya, agla try kar raha hoon...")
            time.sleep(1) # Thoda wait karo
            continue

        try:
            h, w, _ = frame.shape
            scale = RESIZE_WIDTH / w
            resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        except Exception as e:
            print(f"Frame resize error: {e}")
            continue

        results = model(resized_frame)
        person_count = len(results.xyxy[0])
        db_manager.save_crowd_data(person_count)
        predicted_count = prediction_manager.get_future_prediction()

        try:
             sio_client.emit('update_data', {'count': person_count, 'prediction': predicted_count})
             print(f"📡 Data Bheja Gaya -> Count: {person_count}, Prediction: {predicted_count}", end='\r')
        except socketio.exceptions.BadNamespaceError:
             print("Error: Namespace disconnected before emit. Trying to reconnect...", end='\r')
             # The loop will handle reconnection on the next iteration
        except Exception as e:
            print(f"Socket emit error: {e}")


        # --- YEH LINE WINDOW DIKHAYEGI ---
        results.render()
        annotated_frame = results.ims[0].copy()
        cv2.putText(annotated_frame, f'Live Count: {person_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("AuraView - Processing Feed", annotated_frame)
        # ----------------------------------

        # 'q' dabane par band hoga
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if sio_client.connected:
    sio_client.disconnect()

# Cleanup
if cap:
    cap.release()
cv2.destroyAllWindows()
print("\n✅ Script band ho gaya hai.")