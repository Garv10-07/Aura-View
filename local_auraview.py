import torch
import cv2
import yt_dlp # Ab hum sirf ise use karenge
import time

# --- CONFIGURATION ---
YOLO_MODEL_NAME = 'yolov5m' 
# ✏️ YAHAN APNI YOUTUBE LINK KA CLEAN VERSION DAALO ya WEBCAM KE LIYE 0 RAKHO
INPUT_SOURCE = "https://www.youtube.com/watch?v=FyFAqPHBKiQ" 
RESIZE_WIDTH = 720
FRAME_SKIP = 2     

# --- SETUP ---
print(f"🧠 AI Model ({YOLO_MODEL_NAME}) Load Ho Raha Hai...")
try:
    model = torch.hub.load('ultralytics/yolov5', YOLO_MODEL_NAME, pretrained=True, trust_repo=True)
    model.classes = [0] 
    model.conf = 0.5    
    print("✅ Model Load Ho Gaya!")
except Exception as e:
    print(f"❌ Model load karne mein error: {e}")
    exit()

# --- INPUT SOURCE SETUP ---
cap = None
if isinstance(INPUT_SOURCE, str) and ('youtube.com' in INPUT_SOURCE or 'youtu.be' in INPUT_SOURCE):
    try:
        print(f"📡 yt-dlp se YouTube stream ({INPUT_SOURCE}) ka URL nikaal raha hoon...")
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
    print("🎥 Webcam shuru kiya jaa raha hai...")
    cap = cv2.VideoCapture(INPUT_SOURCE)

# --- MAIN LOOP ---
frame_counter = 0
if not cap or not cap.isOpened():
    print("❌ Error: Input source start nahin ho paya.")
else:
    print("\n🚀 Feed shuru ho gaya hai! 'q' dabakar band karein.")
    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Frame nahin mil paya, agla try kar raha hoon...")
            time.sleep(1)
            continue

        frame_counter += 1
        
        # Frame Skipping Logic
        if frame_counter % FRAME_SKIP != 0:
            cv2.imshow("AuraView - Local Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
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
        
        # Video Display
        results.render()
        annotated_frame = results.ims[0].copy()
        cv2.putText(annotated_frame, f'Live People Count: {person_count}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("AuraView - Local Feed", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("✅ Feed band ho gaya hai.")