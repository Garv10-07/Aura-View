# file: test_client.py
import socketio
import time

sio_client = socketio.Client()

print("⏳ Server se connect karne ki koshish...")

try:
    sio_client.connect('http://localhost:8000', socketio_path='/ws')
    print("✅✅✅ Hurray! Server se connect ho gaya!")
    time.sleep(5) # 5 second tak connected raho
    sio_client.disconnect()
    print("👋 Connection band kar diya.")

except socketio.exceptions.ConnectionError as e:
    print("❌❌❌ Connection Fail Ho Gaya. Problem abhi bhi hai.")