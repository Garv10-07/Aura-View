# file: test_server.py
import uvicorn
from fastapi import FastAPI
import socketio

app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, socketio_path='/ws')
app.mount("/ws", socket_app)

@sio.event
async def connect(sid, environ):
    print(f"✅ Hello! Client connected: {sid}")

print("🚀 Test Server shuru ho raha hai... http://127.0.0.1:8000 par")