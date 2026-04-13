import asyncio
import cv2
import json
import base64
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import threading
import time

from src.perception import Perception
from src.planning import Planner
from src.visualization import Visualizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

perception = Perception()
planner = Planner()
visualizer = Visualizer(planner)

# Global shared states
current_cap = None
current_telemetry = {}
latest_frame_bytes = None
active_websockets = []
video_lock = threading.Lock()

# Indicates whether the current source is a static image or a video stream
static_image_mode = False

def processing_loop():
    global current_cap, current_telemetry, latest_frame_bytes, static_image_mode
    
    # Initialize capturing outside the lock to prevent deadlocking if it hangs
    test_path = "Jupyter-Notebook/test.mp4"
    if not os.path.exists(test_path):
        test_path = "Jupyter-Notebook/test5.png"
    
    if os.path.exists(test_path):
        init_cap = cv2.VideoCapture(test_path)
        img_mode = test_path.endswith('.png') or test_path.endswith('.jpg')
    else:
        init_cap = cv2.VideoCapture(0)
        img_mode = False
        
    with video_lock:
        current_cap = init_cap
        static_image_mode = img_mode

    while True:
        # Snap a reference to the active cap to not lock it up 
        with video_lock:
            cap = current_cap
            is_image = static_image_mode
            
        if cap is None:
            time.sleep(0.1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

        # Process Frame
        h, w = frame.shape[:2]
        start_px = (int(0.05 * w), int(0.05 * h))
        
        persons = perception.detect_persons(frame)
        occ = planner.build_occupancy(h, w, obstacle_mask=None)
        
        # Compute Recommendations
        recs = planner.compute_recommendations(persons, start_px, occ, w, h)
        
        # Annotate
        annotated_frame = visualizer.annotate_frame(frame.copy(), persons, recs, start_px, occ)
        
        # Update current telemetry for websockets
        if recs:
            best = recs[0]
            current_telemetry = {
                "active_target": best["person"].get("sector", "N/A"),
                "confidence": round(best["person"]["conf"], 2),
                "path_len": best["path_len"],
                "distance_px": round(best["dist_px"], 2),
                "score": round(best["score"], 2),
                "targets_detected": len(persons)
            }
        else:
            current_telemetry = {
                "active_target": "None",
                "confidence": 0,
                "path_len": 0,
                "distance_px": 0,
                "score": 0,
                "targets_detected": len(persons)
            }
            
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if ret:
            latest_frame_bytes = buffer.tobytes()
        
        # If it's an image, wait a little to avoid 100% CPU on loop
        if is_image:
            time.sleep(0.1)

# Start background thread immediately
bg_thread = threading.Thread(target=processing_loop, daemon=True)
bg_thread.start()

async def get_video_stream():
    global latest_frame_bytes
    while True:
        if latest_frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_bytes + b'\r\n')
        await asyncio.sleep(0.05) # max 20fps for browser stream if inference is faster

@app.get("/api/video_feed")
def video_feed():
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    global current_cap, static_image_mode
    file_location = f"temp_{file.filename}"
    
    # Read files in chunks to prevent OOM for large videos
    with open(file_location, "wb+") as file_object:
        while True:
            chunk = await file.read(1024 * 1024 * 5) # 5MB chunks
            if not chunk:
                break
            file_object.write(chunk)
    
    # Initialize the video outside the lock, because it can block during FFmpeg probe
    new_cap = cv2.VideoCapture(file_location)
    is_img = file.filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    
    with video_lock:
        old_cap = current_cap
        current_cap = new_cap
        static_image_mode = is_img
        
    if old_cap is not None:
        old_cap.release()
        
    return {"info": f"file '{file.filename}' processed"}

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await websocket.send_json(current_telemetry)
            await asyncio.sleep(0.1) # 10 Hz refresh
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
