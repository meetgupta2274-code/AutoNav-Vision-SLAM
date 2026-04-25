# AI-Based Visual SLAM for Autonomous Navigation

This document serves as the comprehensive technical guide and reference manual for the Final Year Project: **AI-Based Visual SLAM for Autonomous Navigation**.

---

## 1. Project Overview
This project simulates the visual perception and spatial reasoning capabilities of an autonomous robotic unit. Originally prototyped as a local Python script, the architecture has been upgraded to a production-grade, Full-Stack Intelligent Dashboard. 

Instead of full dense 3D Point-Cloud SLAM (which relies on LiDAR or Stereo-Depth Odometry), this system models a **Monocular Semantic SLAM** approach. It interprets a standard 2D camera feed, detects human targets in real-time space, maps the visual plane to a topological grid, and mathematically evaluates the optimal navigation paths toward targets while bypassing hypothetical obstacles.

---

## 2. Technology Stack

### Backend Engine (The AI & Core Logic)
- **Language**: Python 3.x
- **FastAPI**: A modern, high-performance web framework used to serve the REST API endpoints and process real-time WebSocket streams without blocking execution.
- **Uvicorn**: An ASGI web server designed to run concurrent asynchronous processes.
- **OpenCV (`cv2`)**: Used for ingesting the camera streams (Webcam or `.mp4`), manipulating the tensor matrices, and drawing target overlays (bounding boxes, trajectory lines) onto the video frame.
- **Ultralytics**: The official framework powering the YOLOv11 model integration.

### Frontend Dashboard (The Telemetry Interface)
- **Library**: React (Bootstrapped via Vite for lightning-fast compilation).
- **Styling**: Tailwind CSS (v3) utilized to create a sleek, "Glassmorphic" Dark Mode aesthetic suited for an autonomous robot's HUD (Heads Up Display).
- **Icons**: `lucide-react` for the modern dashboard iconography.

---

## 3. Core Algorithms Used

### A. Semantic Perception (YOLOv11)
**You Only Look Once (YOLO)** is a state-of-the-art, real-time object detection algorithm. We use the Nano variant (`yolo11n.pt`) optimized for edge-device speed.
- **Role**: Scans every frame of the video to identify human targets, returning their bounding box coordinates (width/height), center centroids, and a categorical confidence probability ranging from 0.0 to 1.0.

### B. Spatial Grid Partitioning
- **Role**: The raw video dimension (e.g., 1920x1080 resolution) is mathematically divided into two conceptual overlays:
  1. **Sector Matrix (3x3)**: Used for high-level semantic zoning (Labeling sectors `A1` through `C3`).
  2. **Navigation Grid (30x30)**: A dense mathematical occupancy grid. Objects and physical space are quantized into a topological map where traversing between cells costs distance units.

### C. A* (A-Star) Pathfinding Algorithm
The A* algorithm provides theoretical optimal routing across the navigation grid.
- **Role**: Once YOLO identifies a target centroid, the A* algorithm plots the geometry of the route from a fixed `START` node to the target's `GOAL` node.
- **Mechanism**: It operates by minimizing the function `f(n) = g(n) + h(n)`:
  - `g(n)`: The actual cost from the start node to node *n*.
  - `h(n)`: The heuristic (Euclidean/Diagonal distance strategy) estimating the cost from node *n* to the final target.
- It dynamically wraps pathing vectors around simulated physical or designated "Off-limit / Obstacle" zones defined in the topological grid.

### D. Multi-Factor Recommendation Engine
An analytical decision-maker ranking the feasibility of multiple targets if a crowd is detected.
- **Role**: Uses custom weighted normalization to determine the mathematically "Optimal" target to pursue.
- It ranks targets based on combining three parameters:
   1. **`W_CONF (0.6)`**: Heavily weighs the AI's confidence that it's a real person.
   2. **`W_PATH (0.3)`**: Weighs the total steps/cells required by the A* Algorithm to reach the target (Lower ETA = Better).
   3. **`W_DIST (0.1)`**: Weighs the raw straight-line pixel proximity.

---

## 4. System Architecture: How It Works

1. **Initialization (`run.bat`)**:
   When the user runs the launcher, two distinct server environments boot concurrently. The backend API opens on Port `8000`, and the React UI binds to Port `5173`.
   
2. **The Daemonic Processing Loop (Backend)**:
   A dedicated background Python thread immediately begins capturing frames from the active `cv2.VideoCapture` source.
   - For every single frame: It invokes YOLO to detect targets, initiates the Pathfinding sequence towards the highest-ranking target, and physically paints those telemetry visualizations onto the frame payload.

3. **Multi-Protocol Streaming**:
   - **MJPEG Visual Stream**: Through an HTTP Multipart response (`/api/video_feed`), the backend rapidly yielded the annotated frames (as `image/jpeg` byte buffers) directly into the browser's `<img>` tag, generating a smooth video playback experience.
   - **WebSocket Telemetry Stream**: Instead of visual data, a concurrent WebSocket connection (`ws://localhost:8000/ws/telemetry`) pulses structured JSON data matching the current frame at 10Hz.

4. **Synchronized Dashboard (Frontend)**:
   The React dashboard captures the MJPEG visually in the center pane while seamlessly mapping the WebSocket JSON data to the corresponding state variables to render the numeric UI logic (such as Confidence %, Active Target counts, and Path Length metrics) on the fly.
