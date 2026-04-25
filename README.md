# AutoNav Vision SLAM

![Banner/Demo Screenshot Placeholder](https://via.placeholder.com/1200x400?text=AutoNav+Vision+SLAM+Dashboard)

> **A Full-Stack Monocular Visual SLAM Dashboard for Autonomous Navigation, powered by YOLOv11 and A* Pathfinding.**

## 📖 Overview
This project simulates the visual perception and spatial reasoning capabilities of an autonomous robotic unit. Upgraded from a local Python script into a **Production-Grade Intelligent Web Dashboard**, this system models a **Monocular Semantic SLAM** approach. 

It ingests standard 2D camera feeds, detects subjects in real-time, projects the visual plane onto a topological coordinate grid, and mathematically computes optimal navigational trajectories around obstacles.

## 🚀 Key Features
* **Real-Time Video Inference**: Ingests Live Webcams or pre-recorded `.mp4` payloads seamlessly.
* **Semantic Perception**: Uses Ultralytics YOLOv11 Nano for edge-optimized, real-time object tracking.
* **Topological A* Routing**: Projects physical space into a 30x30 dense matrix, dynamically computing `f(n) = g(n) + h(n)` vectors to target centroids.
* **Multi-Factor Decision Engine**: Autonomously ranks and targets subjects based on normalized Confidences, Pixel Distances, and Path Trajectories.
* **Telemetry Dashboard**: A React + Vite UI utilizing Glassmorphic aesthetics, consuming backend MJPEG streams and WebSocket data feeds at 10Hz.

---

## 🛠️ Technology Stack

| Architecture Layer | Technologies Used |
| :--- | :--- |
| **Backend API** | Python 3, FastAPI, Uvicorn |
| **Computer Vision** | OpenCV (`cv2`), Ultralytics YOLOv11 |
| **Frontend UI** | React, Vite, Tailwind CSS v3, Lucide-React |
| **Networking** | HTTP Multipart (MJPEG), WebSockets |

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.9+
- Node.js v18+

### 1. Installation
Clone the repository and install the backend and frontend dependencies:

```bash
# Clone the repo
git clone https://github.com/yourusername/AutoNav-Vision-SLAM.git
cd AutoNav-Vision-SLAM

# Backend Setup
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt

# Frontend Setup
cd frontend
npm install
```

### 2. Running the System
For Windows users, simply execute the included batch script to spin up both servers concurrently:
```bash
./run.bat
```

Alternatively, you can manually start the services:
- **Backend**: `cd backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- **Frontend**: `cd frontend && npm run dev`

Navigate your browser to `http://localhost:5173` to access the navigation control UI.

---

## 🧠 System Architecture

1. **Daemonic Processing**: A dedicated background Python thread rapidly ingests frames from the `cv2.VideoCapture` source natively.
2. **Analysis Pipeline**: YOLO detects bounding boxes, while the Grid-Partition system constructs A* trajectory lines. These UI elements are painted onto the frame bytes locally.
3. **Stream Protocol**: Annotated frames are broadcasted over `multipart/x-mixed-replace` endpoints. Concurrently, mathematical data (ETA, Confidence arrays) are dispatched over `/ws/telemetry`.
4. **Reactive Rendering**: The React frontend maps incoming WebSocket integers to dynamic components, syncing perfectly with the inline video player.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! 
Feel free to check [issues page](https://github.com/yourusername/AutoNav-Vision-SLAM/issues).

## 📄 License
This project is open-sourced under the MIT License.
