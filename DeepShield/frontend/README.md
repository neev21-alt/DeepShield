# DeepShield — Frontend Client (Phase 5 Completed)

## Overview
Enterprise React + TypeScript client interface for **DeepShield: Explainable AI-Powered Facial Deepfake Detection**.
Connects seamlessly to the FastAPI backend service providing multi-face detection, Vision Transformer (ViT-B/16) inference, and Grad-CAM visual heatmaps.

## Technology Stack
- **Framework:** React 19 + TypeScript
- **Bundler & Dev Server:** Vite
- **Styling:** Tailwind CSS (Midnight Navy + Electric Blue + Cyan cybersecurity theme)
- **Icons:** Lucide React
- **API Communication:** Native Fetch API with structured TypeScript types matching backend Pydantic schemas

## Features Implemented
1. **Live System Telemetry:** Header badge displaying real-time FastAPI connection status, model loading state, and compute device.
2. **Interactive Dropzone:** Drag-and-drop file ingestion with validation for JPEG/PNG/WEBP and 10MB file size limits.
3. **1-Click Verification Triggers:** Instant sample testing for authentic, manipulated, multi-subject, and no-face imagery.
4. **Multi-Subject Spatial Inspection:** Interactive MediaPipe bounding box overlay map with coordinate tracking.
5. **Score Distribution:** Real vs Fake hypothesis breakdown with uncalibrated probability disclosures.
6. **Grad-CAM Visualizer:** Blended overlay and side-by-side modes with an opacity slider and 14×14 patch grid metadata.
7. **Session History:** In-memory inspection log allowing quick retrieval of previous analyses during an active session.

## Starting the Application

### 1. Start the FastAPI Backend
```bash
cd DeepShield/backend
PORT=8001 python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

### 2. Start the Frontend Dev Server
```bash
# In the frontend directory:
npm run dev

# Or from the project root:
npm run dev
```

The frontend will run on `http://localhost:3000` and automatically proxy API calls to the backend on port 8001.
