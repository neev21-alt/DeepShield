# DeepShield — Development Roadmap

**Development Strategy:** AI-FIRST Engineering Lifecycle  
**Project:** CodeQuest — National Tech Hackathon 2026  
**Team:** Code Ph@nt0m  

---

## Phase Overview & Progress Matrix

| Phase | Description | Status | Target Deliverables |
| :--- | :--- | :---: | :--- |
| **Phase 1** | **Project Initialization & Foundation** | **COMPLETE** | Project layout, config, schemas, FastAPI shell, health check, 501 analyze stub, tests. |
| **Phase 2** | **AI Model Verification & Audit** | Upcoming | Inspect `dima806/deepfake_vs_real_image_detection`, label maps, ViT layers, license, and Grad-CAM viability. |
| **Phase 3** | **Face Detection & Preprocessing** | Planned | MediaPipe BlazeFace integration, multi-face handling, bounding boxes, aspect-ratio padding. |
| **Phase 4** | **Deepfake Classification Engine** | Planned | PyTorch/Transformers inference pipeline, score extraction, threshold calibrations. |
| **Phase 5** | **Grad-CAM Explainability System** | Planned | Transformer-compatible Grad-CAM activation mapping, heatmap generation, OpenCV colormaps. |
| **Phase 6** | **Modular FastAPI Integration** | Planned | Wire services into POST `/analyze`, image upload validation, error responses, database history (SQLite). |
| **Phase 7** | **React Frontend Client** | Planned | React 18/19 + Vite + TypeScript + Tailwind CSS UI: dropzone, face selector, heatmap toggle, report export. |
| **Phase 8** | **Integration, Testing & Deployment** | Planned | End-to-end integration tests, load tests, Docker packaging, deployment documentation. |

---

## Detailed Phase Breakdown

### Phase 1: Project Initialization & Foundation (Current Phase)
- Establish strict directory tree: `backend/app/`, `docs/`, `samples/`, `tests/`.
- Provide environment configuration templates (`.env.example`) and comprehensive `.gitignore`.
- Implement `main.py`, `config.py`, GET `/health`, and POST `/analyze` (501 stub).
- Define Pydantic response and error schemas.
- Set up unit and integration test foundation with `pytest` and `httpx`.

### Phase 2: AI Model Verification & Audit
- Inspect candidate model `dima806/deepfake_vs_real_image_detection`.
- Verify input resolution requirements, normalization parameters, and class ID mapping (`0: Real, 1: Fake` or vice versa).
- Evaluate ViT self-attention blocks for gradient hook extraction.

### Phase 3: Face Detection and Preprocessing Pipeline
- Integrate MediaPipe Face Detection with configurable detection confidence.
- Build multi-face coordinator capable of identifying primary and background subjects.
- Apply contextual boundary padding (15-25%) to preserve hairline and jaw manipulation seams.

### Phase 4: Classification Engine
- Construct isolated inference worker using PyTorch.
- Extract raw model confidence and comparative class scores.
- Document transparency measures (reporting scores without false claims of calibrated posterior probability).

### Phase 5: Grad-CAM Explainability
- Implement gradient extraction on last transformer encoder block.
- Produce spatial heatmap matching input face crop coordinates.
- Render alpha-blended visualization overlays.

### Phase 6: Modular FastAPI Integration
- Transition POST `/analyze` from 501 stub to full orchestrator pipeline.
- Connect `face_service`, `model_service`, and `gradcam_service`.
- Introduce optional SQLite audit storage for historical scans.

### Phase 7: React Frontend Client
- Initialize React + TypeScript application with Vite in `frontend/`.
- Build drag-and-drop file ingestion, face-focus inspector, and interactive Grad-CAM opacity sliders.
- Connect to backend API via typed API client.

### Phase 8: Testing, Hardening & Deployment
- Comprehensive test coverage on diverse benchmark datasets.
- Adversarial robustness testing against JPEG compression and blurring.
- Production containerization with Uvicorn and Gunicorn workers.
