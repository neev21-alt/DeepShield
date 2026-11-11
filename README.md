# DeepShield — Explainable AI-Powered Facial Deepfake Detection

> *"See the fake before it fools you."*

**Event:** CodeQuest — National Tech Hackathon 2026  
**Team:** Code Ph@nt0m  
**Strategy:** AI-FIRST Development Lifecycle  

---

## 1. Problem Statement

Hyper-realistic synthetic faces, deepfake swaps, and diffusion-generated portraits increasingly deceive human observers and automated identity verification systems. Conventional deepfake detection systems act as black boxes: they output opaque classification numbers without revealing which visual features or spatial regions triggered the decision. Consequently, investigators cannot verify whether an algorithm spotted authentic digital artifacts or was simply misled by compression noise, lighting variations, or demographic bias.

---

## 2. Proposed Solution

**DeepShield** addresses this opacity through an **Explainable AI (XAI)** framework tailored for facial forensics. The platform inspects uploaded imagery, locates all human faces using MediaPipe BlazeFace, crops them with boundary-preserving padding, classifies each face using a Vision Transformer architecture, generates spatial activation maps using **Grad-CAM**, and synthesizes an evidentiary narrative report.

---

## 3. Main Features

- **Multi-Subject Face Localization:** Detects and segments multiple faces simultaneously with MediaPipe BlazeFace.
- **Context-Aware Padding:** Expands facial crops to capture blending boundaries at the hairline, jaw, and ears.
- **Vision Transformer Inference:** Evaluates fine-grained synthetic patterns and facial discrepancies.
- **Grad-CAM Saliency Maps:** Visual heatmaps showing exactly which facial regions contributed to the classification verdict.
- **Transparent Score Reporting:** Explicitly distinguishes raw model confidence scores from calibrated posterior probabilities.
- **Structured REST API:** Built with FastAPI, strictly typed Pydantic schemas, and CORS compatibility.
- **Prototype Dashboard (Phase 7):** Responsive React client with heatmap overlay controls and forensic inspection cards.

---

## 4. Technology Stack

### Backend
- **Language & Runtime:** Python 3.11 (Python 3.10+ compatible)
- **Web Framework:** FastAPI
- **ASGI Server:** Uvicorn
- **Data Validation:** Pydantic v2
- **Deep Learning:** PyTorch & Hugging Face Transformers
- **Candidate Checkpoint:** `dima806/deepfake_vs_real_image_detection` (evaluation in Phase 2)

### Computer Vision & Explainability
- **Face Localization:** MediaPipe Face Detection (BlazeFace)
- **Image Processing:** OpenCV (headless), Pillow, NumPy
- **Explainable AI:** Grad-CAM adapted for Vision Transformers

### Frontend (Phase 7)
- **Framework:** React 18 / 19
- **Language:** TypeScript
- **Bundler:** Vite
- **Styling:** Tailwind CSS
- **Icons:** Lucide React

### Database (Phase 6 / Future)
- **Engine:** SQLite

---

## 5. Project Directory Structure

```text
DeepShield/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   └── analyze.py
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── model_service.py
│   │   │   ├── face_service.py
│   │   │   ├── image_service.py
│   │   │   ├── gradcam_service.py
│   │   │   └── explanation_service.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── analysis.py
│   │   │
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── image_utils.py
│   │   │
│   │   └── database/
│   │       └── __init__.py
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_health.py
│   │   ├── test_analyze.py
│   │   └── test_utils.py
│   │
│   ├── samples/
│   │   ├── real/
│   │   └── fake/
│   │
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   └── README.md
│
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── SYSTEM_ARCHITECTURE.md
│   └── DEVELOPMENT_ROADMAP.md
│
├── .gitignore
├── README.md
└── LICENSE
```

---

## 6. Current Development Status

- **Active Stage:** **Phase 1: Project Initialization & Foundation**
- **Foundation Status:** Fully initialized. Directory structure, dependency configuration, Pydantic schemas, and FastAPI application created.
- **Endpoints Active:**
  - `GET /` — API root information and phase status (200 OK)
  - `GET /health` — System status, candidate model ID, `model_loaded: false` (200 OK)
  - `POST /analyze` — Clean pipeline stub returning HTTP 501 Not Implemented (no fake data generated)
- **Frontend Status:** Uninitialized per AI-FIRST development strategy; scheduled for Phase 7.

---

## 7. Installation & Setup Instructions

### Prerequisites
- Python 3.11 installed (Python 3.10+ supported)
- Git

### 1. Clone & Enter Project Directory
```bash
git clone <repository-url>
cd DeepShield/backend
```

### 2. Create and Activate Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
```

### 4. Install Dependencies
```bash
# Core API & validation dependencies (lightweight install)
pip install -r requirements.txt
```

---

## 8. Execution Instructions

### Start Backend Development Server
From the `DeepShield/backend` directory:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Interactive API Documentation
Once running, explore the interactive OpenAPI documentation:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### Running the Test Suite
```bash
pytest tests/ -v
```

---

## 9. Future Implementation Phases

1. **Phase 2:** AI Model Verification (`dima806/deepfake_vs_real_image_detection` architecture, labels, license, ViT layer inspection).
2. **Phase 3:** MediaPipe face detection, crop extraction, and context-padding.
3. **Phase 4:** Deepfake classification engine & inference worker.
4. **Phase 5:** Grad-CAM explainability adaptation for Vision Transformer layers.
5. **Phase 6:** Modular FastAPI integration and optional SQLite persistence.
6. **Phase 7:** React frontend client (drag-and-drop, interactive saliency controls).
7. **Phase 8:** Comprehensive testing, benchmark evaluations, and production containerization.

---

## 10. Limitations & Responsible-Use Disclaimer

> **IMPORTANT DISCLAIMER:**  
> DeepShield is an ongoing prototype developed for research and educational purposes as part of CodeQuest 2026. Automated deepfake detection models can yield false positives or false negatives, particularly when evaluated against unencountered generative architectures, heavy social media compression, motion blur, or unusual lighting conditions. Model outputs must be treated as indicative evidentiary cues and forensic aids, not infallible legal determinations. DeepShield must never be employed as the sole basis for punitive, defamatory, or restrictive actions.
