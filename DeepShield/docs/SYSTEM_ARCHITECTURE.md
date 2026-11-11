# DeepShield — System Architecture

```
+-----------------------------------------------------------------------------------+
|                                  DeepShield Pipeline                              |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ User Image Upload ]                                                            |
|          │                                                                        |
|          ▼                                                                        |
|  [ File Validation ] ──── Validates format (JPEG/PNG/WEBP) & size (<= 10MB)       |
|          │                                                                        |
|          ▼                                                                        |
|  [ MediaPipe Face Detection ] ──── Detects N faces with confidence thresholds     |
|          │                                                                        |
|          ▼                                                                        |
|  [ Face Cropping & Preprocessing ] ──── Normalized square crops with context pad  |
|          │                                                                        |
|          ▼                                                                        |
|  [ PyTorch Deepfake Classification ] ──── Vision Transformer feature extraction   |
|          │                                 (Candidate: dima806/deepfake_vs_real)  |
|          ▼                                                                        |
|  [ Grad-CAM Visualization ] ──── Token gradient saliency mapping onto face crops  |
|          │                                                                        |
|          ▼                                                                        |
|  [ Metadata-Based Explanation ] ──── Translates activation zones & scores into    |
|          │                           structured narrative summaries               |
|          ▼                                                                        |
|  [ Final Detection Report ] ──── Schema-validated JSON response & visual cards    |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## Architecture Components

### 1. Ingestion & File Validation (`api/analyze.py`, `utils/image_utils.py`)
- Sanitizes file headers and magic numbers (PIL / Pillow).
- Verifies image dimensions (minimum 64x64, maximum 4096x4096).
- Rejects corrupt or unsupported formats early with standard HTTP 400 responses.

### 2. Facial Landmark & Detection Engine (`services/face_service.py`)
- Employs **MediaPipe Face Detection** (BlazeFace architecture).
- Operates on full-resolution input images to detect single or multiple faces.
- Computes normalized bounding boxes `[ymin, xmin, ymax, xmax]` with dynamic aspect-ratio compensation.
- Extracts padded sub-crops preserving facial perimeter context (hairline, jawline, ears) where synthetic blending artifacts predominantly cluster.

### 3. Classification Engine (`services/model_service.py`)
- Candidate Architecture: **Hugging Face Transformers / PyTorch** (`dima806/deepfake_vs_real_image_detection`).
- Preprocessing: Standard ViT normalization (mean/std normalization to 224x224).
- Softmax outputs raw model confidence scores for `real` and `fake` labels.
- Guardrails: Non-calibrated outputs are reported transparently as model scores rather than ground-truth probabilities.

### 4. Explainable Saliency: Grad-CAM (`services/gradcam_service.py`)
- Adapted specifically for Vision Transformer architectures (attention rollout / layer gradient saliency).
- Extracts gradients from the final transformer encoder block with respect to the predicted class logit.
- Maps activations onto the input face crop, applying JET/turbo color palette overlays.

### 5. Metadata-Grounded Explanations (`services/explanation_service.py`)
- Combines model scores, activation focal points (e.g. peri-orbital region, nasal bridge, jaw perimeter), and detection metadata.
- Outputs crisp evidentiary summaries for human reviewers.

### 6. Modular FastAPI Backend (`app/main.py`)
- Asynchronous endpoints with Pydantic schema validation.
- CORS-enabled for decoupling from the future React frontend.
- Standardized error envelopes and telemetry-free responses.
