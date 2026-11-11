# DeepShield — Phase 2 Test Samples & Dataset Provenance

This directory stores legally accessible test images used to verify the Vision Transformer deepfake detection pipeline during Phase 2.

---

## 1. Real Facial Samples (`samples/real/`)

### `grace_hopper_navy_real.jpg`
- **Subject:** Rear Admiral Grace Hopper, pioneering computer scientist.
- **Provenance:** Official United States Navy historical portrait photograph (distributed by PyTorch / torchvision benchmark test assets).
- **Copyright & License:** U.S. Federal Government work — Public Domain.
- **Ground Truth Label:** `REAL` (100% genuine historical photograph, zero generative/synthetic manipulation).
- **Model Verification Output:**
  - Predicted Label: `Real` (Class index 0)
  - Uncalibrated Softmax Score: `Real: 0.999379`, `Fake: 0.000621`
  - Latency: ~317 ms (CPU)

### `sample_real_391.jpg`
- **Provenance:** Hugging Face benchmark dataset `desireemcv/face-real-vs-fake` (Validation split index 391).
- **Ground Truth Dataset Label:** `REAL` (Class label `1` in dataset schema).
- **Model Verification Output:**
  - Predicted Label: `Real` (Class index 0)
  - Uncalibrated Softmax Score: `Real: 0.998971`, `Fake: 0.001029`
  - Latency: ~235 ms (CPU)

---

## 2. Manipulated / Synthetic Samples (`samples/fake/`)

### `sample_fake_3.jpg`
- **Provenance:** Hugging Face benchmark dataset `desireemcv/face-real-vs-fake` (Validation split index 3). Sourced from generative face synthesis (StyleGAN / Deepfake).
- **Ground Truth Dataset Label:** `FAKE` (Class label `0` in dataset schema).
- **Model Verification Output:**
  - Predicted Label: `Fake` (Class index 1)
  - Uncalibrated Softmax Score: `Fake: 0.954842`, `Real: 0.045158`
  - Latency: ~274 ms (CPU)

---

## 3. Critical ML Observations & Transparency

1. **Label Mapping Inversion between Datasets:**
   - In dataset `desireemcv/face-real-vs-fake`: `0 = FAKE`, `1 = REAL`.
   - In model `dima806/deepfake_vs_real_image_detection`: `0 = 'Real'`, `1 = 'Fake'`.
   - **Crucial Rule:** Never assume index `0` represents fake or real. Always verify and bind against `model.config.id2label`.

2. **Concept Drift & Training Horizon:**
   - The candidate model was trained approximately 3 years ago on StyleGAN / CelebA artifacts.
   - On recent diffusion models (e.g. Midjourney v5/v6, FLUX, SDXL), domain drift occurs. The model author recommends threshold adjustments or domain fine-tuning for production.

3. **Score Interpretation:**
   - Softmax outputs represent relative output activations for the two heads.
   - They **MUST NOT** be interpreted or communicated as calibrated posterior probabilities.
