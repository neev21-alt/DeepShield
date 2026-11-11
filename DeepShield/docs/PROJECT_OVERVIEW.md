# DeepShield — Project Overview

> **Tagline:** "See the fake before it fools you."  
> **Event:** CodeQuest — National Tech Hackathon 2026  
> **Team:** Code Ph@nt0m  

---

## 1. Executive Summary

**DeepShield** is an Explainable Artificial Intelligence (XAI) platform engineered to detect manipulated facial imagery, generated synthetic portraits, and deepfake media. Rather than operating as an opaque "black-box" classifier, DeepShield pairs high-accuracy vision transformer inference with visual saliency heatmaps (Grad-CAM) and plain-language metadata-grounded explanations.

---

## 2. Problem Statement

Hyper-realistic digital manipulation, neural face swaps, and diffusion-generated portraits present unprecedented challenges across cybersecurity, journalism, digital identity verification, and legal integrity. Modern deepfakes frequently exploit subtle blending boundaries, eye reflection discrepancies, unnatural skin textures, and geometric warping that elude casual human inspection. Existing commercial tools suffer from three acute failure modes:
1. **Opaque Decisions:** Providing binary probability scores without spatial or causal justification.
2. **Fragility to Face Multiplicity:** Failing or crashing when an image contains multiple subjects or background crowds.
3. **Overconfidence:** Presenting uncalibrated model scores as definitive empirical facts.

---

## 3. Target Users

- **Digital Forensics & Incident Response (DFIR) Analysts:** Investigating digital evidence and identity theft incidents.
- **Journalists & Fact-Checking Organizations:** Verifying user-submitted media in breaking news cycles.
- **Identity Verification & KYC Platforms:** Pre-screening profile imagery for synthetic face injections.
- **General Public & Content Creators:** Checking suspicious media before distribution.

---

## 4. Core Workflow

1. **Secure Ingestion & Validation:** Image submitted via multipart payload, validated for dimensions, MIME type, and size limits (default 10 MB).
2. **Face Localization & Normalization:** MediaPipe facial landmark detector identifies all faces, isolates bounding coordinates, and extracts aligned crops.
3. **Deepfake Classification:** PyTorch Vision Transformer analyzes cropped facial regions to produce raw logits for real vs. manipulated classes.
4. **Grad-CAM Explainability:** Saliency gradients backpropagated through transformer attention layers to generate visual activation heatmaps highlighting anomalous regions (e.g. eyes, mouth, blending borders).
5. **Metadata-Grounded Narrative:** Structured findings synthesized into clear, intelligible summaries for non-technical users.
6. **Unified Detection Dossier:** Cohesive JSON payload and visual dashboard presenting per-face verdicts, scores, heatmaps, and evidentiary notes.

---

## 5. Expected Outputs

- **Status & Face Count:** Detection status (`success`, `no_faces_detected`, `error`) and number of faces analyzed.
- **Per-Face Classification:** Bounding boxes, predicted class (`likely_real`, `likely_fake`, `uncertain`), and model score breakdown.
- **Visual Heatmap:** Overlay highlighting specific spatial regions contributing most to the classification decision.
- **Explainability Summary:** Structured observations detailing artifact cues (e.g., boundary blur, iris distortion, asymmetric reflections).
