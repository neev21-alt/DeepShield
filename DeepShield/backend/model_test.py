#!/usr/bin/env python3
"""
DeepShield Model Testing Script (Phase 2 Verification).

Command-line utility to test Vision Transformer inference on arbitrary facial images.
Audits predicted labels, raw uncalibrated softmax scores, and execution latency.

Usage:
    python model_test.py <path_to_image> [--device cpu|cuda]

Example:
    python model_test.py samples/real/grace_hopper_navy_real.jpg
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure backend package is in python search path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from app.services.model_service import (
    ModelService,
    InvalidImageError,
    ModelServiceError,
    model_service,
)


def run_model_test(image_path_str: str, device: str = None) -> int:
    """
    Execute model verification test on the provided image path.
    Returns 0 on success, 1 on error.
    """
    image_path = Path(image_path_str)

    print("=" * 70)
    print("DEEPSHIELD — AI MODEL VERIFICATION CLI (PHASE 2)")
    print("=" * 70)

    # 1. Path existence check
    if not image_path.exists():
        print(f"[ERROR] Image path does not exist: {image_path}", file=sys.stderr)
        return 1

    if not image_path.is_file():
        print(f"[ERROR] Specified path is not a file: {image_path}", file=sys.stderr)
        return 1

    print(f"Target Image:      {image_path.resolve()}")
    print(f"File Size:         {os.path.getsize(image_path):,} bytes")

    # 2. Model service configuration
    if device:
        model_service._device = device

    print(f"Candidate Model:   {model_service.model_name}")
    print(f"Target Device:     {model_service.device}")

    # 3. Model Loading
    print("\n[1/3] Loading Model & Image Processor...")
    if not model_service.is_loaded:
        success = model_service.load_model()
        if not success:
            print(f"[ERROR] Failed to load model weights: {model_service.load_error}", file=sys.stderr)
            return 1

    print("      Model Loaded Successfully.")
    print(f"      Architecture:     {model_service.architecture}")
    print(f"      Total Parameters: {model_service.total_parameters:,}")
    print(f"      Verified id2label: {model_service.id2label}")
    print(f"      Verified label2id: {model_service.label2id}")
    print(f"      Input Resolution: {model_service.image_size['width']}x{model_service.image_size['height']} RGB")
    print(f"      Normalization:    mean={model_service.normalization_params.get('mean')}, std={model_service.normalization_params.get('std')}")

    # 4. Preprocessing & Inference
    print("\n[2/3] Preprocessing Image & Running Inference...")
    try:
        result = model_service.predict(image_path)
    except InvalidImageError as e:
        print(f"[ERROR] Invalid or corrupted image file: {e}", file=sys.stderr)
        return 1
    except ModelServiceError as e:
        print(f"[ERROR] Model service failure: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected inference error: {e}", file=sys.stderr)
        return 1

    # 5. Display Structured Results
    print("\n[3/3] Inference Results:")
    print("-" * 70)
    print(f"  Predicted Label:   {result['predicted_label']}")
    print(f"  Semantic Verdict:  {result['verdict']}")
    print(f"  Confidence:        {result['confidence']:.6f} ({result['confidence']*100:.2f}%)")
    print("  Raw Class Scores:")
    for label, score in result["raw_scores"].items():
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    - {label:<8}: {score:8.6f} |{bar}|")
    print(f"  Inference Time:    {result['inference_time_ms']:.2f} ms")
    print(f"  Model Identifier:  {result['model_identifier']}")
    print(f"  Score Type:        {result['score_type']}")
    print("-" * 70)
    print("  * Note: Scores are uncalibrated softmax confidence outputs derived from")
    print("    Vision Transformer classification logits. They are not calibrated probabilities.")
    print("=" * 70)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepShield Phase 2 — Facial Deepfake Model Inference Tester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_path", help="Path to input facial image (JPEG, PNG, WebP)")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Compute device for inference (defaults to auto-detected GPU if present, else CPU)",
    )

    args = parser.parse_args()
    exit_code = run_model_test(args.image_path, args.device)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
