"""
DeepShield Model Management & Deepfake Classification Service (Phase 2).

Integrates, verifies, and manages the pretrained Vision Transformer (ViT)
facial deepfake classification model: `dima806/deepfake_vs_real_image_detection`.

Architecture:
- ViTForImageClassification (Google ViT base patch16 224 fine-tuned on face forensics)
- Input: 224x224 RGB image normalized with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5]
- Output classes: 2 ('Real' vs 'Fake')

CRITICAL LABEL MAPPING (Verified from checkpoint configuration):
- Class index 0: 'Real'
- Class index 1: 'Fake'

Transparency Notice:
- Model output scores are uncalibrated softmax values derived from classification logits.
- They MUST NOT be presented as empirical posterior probabilities.
"""

import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForImageClassification,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    AutoConfig = None
    AutoImageProcessor = None
    AutoModelForImageClassification = None
    PreTrainedModel = Any  # type: ignore
    PreTrainedTokenizerBase = Any  # type: ignore
    TORCH_AVAILABLE = False

from app.config import settings

logger = logging.getLogger("deepshield.model_service")


class ModelServiceError(Exception):
    """Base exception for model service failures."""
    pass


class ModelNotLoadedError(ModelServiceError):
    """Raised when inference is attempted before the model is loaded."""
    pass


class InvalidImageError(ModelServiceError):
    """Raised when an input image cannot be decoded, is corrupted, or has invalid dimensions."""
    pass


class ModelService:
    """
    Manages Vision Transformer model lifecycle, device allocation,
    tensor preprocessing, and classification inference.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name: str = model_name or settings.model_name
        self._model: Optional[PreTrainedModel] = None
        self._processor: Optional[Any] = None
        self._config: Optional[Any] = None
        self._is_loaded: bool = False
        self._load_error: Optional[str] = None
        self._device: str = "cuda" if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
        
        # Checkpoint verified properties (populated upon load)
        self._architecture: Optional[str] = None
        self._model_type: Optional[str] = None
        self._id2label: Dict[int, str] = {}
        self._label2id: Dict[str, int] = {}
        self._num_labels: int = 0
        self._image_size: Dict[str, int] = {"height": 224, "width": 224}
        self._normalization_params: Dict[str, Any] = {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        }
        self._total_parameters: int = 0

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Indicate whether model weights are resident in memory."""
        return self._is_loaded

    @property
    def load_error(self) -> Optional[str]:
        """Return the last loading error message if load failed."""
        return self._load_error

    @property
    def device(self) -> str:
        """Return the active compute device ('cpu' or 'cuda')."""
        return self._device

    @property
    def architecture(self) -> Optional[str]:
        """Return the verified model architecture class name."""
        return self._architecture

    @property
    def model_type(self) -> Optional[str]:
        """Return the model type (e.g. 'vit')."""
        return self._model_type

    @property
    def id2label(self) -> Dict[int, str]:
        """Return the verified mapping from class index to label string."""
        return dict(self._id2label)

    @property
    def label2id(self) -> Dict[str, int]:
        """Return the verified mapping from label string to class index."""
        return dict(self._label2id)

    @property
    def image_size(self) -> Dict[str, int]:
        """Return expected input spatial dimensions."""
        return dict(self._image_size)

    @property
    def normalization_params(self) -> Dict[str, Any]:
        """Return processor normalization mean and std."""
        return dict(self._normalization_params)

    @property
    def total_parameters(self) -> int:
        """Return total parameter count in the neural network."""
        return self._total_parameters

    # -------------------------------------------------------------------------
    # Lifecycle & Loading
    # -------------------------------------------------------------------------

    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the Vision Transformer model and image processor from Hugging Face.
        
        Requirements met:
        - Loads Hugging Face checkpoint.
        - Loads corresponding image processor.
        - Uses AutoModelForImageClassification.
        - Sets model to evaluation mode (`eval()`).
        - Supports CPU inference and detects GPU if present.
        - Gracefully handles loading failures.
        - Avoids redundant reloading if already in memory.
        
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if self._is_loaded and not force_reload:
            logger.debug("Model '%s' already loaded in memory.", self.model_name)
            return True

        global torch, AutoConfig, AutoImageProcessor, AutoModelForImageClassification, TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            try:
                import torch
                from transformers import (
                    AutoConfig,
                    AutoImageProcessor,
                    AutoModelForImageClassification,
                )
                TORCH_AVAILABLE = True
                self._device = "cuda" if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
            except ImportError as err:
                self._is_loaded = False
                self._load_error = f"Dependencies not yet ready: {err}"
                logger.warning(self._load_error)
                return False

        logger.info("Initializing model load for '%s' on device '%s'...", self.model_name, self._device)
        self._load_error = None

        try:
            # 1. Load Configuration
            config = AutoConfig.from_pretrained(self.model_name)
            self._config = config
            self._model_type = getattr(config, "model_type", "vit")
            self._num_labels = getattr(config, "num_labels", 2)

            # CRITICAL: Verify label mappings from checkpoint
            raw_id2label = getattr(config, "id2label", {0: "Real", 1: "Fake"})
            # Ensure integer keys
            self._id2label = {int(k): str(v) for k, v in raw_id2label.items()}
            raw_label2id = getattr(config, "label2id", {"Real": 0, "Fake": 1})
            self._label2id = {str(k): int(v) for k, v in raw_label2id.items()}

            logger.info("Verified checkpoint label mapping: id2label=%s", self._id2label)

            # 2. Load Image Processor
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            # Extract processor spatial size and normalization
            proc_size = getattr(self._processor, "size", {})
            if isinstance(proc_size, dict):
                h = proc_size.get("height", 224)
                w = proc_size.get("width", 224)
                self._image_size = {"height": h, "width": w}
            elif isinstance(proc_size, int):
                self._image_size = {"height": proc_size, "width": proc_size}

            self._normalization_params = {
                "mean": getattr(self._processor, "image_mean", [0.5, 0.5, 0.5]),
                "std": getattr(self._processor, "image_std", [0.5, 0.5, 0.5]),
                "do_normalize": getattr(self._processor, "do_normalize", True),
                "do_rescale": getattr(self._processor, "do_rescale", True),
            }

            # 3. Load Model Weights
            model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                config=config,
            )
            
            # Place in evaluation mode
            model.eval()
            
            # Move to target device
            model.to(self._device)
            self._model = model

            # Verify architecture & parameters
            self._architecture = model.__class__.__name__
            self._total_parameters = sum(p.numel() for p in model.parameters())

            self._is_loaded = True
            logger.info(
                "Model '%s' loaded successfully. Architecture: %s, Parameters: %d, Device: %s",
                self.model_name,
                self._architecture,
                self._total_parameters,
                self._device,
            )
            return True

        except Exception as exc:
            self._is_loaded = False
            self._load_error = f"{type(exc).__name__}: {str(exc)}"
            logger.error("Failed to load model '%s': %s", self.model_name, self._load_error, exc_info=True)
            return False

    def unload_model(self) -> None:
        """Unload model and free GPU/CPU memory."""
        self._model = None
        self._processor = None
        self._config = None
        self._is_loaded = False
        self._load_error = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model '%s' unloaded from memory.", self.model_name)

    # -------------------------------------------------------------------------
    # Image Ingestion & Preprocessing
    # -------------------------------------------------------------------------

    def _load_pil_image(self, image_input: Union[Image.Image, np.ndarray, bytes, str, Path]) -> Image.Image:
        """
        Safely convert various image input formats into a PIL RGB Image.
        Raises InvalidImageError if input is corrupted or unreadable.
        """
        if isinstance(image_input, Image.Image):
            try:
                return image_input.convert("RGB")
            except Exception as e:
                raise InvalidImageError(f"Failed to convert PIL image to RGB: {e}") from e

        elif isinstance(image_input, np.ndarray):
            try:
                # Expect HWC format
                if image_input.ndim == 2:
                    return Image.fromarray(image_input).convert("RGB")
                elif image_input.ndim == 3:
                    if image_input.shape[2] == 4:
                        return Image.fromarray(image_input, mode="RGBA").convert("RGB")
                    elif image_input.shape[2] == 3:
                        return Image.fromarray(image_input, mode="RGB")
                raise InvalidImageError(f"Unsupported numpy image shape: {image_input.shape}")
            except Exception as e:
                raise InvalidImageError(f"Failed to convert numpy array to PIL Image: {e}") from e

        elif isinstance(image_input, (bytes, bytearray)):
            if len(image_input) == 0:
                raise InvalidImageError("Input image bytes payload is empty (0 bytes).")
            try:
                img = Image.open(io.BytesIO(image_input))
                img.verify()  # Validate structural image integrity
                # Re-open after verify as per PIL documentation
                img = Image.open(io.BytesIO(image_input))
                return img.convert("RGB")
            except Exception as e:
                raise InvalidImageError(f"Corrupted or invalid image bytes: {e}") from e

        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise InvalidImageError(f"Image file does not exist at path: '{path}'")
            try:
                with open(path, "rb") as f:
                    content = f.read()
                return self._load_pil_image(content)
            except InvalidImageError:
                raise
            except Exception as e:
                raise InvalidImageError(f"Failed to read image file from '{path}': {e}") from e

        else:
            raise InvalidImageError(f"Unsupported image input type: {type(image_input)}")

    def preprocess_image(
        self, image_input: Union[Image.Image, np.ndarray, bytes, str, Path]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert input image into preprocessed PyTorch tensors ready for the ViT model.
        
        Returns:
            Dict containing 'pixel_values' tensor on the configured device.
        """
        if not self._is_loaded or self._processor is None:
            loaded = self.load_model()
            if not loaded:
                raise ModelNotLoadedError(f"Model is not loaded. Last error: {self._load_error}")

        pil_img = self._load_pil_image(image_input)
        
        # Basic sanity check on dimensions
        if pil_img.width < 10 or pil_img.height < 10:
            raise InvalidImageError(
                f"Image dimensions ({pil_img.width}x{pil_img.height}) are too small for facial analysis."
            )

        try:
            # Process image to 224x224 normalized tensor
            inputs = self._processor(images=pil_img, return_tensors="pt")
            # Move tensors to active compute device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            return inputs
        except Exception as exc:
            raise InvalidImageError(f"Image preprocessing failed: {exc}") from exc

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def predict(
        self,
        image_input: Union[Image.Image, np.ndarray, bytes, str, Path],
        real_threshold: Optional[float] = None,
        fake_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute forward pass on an input facial crop and return structured classification results.
        
        Requirements met:
        - Loads model if not loaded.
        - Preprocesses image.
        - Applies softmax to raw logits.
        - Maps output indices to verified labels (0: Real, 1: Fake).
        - Applies configurable verdict thresholds (default 0.70):
          * Real >= 0.70 -> likely_real
          * Fake >= 0.70 -> likely_fake (manipulated)
          * Otherwise -> uncertain
        - Returns structured results with execution time and uncalibrated softmax scores.
        
        Returns:
            Dict[str, Any]:
                - predicted_label (str): 'Real' or 'Fake'
                - verdict (str): 'likely_real', 'likely_fake', or 'uncertain'
                - model_score (float): Confidence score for predicted class
                - real_score (float): Uncalibrated score for genuine class
                - fake_score (float): Uncalibrated score for synthetic class
                - raw_scores (Dict[str, float]): Scores for each verified class label
                - confidence (float): Score assigned to the predicted class
                - model_identifier (str): Hugging Face checkpoint name
                - model_architecture (str): Model architecture name (ViTForImageClassification)
                - inference_time_ms (float): Inference execution latency in milliseconds
                - score_type (str): 'uncalibrated_softmax_score'
                - device (str): 'cpu' or 'cuda'
        """
        if not self._is_loaded or self._model is None:
            loaded = self.load_model()
            if not loaded:
                raise ModelNotLoadedError(f"Model could not be loaded: {self._load_error}")

        inputs = self.preprocess_image(image_input)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # shape: (1, num_labels)
            probabilities = torch.softmax(logits, dim=-1)[0]
        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Extract predictions
        pred_idx = int(torch.argmax(probabilities).item())
        pred_label = self._id2label.get(pred_idx, f"Class_{pred_idx}")
        
        # Build raw class score map
        raw_scores: Dict[str, float] = {}
        for idx in range(len(probabilities)):
            label_name = self._id2label.get(idx, f"Class_{idx}")
            raw_scores[label_name] = round(float(probabilities[idx].item()), 6)

        confidence = raw_scores.get(pred_label, float(probabilities[pred_idx].item()))
        real_score = raw_scores.get("Real", 0.0)
        fake_score = raw_scores.get("Fake", 0.0)

        # Determine semantic verdict string using configurable thresholds
        r_thresh = real_threshold if real_threshold is not None else settings.real_threshold
        f_thresh = fake_threshold if fake_threshold is not None else settings.fake_threshold

        if real_score >= r_thresh:
            verdict = "likely_real"
        elif fake_score >= f_thresh:
            verdict = "likely_fake"
        else:
            verdict = "uncertain"

        return {
            "predicted_label": pred_label,
            "verdict": verdict,
            "model_score": confidence,
            "real_score": real_score,
            "fake_score": fake_score,
            "raw_scores": raw_scores,
            "confidence": confidence,
            "model_identifier": self.model_name,
            "model_architecture": self._architecture or "ViTForImageClassification",
            "inference_time_ms": inference_time_ms,
            "score_type": "uncalibrated_softmax_score",
            "device": self._device,
        }

    # -------------------------------------------------------------------------
    # Diagnostic / Audit
    # -------------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        """Return diagnostic overview of checkpoint and loaded status."""
        return {
            "model_identifier": self.model_name,
            "is_loaded": self._is_loaded,
            "load_error": self._load_error,
            "architecture": self._architecture,
            "model_type": self._model_type,
            "device": self._device,
            "id2label": self._id2label,
            "label2id": self._label2id,
            "num_labels": self._num_labels,
            "image_size": self._image_size,
            "normalization_params": self._normalization_params,
            "total_parameters": self._total_parameters,
        }


# Singleton service instance shared across API handlers
model_service = ModelService()
