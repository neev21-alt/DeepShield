"""
Narrative Explanation Generation Service (Future Phase).

Intended Responsibility:
- Synthesize quantitative model scores and spatial Grad-CAM activation zones
  (e.g., peri-orbital regions, lips, jawline blending margins).
- Formulate human-readable evidentiary narratives explaining the decision basis.
- Ground explanations strictly in observable detection metadata and avoid hallucinated facts.
- Future integration: Google Gemini API via metadata-only grounding prompt.

Status:
- Phase 1: Placeholder definition.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExplanationService:
    """Generates structured, human-readable explanations based on model telemetry."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def generate_explanation(self, metadata: Dict[str, Any]) -> str:
        """
        Generate textual explanation grounded in face detection and saliency metrics.
        
        Implementation will be completed in future phase.
        """
        raise NotImplementedError("Metadata-only explanation service will be implemented in future phase.")


explanation_service = ExplanationService()
