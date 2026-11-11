/**
 * DeepShield Frontend Type Definitions
 * Matches FastAPI Pydantic models in backend/app/schemas/analysis.py
 */

export interface BoundingBox {
  x_min: number;
  y_min: number;
  width: number;
  height: number;
  confidence?: number | null;
}

export type Verdict = 'likely_real' | 'likely_fake' | 'uncertain';
export type OverallVerdict = 'likely_real' | 'likely_fake' | 'mixed' | 'uncertain' | 'no_faces_detected';

export interface FaceAnalysisResult {
  face_id: number;
  bounding_box: BoundingBox;
  verdict: Verdict;
  predicted_label?: 'Real' | 'Fake' | null;
  model_score: number;
  real_score: number;
  fake_score: number;
  detection_confidence?: number | null;
  inference_time_ms?: number | null;
  heatmap_available: boolean;
  heatmap_dimensions?: {
    width: number;
    height: number;
  } | null;
  heatmap_base64?: string | null;
  explanation?: string | null;
}

export interface AnalysisResponse {
  status: 'success' | 'no_faces_detected' | 'partial' | 'error';
  message: string;
  faces_detected: number;
  faces: FaceAnalysisResult[];
  overall_verdict: OverallVerdict | null;
  processing_time_ms?: number | null;
  model_identifier: string;
}

export interface HealthResponse {
  status: string;
  backend: string;
  version: string;
  model_loaded: boolean;
  model_identifier: string;
  model_architecture?: string | null;
  device?: string | null;
  timestamp: string;
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  filename: string;
  fileSizeBytes: number;
  previewUrl: string;
  result: AnalysisResponse;
}
