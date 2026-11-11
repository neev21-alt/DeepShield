import { AnalysisResponse, HealthResponse } from '../types/analysis';

// Configurable API base url. Default fallback routes through Vite proxy or relative path.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '');

export async function checkBackendHealth(): Promise<HealthResponse> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/v1/health` : '/api/v1/health';
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      // Try root /health as fallback
      const fallbackUrl = API_BASE_URL ? `${API_BASE_URL}/health` : '/health';
      const fallbackRes = await fetch(fallbackUrl, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });
      if (fallbackRes.ok) {
        return fallbackRes.json();
      }
      throw new Error(`Health check returned status ${response.status}`);
    }

    return response.json();
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Backend unreachable';
    throw new Error(msg);
  }
}

export async function analyzeFacialImage(file: File): Promise<AnalysisResponse> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/v1/analyze` : '/api/v1/analyze';
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      let errorMessage = `Analysis failed with HTTP ${response.status}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          if (typeof errorJson.detail === 'object' && errorJson.detail.message) {
            errorMessage = errorJson.detail.message;
          } else if (typeof errorJson.detail === 'string') {
            errorMessage = errorJson.detail;
          }
        } else if (errorJson.message) {
          errorMessage = errorJson.message;
        }
      } catch {
        // Fallback to status text
        if (response.status === 413) {
          errorMessage = 'Image payload exceeds maximum 10MB limit.';
        } else if (response.status === 400) {
          errorMessage = 'Invalid or corrupted image format. Please upload JPEG, PNG, or WEBP.';
        } else if (response.status === 503) {
          errorMessage = 'Deepfake AI model is currently initializing or unavailable.';
        }
      }
      throw new Error(errorMessage);
    }

    const data: AnalysisResponse = await response.json();
    return data;
  } catch (err: unknown) {
    if (err instanceof Error) {
      throw err;
    }
    throw new Error('Connection to DeepShield backend failed. Ensure API server is running.');
  }
}
