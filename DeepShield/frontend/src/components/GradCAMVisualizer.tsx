import React, { useState } from 'react';
import { Eye, Sliders, Layers, Sparkles, AlertCircle } from 'lucide-react';
import { FaceAnalysisResult } from '../types/analysis';

interface GradCAMVisualizerProps {
  face: FaceAnalysisResult;
}

export const GradCAMVisualizer: React.FC<GradCAMVisualizerProps> = ({ face }) => {
  const [viewMode, setViewMode] = useState<'blended' | 'sideBySide' | 'rawOverlay'>('blended');
  const [opacity, setOpacity] = useState<number>(45);

  const hasHeatmap = face.heatmap_available && Boolean(face.heatmap_base64);
  const heatmapSrc = face.heatmap_base64
    ? `data:image/png;base64,${face.heatmap_base64}`
    : null;

  return (
    <div className="rounded-xl bg-slate-950/70 border border-slate-800 p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-blue-600/20 text-cyan-400 border border-cyan-500/30">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <span>Grad-CAM Visual Explainability</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/60 font-mono">
                14×14 ViT Grid
              </span>
            </h4>
            <p className="text-xs text-slate-400">
              Layer norm gradient backpropagation across 196 spatial patch tokens
            </p>
          </div>
        </div>

        {hasHeatmap && (
          <div className="flex items-center space-x-1.5 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              type="button"
              onClick={() => setViewMode('blended')}
              className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                viewMode === 'blended'
                  ? 'bg-blue-600/30 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Blended Overlay
            </button>
            <button
              type="button"
              onClick={() => setViewMode('sideBySide')}
              className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                viewMode === 'sideBySide'
                  ? 'bg-blue-600/30 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Side-by-Side
            </button>
          </div>
        )}
      </div>

      {hasHeatmap && heatmapSrc ? (
        <div className="space-y-4">
          {viewMode === 'blended' ? (
            <div className="flex flex-col sm:flex-row items-center justify-center gap-6 py-2">
              <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-slate-900 shadow-[0_0_20px_rgba(6,182,212,0.1)] max-w-xs sm:max-w-sm">
                <img
                  src={heatmapSrc}
                  alt={`Grad-CAM Heatmap overlay for Face #${face.face_id}`}
                  className="w-full h-auto object-contain block transition-opacity duration-150"
                  style={{ opacity: opacity / 100 }}
                />
                <div className="absolute top-2 left-2 px-2 py-1 rounded bg-slate-950/80 backdrop-blur-md border border-slate-700/80 text-[10px] font-mono text-cyan-300 flex items-center space-x-1">
                  <Sparkles className="w-3 h-3 text-cyan-400" />
                  <span>JET Colormap Overlay</span>
                </div>
                {face.heatmap_dimensions && (
                  <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-slate-950/80 backdrop-blur-md text-[10px] font-mono text-slate-400">
                    {face.heatmap_dimensions.width}×{face.heatmap_dimensions.height}px
                  </div>
                )}
              </div>

              {/* Opacity Control & Saliency Legend */}
              <div className="w-full sm:w-60 space-y-4">
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400 flex items-center space-x-1">
                      <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                      <span>Heatmap Opacity</span>
                    </span>
                    <span className="text-cyan-300 font-mono font-medium">{opacity}%</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={opacity}
                    onChange={(e) => setOpacity(Number(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>Face Context</span>
                    <span>Balanced</span>
                    <span>Max Saliency</span>
                  </div>
                </div>

                {/* Colormap Legend */}
                <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 space-y-2">
                  <span className="text-[11px] font-medium text-slate-300 block">
                    Feature Activation Scale
                  </span>
                  <div className="h-3 rounded-md bg-gradient-to-r from-blue-700 via-cyan-400 via-yellow-400 to-red-600 shadow-inner" />
                  <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                    <span>Low Influence</span>
                    <span>High Activation</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Side by Side mode */
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 py-2">
              <div className="rounded-xl border border-slate-800 bg-slate-900 p-3 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-slate-200">Face Crop & Coordinates</span>
                  <span className="font-mono text-[11px]">
                    ID #{face.face_id}
                  </span>
                </div>
                <div className="rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center min-h-[160px] border border-slate-800">
                  <img
                    src={heatmapSrc}
                    alt="Facial region crop"
                    className="max-h-56 w-auto object-contain filter grayscale contrast-125"
                  />
                </div>
                <p className="text-[10px] text-slate-400 text-center font-mono">
                  Input dimension normalized to 224×224 RGB
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900 p-3 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold text-cyan-300">Grad-CAM Activation Map</span>
                  <span className="font-mono text-[11px] text-cyan-400">
                    Target: {face.predicted_label || 'Predicted'}
                  </span>
                </div>
                <div className="rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center min-h-[160px] border border-slate-800">
                  <img
                    src={heatmapSrc}
                    alt="Grad-CAM visualization"
                    className="max-h-56 w-auto object-contain"
                  />
                </div>
                <p className="text-[10px] text-slate-400 text-center font-mono">
                  Bicubic interpolated from 14×14 grid
                </p>
              </div>
            </div>
          )}

          {/* Model Explanation Box */}
          <div className="p-3.5 rounded-lg bg-blue-950/20 border border-blue-900/40 text-xs text-slate-300 space-y-1">
            <div className="flex items-center space-x-1.5 text-cyan-300 font-medium">
              <Eye className="w-3.5 h-3.5 text-cyan-400" />
              <span>Diagnostic Explainability Summary</span>
            </div>
            <p className="text-slate-300 text-xs leading-relaxed">
              {face.explanation ||
                "The Grad-CAM visualization highlights image regions that contributed to this prediction. Highlighted areas indicate where Vision Transformer attention weights concentrated."}
            </p>
            <p className="text-[11px] text-slate-500 italic pt-1 border-t border-slate-800/80">
              Notice: Grad-CAM visual cues illustrate feature attribution of the classification head. They do not constitute empirical forensic proof of digital manipulation or absolute authenticity.
            </p>
          </div>
        </div>
      ) : (
        /* Fallback when Grad-CAM is unavailable */
        <div className="flex items-center space-x-3 p-4 rounded-xl bg-slate-900/80 border border-slate-800 text-slate-400">
          <AlertCircle className="w-5 h-5 text-amber-400 shrink-0" />
          <div className="text-xs">
            <p className="font-semibold text-slate-300">Grad-CAM Heatmap Unavailable</p>
            <p className="text-slate-400 mt-0.5">
              {face.explanation ||
                "A visual saliency map could not be derived for this facial crop (crop dimensions too small or gradients flat). Classification scores remain valid."}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
