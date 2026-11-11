import React from 'react';
import { X, Shield, Cpu, Layers, Eye, BookOpen, ExternalLink, Sparkles } from 'lucide-react';
import { HealthResponse } from '../types/analysis';

interface SystemInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: HealthResponse | null;
}

export const SystemInfoModal: React.FC<SystemInfoModalProps> = ({
  isOpen,
  onClose,
  health,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl rounded-2xl bg-slate-900 border border-slate-800 shadow-2xl p-6 sm:p-7 space-y-6 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-blue-600/20 text-cyan-400 border border-cyan-500/30">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white font-mono">
                DEEPSHIELD SYSTEM SPECIFICATION
              </h3>
              <p className="text-xs text-slate-400">
                CodeQuest 2026 Hackathon • Team Code Ph@nt0m
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Live Backend Telemetry Table */}
        <div className="rounded-xl bg-slate-950/70 border border-slate-800 p-4 space-y-3">
          <h4 className="text-xs font-semibold text-cyan-400 uppercase tracking-wider font-mono flex items-center space-x-1.5">
            <Cpu className="w-3.5 h-3.5" />
            <span>Active Backend Status</span>
          </h4>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] text-slate-500 block">Service</span>
              <span className="text-slate-200 font-semibold">{health?.backend || 'FastAPI'}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] text-slate-500 block">Version</span>
              <span className="text-slate-200 font-semibold">{health?.version || '0.1.0'}</span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] text-slate-500 block">Model Status</span>
              <span className="text-emerald-400 font-semibold">
                {health?.model_loaded ? 'Loaded (In-Memory)' : 'Standby / Lazy-Load'}
              </span>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] text-slate-500 block">Compute Device</span>
              <span className="text-cyan-400 font-semibold">{health?.device?.toUpperCase() || 'CPU'}</span>
            </div>
          </div>
        </div>

        {/* Technical Architecture Blocks */}
        <div className="space-y-4 text-xs text-slate-300">
          <div className="space-y-1.5">
            <h4 className="font-semibold text-slate-100 flex items-center space-x-2">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              <span>1. Multi-Subject Face Localization (MediaPipe)</span>
            </h4>
            <p className="text-slate-400 leading-relaxed pl-3.5">
              Images are decoded into standard RGB and passed through MediaPipe BlazeFace. Bounding boxes are computed with 20% contextual boundary padding and clamped to image coordinates to isolate facial features without distortion.
            </p>
          </div>

          <div className="space-y-1.5">
            <h4 className="font-semibold text-slate-100 flex items-center space-x-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
              <span>2. Vision Transformer Inference (ViT-B/16)</span>
            </h4>
            <p className="text-slate-400 leading-relaxed pl-3.5">
              Checkpoint: <code className="text-cyan-300 font-mono">dima806/deepfake_vs_real_image_detection</code>. Each face crop is preprocessed to $224 \times 224$ pixels, tokenized into 196 spatial patch tokens ($14 \times 14$ grid), and evaluated through 12 self-attention transformer layers.
            </p>
          </div>

          <div className="space-y-1.5">
            <h4 className="font-semibold text-slate-100 flex items-center space-x-2">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
              <span>3. Grad-CAM Explainability Engine</span>
            </h4>
            <p className="text-slate-400 leading-relaxed pl-3.5">
              Gradients are captured from <code className="text-cyan-300 font-mono">model.vit.encoder.layer[-1].layernorm_before</code> with respect to the target class logit. Spatial tokens are reshaped into $(14, 14)$, weighted by spatial average gradients, passed through ReLU, normalized, and interpolated to original crop resolution with OpenCV JET alpha blending.
            </p>
          </div>
        </div>

        {/* Scientific & Legal Disclaimer */}
        <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-[11px] text-slate-400 space-y-1">
          <p className="font-semibold text-slate-300">Responsible AI Notice</p>
          <p className="leading-relaxed">
            DeepShield provides technical assistance in assessing media authenticity. Saliency maps illustrate neural activation patterns and must not be used as the sole basis for legal, judicial, or defamatory determinations.
          </p>
        </div>

        {/* Footer */}
        <div className="flex justify-end pt-2">
          <button
            type="button"
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-xs font-medium transition-colors"
          >
            Close Specification
          </button>
        </div>
      </div>
    </div>
  );
};
