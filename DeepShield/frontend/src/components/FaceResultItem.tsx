import React from 'react';
import { ShieldCheck, ShieldAlert, HelpCircle, Clock, Crosshair, BarChart2 } from 'lucide-react';
import { FaceAnalysisResult } from '../types/analysis';
import { GradCAMVisualizer } from './GradCAMVisualizer';

interface FaceResultItemProps {
  face: FaceAnalysisResult;
  isExpanded?: boolean;
}

export const FaceResultItem: React.FC<FaceResultItemProps> = ({ face }) => {
  const isReal = face.verdict === 'likely_real';
  const isFake = face.verdict === 'likely_fake';

  const realPct = (face.real_score * 100).toFixed(1);
  const fakePct = (face.fake_score * 100).toFixed(1);

  return (
    <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-6 shadow-xl backdrop-blur-sm">
      {/* Header Info */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div className="flex items-center space-x-3">
          <div
            className={`w-10 h-10 rounded-xl flex items-center justify-center border shadow-md ${
              isReal
                ? 'bg-emerald-950/60 border-emerald-500/40 text-emerald-400'
                : isFake
                ? 'bg-rose-950/60 border-rose-500/40 text-rose-400'
                : 'bg-amber-950/60 border-amber-500/40 text-amber-400'
            }`}
          >
            {isReal ? (
              <ShieldCheck className="w-5 h-5" />
            ) : isFake ? (
              <ShieldAlert className="w-5 h-5" />
            ) : (
              <HelpCircle className="w-5 h-5" />
            )}
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-base font-bold text-white font-mono">
                Face #{face.face_id}
              </h3>
              <span
                className={`px-2.5 py-0.5 rounded-full text-xs font-semibold tracking-wide border ${
                  isReal
                    ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800/80'
                    : isFake
                    ? 'bg-rose-950/80 text-rose-300 border-rose-800/80'
                    : 'bg-amber-950/80 text-amber-300 border-amber-800/80'
                }`}
              >
                {isReal
                  ? 'Likely Genuine'
                  : isFake
                  ? 'Likely Manipulated'
                  : 'Uncertain'}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Predicted Class:{' '}
              <span className="font-semibold text-slate-200">
                {face.predicted_label || (isReal ? 'Real' : isFake ? 'Fake' : 'Uncertain')}
              </span>
            </p>
          </div>
        </div>

        {/* Telemetry metadata */}
        <div className="flex items-center space-x-4 text-xs font-mono text-slate-400">
          {face.detection_confidence !== null && face.detection_confidence !== undefined && (
            <div className="flex items-center space-x-1" title="MediaPipe BlazeFace detection score">
              <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
              <span>Face Det: {(face.detection_confidence * 100).toFixed(0)}%</span>
            </div>
          )}
          {face.inference_time_ms && (
            <div className="flex items-center space-x-1" title="ViT inference execution latency">
              <Clock className="w-3.5 h-3.5 text-blue-400" />
              <span>{face.inference_time_ms.toFixed(1)} ms</span>
            </div>
          )}
        </div>
      </div>

      {/* Model Score Bars */}
      <div className="space-y-3 p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
        <div className="flex items-center justify-between text-xs text-slate-300">
          <div className="flex items-center space-x-1.5 font-medium">
            <BarChart2 className="w-4 h-4 text-cyan-400" />
            <span>ViT Model Hypothesis Distribution</span>
          </div>
          <span className="text-[11px] font-mono text-slate-500">
            Softmax output logits
          </span>
        </div>

        {/* Score Breakdown Bars */}
        <div className="space-y-2">
          {/* Real Score */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-emerald-400 flex items-center space-x-1">
                <span>Authentic (Real) Hypothesis</span>
              </span>
              <span className="text-slate-200 font-semibold">{realPct}%</span>
            </div>
            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full transition-all duration-500"
                style={{ width: `${realPct}%` }}
              />
            </div>
          </div>

          {/* Fake Score */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-rose-400 flex items-center space-x-1">
                <span>Manipulated (Deepfake) Hypothesis</span>
              </span>
              <span className="text-slate-200 font-semibold">{fakePct}%</span>
            </div>
            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-rose-600 to-rose-400 rounded-full transition-all duration-500"
                style={{ width: `${fakePct}%` }}
              />
            </div>
          </div>
        </div>

        <p className="text-[10px] text-slate-500 leading-tight italic pt-1">
          Disclaimer: Softmax scores represent raw Vision Transformer classifier activation confidence and must not be conflated with calibrated Bayesian posterior probabilities.
        </p>
      </div>

      {/* Grad-CAM Explainability Section */}
      <GradCAMVisualizer face={face} />
    </div>
  );
};
