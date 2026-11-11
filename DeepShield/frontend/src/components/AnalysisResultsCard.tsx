import React, { useState } from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  UserX,
  Clock,
  Layers,
  ArrowLeft,
  Share2,
  CheckCircle2,
  Cpu,
} from 'lucide-react';
import { AnalysisResponse, FaceAnalysisResult } from '../types/analysis';
import { FaceResultItem } from './FaceResultItem';

interface AnalysisResultsCardProps {
  response: AnalysisResponse;
  previewUrl: string | null;
  onReset: () => void;
}

export const AnalysisResultsCard: React.FC<AnalysisResultsCardProps> = ({
  response,
  previewUrl,
  onReset,
}) => {
  const [selectedFaceId, setSelectedFaceId] = useState<number | null>(
    response.faces.length > 0 ? response.faces[0].face_id : null
  );

  const overall = response.overall_verdict;
  const isReal = overall === 'likely_real';
  const isFake = overall === 'likely_fake';
  const isMixed = overall === 'mixed';
  const isUncertain = overall === 'uncertain';
  const isNoFace = response.status === 'no_faces_detected' || response.faces_detected === 0;

  // Selected face object
  const activeFace =
    response.faces.find((f) => f.face_id === selectedFaceId) || response.faces[0];

  return (
    <div className="w-full space-y-6 animate-in fade-in duration-300">
      {/* Top Action Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <button
          type="button"
          onClick={onReset}
          className="flex items-center space-x-2 px-3.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 text-xs font-medium transition-colors"
        >
          <ArrowLeft className="w-4 h-4 text-cyan-400" />
          <span>Analyze Another Image</span>
        </button>

        <div className="flex items-center space-x-3 text-xs font-mono text-slate-400">
          {response.processing_time_ms && (
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800">
              <Clock className="w-3.5 h-3.5 text-blue-400" />
              <span>Latency: {response.processing_time_ms.toFixed(1)} ms</span>
            </div>
          )}
          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 hidden sm:flex">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="truncate max-w-[200px]">{response.model_identifier}</span>
          </div>
        </div>
      </div>

      {/* Executive Verdict Banner */}
      <div
        className={`p-6 sm:p-7 rounded-2xl border backdrop-blur-md relative overflow-hidden transition-all shadow-2xl ${
          isReal
            ? 'bg-gradient-to-r from-emerald-950/40 via-slate-900/90 to-slate-950 border-emerald-500/40'
            : isFake
            ? 'bg-gradient-to-r from-rose-950/40 via-slate-900/90 to-slate-950 border-rose-500/40'
            : isMixed
            ? 'bg-gradient-to-r from-indigo-950/40 via-slate-900/90 to-slate-950 border-indigo-500/40'
            : isUncertain
            ? 'bg-gradient-to-r from-amber-950/40 via-slate-900/90 to-slate-950 border-amber-500/40'
            : 'bg-gradient-to-r from-slate-900/80 to-slate-950 border-slate-800'
        }`}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-5 relative z-10">
          <div className="flex items-start sm:items-center space-x-4">
            <div
              className={`w-14 h-14 rounded-2xl flex items-center justify-center shrink-0 border shadow-lg ${
                isReal
                  ? 'bg-emerald-900/40 border-emerald-500/50 text-emerald-400 shadow-[0_0_20px_rgba(16,185,129,0.2)]'
                  : isFake
                  ? 'bg-rose-900/40 border-rose-500/50 text-rose-400 shadow-[0_0_20px_rgba(239,68,68,0.2)]'
                  : isMixed
                  ? 'bg-indigo-900/40 border-indigo-500/50 text-indigo-400 shadow-[0_0_20px_rgba(99,102,241,0.2)]'
                  : isUncertain
                  ? 'bg-amber-900/40 border-amber-500/50 text-amber-400 shadow-[0_0_20px_rgba(245,158,11,0.2)]'
                  : 'bg-slate-800 border-slate-700 text-slate-400'
              }`}
            >
              {isReal ? (
                <ShieldCheck className="w-8 h-8" />
              ) : isFake ? (
                <ShieldAlert className="w-8 h-8" />
              ) : isMixed ? (
                <Layers className="w-8 h-8" />
              ) : isUncertain ? (
                <AlertTriangle className="w-8 h-8" />
              ) : (
                <UserX className="w-8 h-8" />
              )}
            </div>

            <div>
              <span className="text-xs uppercase tracking-wider font-semibold text-slate-400 font-mono">
                Overall Image Verdict
              </span>
              <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
                {isReal
                  ? 'Likely Authentic Media'
                  : isFake
                  ? 'Likely Manipulated Media'
                  : isMixed
                  ? 'Mixed Manipulated & Authentic Subjects'
                  : isUncertain
                  ? 'Uncertain Classification'
                  : 'No Human Faces Detected'}
              </h2>
              <p className="text-xs text-slate-300 mt-1 max-w-xl">
                {response.message}
              </p>
            </div>
          </div>

          {/* Metric Badges */}
          <div className="flex sm:flex-col items-start sm:items-end justify-between gap-2 border-t sm:border-t-0 border-slate-800/80 pt-3 sm:pt-0">
            <div className="text-right">
              <span className="text-[11px] text-slate-400 block font-mono">Detected Subjects</span>
              <span className="text-lg font-bold text-slate-100 font-mono">
                {response.faces_detected} {response.faces_detected === 1 ? 'Face' : 'Faces'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* No Face Detected Message */}
      {isNoFace && (
        <div className="p-8 rounded-2xl bg-slate-900/60 border border-slate-800 text-center space-y-4">
          <div className="w-12 h-12 rounded-xl bg-slate-800/80 text-slate-400 flex items-center justify-center mx-auto">
            <UserX className="w-6 h-6" />
          </div>
          <div className="max-w-md mx-auto space-y-1">
            <h3 className="text-base font-semibold text-slate-200">No Facial Landmarks Localized</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              MediaPipe BlazeFace did not identify any human faces in this image. Please upload a clear portrait, ID photo, or group photo with visible facial features.
            </p>
          </div>
          <button
            type="button"
            onClick={onReset}
            className="px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold shadow-lg transition-all"
          >
            Upload Another Image
          </button>
        </div>
      )}

      {/* When Faces Exist: Bounding Box Map + Per-Face Details */}
      {!isNoFace && (
        <div className="space-y-6">
          {/* Multi-Face Navigation Pills (if > 1 face) */}
          {response.faces.length > 1 && (
            <div className="flex items-center space-x-2 overflow-x-auto pb-1">
              <span className="text-xs text-slate-400 font-mono shrink-0 mr-1">
                Select Subject:
              </span>
              {response.faces.map((face) => {
                const isFaceSelected = face.face_id === selectedFaceId;
                const isFaceFake = face.verdict === 'likely_fake';
                return (
                  <button
                    key={face.face_id}
                    type="button"
                    onClick={() => setSelectedFaceId(face.face_id)}
                    className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-mono font-medium transition-all ${
                      isFaceSelected
                        ? 'bg-blue-600 text-white border border-blue-400 shadow-[0_0_12px_rgba(37,99,235,0.4)]'
                        : 'bg-slate-900 text-slate-300 hover:bg-slate-800 border border-slate-800'
                    }`}
                  >
                    <span
                      className={`w-2 h-2 rounded-full ${
                        isFaceFake ? 'bg-rose-400' : 'bg-emerald-400'
                      }`}
                    />
                    <span>Face #{face.face_id}</span>
                    <span className="text-[10px] opacity-80 uppercase">
                      ({face.predicted_label || face.verdict})
                    </span>
                  </button>
                );
              })}
            </div>
          )}

          {/* Interactive Spatial Overview with Bounding Box Overlay */}
          {previewUrl && (
            <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-300">
                <span className="font-semibold flex items-center space-x-1.5">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  <span>MediaPipe Facial Localization Map</span>
                </span>
                <span className="text-[11px] font-mono text-slate-500">
                  Normalized coordinates [0.0 – 1.0]
                </span>
              </div>

              {/* Render Image with SVG Bounding Box Overlays */}
              <div className="relative inline-block max-w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
                <img
                  src={previewUrl}
                  alt="Spatial detection map"
                  className="max-h-[380px] w-auto object-contain block mx-auto"
                />

                {/* SVG Overlay */}
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                >
                  {response.faces.map((face) => {
                    const isSelected = face.face_id === selectedFaceId;
                    const isFaceFake = face.verdict === 'likely_fake';
                    const bb = face.bounding_box;
                    const x = bb.x_min * 100;
                    const y = bb.y_min * 100;
                    const w = bb.width * 100;
                    const h = bb.height * 100;

                    return (
                      <g key={face.face_id}>
                        {/* Bounding box rectangle */}
                        <rect
                          x={x}
                          y={y}
                          width={w}
                          height={h}
                          fill={
                            isSelected
                              ? 'rgba(6, 182, 212, 0.15)'
                              : 'transparent'
                          }
                          stroke={
                            isSelected
                              ? '#06b6d4'
                              : isFaceFake
                              ? '#f43f5e'
                              : '#10b981'
                          }
                          strokeWidth={isSelected ? '0.8' : '0.5'}
                          strokeDasharray={isSelected ? 'none' : '1.5, 0.8'}
                        />
                        {/* Face ID Tag */}
                        <rect
                          x={x}
                          y={Math.max(0, y - 4.5)}
                          width={14}
                          height={4.5}
                          fill={isFaceFake ? '#be123c' : '#047857'}
                          rx={0.8}
                        />
                        <text
                          x={x + 1}
                          y={Math.max(3, y - 1.2)}
                          fill="#ffffff"
                          fontSize="3"
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          #{face.face_id}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>
          )}

          {/* Focused Face Analysis Result */}
          {activeFace && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-300 font-mono uppercase tracking-wider">
                  Detailed Diagnostic Breakdown: Face #{activeFace.face_id}
                </h3>
              </div>
              <FaceResultItem face={activeFace} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
