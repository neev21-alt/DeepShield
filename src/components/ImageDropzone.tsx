import React, { useRef, useState, useCallback } from 'react';
import { UploadCloud, FileImage, X, AlertTriangle, ShieldCheck, Zap, Sparkles } from 'lucide-react';

interface ImageDropzoneProps {
  selectedFile: File | null;
  previewUrl: string | null;
  onFileSelect: (file: File) => void;
  onClearFile: () => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  errorMessage: string | null;
  onDismissError: () => void;
  onLoadSample: (sampleType: 'real' | 'fake' | 'multi' | 'none') => void;
}

export const ImageDropzone: React.FC<ImageDropzoneProps> = ({
  selectedFile,
  previewUrl,
  onFileSelect,
  onClearFile,
  onAnalyze,
  isAnalyzing,
  errorMessage,
  onDismissError,
  onLoadSample,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);

      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="w-full">
      {/* Error Alert Banner */}
      {errorMessage && (
        <div className="mb-5 flex items-start justify-between p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-200 backdrop-blur-sm animate-in fade-in slide-in-from-top-2 duration-200">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-semibold text-rose-300">Analysis Error</h4>
              <p className="text-xs text-rose-200/90 mt-0.5">{errorMessage}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onDismissError}
            className="text-rose-400 hover:text-rose-200 p-1 rounded-md transition-colors"
            aria-label="Dismiss error notification"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Main Upload Area / Preview */}
      {!selectedFile || !previewUrl ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative flex flex-col items-center justify-center p-8 sm:p-12 rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200 group ${
            isDragOver
              ? 'border-cyan-400 bg-cyan-950/20 shadow-[0_0_30px_rgba(6,182,212,0.15)] scale-[1.005]'
              : 'border-slate-800 hover:border-slate-700 bg-slate-900/40 hover:bg-slate-900/70'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleFileInputChange}
            className="hidden"
          />

          <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-600/20 to-cyan-500/10 border border-cyan-500/20 text-cyan-400 mb-4 group-hover:scale-105 group-hover:border-cyan-500/40 transition-all shadow-[0_0_15px_rgba(6,182,212,0.1)]">
            <UploadCloud className="w-8 h-8 text-cyan-400" />
          </div>

          <h3 className="text-base sm:text-lg font-semibold text-slate-100 text-center">
            Drop facial portrait here or <span className="text-cyan-400 underline decoration-cyan-500/40 underline-offset-4">browse files</span>
          </h3>
          <p className="text-xs text-slate-400 mt-1.5 text-center max-w-sm">
            Supports JPEG, PNG, or WEBP up to 10MB. Images are evaluated in-memory and never permanently stored.
          </p>

          <div className="flex items-center space-x-4 mt-6 text-[11px] text-slate-500 font-mono">
            <span className="flex items-center space-x-1">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-500/60" />
              <span>MediaPipe Detection</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500/60" />
              <span>ViT-B/16 Classifier</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-500/60" />
              <span>Grad-CAM Explainability</span>
            </span>
          </div>
        </div>
      ) : (
        /* Image Loaded Preview Container */
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-sm">
          <div className="flex flex-col lg:flex-row items-center gap-6">
            {/* Image Thumbnail Frame */}
            <div className="relative group shrink-0">
              <div className="w-48 h-48 sm:w-56 sm:h-56 rounded-xl overflow-hidden border border-slate-700/80 bg-slate-950 flex items-center justify-center shadow-lg">
                <img
                  src={previewUrl}
                  alt="Upload preview"
                  className="w-full h-full object-cover"
                />
              </div>
              <button
                type="button"
                onClick={onClearFile}
                disabled={isAnalyzing}
                className="absolute -top-2 -right-2 p-1.5 rounded-full bg-slate-800 text-slate-300 hover:text-white hover:bg-rose-600 border border-slate-700 transition-all shadow-md focus:outline-none disabled:opacity-50"
                title="Remove image"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Metadata & Launch CTA */}
            <div className="flex-1 w-full space-y-4">
              <div className="space-y-1.5">
                <div className="flex items-center space-x-2">
                  <FileImage className="w-4 h-4 text-cyan-400" />
                  <span className="text-sm font-semibold text-slate-100 truncate max-w-md font-mono">
                    {selectedFile.name}
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400 font-mono">
                  <span className="px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60">
                    {formatFileSize(selectedFile.size)}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700/60 uppercase">
                    {selectedFile.type.split('/')[1] || 'IMAGE'}
                  </span>
                  <span className="text-emerald-400 flex items-center space-x-1">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    <span>Payload verified</span>
                  </span>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-400 space-y-1">
                <p className="flex items-center space-x-1.5 text-slate-300 font-medium">
                  <Zap className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Pipeline Execution Target</span>
                </p>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  FastAPI backend will run BlazeFace bounding box localization, ViT patch tokenization ($14 \times 14$), classification softmax, and generate layer-norm Grad-CAM heatmaps.
                </p>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center gap-3 pt-1">
                <button
                  type="button"
                  onClick={onAnalyze}
                  disabled={isAnalyzing}
                  className="flex-1 min-w-[200px] flex items-center justify-center space-x-2 px-6 py-3 rounded-xl bg-gradient-to-r from-blue-600 via-cyan-600 to-cyan-500 hover:from-blue-500 hover:via-cyan-500 hover:to-cyan-400 text-white font-semibold text-sm shadow-[0_0_20px_rgba(6,182,212,0.25)] hover:shadow-[0_0_25px_rgba(6,182,212,0.4)] disabled:opacity-50 disabled:cursor-not-allowed transition-all active:scale-[0.99]"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      <span>Running ViT Inference...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4 text-cyan-200" />
                      <span>Analyze with DeepShield ViT</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isAnalyzing}
                  className="px-4 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-xs font-medium border border-slate-700 transition-colors disabled:opacity-50"
                >
                  Replace Image
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Test Sample Triggers */}
      <div className="mt-5 pt-4 border-t border-slate-800/80">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider font-mono">
            Quick Verification Samples
          </span>
          <span className="text-[11px] text-slate-500">
            Click to load pre-verified benchmark samples
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <button
            type="button"
            disabled={isAnalyzing}
            onClick={() => onLoadSample('real')}
            className="flex items-center space-x-2 p-2.5 rounded-lg bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-emerald-500/40 text-left transition-all group disabled:opacity-50"
          >
            <span className="w-2 h-2 rounded-full bg-emerald-400 group-hover:shadow-[0_0_8px_rgba(52,211,153,0.6)]" />
            <div className="truncate">
              <p className="text-xs font-medium text-slate-200 truncate">Authentic Portrait</p>
              <p className="text-[10px] text-slate-500">Grace Hopper (Navy)</p>
            </div>
          </button>

          <button
            type="button"
            disabled={isAnalyzing}
            onClick={() => onLoadSample('fake')}
            className="flex items-center space-x-2 p-2.5 rounded-lg bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-rose-500/40 text-left transition-all group disabled:opacity-50"
          >
            <span className="w-2 h-2 rounded-full bg-rose-400 group-hover:shadow-[0_0_8px_rgba(244,63,94,0.6)]" />
            <div className="truncate">
              <p className="text-xs font-medium text-slate-200 truncate">Manipulated Face</p>
              <p className="text-[10px] text-slate-500">Deepfake sample #3</p>
            </div>
          </button>

          <button
            type="button"
            disabled={isAnalyzing}
            onClick={() => onLoadSample('multi')}
            className="flex items-center space-x-2 p-2.5 rounded-lg bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-cyan-500/40 text-left transition-all group disabled:opacity-50"
          >
            <span className="w-2 h-2 rounded-full bg-cyan-400 group-hover:shadow-[0_0_8px_rgba(6,182,212,0.6)]" />
            <div className="truncate">
              <p className="text-xs font-medium text-slate-200 truncate">Multi-Face Group</p>
              <p className="text-[10px] text-slate-500">2 subjects photo</p>
            </div>
          </button>

          <button
            type="button"
            disabled={isAnalyzing}
            onClick={() => onLoadSample('none')}
            className="flex items-center space-x-2 p-2.5 rounded-lg bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-amber-500/40 text-left transition-all group disabled:opacity-50"
          >
            <span className="w-2 h-2 rounded-full bg-amber-400 group-hover:shadow-[0_0_8px_rgba(251,191,36,0.6)]" />
            <div className="truncate">
              <p className="text-xs font-medium text-slate-200 truncate">Zero Faces (Scenery)</p>
              <p className="text-[10px] text-slate-500">No human subjects</p>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};
