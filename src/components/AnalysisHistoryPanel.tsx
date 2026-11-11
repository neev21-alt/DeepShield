import React from 'react';
import { History, Trash2, ArrowUpRight, ShieldCheck, ShieldAlert, AlertTriangle, UserX } from 'lucide-react';
import { HistoryItem } from '../types/analysis';

interface AnalysisHistoryPanelProps {
  history: HistoryItem[];
  onSelectHistoryItem: (item: HistoryItem) => void;
  onClearHistory: () => void;
  onBackToAnalyzer: () => void;
}

export const AnalysisHistoryPanel: React.FC<AnalysisHistoryPanelProps> = ({
  history,
  onSelectHistoryItem,
  onClearHistory,
  onBackToAnalyzer,
}) => {
  return (
    <div className="w-full space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <History className="w-5 h-5 text-cyan-400" />
            <span>Session Inspection History</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Temporary client-side analysis cache. Persists in memory during active session.
          </p>
        </div>

        {history.length > 0 && (
          <button
            type="button"
            onClick={onClearHistory}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-rose-950/40 hover:bg-rose-900/60 text-rose-300 border border-rose-800/50 text-xs font-medium transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear History</span>
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="p-12 text-center rounded-2xl bg-slate-900/40 border border-slate-800 space-y-3">
          <History className="w-10 h-10 text-slate-600 mx-auto" />
          <h3 className="text-sm font-semibold text-slate-300">No session inspections yet</h3>
          <p className="text-xs text-slate-500 max-w-sm mx-auto">
            Uploaded images and their Vision Transformer Grad-CAM results will appear here for quick reference during your active session.
          </p>
          <button
            type="button"
            onClick={onBackToAnalyzer}
            className="mt-2 px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold shadow-md transition-colors"
          >
            Run an Analysis
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {history.map((item) => {
            const verdict = item.result.overall_verdict;
            const isReal = verdict === 'likely_real';
            const isFake = verdict === 'likely_fake';
            const isUncertain = verdict === 'uncertain';

            return (
              <div
                key={item.id}
                onClick={() => onSelectHistoryItem(item)}
                className="group flex items-center justify-between p-4 rounded-xl bg-slate-900/70 hover:bg-slate-850 border border-slate-800 hover:border-slate-700 cursor-pointer transition-all hover:scale-[1.01]"
              >
                <div className="flex items-center space-x-3.5 truncate">
                  <div className="w-14 h-14 rounded-lg overflow-hidden bg-slate-950 border border-slate-800 shrink-0">
                    <img
                      src={item.previewUrl}
                      alt={item.filename}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="truncate space-y-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-mono font-bold text-slate-200 truncate">
                        {item.filename}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2 text-[11px] text-slate-400 font-mono">
                      <span>{item.timestamp}</span>
                      <span>•</span>
                      <span>{item.result.faces_detected} face(s)</span>
                    </div>
                    <div className="flex items-center space-x-1.5">
                      <span
                        className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider border ${
                          isReal
                            ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800'
                            : isFake
                            ? 'bg-rose-950/80 text-rose-300 border-rose-800'
                            : isUncertain
                            ? 'bg-amber-950/80 text-amber-300 border-amber-800'
                            : 'bg-slate-800 text-slate-300 border-slate-700'
                        }`}
                      >
                        {isReal ? (
                          <ShieldCheck className="w-2.5 h-2.5" />
                        ) : isFake ? (
                          <ShieldAlert className="w-2.5 h-2.5" />
                        ) : (
                          <AlertTriangle className="w-2.5 h-2.5" />
                        )}
                        <span>{verdict || item.result.status}</span>
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-2 rounded-lg bg-slate-800/60 group-hover:bg-cyan-600/20 text-slate-400 group-hover:text-cyan-400 transition-colors shrink-0">
                  <ArrowUpRight className="w-4 h-4" />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
