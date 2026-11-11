import React from 'react';
import { Shield, ShieldAlert, Cpu, Activity, RefreshCw, History, Info, Layers } from 'lucide-react';
import { HealthResponse } from '../types/analysis';

interface DashboardHeaderProps {
  health: HealthResponse | null;
  isHealthChecking: boolean;
  onRefreshHealth: () => void;
  activeTab: 'analyzer' | 'history';
  onSelectTab: (tab: 'analyzer' | 'history') => void;
  historyCount: number;
  onOpenInfo: () => void;
}

export const DashboardHeader: React.FC<DashboardHeaderProps> = ({
  health,
  isHealthChecking,
  onRefreshHealth,
  activeTab,
  onSelectTab,
  historyCount,
  onOpenInfo,
}) => {
  const isOnline = health !== null && health.status === 'ok';

  return (
    <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-18">
          {/* Brand Identity */}
          <div className="flex items-center space-x-3.5">
            <div className="relative flex items-center justify-center w-11 h-11 rounded-xl bg-gradient-to-br from-blue-600/30 via-cyan-500/20 to-slate-900 border border-cyan-500/40 shadow-[0_0_20px_rgba(6,182,212,0.15)]">
              <Shield className="w-6 h-6 text-cyan-400" />
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xl font-bold tracking-tight text-white font-mono">
                  DEEP<span className="text-cyan-400">SHIELD</span>
                </span>
                <span className="px-2 py-0.5 text-[10px] uppercase tracking-wider font-semibold rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800/50">
                  ViT-Core
                </span>
              </div>
              <p className="text-xs text-slate-400 tracking-normal font-sans">
                See the fake before it fools you.
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="hidden md:flex items-center space-x-1.5 bg-slate-900/90 p-1.5 rounded-xl border border-slate-800">
            <button
              type="button"
              onClick={() => onSelectTab('analyzer')}
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'analyzer'
                  ? 'bg-blue-600/20 text-cyan-300 border border-cyan-500/30 shadow-[0_0_12px_rgba(6,182,212,0.15)]'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>Inspection Lab</span>
            </button>
            <button
              type="button"
              onClick={() => onSelectTab('history')}
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'history'
                  ? 'bg-blue-600/20 text-cyan-300 border border-cyan-500/30 shadow-[0_0_12px_rgba(6,182,212,0.15)]'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <History className="w-3.5 h-3.5" />
              <span>Session Log</span>
              {historyCount > 0 && (
                <span className="ml-1.5 px-1.5 py-0.2 text-[10px] rounded-full bg-slate-800 text-cyan-300 font-mono">
                  {historyCount}
                </span>
              )}
            </button>
            <button
              type="button"
              onClick={onOpenInfo}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 transition-all"
              title="Architecture details & Grad-CAM specification"
            >
              <Info className="w-3.5 h-3.5" />
              <span>Docs</span>
            </button>
          </nav>

          {/* System Status Indicators */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2.5 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs">
              <span className="relative flex h-2 w-2">
                <span
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                    isOnline ? 'bg-emerald-400' : 'bg-rose-400'
                  }`}
                />
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${
                    isOnline ? 'bg-emerald-500' : 'bg-rose-500'
                  }`}
                />
              </span>
              <span className="text-slate-300 font-mono">
                {isHealthChecking
                  ? 'Checking...'
                  : isOnline
                  ? 'Backend Online'
                  : 'Offline'}
              </span>

              {isOnline && health && (
                <>
                  <span className="text-slate-600">|</span>
                  <div className="flex items-center space-x-1 text-slate-400 text-[11px] font-mono">
                    <Cpu className="w-3 h-3 text-cyan-400" />
                    <span>{health.device?.toUpperCase() || 'CPU'}</span>
                  </div>
                </>
              )}

              <button
                type="button"
                onClick={onRefreshHealth}
                disabled={isHealthChecking}
                aria-label="Refresh backend telemetry"
                className="text-slate-400 hover:text-cyan-400 transition-colors ml-1 p-0.5 rounded focus:outline-none"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isHealthChecking ? 'animate-spin text-cyan-400' : ''}`} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
