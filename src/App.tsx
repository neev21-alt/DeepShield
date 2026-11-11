import React, { useState, useEffect, useCallback } from 'react';
import { DashboardHeader } from './components/DashboardHeader';
import { ImageDropzone } from './components/ImageDropzone';
import { AnalysisResultsCard } from './components/AnalysisResultsCard';
import { AnalysisHistoryPanel } from './components/AnalysisHistoryPanel';
import { SystemInfoModal } from './components/SystemInfoModal';
import { AnalysisResponse, HealthResponse, HistoryItem } from './types/analysis';
import { checkBackendHealth, analyzeFacialImage } from './services/api';
import { getSampleFile } from './services/sampleData';
import { Shield, Sparkles } from 'lucide-react';

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isHealthChecking, setIsHealthChecking] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<'analyzer' | 'history'>('analyzer');
  const [isInfoModalOpen, setIsInfoModalOpen] = useState<boolean>(false);

  // Upload & Analysis State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Session-only in-memory history
  const [history, setHistory] = useState<HistoryItem[]>([]);

  // Health check on initial mount
  const fetchHealth = useCallback(async () => {
    setIsHealthChecking(true);
    try {
      const data = await checkBackendHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setIsHealthChecking(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    // Poll health periodically every 30 seconds
    const timer = setInterval(fetchHealth, 30000);
    return () => clearInterval(timer);
  }, [fetchHealth]);

  // Handle file selection
  const handleFileSelect = useCallback((file: File) => {
    // Client-side validation matching backend
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setErrorMessage('Unsupported file format. Please upload JPEG, PNG, or WEBP.');
      return;
    }
    const maxSizeBytes = 10 * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      setErrorMessage('File size exceeds 10MB limit.');
      return;
    }

    setErrorMessage(null);
    setAnalysisResult(null);
    setSelectedFile(file);

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  }, []);

  // Clear current file
  const handleClearFile = useCallback(() => {
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnalysisResult(null);
    setErrorMessage(null);
  }, [previewUrl]);

  // Load sample file
  const handleLoadSample = useCallback((sampleType: 'real' | 'fake' | 'multi' | 'none') => {
    const file = getSampleFile(sampleType);
    handleFileSelect(file);
  }, [handleFileSelect]);

  // Execute Analysis
  const handleRunAnalysis = async () => {
    if (!selectedFile) {
      setErrorMessage('Please select or upload a facial image first.');
      return;
    }

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const result = await analyzeFacialImage(selectedFile);
      setAnalysisResult(result);

      // Save to temporary session history
      const newHistoryItem: HistoryItem = {
        id: `analysis-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        filename: selectedFile.name,
        fileSizeBytes: selectedFile.size,
        previewUrl: previewUrl || '',
        result,
      };
      setHistory((prev) => [newHistoryItem, ...prev]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Analysis failed. Please check backend connection.';
      setErrorMessage(msg);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Restore analysis from history
  const handleSelectHistoryItem = (item: HistoryItem) => {
    setAnalysisResult(item.result);
    setPreviewUrl(item.previewUrl);
    setSelectedFile(new File([], item.filename));
    setActiveTab('analyzer');
  };

  // Clear session history
  const handleClearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Background Subtle Gradient Blobs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl" />
        <div className="absolute top-1/3 -right-40 w-96 h-96 bg-cyan-600/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 left-1/3 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl" />
      </div>

      {/* Main Top Header */}
      <DashboardHeader
        health={health}
        isHealthChecking={isHealthChecking}
        onRefreshHealth={fetchHealth}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        historyCount={history.length}
        onOpenInfo={() => setIsInfoModalOpen(true)}
      />

      {/* Main Workspace Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {activeTab === 'analyzer' ? (
          <div className="space-y-8">
            {/* Top Introductory Hero (hidden when results are loaded for clean focus) */}
            {!analysisResult && (
              <div className="text-center max-w-2xl mx-auto space-y-3 pb-2">
                <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-900 border border-cyan-500/30 text-cyan-400 text-xs font-mono shadow-[0_0_15px_rgba(6,182,212,0.15)]">
                  <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Explainable AI Deepfake Forensics</span>
                </div>
                <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
                  Multi-Subject Facial Deepfake Detection
                </h1>
                <p className="text-sm text-slate-400 leading-relaxed">
                  Leveraging MediaPipe facial localization, Vision Transformer (ViT-B/16) tokenized classification, and gradient-weighted class activation mapping (Grad-CAM) explainability.
                </p>
              </div>
            )}

            {/* View Switcher: Upload Dropzone vs Results View */}
            {!analysisResult ? (
              <div className="max-w-3xl mx-auto">
                <ImageDropzone
                  selectedFile={selectedFile}
                  previewUrl={previewUrl}
                  onFileSelect={handleFileSelect}
                  onClearFile={handleClearFile}
                  onAnalyze={handleRunAnalysis}
                  isAnalyzing={isAnalyzing}
                  errorMessage={errorMessage}
                  onDismissError={() => setErrorMessage(null)}
                  onLoadSample={handleLoadSample}
                />
              </div>
            ) : (
              <AnalysisResultsCard
                response={analysisResult}
                previewUrl={previewUrl}
                onReset={handleClearFile}
              />
            )}
          </div>
        ) : (
          /* Session History Tab */
          <div className="max-w-4xl mx-auto">
            <AnalysisHistoryPanel
              history={history}
              onSelectHistoryItem={handleSelectHistoryItem}
              onClearHistory={handleClearHistory}
              onBackToAnalyzer={() => setActiveTab('analyzer')}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-5 text-center text-xs text-slate-500 font-mono relative z-10">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <Shield className="w-4 h-4 text-cyan-400/80" />
            <span>DeepShield AI • CodeQuest 2026 National Hackathon</span>
          </div>
          <div>Team Code Ph@nt0m • Pretrained ViT-B/16 Forensic Pipeline</div>
        </div>
      </footer>

      {/* System Architecture Specification Modal */}
      <SystemInfoModal
        isOpen={isInfoModalOpen}
        onClose={() => setIsInfoModalOpen(false)}
        health={health}
      />
    </div>
  );
}
