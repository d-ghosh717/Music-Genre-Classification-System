import React, { useState, useEffect } from 'react';
import { Navigation } from './components/Navigation';
import { ClassificationPage } from './pages/ClassificationPage';
import { PerformancePage } from './pages/PerformancePage';
import { HistoryPage } from './pages/HistoryPage';
import { MethodologyPage } from './pages/MethodologyPage';
import { fetchHealth, HealthStatus } from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('classification');
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((err) => console.warn('Could not fetch health status:', err));
  }, [activeTab]);

  return (
    <div className="min-h-screen bg-[#F9FAFB] flex flex-col font-sans text-[#111827]">
      <Navigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        modelsStatus={
          health
            ? {
                ann: health.ann_model_available,
                cnn: health.cnn_model_available,
              }
            : undefined
        }
      />

      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-8">
        {activeTab === 'classification' && <ClassificationPage />}
        {activeTab === 'performance' && <PerformancePage />}
        {activeTab === 'history' && <HistoryPage />}
        {activeTab === 'methodology' && <MethodologyPage />}
      </main>

      <footer className="border-t border-[#E5E7EB] bg-white py-6">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between text-xs text-[#6B7280] gap-2">
          <span>Music Genre Classification System</span>
          <span className="font-mono text-[11px] text-[#9CA3AF]">
            GTZAN Dataset • ANN (MLP) vs CNN (Mel-Spectrogram)
          </span>
        </div>
      </footer>
    </div>
  );
};

export default App;
