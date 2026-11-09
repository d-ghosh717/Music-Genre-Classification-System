import React from 'react';

interface NavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  modelsStatus?: {
    ann: boolean;
    cnn: boolean;
  };
}

export const Navigation: React.FC<NavigationProps> = ({
  activeTab,
  setActiveTab,
  modelsStatus,
}) => {
  const tabs = [
    { id: 'classification', label: 'Audio Classification' },
    { id: 'performance', label: 'Model Performance' },
    { id: 'history', label: 'Prediction History' },
    { id: 'methodology', label: 'Methodology' },
  ];

  return (
    <header className="border-b border-[#E5E7EB] bg-white sticky top-0 z-40">
      <div className="max-w-6xl mx-auto px-6 py-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-[#111827]">
            Music Genre Classification System
          </h1>
          <p className="text-xs text-[#6B7280] mt-0.5">
            Comparative Analysis of ANN and CNN Models Using GTZAN Dataset
          </p>
        </div>

        <div className="flex items-center gap-4 text-xs font-medium text-[#4B5563]">
          <span className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                modelsStatus?.ann ? 'bg-emerald-500' : 'bg-amber-400'
              }`}
            />
            ANN: {modelsStatus?.ann ? 'Loaded' : 'Not Loaded'}
          </span>
          <span className="text-[#E5E7EB]">|</span>
          <span className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                modelsStatus?.cnn ? 'bg-emerald-500' : 'bg-amber-400'
              }`}
            />
            CNN: {modelsStatus?.cnn ? 'Loaded' : 'Not Loaded'}
          </span>
        </div>
      </div>

      <nav className="max-w-6xl mx-auto px-6 flex gap-8 -mb-px border-t border-[#F3F4F6]">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-3 text-xs sm:text-sm font-medium border-b-2 transition-colors ${
                isActive
                  ? 'border-[#111827] text-[#111827]'
                  : 'border-transparent text-[#6B7280] hover:text-[#111827]'
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
};
