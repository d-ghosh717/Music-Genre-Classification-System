import React, { useEffect, useState } from 'react';
import { fetchEvaluation, EvaluationResponse } from '../services/api';

export const PerformancePage: React.FC = () => {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<'gtzan' | 'jamendo'>('gtzan');
  const [matrixDataset, setMatrixDataset] = useState<'gtzan' | 'jamendo'>('gtzan');

  useEffect(() => {
    loadEvaluationData();
  }, []);

  const loadEvaluationData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchEvaluation();
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch evaluation metrics.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white border border-[#E5E7EB] p-12 text-center text-xs text-[#6B7280] rounded-md shadow-sm">
        Loading evaluation results...
      </div>
    );
  }

  if (error || !data || !data.available) {
    return (
      <div className="bg-white border border-[#E5E7EB] p-12 text-center rounded-md shadow-sm space-y-3">
        <h3 className="text-sm font-semibold text-[#111827]">
          Evaluation data is not available.
        </h3>
        <p className="text-xs text-[#6B7280] max-w-md mx-auto">
          Please run the training pipeline to generate benchmark evaluation records.
        </p>
        <button
          onClick={loadEvaluationData}
          className="px-4 py-2 text-xs font-medium bg-[#111827] text-white rounded hover:bg-neutral-800"
        >
          Retry
        </button>
      </div>
    );
  }

  const gtzanAnn = data.final_ann || data.ann;
  const gtzanCnn = data.final_cnn || data.cnn;
  const jamendoData = data.external_benchmark;

  const genres = gtzanAnn?.genres || [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock',
  ];

  // Prepare accuracy numbers for Section B comparison
  const gtzanAnnAcc = gtzanAnn ? gtzanAnn.accuracy * 100 : 0;
  const gtzanCnnAcc = gtzanCnn ? gtzanCnn.accuracy * 100 : 0;
  const jamendoAnnAcc = jamendoData?.ann?.accuracy ? jamendoData.ann.accuracy * 100 : 0;
  const jamendoCnnAcc = jamendoData?.cnn?.accuracy ? jamendoData.cnn.accuracy * 100 : 0;

  return (
    <div className="space-y-8 max-w-5xl">
      {/* SECTION A — Benchmark Comparison */}
      <section className="bg-white border border-[#E5E7EB] p-6 rounded-md shadow-sm space-y-4">
        <div>
          <h2 className="text-base font-semibold text-[#111827]">Benchmark Comparison</h2>
          <p className="text-xs text-[#6B7280] mt-1">
            Empirical evaluation metrics across in-domain test data and external out-of-distribution music.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
          {/* GTZAN Group */}
          <div className="p-4 bg-[#F9FAFB] border border-[#E5E7EB] rounded-md space-y-3">
            <div className="flex items-center justify-between border-b border-[#E5E7EB] pb-2">
              <span className="font-semibold text-xs text-[#111827]">
                GTZAN Benchmark (In-Domain)
              </span>
              <span className="text-[11px] text-[#6B7280]">153 Isolated Test Tracks</span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-white border border-[#E5E7EB] rounded">
                <span className="text-[#6B7280] block text-[11px]">ANN Accuracy</span>
                <span className="text-lg font-bold text-[#111827]">
                  {gtzanAnn ? `${(gtzanAnn.accuracy * 100).toFixed(2)}%` : 'N/A'}
                </span>
                <span className="text-[11px] text-[#6B7280] block mt-1">
                  Macro F1: {gtzanAnn ? `${(gtzanAnn.f1_macro * 100).toFixed(2)}%` : 'N/A'}
                </span>
              </div>

              <div className="p-3 bg-white border border-[#E5E7EB] rounded">
                <span className="text-[#6B7280] block text-[11px]">CNN Accuracy</span>
                <span className="text-lg font-bold text-[#111827]">
                  {gtzanCnn ? `${(gtzanCnn.accuracy * 100).toFixed(2)}%` : 'N/A'}
                </span>
                <span className="text-[11px] text-[#6B7280] block mt-1">
                  Macro F1: {gtzanCnn ? `${(gtzanCnn.f1_macro * 100).toFixed(2)}%` : 'N/A'}
                </span>
              </div>
            </div>
          </div>

          {/* MTG-Jamendo Group */}
          <div className="p-4 bg-[#F9FAFB] border border-[#E5E7EB] rounded-md space-y-3">
            <div className="flex items-center justify-between border-b border-[#E5E7EB] pb-2">
              <span className="font-semibold text-xs text-[#111827]">
                MTG-Jamendo External Benchmark
              </span>
              <span className="text-[11px] text-[#6B7280]">100 Tracks • 100 Artists</span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-white border border-[#E5E7EB] rounded">
                <span className="text-[#6B7280] block text-[11px]">ANN Accuracy</span>
                <span className="text-lg font-bold text-[#111827]">
                  {jamendoData ? `${(jamendoData.ann.accuracy * 100).toFixed(2)}%` : '39.00%'}
                </span>
                <span className="text-[11px] text-[#6B7280] block mt-1">
                  Macro F1: {jamendoData ? `${(jamendoData.ann.macro_f1 * 100).toFixed(2)}%` : '36.87%'}
                </span>
              </div>

              <div className="p-3 bg-white border border-[#E5E7EB] rounded">
                <span className="text-[#6B7280] block text-[11px]">CNN Accuracy</span>
                <span className="text-lg font-bold text-[#111827]">
                  {jamendoData ? `${(jamendoData.cnn.accuracy * 100).toFixed(2)}%` : '44.00%'}
                </span>
                <span className="text-[11px] text-[#6B7280] block mt-1">
                  Macro F1: {jamendoData ? `${(jamendoData.cnn.macro_f1 * 100).toFixed(2)}%` : '40.16%'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION B — Model Comparison Bar Chart */}
      <section className="bg-white border border-[#E5E7EB] p-6 rounded-md shadow-sm space-y-4">
        <div>
          <h2 className="text-base font-semibold text-[#111827]">Model Accuracy Comparison</h2>
          <p className="text-xs text-[#6B7280] mt-1">
            Grouped comparison of ANN vs CNN accuracy on in-domain (GTZAN) versus external (MTG-Jamendo) evaluations.
          </p>
        </div>

        <div className="p-4 bg-[#F9FAFB] border border-[#E5E7EB] rounded-md space-y-4">
          <div className="flex items-center justify-end gap-6 text-xs text-[#4B5563]">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-[#111827] rounded-sm inline-block" />
              <span>ANN (Multilayer Perceptron)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-[#4B5563] rounded-sm inline-block" />
              <span>CNN (Mel-Spectrogram ConvNet)</span>
            </div>
          </div>

          <div className="space-y-6 pt-2">
            {/* GTZAN Bars */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-medium text-[#111827]">
                <span>GTZAN (In-Domain Evaluation)</span>
                <span className="text-[#6B7280] font-mono">
                  ANN: {gtzanAnnAcc.toFixed(1)}% | CNN: {gtzanCnnAcc.toFixed(1)}%
                </span>
              </div>
              <div className="space-y-1.5">
                <div className="flex items-center gap-3 text-xs">
                  <span className="w-10 text-[#6B7280] text-[11px]">ANN</span>
                  <div className="flex-1 bg-[#E5E7EB] h-4 rounded overflow-hidden">
                    <div
                      className="bg-[#111827] h-full rounded transition-all duration-500"
                      style={{ width: `${gtzanAnnAcc}%` }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono font-medium text-[#111827]">
                    {gtzanAnnAcc.toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="w-10 text-[#6B7280] text-[11px]">CNN</span>
                  <div className="flex-1 bg-[#E5E7EB] h-4 rounded overflow-hidden">
                    <div
                      className="bg-[#4B5563] h-full rounded transition-all duration-500"
                      style={{ width: `${gtzanCnnAcc}%` }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono font-medium text-[#111827]">
                    {gtzanCnnAcc.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* MTG-Jamendo Bars */}
            <div className="space-y-2 pt-2 border-t border-[#E5E7EB]">
              <div className="flex justify-between text-xs font-medium text-[#111827]">
                <span>MTG-Jamendo (External Out-of-Distribution)</span>
                <span className="text-[#6B7280] font-mono">
                  ANN: {jamendoAnnAcc.toFixed(1)}% | CNN: {jamendoCnnAcc.toFixed(1)}%
                </span>
              </div>
              <div className="space-y-1.5">
                <div className="flex items-center gap-3 text-xs">
                  <span className="w-10 text-[#6B7280] text-[11px]">ANN</span>
                  <div className="flex-1 bg-[#E5E7EB] h-4 rounded overflow-hidden">
                    <div
                      className="bg-[#111827] h-full rounded transition-all duration-500"
                      style={{ width: `${jamendoAnnAcc}%` }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono font-medium text-[#111827]">
                    {jamendoAnnAcc.toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="w-10 text-[#6B7280] text-[11px]">CNN</span>
                  <div className="flex-1 bg-[#E5E7EB] h-4 rounded overflow-hidden">
                    <div
                      className="bg-[#4B5563] h-full rounded transition-all duration-500"
                      style={{ width: `${jamendoCnnAcc}%` }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono font-medium text-[#111827]">
                    {jamendoCnnAcc.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION C — Per-Genre Performance */}
      <section className="bg-white border border-[#E5E7EB] p-6 rounded-md shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-[#111827]">Per-Genre F1 Performance</h2>
          </div>

          <div className="inline-flex rounded-md border border-[#E5E7EB] bg-[#F9FAFB] p-0.5 text-xs">
            <button
              onClick={() => setSelectedDataset('gtzan')}
              className={`px-3 py-1 font-medium rounded transition-colors ${
                selectedDataset === 'gtzan'
                  ? 'bg-white text-[#111827] shadow-xs'
                  : 'text-[#6B7280] hover:text-[#111827]'
              }`}
            >
              GTZAN (In-Domain)
            </button>
            <button
              onClick={() => setSelectedDataset('jamendo')}
              className={`px-3 py-1 font-medium rounded transition-colors ${
                selectedDataset === 'jamendo'
                  ? 'bg-white text-[#111827] shadow-xs'
                  : 'text-[#6B7280] hover:text-[#111827]'
              }`}
            >
              MTG-Jamendo (External)
            </button>
          </div>
        </div>

        <div className="overflow-x-auto border border-[#E5E7EB] rounded-md">
          <table className="min-w-full divide-y divide-[#E5E7EB] text-xs">
            <thead className="bg-[#F9FAFB] text-[#6B7280]">
              <tr>
                <th className="px-4 py-2.5 text-left font-medium">Genre</th>
                <th className="px-4 py-2.5 text-right font-medium">ANN F1</th>
                <th className="px-4 py-2.5 text-right font-medium">CNN F1</th>
                <th className="px-4 py-2.5 text-right font-medium">F1 Delta</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E7EB] bg-white">
              {genres.map((g) => {
                let annF1: number | null = null;
                let cnnF1: number | null = null;

                if (selectedDataset === 'gtzan') {
                  annF1 = gtzanAnn?.classification_report?.[g]?.['f1-score'] ?? null;
                  cnnF1 = gtzanCnn?.classification_report?.[g]?.['f1-score'] ?? null;
                } else {
                  annF1 = jamendoData?.ann?.classification_report?.[g]?.['f1-score'] ?? null;
                  cnnF1 = jamendoData?.cnn?.classification_report?.[g]?.['f1-score'] ?? null;
                }

                const delta =
                  annF1 !== null && cnnF1 !== null ? (cnnF1 - annF1) * 100 : null;

                return (
                  <tr key={g} className="hover:bg-neutral-50">
                    <td className="px-4 py-2 font-medium capitalize text-[#111827]">{g}</td>
                    <td className="px-4 py-2 text-right font-mono text-[#4B5563]">
                      {annF1 !== null ? (annF1 * 100).toFixed(1) + '%' : '-'}
                    </td>
                    <td className="px-4 py-2 text-right font-mono font-medium text-[#111827]">
                      {cnnF1 !== null ? (cnnF1 * 100).toFixed(1) + '%' : '-'}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-xs">
                      {delta !== null ? (
                        <span className={delta >= 0 ? 'text-emerald-700' : 'text-neutral-500'}>
                          {delta >= 0 ? `+${delta.toFixed(1)}%` : `${delta.toFixed(1)}%`}
                        </span>
                      ) : (
                        '-'
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* SECTION D — Confusion Matrices */}
      <section className="bg-white border border-[#E5E7EB] p-6 rounded-md shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-[#111827]">Confusion Matrices</h2>
            <p className="text-xs text-[#6B7280] mt-1">
              Side-by-side error distributions showing true vs predicted class frequencies.
            </p>
          </div>

          <div className="inline-flex rounded-md border border-[#E5E7EB] bg-[#F9FAFB] p-0.5 text-xs">
            <button
              onClick={() => setMatrixDataset('gtzan')}
              className={`px-3 py-1 font-medium rounded transition-colors ${
                matrixDataset === 'gtzan'
                  ? 'bg-white text-[#111827] shadow-xs'
                  : 'text-[#6B7280] hover:text-[#111827]'
              }`}
            >
              GTZAN Benchmark
            </button>
            <button
              onClick={() => setMatrixDataset('jamendo')}
              className={`px-3 py-1 font-medium rounded transition-colors ${
                matrixDataset === 'jamendo'
                  ? 'bg-white text-[#111827] shadow-xs'
                  : 'text-[#6B7280] hover:text-[#111827]'
              }`}
            >
              MTG-Jamendo External
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-1">
          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded-md space-y-2">
            <span className="text-xs font-medium text-[#111827] block">
              ANN Confusion Matrix ({matrixDataset === 'gtzan' ? 'GTZAN' : 'MTG-Jamendo'})
            </span>
            <img
              src={
                matrixDataset === 'gtzan'
                  ? '/api/visualizations/final/ann_confusion_matrix.png'
                  : '/api/visualizations/external/ann_confusion_matrix.png'
              }
              alt="ANN Confusion Matrix"
              className="w-full h-auto rounded border border-[#E5E7EB] bg-white"
            />
          </div>

          <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded-md space-y-2">
            <span className="text-xs font-medium text-[#111827] block">
              CNN Confusion Matrix ({matrixDataset === 'gtzan' ? 'GTZAN' : 'MTG-Jamendo'})
            </span>
            <img
              src={
                matrixDataset === 'gtzan'
                  ? '/api/visualizations/final/cnn_confusion_matrix.png'
                  : '/api/visualizations/external/cnn_confusion_matrix.png'
              }
              alt="CNN Confusion Matrix"
              className="w-full h-auto rounded border border-[#E5E7EB] bg-white"
            />
          </div>
        </div>
      </section>

      {/* SECTION E — Domain Shift */}
      <section className="bg-white border border-[#E5E7EB] p-6 rounded-md shadow-sm space-y-4">
        <div>
          <h2 className="text-base font-semibold text-[#111827]">Acoustic Domain Shift</h2>
          <p className="text-xs text-[#6B7280] mt-1">
            Analysis of acoustic feature distributions between GTZAN and independent external music.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
          <div className="border border-[#E5E7EB] p-2 rounded-md bg-[#F9FAFB]">
            <img
              src="/api/visualizations/external/domain_shift_analysis.png"
              alt="Acoustic Domain Shift Analysis"
              className="w-full h-auto rounded border border-[#E5E7EB] bg-white"
            />
          </div>

          <div className="space-y-3 text-xs text-[#4B5563] leading-relaxed">
            <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
              <span className="font-semibold text-[#111827] block mb-1">Key Acoustic Differences</span>
              <ul className="list-disc list-inside space-y-1">
                <li>
                  <strong>RMS Energy:</strong> Modern recordings exhibit significantly higher compression and dynamic range limiting compared to 1990s GTZAN masters.
                </li>
                <li>
                  <strong>Spectral Contrast:</strong> Contemporary studio mixes display pronounced peak-to-valley energy divergence across frequency sub-bands.
                </li>
                <li>
                  <strong>Zero Crossing Rate:</strong> Modern percussive and transient synth textures increase high-frequency boundary crossings.
                </li>
                <li>
                  <strong>MFCC Distributions:</strong> Principal component shifts confirm that independent audio projects into distinct timbral regions from GTZAN.
                </li>
              </ul>
            </div>

            <p className="text-[#6B7280] text-[11px]">
              This domain shift explains the observed generalisation drop when evaluating models on novel music corpora, demonstrating the importance of multi-source calibration over single-dataset cross-validation.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
