import React, { useState, useRef, useEffect, useCallback } from 'react';
import { predictAudioFile, PredictionResponse } from '../services/api';

export const ClassificationPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isWindowDragging, setIsWindowDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounterRef = useRef<number>(0);

  const executeAnalysis = useCallback(async (file: File) => {
    setError(null);
    const validExtensions = ['.wav', '.mp3', '.ogg', '.flac'];
    const hasValidExt = validExtensions.some((ext) =>
      file.name.toLowerCase().endsWith(ext)
    );

    if (!hasValidExt) {
      setError(`Unsupported file format "${file.name}". Please upload an audio file (${validExtensions.join(', ')}).`);
      setSelectedFile(null);
      setPrediction(null);
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError('File size exceeds the 50MB maximum limit.');
      setSelectedFile(null);
      setPrediction(null);
      return;
    }

    setSelectedFile(file);
    setPrediction(null);
    setIsAnalyzing(true);

    try {
      const result = await predictAudioFile(file);
      setPrediction(result);
    } catch (err: any) {
      setError(err.message || 'An error occurred during audio classification.');
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  // Window-wide drag and drop listener: drop anywhere on the page to auto-analyze
  useEffect(() => {
    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      if (e.dataTransfer && e.dataTransfer.types && Array.from(e.dataTransfer.types).includes('Files')) {
        dragCounterRef.current += 1;
        if (dragCounterRef.current === 1) {
          setIsWindowDragging(true);
        }
      }
    };

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
      if (e.dataTransfer) {
        e.dataTransfer.dropEffect = 'copy';
      }
    };

    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault();
      dragCounterRef.current -= 1;
      if (dragCounterRef.current <= 0) {
        dragCounterRef.current = 0;
        setIsWindowDragging(false);
      }
    };

    const handleDrop = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounterRef.current = 0;
      setIsWindowDragging(false);

      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        executeAnalysis(file);
      }
    };

    window.addEventListener('dragenter', handleDragEnter);
    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('dragleave', handleDragLeave);
    window.addEventListener('drop', handleDrop);

    return () => {
      window.removeEventListener('dragenter', handleDragEnter);
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('dragleave', handleDragLeave);
      window.removeEventListener('drop', handleDrop);
    };
  }, [executeAnalysis]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      executeAnalysis(e.target.files[0]);
    }
    e.target.value = '';
  };

  const handleReanalyze = () => {
    if (selectedFile) {
      executeAnalysis(selectedFile);
    }
  };

  // Determine if prediction has uncertainty
  const isUncertain = Boolean(
    prediction?.ann_prediction?.uncertainty?.is_uncertain ||
    prediction?.cnn_prediction?.uncertainty?.is_uncertain ||
    (prediction?.ann_prediction && prediction.ann_prediction.confidence < 0.55) ||
    (prediction?.cnn_prediction && prediction.cnn_prediction.confidence < 0.55)
  );

  const confidenceDifference =
    prediction?.ann_prediction && prediction?.cnn_prediction
      ? Math.abs(prediction.ann_prediction.confidence - prediction.cnn_prediction.confidence) * 100
      : null;

  return (
    <div className="space-y-6 max-w-5xl relative">
      {/* Full-Page Drag & Drop Overlay */}
      {isWindowDragging && (
        <div className="fixed inset-0 z-50 bg-[#111827]/70 backdrop-blur-sm flex items-center justify-center p-6 pointer-events-none transition-all">
          <div className="max-w-md w-full bg-white border-2 border-dashed border-[#111827] rounded-xl p-10 text-center shadow-2xl space-y-4">
            <div className="w-16 h-16 mx-auto bg-neutral-100 rounded-full flex items-center justify-center text-[#111827]">
              <svg className="w-8 h-8 animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-bold text-[#111827]">Drop Audio File Anywhere</h3>
              <p className="text-xs text-[#4B5563] mt-1">
                Auto-analysis will start immediately on drop
              </p>
            </div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-neutral-100 rounded-full text-[11px] font-mono text-[#374151]">
              <span>MP3</span>
              <span>•</span>
              <span>WAV</span>
              <span>•</span>
              <span>FLAC</span>
              <span>•</span>
              <span>OGG</span>
              <span>(Max 50MB)</span>
            </div>
          </div>
        </div>
      )}

      {/* Upload Box */}
      <section className="bg-white border border-[#E5E7EB] p-6 rounded-md shadow-sm space-y-4">
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-[#E5E7EB] hover:border-neutral-400 bg-[#F9FAFB] rounded-md p-8 text-center cursor-pointer transition-colors"
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".wav,.mp3,.ogg,.flac"
            className="hidden"
          />

          <div className="max-w-md mx-auto space-y-2">
            <svg
              className="w-8 h-8 mx-auto text-[#9CA3AF]"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
              />
            </svg>
            <p className="text-sm font-medium text-[#111827]">
              {selectedFile ? selectedFile.name : 'Drop audio file anywhere on the page or click to select'}
            </p>
            <p className="text-xs text-[#6B7280]">
              Supported formats: MP3, WAV, FLAC, OGG (Max 50MB) • Analysis starts automatically
            </p>
          </div>
        </div>

        {isAnalyzing && (
          <div className="flex items-center justify-center gap-3 p-4 bg-neutral-50 border border-[#E5E7EB] rounded-md text-xs text-[#111827]">
            <svg className="animate-spin h-4 w-4 text-[#111827]" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
            </svg>
            <span className="font-medium">
              Analyzing audio with dual ANN and CNN architectures...
            </span>
          </div>
        )}

        {selectedFile && !isAnalyzing && (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 bg-[#F9FAFB] border border-[#E5E7EB] rounded-md text-xs">
            <div>
              <span className="font-medium text-[#111827]">{selectedFile.name}</span>
              <span className="text-[#6B7280] ml-2 font-mono">
                ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
              </span>
            </div>
            <button
              onClick={handleReanalyze}
              disabled={isAnalyzing}
              className="px-4 py-2 text-xs font-medium rounded transition-colors bg-[#111827] text-white hover:bg-neutral-800"
            >
              Re-analyse Track
            </button>
          </div>
        )}

        {error && (
          <div className="p-3 bg-red-50 border border-red-200 text-red-700 text-xs rounded">
            <strong>Error:</strong> {error}
          </div>
        )}
      </section>

      {/* Prediction Results */}
      {prediction && (
        <section className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* ANN Card */}
            <div className="bg-white border border-[#E5E7EB] p-6 rounded-md shadow-sm space-y-4">
              <div className="border-b border-[#E5E7EB] pb-3">
                <span className="text-xs font-mono uppercase tracking-wider text-[#6B7280]">
                  Model 1
                </span>
                <h3 className="text-sm font-semibold text-[#111827]">
                  ANN — Multilayer Perceptron
                </h3>
              </div>

              {prediction.ann_error ? (
                <div className="p-3 bg-amber-50 border border-amber-200 text-amber-800 text-xs rounded">
                  {prediction.ann_error}
                </div>
              ) : prediction.ann_prediction ? (
                <div className="space-y-4">
                  <div className="p-4 bg-[#F9FAFB] border border-[#E5E7EB] rounded flex items-baseline justify-between">
                    <div>
                      <span className="text-xs text-[#6B7280] block">Predicted Genre</span>
                      <span className="text-xl font-bold capitalize text-[#111827]">
                        {prediction.ann_prediction.predicted_genre}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-[#6B7280] block">Confidence</span>
                      <span className="text-lg font-semibold text-[#111827]">
                        {(prediction.ann_prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-xs font-medium text-[#6B7280] uppercase tracking-wider mb-2.5">
                      Top 3 Genre Probabilities
                    </h4>
                    <div className="space-y-2.5">
                      {prediction.ann_prediction.top3.map((item, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="capitalize font-medium text-[#111827]">{item.genre}</span>
                            <span className="text-[#6B7280] font-mono">{(item.probability * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-[#E5E7EB] h-1.5 rounded-full overflow-hidden">
                            <div
                              className="bg-[#111827] h-full rounded-full"
                              style={{ width: `${Math.min(100, Math.max(0, item.probability * 100))}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : null}
            </div>

            {/* CNN Card */}
            <div className="bg-white border border-[#E5E7EB] p-6 rounded-md shadow-sm space-y-4">
              <div className="border-b border-[#E5E7EB] pb-3">
                <span className="text-xs font-mono uppercase tracking-wider text-[#6B7280]">
                  Model 2
                </span>
                <h3 className="text-sm font-semibold text-[#111827]">
                  CNN — Convolutional Neural Network
                </h3>
              </div>

              {prediction.cnn_error ? (
                <div className="p-3 bg-amber-50 border border-amber-200 text-amber-800 text-xs rounded">
                  {prediction.cnn_error}
                </div>
              ) : prediction.cnn_prediction ? (
                <div className="space-y-4">
                  <div className="p-4 bg-[#F9FAFB] border border-[#E5E7EB] rounded flex items-baseline justify-between">
                    <div>
                      <span className="text-xs text-[#6B7280] block">Predicted Genre</span>
                      <span className="text-xl font-bold capitalize text-[#111827]">
                        {prediction.cnn_prediction.predicted_genre}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-[#6B7280] block">Confidence</span>
                      <span className="text-lg font-semibold text-[#111827]">
                        {(prediction.cnn_prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-xs font-medium text-[#6B7280] uppercase tracking-wider mb-2.5">
                      Top 3 Genre Probabilities
                    </h4>
                    <div className="space-y-2.5">
                      {prediction.cnn_prediction.top3.map((item, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span className="capitalize font-medium text-[#111827]">{item.genre}</span>
                            <span className="text-[#6B7280] font-mono">{(item.probability * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-[#E5E7EB] h-1.5 rounded-full overflow-hidden">
                            <div
                              className="bg-[#111827] h-full rounded-full"
                              style={{ width: `${Math.min(100, Math.max(0, item.probability * 100))}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>

          {/* Compact Comparison Section */}
          <div className="bg-white border border-[#E5E7EB] p-6 rounded-md shadow-sm space-y-4">
            <h3 className="text-sm font-semibold text-[#111827] border-b border-[#E5E7EB] pb-2">
              Model Comparison
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
                <span className="text-[#6B7280] block text-[11px]">Model Agreement</span>
                <span
                  className={`font-semibold text-sm ${
                    prediction.models_agree
                      ? 'text-emerald-700'
                      : 'text-[#4B5563]'
                  }`}
                >
                  {prediction.models_agree === true
                    ? 'Both Models Agree'
                    : prediction.models_agree === false
                    ? 'Models Disagree'
                    : 'N/A'}
                </span>
              </div>

              <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
                <span className="text-[#6B7280] block text-[11px]">Confidence Difference</span>
                <span className="font-semibold text-sm text-[#111827]">
                  {confidenceDifference !== null ? `${confidenceDifference.toFixed(1)}%` : 'N/A'}
                </span>
              </div>

              <div className="p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
                <span className="text-[#6B7280] block text-[11px]">Audio Duration</span>
                <span className="font-semibold text-sm text-[#111827]">
                  {prediction.audio_duration ? `${prediction.audio_duration.toFixed(1)} seconds` : 'N/A'}
                </span>
              </div>
            </div>

            {/* Concise Uncertainty / Disagreement Warning */}
            {(isUncertain || prediction.models_agree === false) && (
              <div className="p-3 bg-[#FFFBEB] border border-[#FDE68A] rounded text-xs text-[#92400E] leading-relaxed">
                <strong>Notice:</strong> The models were trained on the 10 GTZAN genres. External audio recordings, hybrid subgenres, or contemporary production styles may result in model disagreement or lower prediction confidence.
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
};
