import React, { useEffect, useState } from 'react';
import { fetchHistory, deleteHistoryItem, fetchHistoryDetail, HistoryItem, HistoryDetail } from '../services/api';

export const HistoryPage: React.FC = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<HistoryDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const records = await fetchHistory();
      setHistory(records);
    } catch (err: any) {
      setError(err.message || 'Failed to load prediction history.');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm(`Delete prediction record #${id}?`)) {
      return;
    }

    try {
      await deleteHistoryItem(id);
      setHistory((prev) => prev.filter((item) => item.id !== id));
      if (selectedRecord?.id === id) {
        setSelectedRecord(null);
      }
    } catch (err: any) {
      alert(`Delete failed: ${err.message}`);
    }
  };

  const handleOpenDetail = async (id: number) => {
    try {
      const detail = await fetchHistoryDetail(id);
      setSelectedRecord(detail);
    } catch (err: any) {
      alert(`Could not load record details: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <section className="bg-white border border-[#E5E7EB] p-6 rounded-md shadow-sm flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-[#111827]">
            Prediction History
          </h2>
          <p className="text-xs text-[#6B7280] mt-0.5">
            Log of previously evaluated audio recordings and dual-model predictions.
          </p>
        </div>
        <button
          onClick={loadHistory}
          className="px-3.5 py-1.5 text-xs font-medium border border-[#E5E7EB] rounded bg-[#F9FAFB] hover:bg-neutral-100 text-[#374151]"
        >
          Refresh Log
        </button>
      </section>

      {error && (
        <div className="p-3 bg-red-50 border border-red-200 text-red-700 text-xs rounded">
          {error}
        </div>
      )}

      {loading ? (
        <div className="bg-white border border-[#E5E7EB] p-12 text-center text-xs text-[#6B7280] rounded-md shadow-sm">
          Loading history records...
        </div>
      ) : history.length === 0 ? (
        <div className="bg-white border border-[#E5E7EB] p-12 text-center text-xs text-[#6B7280] rounded-md shadow-sm">
          No prediction history yet. Upload an audio track on the Classification tab to generate records.
        </div>
      ) : (
        <div className="bg-white border border-[#E5E7EB] rounded-md shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-[#E5E7EB] text-xs">
              <thead className="bg-[#F9FAFB] text-[#6B7280]">
                <tr>
                  <th className="px-4 py-3 text-left font-medium">Filename</th>
                  <th className="px-4 py-3 text-left font-medium">Timestamp</th>
                  <th className="px-4 py-3 text-left font-medium">ANN Genre</th>
                  <th className="px-4 py-3 text-right font-medium">ANN Conf.</th>
                  <th className="px-4 py-3 text-left font-medium">CNN Genre</th>
                  <th className="px-4 py-3 text-right font-medium">CNN Conf.</th>
                  <th className="px-4 py-3 text-right font-medium">Duration</th>
                  <th className="px-4 py-3 text-center font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E5E7EB] bg-white">
                {history.map((item) => (
                  <tr key={item.id} className="hover:bg-neutral-50">
                    <td className="px-4 py-2.5 font-medium text-[#111827]">{item.filename}</td>
                    <td className="px-4 py-2.5 text-[#6B7280] whitespace-nowrap">
                      {new Date(item.timestamp).toLocaleString()}
                    </td>
                    <td className="px-4 py-2.5 capitalize font-medium text-[#111827]">
                      {item.ann_genre || '-'}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-[#4B5563]">
                      {item.ann_confidence ? `${(item.ann_confidence * 100).toFixed(1)}%` : '-'}
                    </td>
                    <td className="px-4 py-2.5 capitalize font-medium text-[#111827]">
                      {item.cnn_genre || '-'}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-[#4B5563]">
                      {item.cnn_confidence ? `${(item.cnn_confidence * 100).toFixed(1)}%` : '-'}
                    </td>
                    <td className="px-4 py-2.5 text-right text-[#6B7280] font-mono">
                      {item.audio_duration ? `${item.audio_duration.toFixed(1)}s` : '-'}
                    </td>
                    <td className="px-4 py-2.5 text-center whitespace-nowrap space-x-2">
                      <button
                        onClick={() => handleOpenDetail(item.id)}
                        className="px-2.5 py-1 text-xs border border-[#E5E7EB] rounded hover:bg-neutral-100 text-[#374151]"
                      >
                        Inspect
                      </button>
                      <button
                        onClick={() => handleDelete(item.id)}
                        className="px-2.5 py-1 text-xs text-red-600 hover:text-red-800"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Record Inspection Modal */}
      {selectedRecord && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center p-4 z-50">
          <div className="bg-white border border-[#E5E7EB] rounded-md p-6 max-w-lg w-full space-y-4 shadow-md">
            <div className="flex justify-between items-center border-b border-[#E5E7EB] pb-3">
              <h3 className="text-sm font-semibold text-[#111827]">
                Record #{selectedRecord.id}: {selectedRecord.filename}
              </h3>
              <button
                onClick={() => setSelectedRecord(null)}
                className="text-neutral-400 hover:text-neutral-600 text-sm font-bold"
              >
                ✕
              </button>
            </div>

            <div className="text-xs space-y-3">
              <div className="grid grid-cols-2 gap-3 p-3 bg-[#F9FAFB] border border-[#E5E7EB] rounded">
                <div>
                  <span className="text-[#6B7280] block text-[11px]">Timestamp</span>
                  <span className="font-mono text-[#111827]">
                    {new Date(selectedRecord.timestamp).toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-[#6B7280] block text-[11px]">Audio Duration</span>
                  <span className="text-[#111827] font-medium font-mono">
                    {selectedRecord.audio_duration?.toFixed(2)} seconds
                  </span>
                </div>
              </div>

              {/* ANN breakdown */}
              {selectedRecord.ann_top3 && (
                <div className="border border-[#E5E7EB] rounded p-3 space-y-1.5">
                  <span className="font-semibold text-[#111827] block mb-1">
                    ANN Top Probabilities (Predicted: {selectedRecord.ann_genre})
                  </span>
                  <div className="space-y-1">
                    {selectedRecord.ann_top3.map((g, idx) => (
                      <div key={idx} className="flex justify-between">
                        <span className="capitalize text-[#374151]">{g.genre}</span>
                        <span className="font-mono text-[#6B7280]">
                          {(g.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* CNN breakdown */}
              {selectedRecord.cnn_top3 && (
                <div className="border border-[#E5E7EB] rounded p-3 space-y-1.5">
                  <span className="font-semibold text-[#111827] block mb-1">
                    CNN Top Probabilities (Predicted: {selectedRecord.cnn_genre})
                  </span>
                  <div className="space-y-1">
                    {selectedRecord.cnn_top3.map((g, idx) => (
                      <div key={idx} className="flex justify-between">
                        <span className="capitalize text-[#374151]">{g.genre}</span>
                        <span className="font-mono text-[#6B7280]">
                          {(g.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="text-right pt-2 border-t border-[#E5E7EB]">
              <button
                onClick={() => setSelectedRecord(null)}
                className="px-4 py-1.5 text-xs font-medium bg-[#111827] text-white rounded hover:bg-neutral-800"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
