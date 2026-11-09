/**
 * API client service for communicating with FastAPI backend.
 */

export interface GenreProbability {
  genre: string;
  probability: number;
}

export interface UncertaintyInfo {
  is_uncertain: boolean;
  entropy: number;
  margin: number;
  reason: string;
}

export interface ModelPrediction {
  predicted_genre: string;
  confidence: number;
  top3: GenreProbability[];
  model_type: string;
  num_segments_evaluated?: number;
  uncertainty?: UncertaintyInfo;
}

export interface PredictionResponse {
  filename: string;
  audio_duration: number;
  ann_prediction: ModelPrediction | null;
  cnn_prediction: ModelPrediction | null;
  models_agree: boolean | null;
  ann_error: string | null;
  cnn_error: string | null;
  uncertainty_status?: string | null;
  out_of_taxonomy_warning?: string | null;
}

export interface HealthStatus {
  status: string;
  ann_model_available: boolean;
  cnn_model_available: boolean;
  dataset_available: boolean;
  model_details?: {
    ann?: {
      loaded: boolean;
      checkpoint_path: string | null;
      scaler_features: number | null;
      is_decontaminated: boolean;
      training_date?: string;
      test_accuracy?: number;
    };
    cnn?: {
      loaded: boolean;
      checkpoint_path: string | null;
      is_decontaminated: boolean;
      training_date?: string;
      test_accuracy?: number;
    };
    active_version?: string;
  };
}

export interface DatasetStatus {
  exists: boolean;
  genres_found: string[];
  genres_missing: string[];
  genre_counts: Record<string, number>;
  total_files: number;
  total_valid: number;
  total_invalid: number;
  is_valid: boolean;
  summary: string;
}

export interface EvaluationMetrics {
  accuracy: number;
  precision_macro: number;
  precision_weighted: number;
  recall_macro: number;
  recall_weighted: number;
  f1_macro: number;
  f1_weighted: number;
  test_loss: number;
  classification_report: Record<string, any>;
  confusion_matrix: number[][];
  genres: string[];
  training_duration_seconds?: number;
  epochs_trained?: number;
}

export interface BaselineModelStats {
  model_name: string;
  accuracy: number;
  f1_macro: number;
  f1_weighted: number;
  test_loss: number;
  feature_count?: number;
  model_size_kb?: number;
}

export interface BaselineVsImproved {
  baseline_ann: BaselineModelStats;
  improved_ann: BaselineModelStats;
  baseline_cnn: BaselineModelStats;
  improved_cnn: BaselineModelStats;
  ann_gain: {
    accuracy_delta: number;
    f1_macro_delta: number;
  };
  cnn_gain: {
    accuracy_delta: number;
    f1_macro_delta: number;
  };
}

export interface ExternalEvaluationItem {
  filename: string;
  source: string;
  actual_genre: string;
  genre_mapping: string;
  is_in_taxonomy: boolean;
  duration_seconds: number;
  ann_prediction: string;
  ann_confidence: number;
  ann_uncertain: boolean;
  cnn_prediction: string;
  cnn_confidence: number;
  cnn_uncertain: boolean;
  uncertainty_status: string;
  out_of_taxonomy_warning?: string | null;
  ann_correct: boolean | null;
  cnn_correct: boolean | null;
}

export interface FinalBenchmark {
  benchmark_type: string;
  clean_split_manifest: string;
  decontamination_summary: {
    total_valid_tracks: number;
    train_tracks: number;
    val_tracks: number;
    test_tracks: number;
    excluded_corrupted: string[];
    excluded_contradictory: string[];
    retained_counterpart: string;
    duplicate_groups_isolated: number;
    leakage_count: number;
  };
  final_ann: {
    model_name: string;
    accuracy: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
    weighted_f1: number;
    test_loss: number;
    evaluated_tracks: number;
  };
  final_cnn: {
    model_name: string;
    accuracy: number;
    macro_precision: number;
    macro_recall: number;
    macro_f1: number;
    weighted_f1: number;
    test_loss: number;
    evaluated_tracks: number;
  };
  architectural_comparison: {
    accuracy_difference: number;
    macro_f1_difference: number;
    superior_model: string;
  };
  historical_evolution: {
    baseline_contaminated: {
      ann_accuracy: number;
      ann_f1_macro: number;
      cnn_accuracy: number;
      cnn_f1_macro: number;
      split_notes: string;
    };
    phase7_improved_contaminated: {
      ann_accuracy: number;
      ann_f1_macro: number;
      cnn_accuracy: number;
      cnn_f1_macro: number;
      split_notes: string;
    };
    phase8_final_clean: {
      ann_accuracy: number;
      ann_f1_macro: number;
      cnn_accuracy: number;
      cnn_f1_macro: number;
      split_notes: string;
    };
  };
}

export interface ExternalBenchmarkResults {
  dataset_name: string;
  num_tracks: number;
  genres: string[];
  agreement_rate: number;
  ann: {
    accuracy: number;
    macro_f1: number;
    weighted_f1: number;
    mean_confidence: number;
    mean_entropy: number;
    classification_report: Record<string, any>;
    confusion_matrix: number[][];
  };
  cnn: {
    accuracy: number;
    macro_f1: number;
    weighted_f1: number;
    mean_confidence: number;
    mean_entropy: number;
    classification_report: Record<string, any>;
    confusion_matrix: number[][];
  };
}

export interface EvaluationResponse {
  available: boolean;
  ann: EvaluationMetrics | null;
  cnn: EvaluationMetrics | null;
  baseline_vs_improved?: BaselineVsImproved | null;
  external_evaluation?: ExternalEvaluationItem[] | null;
  final_benchmark?: FinalBenchmark | null;
  final_ann?: EvaluationMetrics | null;
  final_cnn?: EvaluationMetrics | null;
  decontamination_audit?: any | null;
  external_benchmark?: ExternalBenchmarkResults | null;
}

export interface HistoryItem {
  id: number;
  filename: string;
  timestamp: string;
  ann_genre: string | null;
  cnn_genre: string | null;
  ann_confidence: number | null;
  cnn_confidence: number | null;
  audio_duration: number | null;
}

export interface HistoryDetail extends HistoryItem {
  ann_top3?: GenreProbability[];
  cnn_top3?: GenreProbability[];
}

const BASE_URL = '/api';

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error('Failed to fetch system health status.');
  return res.json();
}

export async function fetchDatasetStatus(): Promise<DatasetStatus> {
  const res = await fetch(`${BASE_URL}/dataset/status`);
  if (!res.ok) throw new Error('Failed to fetch dataset status.');
  return res.json();
}

export async function fetchEvaluation(): Promise<EvaluationResponse> {
  const res = await fetch(`${BASE_URL}/evaluation`);
  if (!res.ok) throw new Error('Failed to fetch model evaluation data.');
  return res.json();
}

export async function fetchHistory(): Promise<HistoryItem[]> {
  const res = await fetch(`${BASE_URL}/history`);
  if (!res.ok) throw new Error('Failed to fetch prediction history.');
  return res.json();
}

export async function fetchHistoryDetail(id: number): Promise<HistoryDetail> {
  const res = await fetch(`${BASE_URL}/history/${id}`);
  if (!res.ok) throw new Error(`Failed to fetch history record ${id}.`);
  return res.json();
}

export async function deleteHistoryItem(id: number): Promise<void> {
  const res = await fetch(`${BASE_URL}/history/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`Failed to delete record ${id}.`);
}

export async function predictAudioFile(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Prediction request failed.' }));
    throw new Error(errorData.detail || 'Prediction request failed.');
  }

  return res.json();
}
