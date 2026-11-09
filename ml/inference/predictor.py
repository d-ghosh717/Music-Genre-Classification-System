"""
Reusable inference engine for music genre classification.
Loads trained ANN and CNN models and evaluates uploaded audio files using
identical preprocessing configurations as used during training.
"""
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ml.config import (
    audio_config,
    dataset_config,
    path_config,
    spectrogram_config,
)
from ml.preprocessing.audio_loader import (
    get_audio_duration,
    load_audio,
    validate_audio_file,
)
from ml.preprocessing.feature_extraction import extract_features
from ml.preprocessing.segmentation import segment_audio
from ml.preprocessing.spectrogram import (
    compute_expected_time_frames,
    generate_mel_spectrogram,
    mel_spectrogram_to_cnn_input,
)

logger = logging.getLogger("predictor")


def compute_uncertainty_metrics(
    probs: np.ndarray,
    confidence_thresh: float = 0.55,
    entropy_thresh: float = 1.65,
    margin_thresh: float = 0.20,
) -> Dict[str, Any]:
    """
    Compute distribution entropy, prediction margin, and uncertainty flag.
    Calibrated empirically on validation set distributions.
    """
    sorted_p = np.sort(probs)[::-1]
    top1 = float(sorted_p[0])
    top2 = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
    margin = float(top1 - top2)
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    reasons = []
    if top1 < confidence_thresh:
        reasons.append(f"Low top-1 confidence ({top1*100:.1f}% < {confidence_thresh*100:.0f}%)")
    if margin < margin_thresh:
        reasons.append(f"Ambiguous margin ({margin*100:.1f}% < {margin_thresh*100:.0f}%)")
    if entropy > entropy_thresh:
        reasons.append(f"High distribution entropy ({entropy:.2f} bits > {entropy_thresh:.2f})")

    is_uncertain = len(reasons) > 0
    return {
        "is_uncertain": is_uncertain,
        "entropy": round(entropy, 3),
        "margin": round(margin, 3),
        "reason": "; ".join(reasons) if is_uncertain else "Confident prediction within GTZAN taxonomy",
    }


class GenrePredictor:
    """
    Inference orchestrator for ANN and CNN models.
    Caches model instances in memory for low-latency predictions.
    """

    def __init__(self):
        self.ann_model: Optional[keras.Model] = None
        self.ann_scaler: Optional[Any] = None
        self.cnn_model: Optional[keras.Model] = None
        self.cnn_config: Optional[Dict] = None
        self.genres: List[str] = sorted(dataset_config.genres)
        self._load_labels()
        self._load_models_if_available()

    def _load_labels(self):
        """Load genre labels from label encoder or fallback to dataset config."""
        label_file = Path(path_config.label_encoder_path)
        if label_file.exists():
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.genres = data.get("genres", sorted(dataset_config.genres))
            except Exception as e:
                logger.warning(f"Could not load label encoder: {e}")

    def _load_models_if_available(self):
        """Attempt to load saved ANN and CNN models."""
        # Load ANN
        ann_path = Path(path_config.ann_model_path)
        scaler_path = Path(path_config.ann_scaler_path)
        if ann_path.exists() and scaler_path.exists():
            try:
                self.ann_model = keras.models.load_model(str(ann_path))
                self.ann_scaler = joblib.load(str(scaler_path))
                logger.info("ANN model and scaler loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading ANN model/scaler: {e}")
                self.ann_model = None
                self.ann_scaler = None

        # Load CNN
        cnn_path = Path(path_config.cnn_model_path)
        cnn_cfg_path = Path(path_config.cnn_config_path)
        if cnn_path.exists():
            try:
                self.cnn_model = keras.models.load_model(str(cnn_path))
                if cnn_cfg_path.exists():
                    with open(cnn_cfg_path, "r", encoding="utf-8") as f:
                        self.cnn_config = json.load(f)
                logger.info("CNN model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading CNN model: {e}")
                self.cnn_model = None

    def is_ann_available(self) -> bool:
        return self.ann_model is not None and self.ann_scaler is not None

    def is_cnn_available(self) -> bool:
        return self.cnn_model is not None

    def get_models_status(self) -> Dict[str, Any]:
        return {
            "ann_available": self.is_ann_available(),
            "cnn_available": self.is_cnn_available(),
            "ann_model_path": path_config.ann_model_path if Path(path_config.ann_model_path).exists() else None,
            "cnn_model_path": path_config.cnn_model_path if Path(path_config.cnn_model_path).exists() else None,
        }

    def get_model_details(self) -> Dict[str, Any]:
        """
        Return comprehensive audit metadata for actively loaded inference checkpoints.
        Enables verification of active model versions, paths, scalers, and training splits.
        """
        ann_meta = {}
        ann_meta_path = Path(path_config.saved_models_dir) / "ann_metadata.json"
        if ann_meta_path.exists():
            try:
                with open(ann_meta_path, "r", encoding="utf-8") as f:
                    ann_meta = json.load(f)
            except Exception:
                pass

        cnn_meta = {}
        cnn_meta_path = Path(path_config.saved_models_dir) / "cnn_metadata.json"
        if cnn_meta_path.exists():
            try:
                with open(cnn_meta_path, "r", encoding="utf-8") as f:
                    cnn_meta = json.load(f)
            except Exception:
                pass

        ann_scaler_features = getattr(self.ann_scaler, "n_features_in_", None) if self.ann_scaler else None

        return {
            "ann": {
                "loaded": self.is_ann_available(),
                "checkpoint_path": str(Path(path_config.ann_model_path).resolve()) if Path(path_config.ann_model_path).exists() else None,
                "scaler_path": str(Path(path_config.ann_scaler_path).resolve()) if Path(path_config.ann_scaler_path).exists() else None,
                "scaler_features": ann_scaler_features,
                "is_decontaminated": ann_meta.get("is_decontaminated_split", False),
                "training_date": ann_meta.get("training_date"),
                "architecture": ann_meta.get("model_architecture", {}).get("type", "ANN"),
                "test_accuracy": ann_meta.get("performance", {}).get("test_accuracy"),
                "test_f1_macro": ann_meta.get("performance", {}).get("test_f1_macro"),
            },
            "cnn": {
                "loaded": self.is_cnn_available(),
                "checkpoint_path": str(Path(path_config.cnn_model_path).resolve()) if Path(path_config.cnn_model_path).exists() else None,
                "config_path": str(Path(path_config.cnn_config_path).resolve()) if Path(path_config.cnn_config_path).exists() else None,
                "input_shape": list(self.cnn_model.input_shape) if self.cnn_model else None,
                "normalization": {
                    "mean": self.cnn_config.get("mean") if self.cnn_config else None,
                    "std": self.cnn_config.get("std") if self.cnn_config else None,
                },
                "is_decontaminated": cnn_meta.get("is_decontaminated_split", False),
                "training_date": cnn_meta.get("training_date"),
                "architecture": cnn_meta.get("model_architecture", {}).get("type", "CNN"),
                "test_accuracy": cnn_meta.get("performance", {}).get("test_accuracy"),
                "test_f1_macro": cnn_meta.get("performance", {}).get("test_f1_macro"),
            },
            "active_version": "Phase 8 Decontaminated Benchmark" if (
                ann_meta.get("is_decontaminated_split") and cnn_meta.get("is_decontaminated_split")
            ) else "Legacy / Unverified",
        }

    def predict_audio(self, file_path: str) -> Dict[str, Any]:
        """
        Execute prediction pipeline for an uploaded audio file.

        Args:
            file_path: Path to the audio file on disk.

        Returns:
            Dictionary with ANN and CNN predictions, top-3 probabilities,
            audio duration, agreement flag, and error fields if applicable.
        """
        # Reload models if they were trained since predictor initialized
        if not self.is_ann_available() or not self.is_cnn_available():
            self._load_models_if_available()

        # Check file validity
        is_valid, msg = validate_audio_file(file_path)
        if not is_valid:
            raise ValueError(f"Invalid audio file: {msg}")

        # Load audio using standard preprocessing configuration
        y, sr = load_audio(file_path, sr=audio_config.sample_rate, mono=audio_config.mono)
        duration = float(len(y) / sr)

        response: Dict[str, Any] = {
            "filename": Path(file_path).name,
            "audio_duration": round(duration, 2),
            "ann_prediction": None,
            "cnn_prediction": None,
            "models_agree": None,
            "ann_error": None,
            "cnn_error": None,
        }

        # 1. ANN Inference
        if self.is_ann_available():
            try:
                features = extract_features(y, sr)
                features_2d = features.reshape(1, -1)
                features_scaled = self.ann_scaler.transform(features_2d)
                ann_probs = self.ann_model.predict(features_scaled, verbose=0)[0]

                top_indices = np.argsort(ann_probs)[::-1][:3]
                predicted_idx = int(top_indices[0])
                predicted_genre = self.genres[predicted_idx]
                confidence = float(ann_probs[predicted_idx])

                top3 = [
                    {"genre": self.genres[idx], "probability": float(ann_probs[idx])}
                    for idx in top_indices
                ]

                ann_unc = compute_uncertainty_metrics(ann_probs)
                response["ann_prediction"] = {
                    "predicted_genre": predicted_genre,
                    "confidence": confidence,
                    "top3": top3,
                    "model_type": "ANN (Statistical Features MLP)",
                    "uncertainty": ann_unc,
                }
            except Exception as e:
                logger.error(f"ANN prediction error: {e}")
                response["ann_error"] = str(e)
        else:
            response["ann_error"] = "ANN model is not trained or model file is missing."

        # 2. CNN Inference
        if self.is_cnn_available():
            try:
                # Segment audio with overlap for inference smoothing
                segments = segment_audio(y, sr, overlap=audio_config.segment_overlap)
                expected_frames = compute_expected_time_frames()
                
                spec_batch = []
                for seg in segments:
                    mel_spec = generate_mel_spectrogram(seg, sr)
                    cnn_in = mel_spectrogram_to_cnn_input(mel_spec, expected_time_frames=expected_frames)
                    spec_batch.append(cnn_in)

                X_batch = np.array(spec_batch, dtype=np.float32)

                # Normalize using saved training stats
                if self.cnn_config and "mean" in self.cnn_config and "std" in self.cnn_config:
                    m = self.cnn_config["mean"]
                    s = self.cnn_config["std"]
                    X_batch = (X_batch - m) / s

                # Segment-level predictions
                segment_probs = self.cnn_model.predict(X_batch, verbose=0)
                # Aggregate segment probabilities to generate track prediction
                avg_probs = np.mean(segment_probs, axis=0)

                top_indices = np.argsort(avg_probs)[::-1][:3]
                predicted_idx = int(top_indices[0])
                predicted_genre = self.genres[predicted_idx]
                confidence = float(avg_probs[predicted_idx])

                top3 = [
                    {"genre": self.genres[idx], "probability": float(avg_probs[idx])}
                    for idx in top_indices
                ]

                cnn_unc = compute_uncertainty_metrics(avg_probs)
                response["cnn_prediction"] = {
                    "predicted_genre": predicted_genre,
                    "confidence": confidence,
                    "top3": top3,
                    "model_type": "CNN (Mel-Spectrogram ConvNet)",
                    "num_segments_evaluated": len(segments),
                    "uncertainty": cnn_unc,
                }
            except Exception as e:
                logger.error(f"CNN prediction error: {e}")
                response["cnn_error"] = str(e)
        else:
            response["cnn_error"] = "CNN model is not trained or model file is missing."

        # Model comparison & agreement
        if response["ann_prediction"] and response["cnn_prediction"]:
            ann_g = response["ann_prediction"]["predicted_genre"]
            cnn_g = response["cnn_prediction"]["predicted_genre"]
            models_agree = (ann_g == cnn_g)
            response["models_agree"] = models_agree

            ann_u = response["ann_prediction"]["uncertainty"]["is_uncertain"]
            cnn_u = response["cnn_prediction"]["uncertainty"]["is_uncertain"]

            if ann_u and cnn_u:
                response["uncertainty_status"] = "High Uncertainty / Potential Out-of-Taxonomy"
                response["out_of_taxonomy_warning"] = (
                    "Both neural networks display diffuse probability distributions and high entropy. "
                    "The audio is likely outside the 10-class GTZAN taxonomy (e.g. EDM, Ambient, or Hybrid Fusion) "
                    "or exhibits out-of-distribution acoustic production characteristics."
                )
            elif not models_agree:
                response["uncertainty_status"] = "Architectural Disagreement"
                response["out_of_taxonomy_warning"] = (
                    "The ANN and CNN architectures arrived at conflicting genre classifications. "
                    "Inspect the top-3 distributions for both models."
                )
            elif ann_u or cnn_u:
                response["uncertainty_status"] = "Moderate Uncertainty"
                response["out_of_taxonomy_warning"] = (
                    "Both models agree on the top genre, but one model assigns marginal confidence."
                )
            else:
                response["uncertainty_status"] = "Confident Consensus"
                response["out_of_taxonomy_warning"] = None

        return response


# Global singleton instance for backend reuse
_global_predictor: Optional[GenrePredictor] = None


def predict_ann(file_path: str) -> Dict[str, Any]:
    """
    Dedicated ANN prediction function.
    Accepts an audio file path, extracts 61 features, scales using the fitted
    StandardScaler, predicts using the trained ANN, and returns top genres
    and normalized probabilities.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        Dict with predicted_genre, confidence, top3, all_probabilities,
        and audio metadata.
    """
    predictor = get_predictor()
    if not predictor.is_ann_available():
        predictor._load_models_if_available()
    if not predictor.is_ann_available():
        raise RuntimeError("ANN model or scaler is missing. Please train the ANN first.")

    is_valid, msg = validate_audio_file(file_path)
    if not is_valid:
        raise ValueError(f"Invalid audio file: {msg}")

    y, sr = load_audio(file_path, sr=audio_config.sample_rate, mono=audio_config.mono)
    duration = float(len(y) / sr)

    features = extract_features(y, sr)
    expected_dim = getattr(predictor.ann_scaler, "n_features_in_", features.shape[0])
    if features.shape[0] != expected_dim:
        raise ValueError(f"Extracted feature dimension {features.shape[0]} does not match expected {expected_dim}.")

    features_2d = features.reshape(1, -1)
    features_scaled = predictor.ann_scaler.transform(features_2d)
    probs = predictor.ann_model.predict(features_scaled, verbose=0)[0]

    # Validate that probabilities sum approximately to 1.0
    prob_sum = float(np.sum(probs))
    if abs(prob_sum - 1.0) > 0.01:
        raise ValueError(f"Output probabilities do not sum to 1.0 (sum={prob_sum})")

    top_indices = np.argsort(probs)[::-1][:3]
    top_genre = predictor.genres[int(top_indices[0])]
    confidence = float(probs[int(top_indices[0])])

    top3 = [
        {"genre": predictor.genres[idx], "probability": float(probs[idx])}
        for idx in top_indices
    ]

    all_probabilities = {
        predictor.genres[i]: float(probs[i]) for i in range(len(predictor.genres))
    }

    return {
        "filename": Path(file_path).name,
        "audio_duration": round(duration, 2),
        "predicted_genre": top_genre,
        "confidence": confidence,
        "top3": top3,
        "all_probabilities": all_probabilities,
        "feature_count": len(features),
        "model_type": "ANN (Statistical Features MLP)",
    }


def predict_cnn(file_path: str) -> Dict[str, Any]:
    """
    Dedicated CNN prediction function with track-level segment probability aggregation.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        Dict with predicted_genre, confidence, top3, all_probabilities,
        num_segments, audio_duration, and filename.
    """
    predictor = get_predictor()
    if not predictor.is_cnn_available():
        predictor._load_models_if_available()
    if not predictor.is_cnn_available():
        raise RuntimeError("CNN model is missing. Please train the CNN first.")

    is_valid, msg = validate_audio_file(file_path)
    if not is_valid:
        raise ValueError(f"Invalid audio file: {msg}")

    y, sr = load_audio(file_path, sr=audio_config.sample_rate, mono=audio_config.mono)
    duration = float(len(y) / sr)

    # Segment audio
    segments = segment_audio(y, sr)
    expected_frames = compute_expected_time_frames()
    
    spec_batch = []
    for seg in segments:
        mel_spec = generate_mel_spectrogram(seg, sr)
        cnn_in = mel_spectrogram_to_cnn_input(mel_spec, expected_time_frames=expected_frames)
        spec_batch.append(cnn_in)

    X_batch = np.array(spec_batch, dtype=np.float32)

    # Normalize using saved training stats
    if predictor.cnn_config and "mean" in predictor.cnn_config and "std" in predictor.cnn_config:
        m = predictor.cnn_config["mean"]
        s = predictor.cnn_config["std"]
        X_batch = (X_batch - m) / s

    # Segment-level predictions
    segment_probs = predictor.cnn_model.predict(X_batch, verbose=0)
    # Aggregate segment probabilities via mean to generate track prediction
    avg_probs = np.mean(segment_probs, axis=0)

    # Validate that probabilities sum approximately to 1.0
    prob_sum = float(np.sum(avg_probs))
    if abs(prob_sum - 1.0) > 0.01:
        avg_probs = avg_probs / prob_sum

    top_indices = np.argsort(avg_probs)[::-1][:3]
    top_genre = predictor.genres[int(top_indices[0])]
    confidence = float(avg_probs[int(top_indices[0])])

    top3 = [
        {"genre": predictor.genres[idx], "probability": float(avg_probs[idx])}
        for idx in top_indices
    ]

    all_probabilities = {
        predictor.genres[i]: float(avg_probs[i]) for i in range(len(predictor.genres))
    }

    return {
        "filename": Path(file_path).name,
        "audio_duration": round(duration, 2),
        "predicted_genre": top_genre,
        "confidence": confidence,
        "top3": top3,
        "all_probabilities": all_probabilities,
        "num_segments": len(segments),
        "model_type": "CNN (Mel-Spectrogram ConvNet)",
    }


def predict_both(file_path: str) -> Dict[str, Any]:
    """
    Simultaneous prediction comparing ANN and CNN models.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        Structured dictionary matching project specification:
        {
            "ann": { "predicted_genre": str, "confidence": float, "top_predictions": list },
            "cnn": { "predicted_genre": str, "confidence": float, "top_predictions": list },
            "agreement": bool,
            "filename": str,
            "audio_duration": float
        }
    """
    ann_res = predict_ann(file_path)
    cnn_res = predict_cnn(file_path)

    agreement = (ann_res["predicted_genre"] == cnn_res["predicted_genre"])

    return {
        "ann": {
            "predicted_genre": ann_res["predicted_genre"],
            "confidence": ann_res["confidence"],
            "top_predictions": ann_res["top3"],
        },
        "cnn": {
            "predicted_genre": cnn_res["predicted_genre"],
            "confidence": cnn_res["confidence"],
            "top_predictions": cnn_res["top3"],
        },
        "agreement": agreement,
        "confidence_difference": abs(ann_res["confidence"] - cnn_res["confidence"]),
        "filename": Path(file_path).name,
        "audio_duration": ann_res["audio_duration"],
    }


def get_predictor() -> GenrePredictor:
    """Get or create singleton GenrePredictor."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = GenrePredictor()
    return _global_predictor
