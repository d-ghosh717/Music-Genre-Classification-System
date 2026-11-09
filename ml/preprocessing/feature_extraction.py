"""
Audio feature extraction for the ANN model.
Extracts MFCCs, chroma, spectral features, ZCR, RMS, and tempo,
then computes statistical summaries to form a fixed-length feature vector.
"""
import logging
from typing import Dict, Optional

import librosa
import numpy as np

from ml.config import audio_config, feature_config

logger = logging.getLogger(__name__)


def extract_features(
    y: np.ndarray,
    sr: int,
    config: Optional[object] = None,
) -> np.ndarray:
    """
    Extract a fixed-length feature vector from an audio signal.
    
    For time-varying features, computes mean and standard deviation.
    
    Args:
        y: Audio signal (mono, resampled).
        sr: Sample rate.
        config: FeatureConfig override. Defaults to global feature_config.
    
    Returns:
        1-D numpy array representing the feature vector.
    
    Raises:
        ValueError: If the audio signal is too short for feature extraction.
    """
    cfg = config or feature_config
    
    if len(y) < cfg.n_fft:
        raise ValueError(
            f"Audio signal too short ({len(y)} samples) for feature extraction "
            f"(minimum {cfg.n_fft} samples required)."
        )
    
    if getattr(cfg, "normalize_audio", True):
        from ml.preprocessing.audio_loader import normalize_audio
        y = normalize_audio(y)

    features = {}
    
    # 1. MFCCs — Mel-Frequency Cepstral Coefficients
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    features["mfcc_mean"] = np.mean(mfccs, axis=1)
    features["mfcc_std"] = np.std(mfccs, axis=1)
    
    # 2. Chroma STFT
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    features["chroma_mean"] = np.mean(chroma, axis=1)
    features["chroma_std"] = np.std(chroma, axis=1)
    
    # 3. Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    features["spectral_centroid_mean"] = np.mean(spectral_centroid)
    features["spectral_centroid_std"] = np.std(spectral_centroid)
    
    # 4. Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
    features["spectral_bandwidth_std"] = np.std(spectral_bandwidth)
    
    # 5. Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
    features["spectral_rolloff_std"] = np.std(spectral_rolloff)
    
    # 6. Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)
    
    # 7. RMS energy
    rms = librosa.feature.rms(
        y=y,
        frame_length=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)
    
    # 8. Tempo (single scalar value)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa may return an array; ensure scalar
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    features["tempo"] = float(tempo)
    
    # Base feature components (61 features)
    components = [
        features["mfcc_mean"],       # 13
        features["mfcc_std"],        # 13
        features["chroma_mean"],     # 12
        features["chroma_std"],      # 12
        np.array([features["spectral_centroid_mean"]]),   # 1
        np.array([features["spectral_centroid_std"]]),    # 1
        np.array([features["spectral_bandwidth_mean"]]),  # 1
        np.array([features["spectral_bandwidth_std"]]),   # 1
        np.array([features["spectral_rolloff_mean"]]),    # 1
        np.array([features["spectral_rolloff_std"]]),     # 1
        np.array([features["zcr_mean"]]),                 # 1
        np.array([features["zcr_std"]]),                  # 1
        np.array([features["rms_mean"]]),                 # 1
        np.array([features["rms_std"]]),                  # 1
        np.array([features["tempo"]]),                    # 1
    ]

    # 9. Spectral Contrast (optional extension, 7 bands * 2 = 14 features)
    if getattr(cfg, "include_contrast", False):
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        components.append(np.mean(contrast, axis=1))  # 7
        components.append(np.std(contrast, axis=1))   # 7
    
    # Concatenate all features into a single vector
    feature_vector = np.concatenate(components)
    return feature_vector.astype(np.float32)


def get_feature_names(config: Optional[object] = None) -> list:
    """Return ordered list of feature names matching the feature vector layout."""
    cfg = config or feature_config
    names = []
    for i in range(cfg.n_mfcc):
        names.append(f"mfcc_{i+1}_mean")
    for i in range(cfg.n_mfcc):
        names.append(f"mfcc_{i+1}_std")
    for i in range(cfg.n_chroma):
        names.append(f"chroma_{i+1}_mean")
    for i in range(cfg.n_chroma):
        names.append(f"chroma_{i+1}_std")
    names.extend([
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "zcr_mean", "zcr_std",
        "rms_mean", "rms_std",
        "tempo",
    ])
    if getattr(cfg, "include_contrast", False):
        for i in range(7):
            names.append(f"spectral_contrast_band_{i+1}_mean")
        for i in range(7):
            names.append(f"spectral_contrast_band_{i+1}_std")
    return names


FEATURE_VECTOR_LENGTH = 75 if getattr(feature_config, "include_contrast", False) else 61
