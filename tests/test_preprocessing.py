"""
Unit tests for audio preprocessing and feature extraction modules using synthetic signals.
"""
import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from ml.config import audio_config, feature_config, spectrogram_config
from ml.preprocessing.audio_loader import load_audio, normalize_audio, validate_audio_file
from ml.preprocessing.feature_extraction import extract_features, FEATURE_VECTOR_LENGTH
from ml.preprocessing.segmentation import segment_audio
from ml.preprocessing.spectrogram import (
    compute_expected_time_frames,
    generate_mel_spectrogram,
    mel_spectrogram_to_cnn_input,
)


@pytest.fixture
def synthetic_audio():
    """Generate a 5-second 440Hz sinusoidal test audio tone at 22050 Hz."""
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave with harmonics
    y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    return y.astype(np.float32), sr


@pytest.fixture
def synthetic_audio_file(synthetic_audio):
    """Write synthetic audio to a temporary WAV file."""
    y, sr = synthetic_audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    sf.write(path, y, sr)
    yield path
    Path(path).unlink(missing_ok=True)


def test_audio_loader_valid(synthetic_audio_file):
    """Test loading a valid audio file."""
    y, sr = load_audio(synthetic_audio_file)
    assert sr == audio_config.sample_rate
    assert len(y) > 0
    assert isinstance(y, np.ndarray)


def test_audio_loader_nonexistent():
    """Test error handling for nonexistent file."""
    with pytest.raises(FileNotFoundError):
        load_audio("/path/to/definitely_nonexistent_audio.wav")


def test_audio_normalization(synthetic_audio):
    """Test peak normalization bounds."""
    y, _ = synthetic_audio
    y_scaled = y * 5.0
    normed = normalize_audio(y_scaled)
    assert np.max(np.abs(normed)) <= 1.0001


def test_feature_extraction_dimensions(synthetic_audio):
    """Verify that extract_features produces a vector of exactly FEATURE_VECTOR_LENGTH (61)."""
    y, sr = synthetic_audio
    feat = extract_features(y, sr)
    assert feat.ndim == 1
    assert len(feat) == FEATURE_VECTOR_LENGTH
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()


def test_spectrogram_generation(synthetic_audio):
    """Verify Mel-spectrogram output shape and log scaling."""
    y, sr = synthetic_audio
    spec = generate_mel_spectrogram(y, sr)
    assert spec.ndim == 2
    assert spec.shape[0] == spectrogram_config.n_mels
    # Values should be non-positive dB (max is 0 dB)
    assert np.max(spec) <= 0.0001


def test_mel_spectrogram_to_cnn_input(synthetic_audio):
    """Verify CNN input formatting with channel dimension."""
    y, sr = synthetic_audio
    spec = generate_mel_spectrogram(y, sr)
    cnn_in = mel_spectrogram_to_cnn_input(spec, expected_time_frames=130)
    assert cnn_in.shape == (spectrogram_config.n_mels, 130, 1)
    assert cnn_in.dtype == np.float32


def test_segmentation_duration_and_overlap(synthetic_audio):
    """Verify segmentation splits audio properly according to duration & overlap settings."""
    y, sr = synthetic_audio
    segments = segment_audio(y, sr, segment_duration=3.0, overlap=0.5)
    # 5s total with 3s window and 1.5s hop -> starts at 0.0, 1.5 -> 2 segments
    assert len(segments) == 2
    for seg in segments:
        assert len(seg) == int(3.0 * sr)


def test_uncertainty_metrics_confident():
    """Verify uncertainty detection identifies peaked distributions as confident."""
    from ml.inference.predictor import compute_uncertainty_metrics
    # Peaked distribution (90% confident)
    probs = np.array([0.90, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005])
    metrics = compute_uncertainty_metrics(probs)
    assert metrics["is_uncertain"] is False
    assert metrics["margin"] > 0.8
    assert metrics["entropy"] < 1.0


def test_uncertainty_metrics_ambiguous_and_out_of_taxonomy():
    """Verify uncertainty detection flags flat or diffuse distributions."""
    from ml.inference.predictor import compute_uncertainty_metrics
    # Flat / ambiguous distribution (simulating out-of-taxonomy track)
    probs = np.array([0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.04, 0.03, 0.03])
    metrics = compute_uncertainty_metrics(probs)
    assert metrics["is_uncertain"] is True
    assert metrics["entropy"] > 1.65
    assert len(metrics["reason"]) > 0
