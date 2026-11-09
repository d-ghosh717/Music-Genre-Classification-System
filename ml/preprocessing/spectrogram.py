"""
Mel-spectrogram generation for the CNN model.
Converts audio signals into log-scaled Mel-spectrograms with consistent shape.
"""
import logging
from typing import Optional

import librosa
import numpy as np

from ml.config import audio_config, spectrogram_config

logger = logging.getLogger(__name__)


def generate_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    config: Optional[object] = None,
) -> np.ndarray:
    """
    Generate a log-scaled Mel-spectrogram from an audio signal.
    
    Args:
        y: Audio signal (mono, resampled).
        sr: Sample rate.
        config: SpectrogramConfig override. Defaults to global spectrogram_config.
    
    Returns:
        2-D numpy array of shape (n_mels, time_frames) — log-scaled power.
    """
    cfg = config or spectrogram_config
    
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        fmax=cfg.fmax,
    )
    
    # Convert to log scale (dB). Add small epsilon to avoid log(0).
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec


def mel_spectrogram_to_cnn_input(
    log_mel_spec: np.ndarray,
    expected_time_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Prepare a Mel-spectrogram for CNN input by padding/truncating to a fixed width
    and adding a channel dimension.
    
    Args:
        log_mel_spec: 2-D array of shape (n_mels, time_frames).
        expected_time_frames: Target number of time frames. If None, computed from
            segment_duration and hop_length.
    
    Returns:
        3-D numpy array of shape (n_mels, time_frames, 1).
    """
    if expected_time_frames is None:
        # Compute expected frames from segment duration
        expected_samples = int(audio_config.segment_duration * audio_config.sample_rate)
        expected_time_frames = 1 + expected_samples // spectrogram_config.hop_length
    
    n_mels, current_frames = log_mel_spec.shape
    
    if current_frames < expected_time_frames:
        # Pad with the minimum value (silence in dB scale)
        pad_width = expected_time_frames - current_frames
        log_mel_spec = np.pad(
            log_mel_spec,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=log_mel_spec.min(),
        )
    elif current_frames > expected_time_frames:
        # Truncate
        log_mel_spec = log_mel_spec[:, :expected_time_frames]
    
    # Add channel dimension for Conv2D: (n_mels, time_frames, 1)
    return log_mel_spec[..., np.newaxis].astype(np.float32)


def compute_expected_time_frames(segment_duration: Optional[float] = None) -> int:
    """Calculate the expected number of time frames for a single segment."""
    duration = segment_duration or audio_config.segment_duration
    expected_samples = int(duration * audio_config.sample_rate)
    return 1 + expected_samples // spectrogram_config.hop_length


def augment_spectrogram(
    spec: np.ndarray,
    freq_mask_max: int = 12,
    time_mask_max: int = 16,
    noise_level: float = 0.03,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply training-only SpecAugment (frequency masking, time masking, and subtle Gaussian noise).

    Args:
        spec: Spectrogram array of shape (n_mels, time_frames, 1).
        freq_mask_max: Maximum mel channels to mask.
        time_mask_max: Maximum time steps to mask.
        noise_level: Scale of additive Gaussian noise.
        rng: Optional random generator.

    Returns:
        Augmented copy of input spectrogram.
    """
    generator = rng or np.random.default_rng()
    augmented = spec.copy()

    n_mels = augmented.shape[0]
    time_frames = augmented.shape[1]

    # 1. Frequency masking
    if freq_mask_max > 0 and n_mels > freq_mask_max:
        f = generator.integers(0, freq_mask_max + 1)
        if f > 0:
            f0 = generator.integers(0, n_mels - f)
            augmented[f0 : f0 + f, :, :] = 0.0

    # 2. Time masking
    if time_mask_max > 0 and time_frames > time_mask_max:
        t = generator.integers(0, time_mask_max + 1)
        if t > 0:
            t0 = generator.integers(0, time_frames - t)
            augmented[:, t0 : t0 + t, :] = 0.0

    # 3. Controlled Gaussian noise
    if noise_level > 0.0:
        noise = generator.normal(0, noise_level, size=augmented.shape).astype(np.float32)
        augmented = augmented + noise

    return augmented
