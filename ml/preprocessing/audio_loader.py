"""
Audio loading and validation utilities.
Handles reading audio files, converting to mono, resampling, and validation.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np

from ml.config import audio_config

logger = logging.getLogger(__name__)


def load_audio(
    file_path: str,
    sr: Optional[int] = None,
    mono: Optional[bool] = None,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file with consistent preprocessing.
    
    Args:
        file_path: Path to the audio file.
        sr: Target sample rate. Defaults to audio_config.sample_rate.
        mono: Whether to convert to mono. Defaults to audio_config.mono.
        duration: Maximum duration to load (seconds). None loads entire file.
        offset: Start reading after this time (seconds).
    
    Returns:
        Tuple of (audio signal as numpy array, sample rate).
    
    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the file cannot be decoded as audio.
    """
    sr = sr if sr is not None else audio_config.sample_rate
    mono = mono if mono is not None else audio_config.mono
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    if path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {file_path}")
    
    try:
        y, loaded_sr = librosa.load(
            str(path),
            sr=sr,
            mono=mono,
            duration=duration,
            offset=offset,
        )
    except Exception as e:
        raise ValueError(f"Failed to load audio file '{file_path}': {e}")
    
    if y is None or len(y) == 0:
        raise ValueError(f"Audio file produced empty signal: {file_path}")
    
    return y, loaded_sr


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds without loading the full signal.
    
    Args:
        file_path: Path to the audio file.
    
    Returns:
        Duration in seconds.
    """
    try:
        return librosa.get_duration(path=file_path)
    except Exception as e:
        raise ValueError(f"Cannot determine duration of '{file_path}': {e}")


def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """
    Check whether an audio file is valid and readable.
    
    Args:
        file_path: Path to the audio file.
    
    Returns:
        Tuple of (is_valid, message).
    """
    path = Path(file_path)
    
    if not path.exists():
        return False, f"File does not exist: {file_path}"
    
    if path.stat().st_size == 0:
        return False, f"File is empty (0 bytes): {file_path}"
    
    try:
        y, sr = librosa.load(str(path), sr=None, duration=1.0)
        if y is None or len(y) == 0:
            return False, f"File produced empty audio signal: {file_path}"
        return True, "OK"
    except Exception as e:
        return False, f"Cannot read audio: {e}"


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """
    Peak-normalize an audio signal to [-1, 1].
    
    Args:
        y: Audio signal.
    
    Returns:
        Normalized audio signal.
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        return y / peak
    return y
