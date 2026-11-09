"""
Audio segmentation utilities.
Splits audio tracks into shorter, optionally overlapping segments for CNN training.
Ensures track-level split integrity to prevent data leakage.
"""
import logging
from typing import List, Tuple

import numpy as np

from ml.config import audio_config

logger = logging.getLogger(__name__)


def segment_audio(
    y: np.ndarray,
    sr: int,
    segment_duration: float = None,
    overlap: float = None,
) -> List[np.ndarray]:
    """
    Split an audio signal into fixed-length segments with optional overlap.
    
    Args:
        y: Audio signal.
        sr: Sample rate.
        segment_duration: Duration of each segment in seconds.
            Defaults to audio_config.segment_duration.
        overlap: Overlap ratio between segments (0.0 to <1.0).
            Defaults to audio_config.segment_overlap.
    
    Returns:
        List of audio segments as numpy arrays.
    """
    segment_duration = segment_duration or audio_config.segment_duration
    overlap = overlap if overlap is not None else audio_config.segment_overlap
    
    segment_samples = int(segment_duration * sr)
    hop_samples = int(segment_samples * (1.0 - overlap))
    
    if hop_samples <= 0:
        raise ValueError(f"Overlap {overlap} is too large; hop size would be <= 0.")
    
    segments = []
    start = 0
    while start + segment_samples <= len(y):
        segment = y[start : start + segment_samples]
        segments.append(segment)
        start += hop_samples
    
    # If we have no segments (audio shorter than one segment), use the entire audio
    # padded to the segment length
    if len(segments) == 0 and len(y) > 0:
        padded = np.zeros(segment_samples, dtype=y.dtype)
        padded[: len(y)] = y
        segments.append(padded)
        logger.warning(
            f"Audio shorter than segment duration ({len(y)/sr:.2f}s < {segment_duration}s). "
            f"Padded to {segment_duration}s."
        )
    
    return segments


def segment_audio_for_inference(
    y: np.ndarray,
    sr: int,
) -> List[np.ndarray]:
    """
    Segment audio for inference.
    Uses the same parameters as training segmentation for consistency.
    
    Args:
        y: Audio signal.
        sr: Sample rate.
    
    Returns:
        List of audio segments.
    """
    return segment_audio(y, sr)
