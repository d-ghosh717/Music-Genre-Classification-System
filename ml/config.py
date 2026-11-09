"""
Central configuration for the Music Genre Classification ML pipeline.
All preprocessing, model, and training parameters are defined here
to ensure consistency between training and inference.
"""
import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class AudioConfig:
    """Audio preprocessing parameters."""
    sample_rate: int = 22050
    duration: float = 30.0  # Expected duration of GTZAN tracks in seconds
    mono: bool = True
    
    # Segmentation for CNN (3.0s segments, 130 frames)
    segment_duration: float = 3.0  # Duration of each segment in seconds
    segment_overlap: float = 0.5   # Overlap ratio between segments (50% for inference smoothing)


@dataclass
class FeatureConfig:
    """Feature extraction parameters for ANN."""
    n_mfcc: int = 13
    n_chroma: int = 12
    n_fft: int = 2048
    hop_length: int = 512
    
    # Feature enhancements
    include_contrast: bool = True   # 7 bands * 2 (mean + std) = 14 features
    normalize_audio: bool = True    # Peak-normalize raw audio before feature extraction
    
    # Total features:
    # 61 base + 14 spectral contrast = 75 features


@dataclass
class SpectrogramConfig:
    """Mel-spectrogram parameters for CNN."""
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmax: int = 8000  # Maximum frequency for Mel filter


@dataclass 
class DatasetConfig:
    """Dataset paths and split ratios."""
    dataset_dir: str = str(os.environ.get(
        "GTZAN_DATASET_DIR",
        str(PROJECT_ROOT / "dataset" / "genres_original")
    ))
    
    genres: List[str] = field(default_factory=lambda: [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    ])
    
    # Stratified split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    random_seed: int = 42
    
    audio_extensions: List[str] = field(default_factory=lambda: [".wav", ".au"])


@dataclass
class ANNConfig:
    """ANN model and training hyperparameters."""
    # Architecture
    layer_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rates: List[float] = field(default_factory=lambda: [0.25, 0.2, 0.2])
    use_batch_norm: bool = True
    
    # Training
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Callbacks
    early_stopping_patience: int = 20
    lr_reduce_patience: int = 10
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class CNNConfig:
    """CNN model and training hyperparameters."""
    # Architecture
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: tuple = (3, 3)
    pool_size: tuple = (2, 2)
    dense_units: int = 128
    dropout_rate: float = 0.4
    use_batch_norm: bool = True
    
    # Training
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Callbacks
    early_stopping_patience: int = 20
    lr_reduce_patience: int = 10
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class PathConfig:
    """Output paths for models, results, and artifacts."""
    saved_models_dir: str = str(PROJECT_ROOT / "ml" / "saved_models")
    results_dir: str = str(PROJECT_ROOT / "results")
    
    # Model files
    ann_model_path: str = str(PROJECT_ROOT / "ml" / "saved_models" / "ann_model.keras")
    ann_best_path: str = str(PROJECT_ROOT / "ml" / "saved_models" / "ann_best.keras")
    ann_scaler_path: str = str(PROJECT_ROOT / "ml" / "saved_models" / "ann_scaler.pkl")
    ann_config_path: str = str(PROJECT_ROOT / "ml" / "saved_models" / "ann_config.json")
    ann_metadata_path: str = str(PROJECT_ROOT / "ml" / "saved_models" / "ann_metadata.json")
    
    cnn_model_path: str = str(PROJECT_ROOT / "ml" / "saved_models" / "cnn_model.keras")
    cnn_best_path: str = str(PROJECT_ROOT / "ml" / "saved_models" / "cnn_best.keras")
    cnn_config_path: str = str(PROJECT_ROOT / "ml" / "saved_models" / "cnn_config.json")
    cnn_metadata_path: str = str(PROJECT_ROOT / "ml" / "saved_models" / "cnn_metadata.json")
    
    # Label encoder
    label_encoder_path: str = str(PROJECT_ROOT / "ml" / "saved_models" / "label_encoder.json")
    
    # Results subdirectories
    ann_results_dir: str = str(PROJECT_ROOT / "results" / "ann")
    cnn_results_dir: str = str(PROJECT_ROOT / "results" / "cnn")
    comparison_results_dir: str = str(PROJECT_ROOT / "results" / "comparison")


# Global configuration instances
audio_config = AudioConfig()
feature_config = FeatureConfig()
spectrogram_config = SpectrogramConfig()
dataset_config = DatasetConfig()
ann_config = ANNConfig()
cnn_config = CNNConfig()
path_config = PathConfig()

NUM_GENRES = len(dataset_config.genres)
