"""
CNN Model Training Pipeline for Music Genre Classification.

Operates directly on log Mel-spectrogram patches.
CRITICAL DATA LEAKAGE PREVENTION: Splits the dataset into train/val/test at the
track level BEFORE extracting segments. All segments from a given track stay
in the same split.

During evaluation and inference, segment probabilities are aggregated (averaged)
at the original track level to produce a single 10-genre probability distribution
per original track, matching the ANN evaluation methodology.
"""
import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ml.config import (
    audio_config,
    cnn_config,
    dataset_config,
    path_config,
    spectrogram_config,
    NUM_GENRES,
)
from ml.data_splitter import split_dataset, load_clean_split, print_split_summary
from ml.dataset_validator import validate_dataset
from ml.evaluation.evaluate import compute_classification_metrics, save_metrics_report
from ml.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_training_history,
    plot_model_comparison,
)
from ml.models.cnn_model import build_cnn_model
from ml.preprocessing.audio_loader import load_audio
from ml.preprocessing.segmentation import segment_audio
from ml.preprocessing.spectrogram import (
    augment_spectrogram,
    compute_expected_time_frames,
    generate_mel_spectrogram,
    mel_spectrogram_to_cnn_input,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_cnn")


def process_track_to_spectrogram_segments(
    audio_path: str,
    sr: int = audio_config.sample_rate,
    expected_frames: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Load an audio track, partition into audio segments, and compute
    a log Mel-spectrogram for each segment.

    Returns:
        List of 3D arrays of shape (128, expected_frames, 1).
    """
    if expected_frames is None:
        expected_frames = compute_expected_time_frames(audio_config.segment_duration)
    y, _ = load_audio(audio_path, sr=sr)
    segments = segment_audio(y, sr)
    spec_segments = []

    for seg in segments:
        mel_spec = generate_mel_spectrogram(seg, sr)
        cnn_in = mel_spectrogram_to_cnn_input(mel_spec, expected_time_frames=expected_frames)
        spec_segments.append(cnn_in)

    return spec_segments


def extract_cnn_data_for_split(
    file_list: List[Tuple[str, str]],
    genre_to_idx: Dict[str, int],
    expected_frames: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Process audio tracks into segments.
    Records track_ids for track-level aggregation during testing.

    Returns:
        X: (num_segments, 128, expected_frames, 1)
        y: (num_segments,)
        track_ids: (num_segments,) mapping each segment back to its source track index.
    """
    X_list = []
    y_list = []
    track_ids = []

    total = len(file_list)
    logger.info(f"Extracting Mel-spectrogram segments for {total} tracks...")

    for track_idx, (path, genre) in enumerate(file_list):
        if (track_idx + 1) % 100 == 0 or (track_idx + 1) == total:
            logger.info(f"  Processed {track_idx + 1}/{total} tracks")

        try:
            segments = process_track_to_spectrogram_segments(path, expected_frames=expected_frames)
            target = genre_to_idx[genre]
            for seg in segments:
                X_list.append(seg)
                y_list.append(target)
                track_ids.append(track_idx)
        except Exception as e:
            logger.warning(f"Error processing {path}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, track_ids


def train_cnn(
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = None,
    use_augmentation: bool = True,
    cache_data: bool = True,
    use_clean_split: bool = True,
) -> Dict:
    """
    Execute end-to-end CNN training pipeline.
    """
    lr = learning_rate or cnn_config.learning_rate

    # 1. Dataset validation
    logger.info("Validating dataset integrity...")
    report = validate_dataset(verbose=False)
    if not report.is_valid:
        raise RuntimeError(f"Dataset validation failed: {report.summary}")

    genres = sorted(dataset_config.genres)
    genre_to_idx = {g: i for i, g in enumerate(genres)}

    # Ensure output directories exist
    os.makedirs(path_config.saved_models_dir, exist_ok=True)
    os.makedirs(path_config.cnn_results_dir, exist_ok=True)
    os.makedirs(path_config.comparison_results_dir, exist_ok=True)
    final_results_dir = Path(path_config.results_dir) / "final"
    final_results_dir.mkdir(parents=True, exist_ok=True)

    # 2. Track-level split
    if use_clean_split:
        logger.info("Loading verified DECONTAMINATED grouped dataset split (Phase 8)...")
        splits = load_clean_split()
    else:
        logger.info("Splitting dataset into legacy baseline train, val, and test partitions...")
        splits = split_dataset()
    print_split_summary(splits)

    # Critical Data Leakage Verification
    train_paths = {p for p, _ in splits["train"]}
    val_paths = {p for p, _ in splits["val"]}
    test_paths = {p for p, _ in splits["test"]}
    assert len(train_paths & val_paths) == 0, "Data leakage detected: train and val overlap!"
    assert len(train_paths & test_paths) == 0, "Data leakage detected: train and test overlap!"
    assert len(val_paths & test_paths) == 0, "Data leakage detected: val and test overlap!"
    logger.info("Track-level split integrity verified: zero track overlap across partitions.")

    expected_frames = compute_expected_time_frames()
    logger.info(f"Segment duration: {audio_config.segment_duration}s, expected time frames: {expected_frames}")
    input_shape = (spectrogram_config.n_mels, expected_frames, 1)

    # 3. Spectrogram extraction / Caching
    import gc
    cache_dir = Path(path_config.saved_models_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = "cnn_spectrograms_clean.npz" if use_clean_split else "cnn_spectrograms_cache.npz"
    cnn_cache_file = cache_dir / cache_name

    cache_valid = False
    if cache_data and cnn_cache_file.exists():
        try:
            data = np.load(cnn_cache_file, allow_pickle=True)
            if (
                "expected_frames" in data
                and int(data["expected_frames"]) == expected_frames
                and "X_train" in data
            ):
                cache_valid = True
                X_train = data["X_train"]
                y_train = data["y_train"]
                X_val = data["X_val"]
                y_val = data["y_val"]
                # Defer loading X_test to save memory during training
                logger.info(f"Loaded train/val cached spectrogram data from {cnn_cache_file} (frames={expected_frames})")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}. Extracting fresh spectrograms...")

    if not cache_valid:
        logger.info("Extracting spectrograms for Train partition...")
        X_train, y_train, _ = extract_cnn_data_for_split(splits["train"], genre_to_idx, expected_frames)
        logger.info("Extracting spectrograms for Val partition...")
        X_val, y_val, _ = extract_cnn_data_for_split(splits["val"], genre_to_idx, expected_frames)
        logger.info("Extracting spectrograms for Test partition...")
        X_test, y_test, track_ids_test = extract_cnn_data_for_split(splits["test"], genre_to_idx, expected_frames)

        if cache_data:
            np.savez_compressed(
                cnn_cache_file,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                track_ids_test=np.array(track_ids_test),
                expected_frames=expected_frames,
                segment_duration=audio_config.segment_duration,
            )
            logger.info(f"Spectrogram data cached to {cnn_cache_file}")
            del X_test, y_test, track_ids_test
            gc.collect()

    logger.info(f"Train/Val dataset shapes: Train={X_train.shape}, Val={X_val.shape}")

    # 4. Normalization using training set statistics only (in-place to prevent OOM)
    train_mean = float(np.mean(X_train))
    train_std = float(np.std(X_train)) + 1e-6
    logger.info(f"Spectrogram normalization: mean={train_mean:.2f}, std={train_std:.2f}")

    X_train -= train_mean
    X_train /= train_std
    X_val -= train_mean
    X_val /= train_std

    # 5. Data Augmentation (Training Only, in-place)
    if use_augmentation:
        logger.info("Applying SpecAugment data augmentation (time & freq masking) to training samples...")
        rng = np.random.default_rng(seed=dataset_config.random_seed)
        # Augment 50% of training samples
        indices = rng.choice(len(X_train), size=int(0.5 * len(X_train)), replace=False)
        for idx in indices:
            X_train[idx] = augment_spectrogram(X_train[idx], rng=rng)

    # Save normalization stats for inference
    cnn_norm_stats = {
        "mean": train_mean,
        "std": train_std,
        "input_shape": list(input_shape),
        "segment_duration": audio_config.segment_duration,
        "segment_overlap": audio_config.segment_overlap,
        "expected_frames": expected_frames,
    }
    with open(path_config.cnn_config_path, "w", encoding="utf-8") as f:
        json.dump(cnn_norm_stats, f, indent=2)

    # 6. Build CNN model
    logger.info(f"Building CNN model (input_shape={input_shape}, num_classes={NUM_GENRES})...")
    model = build_cnn_model(input_shape=input_shape, num_classes=NUM_GENRES, learning_rate=lr)
    model.summary(print_fn=logger.info)

    # 7. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=path_config.cnn_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 8. Training
    logger.info(f"Starting CNN training for up to {epochs} epochs (batch_size={batch_size})...")
    start_time = time.time()
    try:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=1,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current checkpoint...")
        history = None
    training_duration = time.time() - start_time
    logger.info(f"CNN training completed in {training_duration:.2f} seconds.")

    # Free training/validation memory before testing
    del X_train, y_train, X_val, y_val
    gc.collect()

    # Ensure best model is loaded
    if Path(path_config.cnn_model_path).exists():
        model = keras.models.load_model(path_config.cnn_model_path)
    logger.info(f"Model saved to {path_config.cnn_model_path}")

    # 9. Track-Level Evaluation on held-out test tracks
    logger.info("Loading held-out test split from cache...")
    test_cache = np.load(cnn_cache_file, allow_pickle=True)
    X_test = test_cache["X_test"]
    y_test = test_cache["y_test"]
    track_ids_test = test_cache["track_ids_test"].tolist()
    del test_cache
    gc.collect()

    X_test -= train_mean
    X_test /= train_std

    logger.info("Evaluating CNN with track-level segment probability aggregation on held-out test set...")
    segment_probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    test_loss, _ = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

    # Aggregate segment probabilities per original track
    unique_tracks = sorted(list(set(track_ids_test)))
    y_test_track_true = []
    y_test_track_pred = []

    for t_id in unique_tracks:
        mask = np.array(track_ids_test) == t_id
        # Average segment probabilities for this track
        avg_probs = np.mean(segment_probs[mask], axis=0)
        track_pred = int(np.argmax(avg_probs))
        track_true = int(y_test[mask][0])

        y_test_track_true.append(track_true)
        y_test_track_pred.append(track_pred)

    y_test_track_true = np.array(y_test_track_true)
    y_test_track_pred = np.array(y_test_track_pred)

    metrics = compute_classification_metrics(
        y_true=y_test_track_true,
        y_pred=y_test_track_pred,
        genres=genres,
        test_loss=test_loss,
    )
    metrics["training_duration_seconds"] = training_duration
    metrics["epochs_trained"] = len(history.history["loss"]) if history else 0
    metrics["num_test_tracks"] = len(unique_tracks)
    metrics["num_test_segments"] = len(y_test)

    logger.info("=" * 50)
    logger.info("CNN TEST RESULTS (Track-Level Aggregation):")
    logger.info(f"  Test Accuracy:     {metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1-Score:    {metrics['f1_macro']:.4f}")
    logger.info(f"  Weighted F1-Score: {metrics['f1_weighted']:.4f}")
    logger.info(f"  Test Loss:         {metrics['test_loss']:.4f}")
    logger.info(f"  Evaluated Tracks:  {len(unique_tracks)}")
    logger.info("=" * 50)

    # Save metrics JSON & text report
    metrics_path = Path(path_config.cnn_results_dir) / "metrics.json"
    save_metrics_report(metrics, str(metrics_path))

    from sklearn.metrics import classification_report as sk_classification_report
    text_report = sk_classification_report(y_test_track_true, y_test_track_pred, target_names=genres, zero_division=0)
    text_report_path = Path(path_config.cnn_results_dir) / "classification_report.txt"
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    logger.info(f"Metrics saved to {metrics_path} and {text_report_path}")

    # 10. Visualizations
    logger.info("Generating CNN evaluation visualizations...")
    if history:
        plot_training_history(history.history, "cnn", path_config.cnn_results_dir)
    plot_confusion_matrix(metrics["confusion_matrix"], genres, "cnn", path_config.cnn_results_dir)

    # If decontaminated clean split, also record to results/final/ and backup models
    if use_clean_split:
        import shutil
        final_dir = Path(path_config.results_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        save_metrics_report(metrics, str(final_dir / "cnn_metrics.json"))
        with open(final_dir / "cnn_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(text_report)
        plot_confusion_matrix(metrics["confusion_matrix"], genres, "cnn", str(final_dir))
        if history:
            plot_training_history(history.history, "cnn", str(final_dir))

        final_models_dir = Path(path_config.saved_models_dir) / "final_cnn"
        final_models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(path_config.cnn_best_path, final_models_dir / "cnn_best.keras")
        shutil.copy(path_config.cnn_model_path, final_models_dir / "cnn_model.keras")
        shutil.copy(path_config.cnn_config_path, final_models_dir / "cnn_config.json")

    # 11. Model Comparison (ANN vs CNN)
    ann_metrics_file = Path(path_config.ann_results_dir) / "metrics.json"
    if ann_metrics_file.exists():
        try:
            with open(ann_metrics_file, "r", encoding="utf-8") as f:
                ann_metrics = json.load(f)
            logger.info("Generating ANN vs CNN model comparison charts...")
            plot_model_comparison(ann_metrics, metrics, path_config.comparison_results_dir)
            
            # Save structured comparison JSON
            comparison_data = {
                "ann": {
                    "accuracy": ann_metrics.get("accuracy"),
                    "f1_macro": ann_metrics.get("f1_macro"),
                    "f1_weighted": ann_metrics.get("f1_weighted"),
                    "test_loss": ann_metrics.get("test_loss"),
                },
                "cnn": {
                    "accuracy": metrics.get("accuracy"),
                    "f1_macro": metrics.get("f1_macro"),
                    "f1_weighted": metrics.get("f1_weighted"),
                    "test_loss": metrics.get("test_loss"),
                },
                "comparison": {
                    "accuracy_difference": round(metrics.get("accuracy", 0) - ann_metrics.get("accuracy", 0), 4),
                    "f1_difference": round(metrics.get("f1_macro", 0) - ann_metrics.get("f1_macro", 0), 4),
                    "superior_model": "CNN" if metrics.get("accuracy", 0) > ann_metrics.get("accuracy", 0) else "ANN",
                },
            }
            comp_json_path = Path(path_config.comparison_results_dir) / "comparison_metrics.json"
            with open(comp_json_path, "w", encoding="utf-8") as f:
                json.dump(comparison_data, f, indent=2)
            logger.info(f"Comparison summary saved to {comp_json_path}")
        except Exception as e:
            logger.warning(f"Could not generate comparison: {e}")

    # 12. Save CNN Metadata
    cnn_metadata = {
        "class_labels": genres,
        "input_shape": list(input_shape),
        "spectrogram_configuration": {
            "n_mels": spectrogram_config.n_mels,
            "n_fft": spectrogram_config.n_fft,
            "hop_length": spectrogram_config.hop_length,
            "fmax": spectrogram_config.fmax,
            "sample_rate": audio_config.sample_rate,
        },
        "segment_duration": audio_config.segment_duration,
        "segment_overlap": audio_config.segment_overlap,
        "model_architecture": {
            "type": "Convolutional Neural Network (CNN)",
            "conv_filters": cnn_config.conv_filters,
            "kernel_size": list(cnn_config.kernel_size),
            "pool_size": list(cnn_config.pool_size),
            "dense_units": cnn_config.dense_units,
            "dropout_rate": cnn_config.dropout_rate,
            "use_batch_norm": cnn_config.use_batch_norm,
            "summary": "Conv2D(32, 3x3) -> BN -> MaxPool(2x2) -> Conv2D(64, 3x3) -> BN -> MaxPool(2x2) -> Conv2D(128, 3x3) -> BN -> MaxPool(2x2) -> GAP -> Dense(128) -> Dropout(0.4) -> Dense(10, Softmax)",
        },
        "training_configuration": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "loss": "sparse_categorical_crossentropy",
            "early_stopping_patience": 12,
            "augmentation_used": use_augmentation,
        },
        "is_decontaminated_split": use_clean_split,
        "dataset_split_information": {
            "train_tracks": len(splits["train"]),
            "val_tracks": len(splits["val"]),
            "test_tracks": len(splits["test"]),
            "split_type": "decontaminated_grouped_stratified" if use_clean_split else "legacy_stratified",
            "clean_splits_manifest": "dataset/clean_splits.json" if use_clean_split else None,
            "random_seed": dataset_config.random_seed,
        },
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "performance": {
            "test_accuracy": metrics["accuracy"],
            "test_f1_macro": metrics["f1_macro"],
            "test_f1_weighted": metrics["f1_weighted"],
            "test_loss": metrics["test_loss"],
            "epochs_trained": metrics["epochs_trained"],
            "training_duration_seconds": round(training_duration, 2),
        },
        "relative_paths": {
            "model_path": "ml/saved_models/cnn_model.keras",
            "best_model_path": "ml/saved_models/cnn_best.keras",
            "metadata_path": "ml/saved_models/cnn_metadata.json",
        },
    }
    with open(path_config.cnn_metadata_path, "w", encoding="utf-8") as f:
        json.dump(cnn_metadata, f, indent=2)
    if use_clean_split:
        with open(final_models_dir / "cnn_metadata.json", "w", encoding="utf-8") as f:
            json.dump(cnn_metadata, f, indent=2)
    logger.info(f"CNN metadata saved to {path_config.cnn_metadata_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN music genre classifier on Mel-spectrograms.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no-aug", action="store_true", help="Disable SpecAugment data augmentation")
    parser.add_argument("--no-cache", action="store_true", help="Disable spectrogram caching")
    parser.add_argument("--baseline-split", action="store_true", help="Use legacy contaminated baseline split instead of clean split")
    args = parser.parse_args()

    train_cnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_augmentation=not args.no_aug,
        cache_data=not args.no_cache,
        use_clean_split=not args.baseline_split,
    )
