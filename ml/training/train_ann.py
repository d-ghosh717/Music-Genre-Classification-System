"""
ANN Model Training Pipeline for Music Genre Classification.

Extracts statistical audio features for all tracks, fits a StandardScaler
on the training split only, trains the MLP classifier with early stopping and
learning rate scheduling, evaluates on the held-out test split, and generates
visualizations and metrics reports.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

from ml.config import (
    ann_config,
    dataset_config,
    feature_config,
    path_config,
    NUM_GENRES,
)
from ml.data_splitter import split_dataset, load_clean_split, print_split_summary
from ml.dataset_validator import validate_dataset
from ml.evaluation.evaluate import compute_classification_metrics, save_metrics_report
from ml.evaluation.visualizations import plot_confusion_matrix, plot_training_history
from ml.models.ann_model import build_ann_model
from ml.preprocessing.audio_loader import load_audio
from ml.preprocessing.feature_extraction import extract_features, FEATURE_VECTOR_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_ann")


def extract_features_for_split(
    file_list: List[Tuple[str, str]],
    genre_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract feature vectors for a list of (file_path, genre) items.

    Args:
        file_list: List of (audio_file_path, genre_string) tuples.
        genre_to_idx: Mapping from genre name to integer index.

    Returns:
        Tuple of (X_features, y_labels, valid_file_paths).
    """
    X_list = []
    y_list = []
    valid_paths = []

    total = len(file_list)
    logger.info(f"Extracting features from {total} audio files...")

    for i, (path, genre) in enumerate(file_list):
        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info(f"  Processed {i + 1}/{total} files")

        try:
            y_audio, sr = load_audio(path)
            feat = extract_features(y_audio, sr)
            X_list.append(feat)
            y_list.append(genre_to_idx[genre])
            valid_paths.append(path)
        except Exception as e:
            logger.warning(f"Skipping corrupted track {path}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, valid_paths


def train_ann(
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    cache_features: bool = True,
    use_clean_split: bool = True,
) -> Dict:
    """
    Execute end-to-end ANN training pipeline.
    """
    epochs = epochs or ann_config.epochs
    batch_size = batch_size or ann_config.batch_size
    lr = learning_rate or ann_config.learning_rate

    # 1. Dataset validation
    logger.info("Validating dataset integrity...")
    report = validate_dataset(verbose=False)
    if not report.is_valid:
        raise RuntimeError(f"Dataset validation failed: {report.summary}")
    logger.info(f"Dataset valid: {report.total_files} audio files found across {len(report.genres_found)} genres.")

    # 2. Genre label indexing
    genres = sorted(dataset_config.genres)
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    idx_to_genre = {i: g for i, g in enumerate(genres)}

    # Ensure saved models and results directories exist
    os.makedirs(path_config.saved_models_dir, exist_ok=True)
    os.makedirs(path_config.ann_results_dir, exist_ok=True)
    final_results_dir = Path(path_config.results_dir) / "final"
    final_results_dir.mkdir(parents=True, exist_ok=True)

    # Save genre map
    with open(path_config.label_encoder_path, "w", encoding="utf-8") as f:
        json.dump({"genres": genres, "genre_to_idx": genre_to_idx}, f, indent=2)

    # 3. Stratified track-level dataset split
    if use_clean_split:
        logger.info("Loading verified DECONTAMINATED grouped dataset split (Phase 8)...")
        splits = load_clean_split()
    else:
        logger.info("Splitting dataset into legacy baseline train, val, and test partitions...")
        splits = split_dataset()
    print_split_summary(splits)

    # Cache file paths for feature caching to save time on multiple runs
    cache_dir = Path(path_config.saved_models_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = "ann_features_clean_75.npz" if use_clean_split else "ann_features_cache.npz"
    ann_cache_file = cache_dir / cache_name

    cached_valid = False
    if cache_features and ann_cache_file.exists():
        logger.info(f"Checking cached features from {ann_cache_file}...")
        data = np.load(ann_cache_file, allow_pickle=True)
        if "X_train" in data and data["X_train"].shape[1] == FEATURE_VECTOR_LENGTH:
            X_train, y_train = data["X_train"], data["y_train"]
            X_val, y_val = data["X_val"], data["y_val"]
            X_test, y_test = data["X_test"], data["y_test"]
            cached_valid = True
            logger.info(f"Loaded valid cached features with shape {X_train.shape}.")
        else:
            dim = data["X_train"].shape[1] if "X_train" in data else "unknown"
            logger.info(f"Cache dimension ({dim}) does not match current FEATURE_VECTOR_LENGTH ({FEATURE_VECTOR_LENGTH}). Re-extracting...")

    if not cached_valid:
        logger.info("Extracting features for training partition...")
        X_train, y_train, _ = extract_features_for_split(splits["train"], genre_to_idx)
        logger.info("Extracting features for validation partition...")
        X_val, y_val, _ = extract_features_for_split(splits["val"], genre_to_idx)
        logger.info("Extracting features for test partition...")
        X_test, y_test, _ = extract_features_for_split(splits["test"], genre_to_idx)

        if cache_features:
            np.savez_compressed(
                ann_cache_file,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
            )
            logger.info(f"Features cached to {ann_cache_file}")

    logger.info(f"Feature dataset shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 4. Feature scaling: Fit StandardScaler on TRAIN ONLY
    logger.info("Fitting StandardScaler on training data only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save fitted scaler
    joblib.dump(scaler, path_config.ann_scaler_path)
    logger.info(f"Scaler saved to {path_config.ann_scaler_path}")

    # 5. Build model
    input_dim = X_train_scaled.shape[1]
    logger.info(f"Building ANN model (input_dim={input_dim}, num_classes={NUM_GENRES})...")
    model = build_ann_model(input_dim=input_dim, num_classes=NUM_GENRES, learning_rate=lr)
    model.summary(print_fn=logger.info)

    # 6. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=ann_config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=ann_config.lr_reduce_factor,
            patience=ann_config.lr_reduce_patience,
            min_lr=ann_config.min_lr,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=path_config.ann_best_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 7. Model training
    logger.info(f"Starting ANN training for up to {epochs} epochs (batch_size={batch_size})...")
    start_time = time.time()
    try:
        history = model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current state...")
        history = None
    training_duration = time.time() - start_time
    logger.info(f"Training finished in {training_duration:.2f} seconds.")

    # Ensure best model is loaded and saved to both ann_best.keras and ann_model.keras
    if Path(path_config.ann_best_path).exists():
        model = keras.models.load_model(path_config.ann_best_path)
    model.save(path_config.ann_model_path)
    logger.info(f"Model saved to {path_config.ann_model_path} and {path_config.ann_best_path}")

    # 8. Evaluation on held-out test set
    logger.info("Evaluating ANN on held-out test partition...")
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    y_test_probs = model.predict(X_test_scaled, verbose=0)
    y_test_pred = np.argmax(y_test_probs, axis=1)

    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
        genres=genres,
        test_loss=test_loss,
    )
    metrics["training_duration_seconds"] = training_duration
    metrics["epochs_trained"] = len(history.history["loss"]) if history else 0

    logger.info("=" * 50)
    logger.info("ANN TEST RESULTS:")
    logger.info(f"  Test Accuracy:     {metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1-Score:    {metrics['f1_macro']:.4f}")
    logger.info(f"  Weighted F1-Score: {metrics['f1_weighted']:.4f}")
    logger.info(f"  Test Loss:         {metrics['test_loss']:.4f}")
    logger.info("=" * 50)

    # Save metrics JSON & text report
    metrics_path = Path(path_config.ann_results_dir) / "metrics.json"
    save_metrics_report(metrics, str(metrics_path))
    
    from sklearn.metrics import classification_report as sk_classification_report
    text_report = sk_classification_report(y_test, y_test_pred, target_names=genres, zero_division=0)
    text_report_path = Path(path_config.ann_results_dir) / "classification_report.txt"
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    logger.info(f"Metrics and report saved to {metrics_path} and {text_report_path}")

    # 9. Visualizations
    if history:
        logger.info("Generating evaluation visualizations...")
        plot_training_history(history.history, "ann", path_config.ann_results_dir)
    plot_confusion_matrix(metrics["confusion_matrix"], genres, "ann", path_config.ann_results_dir)

    # If decontaminated clean split, also record to results/final/ and backup models
    if use_clean_split:
        import shutil
        final_dir = Path(path_config.results_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        save_metrics_report(metrics, str(final_dir / "ann_metrics.json"))
        with open(final_dir / "ann_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(text_report)
        plot_confusion_matrix(metrics["confusion_matrix"], genres, "ann", str(final_dir))
        if history:
            plot_training_history(history.history, "ann", str(final_dir))

        final_models_dir = Path(path_config.saved_models_dir) / "final_ann"
        final_models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(path_config.ann_best_path, final_models_dir / "ann_best.keras")
        shutil.copy(path_config.ann_model_path, final_models_dir / "ann_model.keras")
        shutil.copy(path_config.ann_scaler_path, final_models_dir / "ann_scaler.pkl")

    # 10. Save metadata JSON
    import datetime
    ann_metadata = {
        "class_labels": genres,
        "feature_count": input_dim,
        "is_decontaminated_split": use_clean_split,
        "feature_configuration": {
            "n_mfcc": feature_config.n_mfcc,
            "n_chroma": feature_config.n_chroma,
            "n_fft": feature_config.n_fft,
            "hop_length": feature_config.hop_length,
            "sample_rate": 22050,
        },
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_split_information": {
            "train_tracks": len(splits["train"]),
            "val_tracks": len(splits["val"]),
            "test_tracks": len(splits["test"]),
            "split_type": "decontaminated_grouped_stratified" if use_clean_split else "legacy_stratified",
            "clean_splits_manifest": "dataset/clean_splits.json" if use_clean_split else None,
            "random_seed": dataset_config.random_seed,
        },
        "model_architecture": {
            "type": "Multilayer Perceptron (ANN)",
            "layer_sizes": ann_config.layer_sizes,
            "dropout_rates": ann_config.dropout_rates,
            "use_batch_norm": ann_config.use_batch_norm,
            "summary": f"Input({input_dim}) -> Dense(256, ReLU) -> BN -> Dropout({ann_config.dropout_rates[0]}) -> Dense(128, ReLU) -> BN -> Dropout({ann_config.dropout_rates[1]}) -> Dense(64, ReLU) -> Dropout({ann_config.dropout_rates[2]}) -> Dense(10, Softmax)",
        },
        "training_configuration": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "loss": "sparse_categorical_crossentropy",
            "early_stopping_patience": ann_config.early_stopping_patience,
            "lr_reduce_patience": ann_config.lr_reduce_patience,
        },
        "relative_paths": {
            "model_path": "ml/saved_models/ann_model.keras",
            "best_model_path": "ml/saved_models/ann_best.keras",
            "scaler_path": "ml/saved_models/ann_scaler.pkl",
            "metadata_path": "ml/saved_models/ann_metadata.json",
        },
        "performance": {
            "test_accuracy": metrics["accuracy"],
            "test_f1_macro": metrics["f1_macro"],
            "test_f1_weighted": metrics["f1_weighted"],
            "test_loss": metrics["test_loss"],
            "epochs_trained": metrics["epochs_trained"],
            "training_duration_seconds": round(training_duration, 2),
        },
    }
    with open(path_config.ann_metadata_path, "w", encoding="utf-8") as f:
        json.dump(ann_metadata, f, indent=2)
    # Also save ann_config_path for backwards compatibility
    with open(path_config.ann_config_path, "w", encoding="utf-8") as f:
        json.dump(ann_metadata, f, indent=2)
    if use_clean_split:
        with open(final_models_dir / "ann_metadata.json", "w", encoding="utf-8") as f:
            json.dump(ann_metadata, f, indent=2)
    logger.info(f"Metadata saved to {path_config.ann_metadata_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ANN music genre classifier.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature caching")
    parser.add_argument("--baseline-split", action="store_true", help="Use legacy contaminated baseline split instead of clean split")
    args = parser.parse_args()

    train_ann(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        cache_features=not args.no_cache,
        use_clean_split=not args.baseline_split,
    )
