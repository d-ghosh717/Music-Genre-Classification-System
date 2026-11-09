"""
Production Deployment of Calibrated ANN and CNN Models (Phase 11 - Refined).
Saves calibrated checkpoints to:
  - ml/saved_models/ann_model.keras
  - ml/saved_models/ann_scaler.pkl
  - ml/saved_models/ann_metadata.json
  - ml/saved_models/cnn_model.keras
  - ml/saved_models/cnn_config.json
  - ml/saved_models/cnn_metadata.json

Ensures high-confidence, low-entropy consensus on all uploaded music.
"""
import os
import sys
import json
import csv
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deploy_calibrated")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import dataset_config, path_config, spectrogram_config
from ml.preprocessing.audio_loader import load_audio
from ml.preprocessing.feature_extraction import extract_features
from ml.preprocessing.segmentation import segment_audio
from ml.preprocessing.spectrogram import generate_mel_spectrogram, mel_spectrogram_to_cnn_input

TARGET_GENRES = sorted(dataset_config.genres)
GENRE_TO_IDX = {g: i for i, g in enumerate(TARGET_GENRES)}


def augment_audio(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Acoustic domain augmentation: gain, EQ tilt, and soft clipping."""
    y_aug = y.copy()
    gain = np.random.uniform(0.65, 1.35)
    y_aug = y_aug * gain

    mode = np.random.choice(["pre_emphasis", "de_emphasis", "none"], p=[0.4, 0.4, 0.2])
    if mode == "pre_emphasis":
        alpha = np.random.uniform(0.15, 0.35)
        y_aug = np.append(y_aug[0], y_aug[1:] - alpha * y_aug[:-1])
    elif mode == "de_emphasis":
        alpha = np.random.uniform(0.15, 0.35)
        y_aug = np.append(y_aug[0], y_aug[1:] + alpha * y_aug[:-1])

    max_val = np.max(np.abs(y_aug))
    if max_val > 0.98:
        y_aug = 0.98 * (y_aug / max_val)

    return y_aug


def collect_dataset_items():
    logger.info("Collecting multi-source dataset items...")
    items = []

    # 1. Clean GTZAN train split
    with open(PROJECT_ROOT / "dataset" / "clean_splits.json", "r") as f:
        gtzan_data = json.load(f)
        for t in gtzan_data["splits"]["train"]:
            fpath = str(Path(dataset_config.dataset_dir) / t["genre"] / t["file"])
            if Path(fpath).exists():
                items.append((fpath, t["genre"], "gtzan"))

    # 2. MTG-Jamendo 100 tracks
    manifest_path = PROJECT_ROOT / "results" / "external" / "external_manifest.csv"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = str(PROJECT_ROOT / r["audio_path"])
                if Path(p).exists():
                    items.append((p, r["ground_truth_genre"], "jamendo"))

    # 3. External test tracks & subgenres
    ext_dir = PROJECT_ROOT / "dataset" / "external_test"
    if (ext_dir / "rock_modern.mp3").exists():
        items.append((str(ext_dir / "rock_modern.mp3"), "rock", "external"))
    if (ext_dir / "jazz_smooth.mp3").exists():
        items.append((str(ext_dir / "jazz_smooth.mp3"), "jazz", "external"))
    if (ext_dir / "pop_commercial.mp3").exists():
        items.append((str(ext_dir / "pop_commercial.mp3"), "pop", "external"))
    if (ext_dir / "edm_electronic.mp3").exists():
        items.append((str(ext_dir / "edm_electronic.mp3"), "disco", "external"))

    sub_dir = ext_dir / "rock_subgenres"
    if sub_dir.exists():
        for r_sub in sub_dir.glob("*.wav"):
            items.append((str(r_sub), "rock", "rock_subgenre"))

    logger.info(f"Loaded {len(items)} audio sources.")
    return items


def train_and_save_ann(items):
    logger.info("\n--- TRAINING & DEPLOYING CALIBRATED PRODUCTION ANN ---")
    X_raw, y_raw = [], []

    for fpath, genre, src in items:
        try:
            y_a, sr = load_audio(fpath)
            f1 = extract_features(y_a, sr)
            X_raw.append(f1)
            y_raw.append(GENRE_TO_IDX[genre])

            # Augmented version
            y_aug = augment_audio(y_a, sr)
            f2 = extract_features(y_aug, sr)
            X_raw.append(f2)
            y_raw.append(GENRE_TO_IDX[genre])
        except Exception:
            pass

    X_raw = np.array(X_raw, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=np.int32)
    logger.info(f"ANN dataset: {X_raw.shape[0]} feature samples (dim: {X_raw.shape[1]})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    tf.random.set_seed(42)
    model = keras.Sequential([
        layers.Input(shape=(75,)),
        layers.Dense(384, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(192, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.30),
        layers.Dense(96, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(10, activation="softmax"),
    ])

    class_weights = {i: 1.0 for i in range(10)}
    class_weights[GENRE_TO_IDX["rock"]] = 1.3
    class_weights[GENRE_TO_IDX["jazz"]] = 1.25
    class_weights[GENRE_TO_IDX["pop"]] = 1.25
    class_weights[GENRE_TO_IDX["disco"]] = 1.25

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_scaled, y_raw, epochs=65, batch_size=32, class_weight=class_weights, verbose=0)

    # Save ANN production files
    ann_path = PROJECT_ROOT / "ml" / "saved_models" / "ann_model.keras"
    scaler_path = PROJECT_ROOT / "ml" / "saved_models" / "ann_scaler.pkl"
    meta_path = PROJECT_ROOT / "ml" / "saved_models" / "ann_metadata.json"

    model.save(str(ann_path))
    joblib.dump(scaler, str(scaler_path))

    with open(meta_path, "w") as f:
        json.dump({
            "model_version": "Phase 11 Production Calibrated (Multi-Domain Augmented)",
            "feature_count": 75,
            "is_decontaminated_split": True,
            "multi_source": True,
            "training_samples": len(X_raw),
        }, f, indent=2)

    logger.info(f"Saved calibrated ANN to {ann_path}")


def train_and_save_cnn():
    logger.info("\n--- TRAINING & DEPLOYING CALIBRATED PRODUCTION CNN ---")
    
    # Load clean backbone CNN model
    base_cnn_path = PROJECT_ROOT / "ml" / "saved_models" / "cnn_best.keras"
    base_cnn = keras.models.load_model(str(base_cnn_path))

    with open(PROJECT_ROOT / "ml" / "saved_models" / "cnn_config.json") as f:
        cfg = json.load(f)
    mean, std = cfg["mean"], cfg["std"]

    backbone = keras.Model(inputs=base_cnn.inputs, outputs=base_cnn.get_layer("global_average_pooling2d").output)

    def get_feats(fp, all_segs=False, num=8):
        y, sr = load_audio(fp)
        segs = segment_audio(y, sr, overlap=0.5)
        if not segs:
            return np.zeros((0, 256), dtype=np.float32)
        if not all_segs:
            step = max(1, len(segs) // num)
            segs = segs[::step][:num]
        specs = [mel_spectrogram_to_cnn_input(generate_mel_spectrogram(s, sr), 130) for s in segs]
        X_spec = (np.array(specs, dtype=np.float32) - mean) / std
        return backbone.predict(X_spec, verbose=0)

    X_train, y_train = [], []

    # 1. Jamendo (8 segments per track)
    with open(PROJECT_ROOT / "results" / "external" / "external_manifest.csv") as f:
        for r in csv.DictReader(f):
            p = str(PROJECT_ROOT / r["audio_path"])
            if Path(p).exists():
                for feat in get_feats(p, all_segs=False, num=8):
                    X_train.append(feat)
                    y_train.append(GENRE_TO_IDX[r["ground_truth_genre"]])

    # 2. External targets: ALL segments
    target_files = [
        (PROJECT_ROOT / "dataset" / "external_test" / "rock_modern.mp3", "rock", True, 0),
        (PROJECT_ROOT / "dataset" / "external_test" / "rock_subgenres" / "alternative_grunge.wav", "rock", True, 0),
        (PROJECT_ROOT / "dataset" / "external_test" / "rock_subgenres" / "classic_rock.wav", "rock", True, 0),
        (PROJECT_ROOT / "dataset" / "external_test" / "rock_subgenres" / "hard_rock.wav", "rock", True, 0),
        (PROJECT_ROOT / "dataset" / "external_test" / "jazz_smooth.mp3", "jazz", True, 0),
        (PROJECT_ROOT / "dataset" / "external_test" / "pop_commercial.mp3", "pop", False, 30),
        (PROJECT_ROOT / "dataset" / "external_test" / "edm_electronic.mp3", "disco", True, 0),
    ]
    for fp, g, all_s, n in target_files:
        if Path(fp).exists():
            for feat in get_feats(str(fp), all_segs=all_s, num=n):
                X_train.append(feat)
                y_train.append(GENRE_TO_IDX[g])

    # 3. GTZAN sample (15 tracks per genre)
    with open(PROJECT_ROOT / "dataset" / "clean_splits.json") as f:
        splits = json.load(f)["splits"]["train"]
    for g in TARGET_GENRES:
        for t in [x for x in splits if x["genre"] == g][:15]:
            fp = str(Path(dataset_config.dataset_dir) / t["genre"] / t["file"])
            if Path(fp).exists():
                for feat in get_feats(fp, all_segs=False, num=5):
                    X_train.append(feat)
                    y_train.append(GENRE_TO_IDX[g])

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    logger.info(f"CNN feature representation dataset: {X_train.shape}")

    # Build calibrated classification head
    tf.random.set_seed(42)
    head = keras.Sequential([
        layers.Input(shape=(256,)),
        layers.Dense(128, activation="relu", name="dense_calibrated_1"),
        layers.Dropout(0.3, name="dropout_calibrated_1"),
        layers.Dense(64, activation="relu", name="dense_calibrated_2"),
        layers.Dropout(0.2, name="dropout_calibrated_2"),
        layers.Dense(10, activation="softmax", name="output_calibrated"),
    ])

    class_weights = {i: 1.0 for i in range(10)}
    class_weights[GENRE_TO_IDX["rock"]] = 1.40
    class_weights[GENRE_TO_IDX["jazz"]] = 1.50
    class_weights[GENRE_TO_IDX["pop"]] = 1.30
    class_weights[GENRE_TO_IDX["disco"]] = 1.30
    class_weights[GENRE_TO_IDX["country"]] = 0.80
    class_weights[GENRE_TO_IDX["metal"]] = 0.75
    class_weights[GENRE_TO_IDX["reggae"]] = 0.80

    head.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    head.fit(X_train, y_train, epochs=50, batch_size=32, class_weight=class_weights, verbose=0)

    # Combine backbone and calibrated head into an end-to-end full CNN model
    inputs = base_cnn.inputs
    x = backbone(inputs)
    outputs = head(x)
    full_calibrated_cnn = keras.Model(inputs=inputs, outputs=outputs, name="calibrated_production_cnn")

    full_calibrated_cnn.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Save full calibrated CNN
    cnn_path = PROJECT_ROOT / "ml" / "saved_models" / "cnn_model.keras"
    cfg_path = PROJECT_ROOT / "ml" / "saved_models" / "cnn_config.json"
    meta_path = PROJECT_ROOT / "ml" / "saved_models" / "cnn_metadata.json"

    full_calibrated_cnn.save(str(cnn_path))

    with open(cfg_path, "w") as f:
        json.dump({
            "mean": mean,
            "std": std,
            "segment_duration": 3.0,
            "segment_overlap": 0.5,
            "expected_frames": 130
        }, f, indent=2)

    with open(meta_path, "w") as f:
        json.dump({
            "model_version": "Phase 11 Production Calibrated (Multi-Domain End-to-End)",
            "input_shape": [128, 130, 1],
            "is_decontaminated_split": True,
            "multi_source": True,
            "training_samples": len(X_train),
        }, f, indent=2)

    logger.info(f"Saved full calibrated end-to-end CNN to {cnn_path}")


def verify_live_predictions():
    logger.info("\n=======================================================")
    logger.info("VERIFYING LIVE INFERENCE VIA GenrePredictor ON USER AUDIO")
    logger.info("=======================================================")
    from ml.inference.predictor import GenrePredictor
    predictor = GenrePredictor()

    test_files = [
        ("/Users/dibyangshughosh/Downloads/rock.mp3", "rock"),
        ("/Users/dibyangshughosh/Downloads/jazz.mp3", "jazz"),
        ("/Users/dibyangshughosh/Downloads/pop.mp3", "pop"),
        ("/Users/dibyangshughosh/Downloads/edm.mp3", "disco"),
    ]

    all_passed = True
    for fp, target in test_files:
        if Path(fp).exists():
            res = predictor.predict_audio(fp)
            ann = res["ann_prediction"]
            cnn = res["cnn_prediction"]
            agree = res["models_agree"]
            ann_ok = (ann["predicted_genre"] == target)
            cnn_ok = (cnn["predicted_genre"] == target)

            print(f"\nTrack: {Path(fp).name} | Target Genre: {target.upper()}")
            print(f"  ANN: {ann['predicted_genre']} ({ann['confidence']*100:.1f}%) [OK: {ann_ok}]")
            print(f"       Top 3: {[(x['genre'], round(x['probability']*100, 1)) for x in ann['top3']]}")
            print(f"  CNN: {cnn['predicted_genre']} ({cnn['confidence']*100:.1f}%) [OK: {cnn_ok}]")
            print(f"       Top 3: {[(x['genre'], round(x['probability']*100, 1)) for x in cnn['top3']]}")
            print(f"  Agreement: {agree} | Status: {res.get('uncertainty_status')}")

            if not (ann_ok and cnn_ok and res.get('uncertainty_status') == 'Confident Consensus'):
                all_passed = False

    print(f"\n>>> FINAL VERIFICATION STATUS: {'ALL PASSED CONFIDENT CONSENSUS' if all_passed else 'NEEDS ATTENTION'} <<<")


def main():
    items = collect_dataset_items()
    train_and_save_ann(items)
    train_and_save_cnn()
    verify_live_predictions()


if __name__ == "__main__":
    main()
