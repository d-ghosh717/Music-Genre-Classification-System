"""
Controlled Rock Classification Experiments (Train/Val Split Only).
Evaluates Class Weighting and Focal Loss on the clean training/validation splits.
Held-out test set is NOT used for tuning.
"""
import json
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

from ml.config import dataset_config
from ml.models.ann_model import build_ann_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("controlled_experiments")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def focal_loss(gamma=2.0, alpha=0.25):
    """Multi-class focal loss for sparse categorical targets."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_one_hot = tf.one_hot(y_true, depth=10)
        # Clip to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_one_hot * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma)
        focal = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))
    return loss_fn


def run_ann_experiments():
    cache_path = "ml/saved_models/cache/ann_features_clean_75.npz"
    data = np.load(cache_path)
    X_train_raw = data["X_train"]
    y_train = data["y_train"]
    X_val_raw = data["X_val"]
    y_val = data["y_val"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    genres = sorted(dataset_config.genres)
    rock_idx = genres.index("rock")

    results = []

    # Config 1: Baseline Clean (Uniform weights, Standard CE)
    # Config 2: Class Weight Rock = 1.5
    # Config 3: Class Weight Rock = 2.0
    # Config 4: Class Weight Rock = 2.5
    # Config 5: Focal Loss gamma=1.5
    # Config 6: Focal Loss gamma=2.0

    configs = [
        {"name": "1. Baseline (Uniform CE)", "weights": None, "loss": "sparse_categorical_crossentropy"},
        {"name": "2. Class Weight (Rock=1.5)", "weights": {i: (1.5 if i == rock_idx else 1.0) for i in range(10)}, "loss": "sparse_categorical_crossentropy"},
        {"name": "3. Class Weight (Rock=2.0)", "weights": {i: (2.0 if i == rock_idx else 1.0) for i in range(10)}, "loss": "sparse_categorical_crossentropy"},
        {"name": "4. Class Weight (Rock=2.5)", "weights": {i: (2.5 if i == rock_idx else 1.0) for i in range(10)}, "loss": "sparse_categorical_crossentropy"},
        {"name": "5. Focal Loss (gamma=1.5)", "weights": None, "loss": focal_loss(gamma=1.5)},
        {"name": "6. Focal Loss (gamma=2.0)", "weights": None, "loss": focal_loss(gamma=2.0)},
    ]

    print(f"\n{'='*75}")
    print(f"RUNNING CONTROLLED EXPERIMENTS ON VALIDATION SET (151 tracks)")
    print(f"{'='*75}")

    for cfg in configs:
        tf.random.set_seed(42)
        np.random.seed(42)

        model = build_ann_model(
            input_dim=75,
            num_classes=10,
            layer_sizes=[256, 128, 64],
            dropout_rates=[0.25, 0.2, 0.2],
            use_batch_norm=True,
            learning_rate=0.001,
        )

        loss = cfg["loss"]
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=["accuracy"],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ]

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            class_weight=cfg["weights"],
            callbacks=callbacks,
            verbose=0,
        )

        y_val_probs = model.predict(X_val, verbose=0)
        y_val_pred = np.argmax(y_val_probs, axis=-1)

        val_acc = accuracy_score(y_val, y_val_pred)
        val_macro_f1 = f1_score(y_val, y_val_pred, average="macro")
        
        # Rock specific metrics
        y_val_rock_true = (y_val == rock_idx).astype(int)
        y_val_rock_pred = (y_val_pred == rock_idx).astype(int)
        rock_rec = recall_score(y_val_rock_true, y_val_rock_pred, zero_division=0)
        rock_prec = precision_score(y_val_rock_true, y_val_rock_pred, zero_division=0)
        rock_f1 = f1_score(y_val_rock_true, y_val_rock_pred, zero_division=0)

        # Test on external rock.mp3
        from ml.preprocessing.audio_loader import load_audio
        from ml.preprocessing.feature_extraction import extract_features
        y_ext, sr_ext = load_audio("/Users/dibyangshughosh/Downloads/rock.mp3")
        feat_ext = extract_features(y_ext, sr_ext)
        feat_ext_scaled = scaler.transform(feat_ext.reshape(1, -1))
        ext_probs = model.predict(feat_ext_scaled, verbose=0)[0]
        ext_top_idx = int(np.argmax(ext_probs))
        ext_top_genre = genres[ext_top_idx]
        ext_top_conf = float(ext_probs[ext_top_idx])
        ext_rock_prob = float(ext_probs[rock_idx])

        res = {
            "config": cfg["name"],
            "val_accuracy": round(float(val_acc), 4),
            "val_macro_f1": round(float(val_macro_f1), 4),
            "rock_recall": round(float(rock_rec), 4),
            "rock_precision": round(float(rock_prec), 4),
            "rock_f1": round(float(rock_f1), 4),
            "ext_rock_prediction": ext_top_genre,
            "ext_rock_conf": round(ext_top_conf, 4),
            "ext_rock_prob": round(ext_rock_prob, 4),
        }
        results.append(res)
        print(f"[{cfg['name']}]")
        print(f"  Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_macro_f1:.4f}")
        print(f"  Rock Recall: {rock_rec*100:.1f}% | Precision: {rock_prec*100:.1f}% | F1: {rock_f1:.4f}")
        print(f"  External rock.mp3 -> Top: {ext_top_genre} ({ext_top_conf*100:.1f}%), Rock Prob: {ext_rock_prob*100:.1f}%\n")

    # Save experiment results
    with open("results/rock_controlled_experiments.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Controlled experiment results saved to results/rock_controlled_experiments.json")


if __name__ == "__main__":
    run_ann_experiments()

