"""
Phase 10: Evaluation of Unchanged Phase 8 ANN & CNN Checkpoints
on the 100-Track MTG-Jamendo External Dataset.

Computes:
- Per-class precision, recall, F1
- Overall accuracy, macro F1, weighted F1
- Confusion matrices
- Calibration and prediction entropy
- ANN/CNN agreement
- Track-level predictions CSV
- Visualizations: ann_confusion_matrix.png, cnn_confusion_matrix.png, genre_comparison.png
"""
import csv
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml.config import dataset_config, path_config
from ml.inference.predictor import GenrePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_external")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate_external_dataset():
    manifest_path = PROJECT_ROOT / "results" / "external" / "external_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tracks = list(reader)

    logger.info(f"Loaded {len(tracks)} tracks from external manifest.")
    genres = sorted(dataset_config.genres)
    genre_to_idx = {g: i for i, g in enumerate(genres)}

    predictor = GenrePredictor()

    y_true = []
    y_pred_ann = []
    y_pred_cnn = []
    
    ann_probs_list = []
    cnn_probs_list = []

    detailed_predictions = []

    for i, track in enumerate(tracks):
        audio_path = str(PROJECT_ROOT / track["audio_path"])
        ground_truth = track["ground_truth_genre"]
        y_true.append(ground_truth)

        pred = predictor.predict_audio(audio_path)
        ann = pred["ann_prediction"]
        cnn = pred["cnn_prediction"]

        ann_pred = ann["predicted_genre"]
        ann_conf = ann["confidence"]
        ann_entropy = ann["uncertainty"]["entropy"]

        cnn_pred = cnn["predicted_genre"]
        cnn_conf = cnn["confidence"]
        cnn_entropy = cnn["uncertainty"]["entropy"]

        y_pred_ann.append(ann_pred)
        y_pred_cnn.append(cnn_pred)

        detailed_predictions.append({
            "track_id": track["track_id"],
            "track_name": track["track_name"],
            "artist_name": track["artist_name"],
            "ground_truth": ground_truth,
            "ann_predicted": ann_pred,
            "ann_confidence": round(ann_conf, 4),
            "ann_entropy": ann_entropy,
            "ann_correct": ann_pred == ground_truth,
            "cnn_predicted": cnn_pred,
            "cnn_confidence": round(cnn_conf, 4),
            "cnn_entropy": cnn_entropy,
            "cnn_correct": cnn_pred == ground_truth,
            "models_agree": ann_pred == cnn_pred,
            "uncertainty_status": pred["uncertainty_status"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(tracks):
            logger.info(f"Evaluated {i + 1}/{len(tracks)} tracks...")

    # Save detailed predictions CSV
    pred_csv_path = PROJECT_ROOT / "results" / "external" / "external_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "track_id",
            "track_name",
            "artist_name",
            "ground_truth",
            "ann_predicted",
            "ann_confidence",
            "ann_entropy",
            "ann_correct",
            "cnn_predicted",
            "cnn_confidence",
            "cnn_entropy",
            "cnn_correct",
            "models_agree",
            "uncertainty_status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_predictions)
    logger.info(f"Saved detailed predictions to {pred_csv_path}")

    # Metrics computation
    ann_acc = accuracy_score(y_true, y_pred_ann)
    cnn_acc = accuracy_score(y_true, y_pred_cnn)

    ann_macro_f1 = f1_score(y_true, y_pred_ann, labels=genres, average="macro", zero_division=0)
    cnn_macro_f1 = f1_score(y_true, y_pred_cnn, labels=genres, average="macro", zero_division=0)

    ann_weighted_f1 = f1_score(y_true, y_pred_ann, labels=genres, average="weighted", zero_division=0)
    cnn_weighted_f1 = f1_score(y_true, y_pred_cnn, labels=genres, average="weighted", zero_division=0)

    ann_report = classification_report(y_true, y_pred_ann, labels=genres, output_dict=True, zero_division=0)
    cnn_report = classification_report(y_true, y_pred_cnn, labels=genres, output_dict=True, zero_division=0)

    ann_cm = confusion_matrix(y_true, y_pred_ann, labels=genres)
    cnn_cm = confusion_matrix(y_true, y_pred_cnn, labels=genres)

    agreement_rate = float(np.mean([d["models_agree"] for d in detailed_predictions]))

    # Average confidence & entropy
    ann_mean_conf = float(np.mean([d["ann_confidence"] for d in detailed_predictions]))
    cnn_mean_conf = float(np.mean([d["cnn_confidence"] for d in detailed_predictions]))

    ann_mean_entropy = float(np.mean([d["ann_entropy"] for d in detailed_predictions]))
    cnn_mean_entropy = float(np.mean([d["cnn_entropy"] for d in detailed_predictions]))

    metrics_payload = {
        "dataset_name": "MTG-Jamendo External Benchmark (100 Distinct Artists)",
        "num_tracks": len(tracks),
        "genres": genres,
        "agreement_rate": round(agreement_rate, 4),
        "ann": {
            "accuracy": round(float(ann_acc), 4),
            "macro_f1": round(float(ann_macro_f1), 4),
            "weighted_f1": round(float(ann_weighted_f1), 4),
            "mean_confidence": round(ann_mean_conf, 4),
            "mean_entropy": round(ann_mean_entropy, 4),
            "classification_report": ann_report,
            "confusion_matrix": ann_cm.tolist(),
        },
        "cnn": {
            "accuracy": round(float(cnn_acc), 4),
            "macro_f1": round(float(cnn_macro_f1), 4),
            "weighted_f1": round(float(cnn_weighted_f1), 4),
            "mean_confidence": round(cnn_mean_conf, 4),
            "mean_entropy": round(cnn_mean_entropy, 4),
            "classification_report": cnn_report,
            "confusion_matrix": cnn_cm.tolist(),
        },
    }

    # Save metrics JSON
    metrics_json_path = PROJECT_ROOT / "results" / "external" / "external_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved metrics JSON to {metrics_json_path}")

    # Save classification report text
    report_txt_path = PROJECT_ROOT / "results" / "external" / "external_classification_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=========================================================================\n")
        f.write("      PHASE 10: EXTERNAL BENCHMARK EVALUATION (MTG-JAMENDO 100 TRACKS)   \n")
        f.write("=========================================================================\n\n")
        f.write(f"Dataset: MTG-Jamendo (100 tracks, 10 per genre, 100 distinct artists)\n")
        f.write(f"Model Agreement Rate: {agreement_rate*100:.2f}%\n\n")
        f.write("--- ANN (Multilayer Perceptron) Results ---\n")
        f.write(f"Accuracy   : {ann_acc*100:.2f}%\n")
        f.write(f"Macro F1   : {ann_macro_f1:.4f}\n")
        f.write(f"Weighted F1: {ann_weighted_f1:.4f}\n")
        f.write(f"Mean Conf  : {ann_mean_conf*100:.1f}%\n")
        f.write(f"Mean Entr  : {ann_mean_entropy:.2f} bits\n\n")
        f.write(classification_report(y_true, y_pred_ann, labels=genres, zero_division=0))
        f.write("\n\n--- CNN (Convolutional Neural Network) Results ---\n")
        f.write(f"Accuracy   : {cnn_acc*100:.2f}%\n")
        f.write(f"Macro F1   : {cnn_macro_f1:.4f}\n")
        f.write(f"Weighted F1: {cnn_weighted_f1:.4f}\n")
        f.write(f"Mean Conf  : {cnn_mean_conf*100:.1f}%\n")
        f.write(f"Mean Entr  : {cnn_mean_entropy:.2f} bits\n\n")
        f.write(classification_report(y_true, y_pred_cnn, labels=genres, zero_division=0))

    logger.info(f"Saved text report to {report_txt_path}")

    # Plot Confusion Matrices
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # ANN Confusion Matrix
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(ann_cm, annot=True, fmt="d", cmap="Blues", xticklabels=genres, yticklabels=genres, ax=ax)
    ax.set_title(f"External MTG-Jamendo: ANN Confusion Matrix (Acc: {ann_acc*100:.1f}%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Genre", fontsize=11)
    ax.set_ylabel("True Genre", fontsize=11)
    plt.tight_layout()
    ann_cm_path = PROJECT_ROOT / "results" / "external" / "ann_confusion_matrix.png"
    plt.savefig(ann_cm_path, dpi=200)
    plt.close()

    # CNN Confusion Matrix
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cnn_cm, annot=True, fmt="d", cmap="Purples", xticklabels=genres, yticklabels=genres, ax=ax)
    ax.set_title(f"External MTG-Jamendo: CNN Confusion Matrix (Acc: {cnn_acc*100:.1f}%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Genre", fontsize=11)
    ax.set_ylabel("True Genre", fontsize=11)
    plt.tight_layout()
    cnn_cm_path = PROJECT_ROOT / "results" / "external" / "cnn_confusion_matrix.png"
    plt.savefig(cnn_cm_path, dpi=200)
    plt.close()

    # Per-Genre Comparison Chart (GTZAN vs External)
    # Load GTZAN final metrics
    gtzan_ann_path = PROJECT_ROOT / "results" / "final" / "ann_metrics.json"
    gtzan_cnn_path = PROJECT_ROOT / "results" / "final" / "cnn_metrics.json"
    with open(gtzan_ann_path, "r") as f:
        gtzan_ann = json.load(f)["classification_report"]
    with open(gtzan_cnn_path, "r") as f:
        gtzan_cnn = json.load(f)["classification_report"]

    x = np.arange(len(genres))
    width = 0.20

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - 1.5 * width, [gtzan_ann[g]["f1-score"] for g in genres], width, label="GTZAN ANN F1", color="#3b82f6", alpha=0.85)
    ax.bar(x - 0.5 * width, [ann_report[g]["f1-score"] for g in genres], width, label="External Jamendo ANN F1", color="#93c5fd", alpha=0.85)
    ax.bar(x + 0.5 * width, [gtzan_cnn[g]["f1-score"] for g in genres], width, label="GTZAN CNN F1", color="#8b5cf6", alpha=0.85)
    ax.bar(x + 1.5 * width, [cnn_report[g]["f1-score"] for g in genres], width, label="External Jamendo CNN F1", color="#c4b5fd", alpha=0.85)

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Generalisation Gap: GTZAN Clean Benchmark vs External MTG-Jamendo Benchmark", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in genres], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    comp_path = PROJECT_ROOT / "results" / "external" / "genre_comparison.png"
    plt.savefig(comp_path, dpi=200)
    plt.close()

    logger.info("Saved all evaluation figures.")
    return metrics_payload


if __name__ == "__main__":
    evaluate_external_dataset()

