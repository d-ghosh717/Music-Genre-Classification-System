"""
Plotting and visualization utilities for model training curves, confusion matrices,
and architectural comparison charts.
"""
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ml.config import path_config


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str,
    output_dir: str,
) -> Tuple[str, str]:
    """
    Plot and save training & validation accuracy and loss curves.

    Args:
        history: Keras history.history dict (keys: 'accuracy', 'val_accuracy', 'loss', 'val_loss').
        model_name: "ann" or "cnn"
        output_dir: Directory to save plots.

    Returns:
        Tuple of (accuracy_plot_path, loss_plot_path).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("accuracy", [])) + 1)

    # 1. Combined Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("accuracy", []), "b-", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, history.get("val_accuracy", []), "r--", label="Validation Accuracy", linewidth=2)
    plt.title(f"{model_name.upper()} Model: Training & Validation Accuracy", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    acc_path = out_path / f"{model_name.lower()}_training_accuracy.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()

    # 1b. Separate Validation Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("val_accuracy", []), "r--", label="Validation Accuracy", linewidth=2)
    plt.title(f"{model_name.upper()} Model: Validation Accuracy", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    val_acc_path = out_path / f"{model_name.lower()}_validation_accuracy.png"
    plt.savefig(val_acc_path, dpi=300)
    plt.close()

    # 2. Combined Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("loss", []), "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, history.get("val_loss", []), "r--", label="Validation Loss", linewidth=2)
    plt.title(f"{model_name.upper()} Model: Training & Validation Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    loss_path = out_path / f"{model_name.lower()}_training_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # 2b. Separate Validation Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("val_loss", []), "r--", label="Validation Loss", linewidth=2)
    plt.title(f"{model_name.upper()} Model: Validation Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    val_loss_path = out_path / f"{model_name.lower()}_validation_loss.png"
    plt.savefig(val_loss_path, dpi=300)
    plt.close()

    return str(acc_path), str(loss_path)


def plot_confusion_matrix(
    cm: List[List[int]],
    genres: List[str],
    model_name: str,
    output_dir: str,
) -> str:
    """
    Plot and save confusion matrix heatmap.

    Args:
        cm: Confusion matrix 2D list or array.
        genres: List of class genre labels.
        model_name: "ann" or "cnn".
        output_dir: Directory to save figure.

    Returns:
        Path to saved confusion matrix figure.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cm_array = np.array(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=genres,
        yticklabels=genres,
        cbar=True,
        square=True,
    )
    plt.title(f"{model_name.upper()} Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig_path = out_path / f"{model_name.lower()}_confusion_matrix.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return str(fig_path)


def plot_model_comparison(
    ann_metrics: Dict,
    cnn_metrics: Dict,
    output_dir: str,
) -> List[str]:
    """
    Generate comparison bar charts between ANN and CNN:
    1. Overall Accuracy and Macro F1 comparison
    2. Per-Genre F1-Score breakdown comparison

    Args:
        ann_metrics: Metrics dictionary for ANN.
        cnn_metrics: Metrics dictionary for CNN.
        output_dir: Output directory for comparison plots.

    Returns:
        List of generated image paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. Overall Metrics Bar Chart
    metrics_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    ann_vals = [
        ann_metrics.get("accuracy", 0.0),
        ann_metrics.get("precision_macro", 0.0),
        ann_metrics.get("recall_macro", 0.0),
        ann_metrics.get("f1_macro", 0.0),
        ann_metrics.get("f1_weighted", 0.0),
    ]
    cnn_vals = [
        cnn_metrics.get("accuracy", 0.0),
        cnn_metrics.get("precision_macro", 0.0),
        cnn_metrics.get("recall_macro", 0.0),
        cnn_metrics.get("f1_macro", 0.0),
        cnn_metrics.get("f1_weighted", 0.0),
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    rects1 = plt.bar(x - width/2, ann_vals, width, label="ANN (Tabular Features)", color="#3B82F6")
    rects2 = plt.bar(x + width/2, cnn_vals, width, label="CNN (Mel-Spectrograms)", color="#10B981")

    plt.ylabel("Score (0.0 - 1.0)", fontsize=11)
    plt.title("Architecture Comparison: ANN vs CNN Performance", fontsize=13, fontweight="bold")
    plt.xticks(x, metrics_names, fontsize=10)
    plt.ylim([0, 1.1])
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")

    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        plt.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        plt.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    comp_path1 = out_path / "model_metrics_comparison.png"
    plt.savefig(comp_path1, dpi=300)
    plt.close()
    generated.append(str(comp_path1))

    # 1b. Accuracy-Only Comparison
    plt.figure(figsize=(6, 5))
    acc_labels = ["ANN", "CNN"]
    acc_values = [ann_metrics.get("accuracy", 0.0), cnn_metrics.get("accuracy", 0.0)]
    colors = ["#3B82F6", "#10B981"]
    bars = plt.bar(acc_labels, acc_values, color=colors, width=0.45)
    plt.ylabel("Accuracy", fontsize=11)
    plt.title("Test Accuracy Comparison: ANN vs CNN", fontsize=13, fontweight="bold")
    plt.ylim([0, 1.1])
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars:
        h = b.get_height()
        plt.annotate(f"{h*100:.1f}%", xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    acc_comp_path = out_path / "model_accuracy_comparison.png"
    plt.savefig(acc_comp_path, dpi=300)
    plt.close()
    generated.append(str(acc_comp_path))

    # 1c. Macro F1-Only Comparison
    plt.figure(figsize=(6, 5))
    f1_labels = ["ANN", "CNN"]
    f1_values = [ann_metrics.get("f1_macro", 0.0), cnn_metrics.get("f1_macro", 0.0)]
    bars_f1 = plt.bar(f1_labels, f1_values, color=colors, width=0.45)
    plt.ylabel("Macro F1-Score", fontsize=11)
    plt.title("Macro F1 Comparison: ANN vs CNN", fontsize=13, fontweight="bold")
    plt.ylim([0, 1.1])
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars_f1:
        h = b.get_height()
        plt.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    f1_comp_path = out_path / "model_f1_comparison.png"
    plt.savefig(f1_comp_path, dpi=300)
    plt.close()
    generated.append(str(f1_comp_path))
    generated.append(str(comp_path1))

    # 2. Per-Genre F1-Score Breakdown
    genres = ann_metrics.get("genres", [])
    if genres:
        ann_f1s = [ann_metrics.get("classification_report", {}).get(g, {}).get("f1-score", 0.0) for g in genres]
        cnn_f1s = [cnn_metrics.get("classification_report", {}).get(g, {}).get("f1-score", 0.0) for g in genres]

        x_g = np.arange(len(genres))
        plt.figure(figsize=(12, 6))
        plt.bar(x_g - width/2, ann_f1s, width, label="ANN", color="#3B82F6")
        plt.bar(x_g + width/2, cnn_f1s, width, label="CNN", color="#10B981")

        plt.ylabel("F1-Score", fontsize=11)
        plt.title("Per-Genre F1-Score Comparison: ANN vs CNN", fontsize=13, fontweight="bold")
        plt.xticks(x_g, genres, rotation=45, ha="right", fontsize=10)
        plt.ylim([0, 1.1])
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()

        comp_path2 = out_path / "per_genre_f1_comparison.png"
        plt.savefig(comp_path2, dpi=300)
        plt.close()
        generated.append(str(comp_path2))

    return generated
