"""
Evaluation metrics calculation module for ANN and CNN genre classifiers.
Computes standard classification metrics using scikit-learn without synthetic or hardcoded metrics.
"""
from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from ml.config import dataset_config, path_config


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genres: Optional[List[str]] = None,
    test_loss: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute rigorous evaluation metrics for multi-class classification.

    Args:
        y_true: Ground truth integer class indices (N,).
        y_pred: Predicted integer class indices (N,).
        genres: List of class names ordered by index.
        test_loss: Model loss evaluated on the test set.

    Returns:
        Dictionary containing all evaluation metrics and confusion matrix.
    """
    genres = genres or dataset_config.genres

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        target_names=genres,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "test_loss": float(test_loss),
        "classification_report": report,
        "confusion_matrix": cm,
        "genres": genres,
    }


def save_metrics_report(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics dictionary to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_metrics_report(filepath: str) -> Optional[Dict[str, Any]]:
    """Load metrics report from JSON if it exists."""
    path = Path(filepath)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
