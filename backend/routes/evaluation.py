"""
API routes for model evaluation results and visualizations.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Path as FPath
from fastapi.responses import FileResponse

from ml.config import path_config
from ml.evaluation.evaluate import load_metrics_report
from ml.evaluation.visualizations import plot_model_comparison

router = APIRouter(prefix="/api", tags=["Evaluation"])


@router.get("/evaluation")
async def get_evaluation_metrics():
    """
    Returns saved evaluation metrics for ANN and CNN models evaluated on the held-out test split.
    """
    ann_metrics_file = Path(path_config.ann_results_dir) / "metrics.json"
    cnn_metrics_file = Path(path_config.cnn_results_dir) / "metrics.json"

    ann_metrics = load_metrics_report(str(ann_metrics_file))
    cnn_metrics = load_metrics_report(str(cnn_metrics_file))

    # If both models have evaluated, ensure comparison plots exist
    if ann_metrics and cnn_metrics:
        plot_model_comparison(ann_metrics, cnn_metrics, path_config.comparison_results_dir)

    is_available = (ann_metrics is not None) or (cnn_metrics is not None)

    import json
    comparison_file = Path(path_config.comparison_results_dir) / "baseline_vs_improved.json"
    baseline_vs_improved = None
    if comparison_file.exists():
        try:
            with open(comparison_file, "r") as f:
                baseline_vs_improved = json.load(f)
        except Exception:
            pass

    external_file = Path(path_config.results_dir) / "external_evaluation.json"
    external_evaluation = None
    if external_file.exists():
        try:
            with open(external_file, "r") as f:
                external_evaluation = json.load(f)
        except Exception:
            pass

    final_comparison_file = Path(path_config.results_dir) / "final" / "comparison.json"
    final_benchmark = None
    if final_comparison_file.exists():
        try:
            with open(final_comparison_file, "r") as f:
                final_benchmark = json.load(f)
        except Exception:
            pass

    final_ann_file = Path(path_config.results_dir) / "final" / "ann_metrics.json"
    final_ann = None
    if final_ann_file.exists():
        final_ann = load_metrics_report(str(final_ann_file))

    final_cnn_file = Path(path_config.results_dir) / "final" / "cnn_metrics.json"
    final_cnn = None
    if final_cnn_file.exists():
        final_cnn = load_metrics_report(str(final_cnn_file))

    audit_file = Path(path_config.results_dir) / "decontamination_audit.json"
    decontamination_audit = None
    if audit_file.exists():
        try:
            with open(audit_file, "r") as f:
                decontamination_audit = json.load(f)
        except Exception:
            pass

    external_benchmark_file = Path(path_config.results_dir) / "external" / "external_metrics.json"
    external_benchmark = None
    if external_benchmark_file.exists():
        try:
            with open(external_benchmark_file, "r") as f:
                external_benchmark = json.load(f)
        except Exception:
            pass

    return {
        "available": is_available,
        "ann": ann_metrics,
        "cnn": cnn_metrics,
        "baseline_vs_improved": baseline_vs_improved,
        "external_evaluation": external_evaluation,
        "final_benchmark": final_benchmark,
        "final_ann": final_ann,
        "final_cnn": final_cnn,
        "decontamination_audit": decontamination_audit,
        "external_benchmark": external_benchmark,
    }


@router.get("/visualizations/{model}/{filename}")
async def get_visualization_image(
    model: str = FPath(..., description="'ann', 'cnn', 'comparison', 'final', or 'external'"),
    filename: str = FPath(..., description="Image filename, e.g. 'ann_confusion_matrix.png'"),
):
    """
    Safely serve saved evaluation visualization plots.
    Prevents path traversal attacks.
    """
    allowed_models = {"ann", "cnn", "comparison", "final", "external"}
    if model not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Invalid model category '{model}'.")

    # Sanitize filename
    safe_filename = Path(filename).name
    if not safe_filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="Only PNG images are served.")

    target_dir = Path(path_config.results_dir) / model
    file_path = (target_dir / safe_filename).resolve()

    # Ensure path stays within target_dir
    if not str(file_path).startswith(str(target_dir.resolve())):
        raise HTTPException(status_code=403, detail="Forbidden file path.")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Visualization '{filename}' not found for '{model}'.")

    return FileResponse(path=str(file_path), media_type="image/png")
