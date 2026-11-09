"""
Final Benchmark Comparison Generator for Phase 8.
Generates results/final/comparison.json and comparison visualizations.
"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import path_config
from ml.evaluation.visualizations import plot_model_comparison


def generate_final_comparison():
    final_dir = Path(path_config.results_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    ann_final_file = final_dir / "ann_metrics.json"
    cnn_final_file = final_dir / "cnn_metrics.json"

    with open(ann_final_file, "r") as f:
        ann_final = json.load(f)

    with open(cnn_final_file, "r") as f:
        cnn_final = json.load(f)

    # 1. Standard final plots
    plot_model_comparison(ann_final, cnn_final, str(final_dir))

    # 2. Structured final comparison JSON
    comparison_data = {
        "benchmark_type": "Phase 8 Decontaminated & Grouped Stratified (Zero Leakage)",
        "clean_split_manifest": "dataset/clean_splits.json",
        "decontamination_summary": {
            "total_valid_tracks": 998,
            "train_tracks": 694,
            "val_tracks": 151,
            "test_tracks": 153,
            "excluded_corrupted": ["jazz.00054.wav"],
            "excluded_contradictory": ["metal.00058.wav"],
            "retained_counterpart": "rock.00016.wav (Queen - 'Tie Your Mother Down')",
            "duplicate_groups_isolated": 30,
            "leakage_count": 0,
        },
        "final_ann": {
            "model_name": "Improved ANN (75 Features, Decontaminated Split)",
            "accuracy": ann_final["accuracy"],
            "macro_precision": ann_final["precision_macro"],
            "macro_recall": ann_final["recall_macro"],
            "macro_f1": ann_final["f1_macro"],
            "weighted_f1": ann_final["f1_weighted"],
            "test_loss": ann_final["test_loss"],
            "evaluated_tracks": ann_final.get("evaluated_tracks", 153),
        },
        "final_cnn": {
            "model_name": "2D CNN (Mel-Spectrograms, Decontaminated Split)",
            "accuracy": cnn_final["accuracy"],
            "macro_precision": cnn_final["precision_macro"],
            "macro_recall": cnn_final["recall_macro"],
            "macro_f1": cnn_final["f1_macro"],
            "weighted_f1": cnn_final["f1_weighted"],
            "test_loss": cnn_final["test_loss"],
            "evaluated_tracks": cnn_final.get("evaluated_tracks", 153),
        },
        "architectural_comparison": {
            "accuracy_difference": round(cnn_final["accuracy"] - ann_final["accuracy"], 4),
            "macro_f1_difference": round(cnn_final["f1_macro"] - ann_final["f1_macro"], 4),
            "superior_model": "CNN" if cnn_final["accuracy"] > ann_final["accuracy"] else "ANN",
        },
        "historical_evolution": {
            "baseline_contaminated": {
                "ann_accuracy": 0.6800,
                "ann_f1_macro": 0.6667,
                "cnn_accuracy": 0.8733,
                "cnn_f1_macro": 0.8715,
                "split_notes": "Contains 7 duplicate groups leaking across train/val/test partitions (Sturm 2013).",
            },
            "phase7_improved_contaminated": {
                "ann_accuracy": 0.7333,
                "ann_f1_macro": 0.7312,
                "cnn_accuracy": 0.8667,
                "cnn_f1_macro": 0.8646,
                "split_notes": "75 features added; evaluated on identical contaminated split for controlled ablation.",
            },
            "phase8_final_clean": {
                "ann_accuracy": ann_final["accuracy"],
                "ann_f1_macro": ann_final["f1_macro"],
                "cnn_accuracy": cnn_final["accuracy"],
                "cnn_f1_macro": cnn_final["f1_macro"],
                "split_notes": "Zero leakage; all duplicate groups grouped atomically; contradictory label resolved.",
            }
        }
    }

    out_comp = final_dir / "comparison.json"
    with open(out_comp, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Final comparison JSON saved to: {out_comp}")

    # 3. Create high-resolution benchmark evolution comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    stages = ["Historical Baseline\n(Contaminated)", "Phase 7 Improved\n(Contaminated Split)", "Phase 8 Final Benchmark\n(Zero-Leakage Clean)"]
    x = np.arange(len(stages))
    width = 0.35

    ann_accs = [68.00, 73.33, ann_final["accuracy"] * 100]
    cnn_accs = [87.33, 86.67, cnn_final["accuracy"] * 100]

    rects1 = ax1.bar(x - width/2, ann_accs, width, label="ANN (Tabular)", color="#64748b")
    rects2 = ax1.bar(x + width/2, cnn_accs, width, label="CNN (Mel-Spectrograms)", color="#0f766e")

    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Benchmark Evolution: Test Accuracy Across Pipeline Stages")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend()

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # F1 Plot
    ann_f1s = [66.67, 73.12, ann_final["f1_macro"] * 100]
    cnn_f1s = [87.15, 86.46, cnn_final["f1_macro"] * 100]

    rects3 = ax2.bar(x - width/2, ann_f1s, width, label="ANN (Tabular)", color="#64748b")
    rects4 = ax2.bar(x + width/2, cnn_f1s, width, label="CNN (Mel-Spectrograms)", color="#0f766e")

    ax2.set_ylabel("Macro F1-Score (%)")
    ax2.set_title("Benchmark Evolution: Macro F1-Score Across Pipeline Stages")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend()

    for r in rects3:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for r in rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_path = final_dir / "benchmark_evolution_comparison.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Benchmark evolution chart saved to: {chart_path}")


if __name__ == "__main__":
    generate_final_comparison()

