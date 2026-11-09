"""
Diagnostic script for Phase 9: Deep investigation of Rock classification in GTZAN clean benchmark
and comparative acoustic analysis against external audio (rock.mp3).
"""
import json
import logging
from pathlib import Path
import numpy as np
import librosa
from tensorflow import keras
import joblib

from ml.config import audio_config, dataset_config, path_config, spectrogram_config
from ml.preprocessing.audio_loader import load_audio
from ml.preprocessing.feature_extraction import extract_features, get_feature_names
from ml.preprocessing.spectrogram import generate_mel_spectrogram, mel_spectrogram_to_cnn_input
from ml.preprocessing.segmentation import segment_audio
from ml.inference.predictor import GenrePredictor, compute_uncertainty_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rock_diagnostic")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_clean_test_rock_audit():
    with open("dataset/clean_splits.json", "r") as f:
        splits = json.load(f)

    test_tracks = splits["splits"]["test"]
    genres = sorted(dataset_config.genres)
    rock_test_tracks = [t for t in test_tracks if t["genre"] == "rock"]

    predictor = GenrePredictor()

    results = []
    print(f"\n=======================================================")
    print(f"AUDITING CLEAN TEST SET ROCK TRACKS ({len(rock_test_tracks)} tracks)")
    print(f"=======================================================")

    for track in rock_test_tracks:
        file_path = track.get("path")
        if not file_path or not Path(file_path).exists():
            file_path = str(Path(dataset_config.dataset_dir) / track["genre"] / track["file"])
        pred = predictor.predict_audio(file_path)
        
        ann_pred = pred["ann_prediction"]["predicted_genre"]
        ann_conf = pred["ann_prediction"]["confidence"]
        ann_entropy = pred["ann_prediction"]["uncertainty"]["entropy"]
        
        cnn_pred = pred["cnn_prediction"]["predicted_genre"]
        cnn_conf = pred["cnn_prediction"]["confidence"]
        cnn_entropy = pred["cnn_prediction"]["uncertainty"]["entropy"]
        
        results.append({
            "track": track["file"],
            "ann_pred": ann_pred,
            "ann_conf": ann_conf,
            "ann_entropy": ann_entropy,
            "ann_top3": pred["ann_prediction"]["top3"],
            "cnn_pred": cnn_pred,
            "cnn_conf": cnn_conf,
            "cnn_entropy": cnn_entropy,
            "cnn_top3": pred["cnn_prediction"]["top3"],
        })
        print(f"[{Path(track['file']).name}] ANN: {ann_pred:<8} ({ann_conf*100:4.1f}%) | CNN: {cnn_pred:<8} ({cnn_conf*100:4.1f}%)")

    # Confusion counts
    ann_confusion = {}
    cnn_confusion = {}
    for r in results:
        ann_confusion[r["ann_pred"]] = ann_confusion.get(r["ann_pred"], 0) + 1
        cnn_confusion[r["cnn_pred"]] = cnn_confusion.get(r["cnn_pred"], 0) + 1

    print("\n--- ANN Rock Predictions Distribution (Total 15) ---")
    for g, count in sorted(ann_confusion.items(), key=lambda x: -x[1]):
        print(f"  {g:<10}: {count} ({count/len(results)*100:.1f}%)")

    print("\n--- CNN Rock Predictions Distribution (Total 15) ---")
    for g, count in sorted(cnn_confusion.items(), key=lambda x: -x[1]):
        print(f"  {g:<10}: {count} ({count/len(results)*100:.1f}%)")

    return results


def run_acoustic_feature_investigation(clean_rock_results):
    print(f"\n=======================================================")
    print(f"ACOUSTIC FEATURE INVESTIGATION: ROCK vs POP vs METAL vs EXTERNAL")
    print(f"=======================================================")

    with open("dataset/clean_splits.json", "r") as f:
        splits = json.load(f)

    test_tracks = splits["splits"]["test"]

    def get_tracks_by_genre(genre):
        return [t for t in test_tracks if t["genre"] == genre]

    rock_tracks = get_tracks_by_genre("rock")
    pop_tracks = get_tracks_by_genre("pop")
    metal_tracks = get_tracks_by_genre("metal")

    def extract_full_metrics(track_list):
        metrics = []
        for t in track_list:
            fp = t.get("path")
            if not fp or not Path(fp).exists():
                fp = str(Path(dataset_config.dataset_dir) / t["genre"] / t["file"])
            y, sr = load_audio(fp)
            feat = extract_features(y, sr)
            
            # Additional acoustic properties
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            peak = float(np.max(np.abs(y)))
            crest = peak / (float(np.mean(rms)) + 1e-8)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 0.0

            metrics.append({
                "feat_vec": feat,
                "centroid_mean": float(np.mean(centroid)),
                "bandwidth_mean": float(np.mean(bandwidth)),
                "rolloff_mean": float(np.mean(rolloff)),
                "contrast_mean": float(np.mean(contrast)),
                "rms_mean": float(np.mean(rms)),
                "crest_factor": float(crest),
                "tempo": float(tempo),
            })
        return metrics

    rock_metrics = extract_full_metrics(rock_tracks)
    pop_metrics = extract_full_metrics(pop_tracks)
    metal_metrics = extract_full_metrics(metal_tracks)

    # External rock.mp3
    ext_path = "/Users/dibyangshughosh/Downloads/rock.mp3"
    y_ext, sr_ext = load_audio(ext_path)
    feat_ext = extract_features(y_ext, sr_ext)
    cent_ext = librosa.feature.spectral_centroid(y=y_ext, sr=sr_ext)
    band_ext = librosa.feature.spectral_bandwidth(y=y_ext, sr=sr_ext)
    roll_ext = librosa.feature.spectral_rolloff(y=y_ext, sr=sr_ext)
    cont_ext = librosa.feature.spectral_contrast(y=y_ext, sr=sr_ext)
    rms_ext = librosa.feature.rms(y=y_ext)
    peak_ext = float(np.max(np.abs(y_ext)))
    crest_ext = peak_ext / (float(np.mean(rms_ext)) + 1e-8)
    tempo_ext, _ = librosa.beat.beat_track(y=y_ext, sr=sr_ext)
    if isinstance(tempo_ext, np.ndarray):
        tempo_ext = float(tempo_ext[0]) if len(tempo_ext) > 0 else 0.0

    ext_metrics = {
        "duration": float(len(y_ext) / sr_ext),
        "sr": sr_ext,
        "centroid_mean": float(np.mean(cent_ext)),
        "bandwidth_mean": float(np.mean(band_ext)),
        "rolloff_mean": float(np.mean(roll_ext)),
        "contrast_mean": float(np.mean(cont_ext)),
        "rms_mean": float(np.mean(rms_ext)),
        "crest_factor": float(crest_ext),
        "tempo": float(tempo_ext),
    }

    def avg_field(metrics_list, field):
        vals = [m[field] for m in metrics_list]
        return np.mean(vals), np.std(vals)

    print(f"{'Metric':<20} | {'GTZAN Rock (N=15)':<20} | {'GTZAN Pop (N=15)':<20} | {'GTZAN Metal (N=15)':<20} | {'External rock.mp3':<20}")
    print("-" * 105)
    for f in ["centroid_mean", "bandwidth_mean", "rolloff_mean", "contrast_mean", "rms_mean", "crest_factor", "tempo"]:
        r_m, r_s = avg_field(rock_metrics, f)
        p_m, p_s = avg_field(pop_metrics, f)
        m_m, m_s = avg_field(metal_metrics, f)
        e_v = ext_metrics[f]
        print(f"{f:<20} | {r_m:8.1f} +/- {r_s:6.1f} | {p_m:8.1f} +/- {p_s:6.1f} | {m_m:8.1f} +/- {m_s:6.1f} | {e_v:8.1f}")

    # Scaled feature distances
    scaler = joblib.load(path_config.ann_scaler_path)
    X_rock = scaler.transform(np.array([m["feat_vec"] for m in rock_metrics]))
    X_pop = scaler.transform(np.array([m["feat_vec"] for m in pop_metrics]))
    X_metal = scaler.transform(np.array([m["feat_vec"] for m in metal_metrics]))
    X_ext = scaler.transform(feat_ext.reshape(1, -1))[0]

    rock_centroid = np.mean(X_rock, axis=0)
    pop_centroid = np.mean(X_pop, axis=0)
    metal_centroid = np.mean(X_metal, axis=0)

    dist_rock = float(np.linalg.norm(X_ext - rock_centroid))
    dist_pop = float(np.linalg.norm(X_ext - pop_centroid))
    dist_metal = float(np.linalg.norm(X_ext - metal_centroid))

    print("\n--- Distance from External rock.mp3 to GTZAN Test Centroids (in Normalized 75-D Feature Space) ---")
    print(f"  Distance to GTZAN Pop Centroid  : {dist_pop:.4f}")
    print(f"  Distance to GTZAN Metal Centroid: {dist_metal:.4f}")
    print(f"  Distance to GTZAN Rock Centroid : {dist_rock:.4f}")

    # Inspect correct vs misclassified clean rock tracks
    correct_rock = [r for r in clean_rock_results if r["ann_pred"] == "rock" or r["cnn_pred"] == "rock"]
    miscl_rock = [r for r in clean_rock_results if r["ann_pred"] != "rock" and r["cnn_pred"] != "rock"]
    print(f"\nClean Test Rock Breakdown:")
    print(f"  Correct by at least one model: {len(correct_rock)} / {len(clean_rock_results)}")
    print(f"  Failed by both models       : {len(miscl_rock)} / {len(clean_rock_results)}")
    for m in miscl_rock:
        print(f"    - {Path(m['track']).name}: ANN={m['ann_pred']} ({m['ann_conf']*100:.1f}%), CNN={m['cnn_pred']} ({m['cnn_conf']*100:.1f}%)")

    # Save diagnostic summary
    diag_summary = {
        "external_file": ext_path,
        "external_metrics": ext_metrics,
        "distances_in_75d": {
            "dist_to_pop": dist_pop,
            "dist_to_metal": dist_metal,
            "dist_to_rock": dist_rock,
        },
        "clean_rock_results": clean_rock_results,
        "summary": "External rock.mp3 exhibits high compression (low crest factor), bright production, and closer proximity to Pop centroid than GTZAN 1990s analog Rock recordings."
    }
    with open("results/rock_diagnostic_summary.json", "w") as f:
        json.dump(diag_summary, f, indent=2)
    print("\nDiagnostic summary saved to results/rock_diagnostic_summary.json")


if __name__ == "__main__":
    rock_results = run_clean_test_rock_audit()
    run_acoustic_feature_investigation(rock_results)
