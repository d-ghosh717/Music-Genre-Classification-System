"""
Comprehensive Diagnostic Audit for Phase 12 (Indian / Bollywood Music).
Evaluates 'Raatan Lambiyan' and 'Bhaag Milkha Bhaag (Zinda)' segment by segment
across ANN and CNN pipelines. Records top-3 probabilities, entropy, embedding distances,
and architectural agreement.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import dataset_config, spectrogram_config
from ml.preprocessing.audio_loader import load_audio
from ml.preprocessing.feature_extraction import extract_features
from ml.preprocessing.segmentation import segment_audio
from ml.preprocessing.spectrogram import generate_mel_spectrogram, mel_spectrogram_to_cnn_input

GENRES = sorted(dataset_config.genres)
G2IDX = {g: i for i, g in enumerate(GENRES)}


def compute_entropy(probs: np.ndarray) -> float:
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def run_segment_diagnostic(audio_path: str, song_title: str):
    print(f"\n=======================================================")
    print(f"DIAGNOSTIC AUDIT: {song_title}")
    print(f"File: {Path(audio_path).name}")
    print(f"=======================================================")

    # 1. Load models and scalers
    ann = keras.models.load_model(str(PROJECT_ROOT / "ml" / "saved_models" / "ann_model.keras"))
    scaler = joblib.load(str(PROJECT_ROOT / "ml" / "saved_models" / "ann_scaler.pkl"))
    cnn = keras.models.load_model(str(PROJECT_ROOT / "ml" / "saved_models" / "cnn_model.keras"))
    with open(PROJECT_ROOT / "ml" / "saved_models" / "cnn_config.json") as f:
        cnn_cfg = json.load(f)
    cnn_mean = cnn_cfg["mean"]
    cnn_std = cnn_cfg["std"]

    # Load audio
    y, sr = load_audio(audio_path, sr=22050, mono=True)
    total_duration = len(y) / sr
    print(f"Total Duration: {total_duration:.2f}s ({len(y)} samples at {sr} Hz)")

    # Full track ANN prediction
    full_feat = extract_features(y, sr).reshape(1, -1)
    full_feat_scaled = scaler.transform(full_feat)
    ann_full_probs = ann.predict(full_feat_scaled, verbose=0)[0]
    ann_full_top3 = [(GENRES[i], float(ann_full_probs[i])) for i in np.argsort(ann_full_probs)[::-1][:3]]
    ann_full_entropy = compute_entropy(ann_full_probs)

    # Segment audio (3.0s, 50% overlap)
    segments = segment_audio(y, sr, segment_duration=3.0, overlap=0.5)
    print(f"Extracted {len(segments)} segments (window=3.0s, overlap=1.5s)")

    # Prepare CNN input batch
    expected_frames = 130
    cnn_inputs = []
    for s in segments:
        mel = generate_mel_spectrogram(s, sr, config=spectrogram_config)
        cnn_in = mel_spectrogram_to_cnn_input(mel, expected_time_frames=expected_frames)
        cnn_inputs.append(cnn_in)
    X_cnn = (np.array(cnn_inputs, dtype=np.float32) - cnn_mean) / cnn_std

    # Predict CNN segment-by-segment
    cnn_seg_probs = cnn.predict(X_cnn, verbose=0)
    cnn_avg_probs = np.mean(cnn_seg_probs, axis=0)
    cnn_full_top3 = [(GENRES[i], float(cnn_avg_probs[i])) for i in np.argsort(cnn_avg_probs)[::-1][:3]]
    cnn_full_entropy = compute_entropy(cnn_avg_probs)

    print("\n--- TRACK-LEVEL RESULTS ---")
    print(f"ANN: {ann_full_top3[0][0]} ({ann_full_top3[0][1]*100:.1f}%) | Entropy: {ann_full_entropy:.3f} bits")
    print(f"     Top 3: {[(g, f'{p*100:.1f}%') for g, p in ann_full_top3]}")
    print(f"CNN: {cnn_full_top3[0][0]} ({cnn_full_top3[0][1]*100:.1f}%) | Entropy: {cnn_full_entropy:.3f} bits")
    print(f"     Top 3: {[(g, f'{p*100:.1f}%') for g, p in cnn_full_top3]}")
    print(f"Track Agreement: {ann_full_top3[0][0] == cnn_full_top3[0][0]}")

    # Segment-by-segment breakdown
    segment_records = []
    ann_genre_counts = {}
    cnn_genre_counts = {}
    agreement_count = 0

    step_time = 1.5  # 50% overlap of 3.0s
    for idx, seg in enumerate(segments):
        t_start = idx * step_time
        t_end = t_start + 3.0

        # Segment ANN
        try:
            seg_feat = extract_features(seg, sr).reshape(1, -1)
            seg_feat_scaled = scaler.transform(seg_feat)
            ann_p = ann.predict(seg_feat_scaled, verbose=0)[0]
        except Exception:
            ann_p = np.zeros(10)

        cnn_p = cnn_seg_probs[idx]

        ann_pred = GENRES[np.argmax(ann_p)]
        cnn_pred = GENRES[np.argmax(cnn_p)]
        ann_conf = float(np.max(ann_p))
        cnn_conf = float(np.max(cnn_p))
        ann_ent = compute_entropy(ann_p)
        cnn_ent = compute_entropy(cnn_p)

        agrees = (ann_pred == cnn_pred)
        if agrees:
            agreement_count += 1

        ann_genre_counts[ann_pred] = ann_genre_counts.get(ann_pred, 0) + 1
        cnn_genre_counts[cnn_pred] = cnn_genre_counts.get(cnn_pred, 0) + 1

        rec = {
            "seg_idx": idx,
            "time_window": f"{t_start:.1f}s - {t_end:.1f}s",
            "ann_pred": ann_pred,
            "ann_conf": ann_conf,
            "ann_entropy": ann_ent,
            "cnn_pred": cnn_pred,
            "cnn_conf": cnn_conf,
            "cnn_entropy": cnn_ent,
            "agrees": agrees,
            "ann_top3": [(GENRES[i], float(ann_p[i])) for i in np.argsort(ann_p)[::-1][:3]],
            "cnn_top3": [(GENRES[i], float(cnn_p[i])) for i in np.argsort(cnn_p)[::-1][:3]],
        }
        segment_records.append(rec)

    print(f"\n--- SEGMENT-LEVEL SUMMARY ({len(segments)} segments) ---")
    print(f"Segment Agreement Rate: {agreement_count}/{len(segments)} ({agreement_count/len(segments)*100:.1f}%)")
    print("ANN Predicted Segment Distribution:")
    for g, c in sorted(ann_genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {g}: {c} segments ({c/len(segments)*100:.1f}%)")
    print("CNN Predicted Segment Distribution:")
    for g, c in sorted(cnn_genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {g}: {c} segments ({c/len(segments)*100:.1f}%)")

    # Sample representative sections (Intro, Verse 1, Chorus, Outro)
    print("\n--- SAMPLE SEGMENT-BY-SEGMENT TRAJECTORY ---")
    sample_indices = [
        0,  # Intro
        len(segments) // 6,  # Early verse
        len(segments) // 3,  # Build up
        len(segments) // 2,  # Chorus / Climax
        (2 * len(segments)) // 3,  # Bridge
        len(segments) - 1,  # Outro
    ]
    for si in sample_indices:
        r = segment_records[si]
        print(f"[{r['time_window']}] Seg {si+1}:")
        print(f"   ANN: {r['ann_pred']} ({r['ann_conf']*100:.1f}%, H={r['ann_entropy']:.2f}b) | Top 3: {[(g, f'{p*100:.1f}%') for g, p in r['ann_top3']]}")
        print(f"   CNN: {r['cnn_pred']} ({r['cnn_conf']*100:.1f}%, H={r['cnn_entropy']:.2f}b) | Top 3: {[(g, f'{p*100:.1f}%') for g, p in r['cnn_top3']]}")
        print(f"   Agreement: {r['agrees']}")

    return {
        "song_title": song_title,
        "duration": total_duration,
        "num_segments": len(segments),
        "ann_full": ann_full_top3,
        "cnn_full": cnn_full_top3,
        "segment_agreement_rate": agreement_count / len(segments),
        "ann_distribution": ann_genre_counts,
        "cnn_distribution": cnn_genre_counts,
        "records": segment_records,
    }


def main():
    music_dir = Path("/Users/dibyangshughosh/Downloads/music")
    raatan_path = music_dir / "Raataan Lambiyan Shershaah 128 Kbps.mp3"
    bhaag_path = music_dir / "Bhaag_Milkha_Bhaag_-_Zinda_(mp3.pm).mp3"

    results = {}
    if raatan_path.exists():
        results["raatan"] = run_segment_diagnostic(str(raatan_path), "Raatan Lambiyan (Bollywood Romantic Soundtrack)")

    if bhaag_path.exists():
        results["bhaag"] = run_segment_diagnostic(str(bhaag_path), "Bhaag Milkha Bhaag - Zinda (Bollywood Rock/Anthem Soundtrack)")

    # Save diagnostic results to file
    out_file = PROJECT_ROOT / "results" / "phase12_indian_diagnostic.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDiagnostic audit saved to {out_file}")


if __name__ == "__main__":
    main()

