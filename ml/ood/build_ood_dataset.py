"""
Assembles the Out-of-Distribution (OOD) Calibration and Validation Dataset.
Selects independent recordings strictly outside the GTZAN 10-genre taxonomy:
  - Folk
  - Ambient
  - Electronic / EDM
  - Soundtrack / Orchestral
Guarantees 100% isolation from the Phase 10 external benchmark tracks and artists.
"""
import os
import sys
import csv
import json
import logging
import urllib.request
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_ood")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OOD_DATASET_DIR = PROJECT_ROOT / "dataset" / "ood_calibration"

OOD_GENRES = ["folk", "ambient", "soundtrack", "electronic"]


def load_phase10_exclusions():
    manifest_file = PROJECT_ROOT / "results" / "external" / "external_manifest.csv"
    excluded_tracks = set()
    excluded_artists = set()
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                excluded_tracks.add(r["track_id"])
                excluded_artists.add(r["artist_id"])
    logger.info(f"Loaded {len(excluded_tracks)} Phase 10 benchmark tracks to exclude.")
    return excluded_tracks, excluded_artists


def assemble_ood_calibration_dataset(target_per_genre: int = 10):
    OOD_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = OOD_DATASET_DIR / "raw"
    audio_dir = OOD_DATASET_DIR / "audio_30s"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    excluded_tracks, excluded_artists = load_phase10_exclusions()

    metadata_path = PROJECT_ROOT / "dataset" / "external_jamendo" / "metadata" / "autotagging_genre.tsv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")

    # Find candidates
    candidates = {g: [] for g in OOD_GENRES}
    gtzan_genres = {
        "genre---blues", "genre---classical", "genre---country",
        "genre---disco", "genre---hiphop", "genre---jazz",
        "genre---metal", "genre---pop", "genre---reggae", "genre---rock"
    }

    with open(metadata_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tid = row.get("TRACK_ID")
            dur = float(row.get("DURATION", 0))
            if dur < 35.0 or tid in excluded_tracks:
                continue

            tags = set(t.strip() for t in row.get("TAGS", "").split("\t") if t.strip())
            # Must not contain any GTZAN genre
            if tags.intersection(gtzan_genres):
                continue

            for g in OOD_GENRES:
                if f"genre---{g}" in tags:
                    num_id = tid.replace("track_", "")
                    candidates[g].append({
                        "track_id": tid,
                        "numeric_id": num_id,
                        "genre": g,
                        "duration": dur,
                    })

    manifest_rows = []
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    # 1. Download and process MTG-Jamendo OOD tracks
    for g in OOD_GENRES:
        g_dir = audio_dir / g
        g_dir.mkdir(parents=True, exist_ok=True)
        g_cands = candidates[g]
        count = 0

        for cand in g_cands:
            if count >= target_per_genre:
                break

            tid = cand["track_id"]
            num_id = cand["numeric_id"]
            raw_path = raw_dir / f"{tid}.mp3"
            processed_path = g_dir / f"{tid}.wav"

            # Download if not present
            if not raw_path.exists():
                url = f"https://storage.jamendo.com/download/track/{num_id}/mp32/"
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=15) as resp, open(raw_path, "wb") as f_out:
                        f_out.write(resp.read())
                except Exception as e:
                    if raw_path.exists():
                        raw_path.unlink()
                    continue

            # Process 30s center crop
            try:
                y, sr = librosa.load(str(raw_path), sr=22050, mono=True)
                if len(y) / sr < 30.0:
                    continue

                # 30s center window
                target_len = int(30.0 * 22050)
                start = max(0, (len(y) - target_len) // 2)
                y_30s = y[start : start + target_len]

                # Peak normalize
                max_amp = np.max(np.abs(y_30s))
                if max_amp > 0:
                    y_30s = y_30s / max_amp

                sf.write(str(processed_path), y_30s, 22050)

                manifest_rows.append({
                    "track_id": tid,
                    "genre": g,
                    "source": "MTG-Jamendo",
                    "audio_path": str(processed_path.relative_to(PROJECT_ROOT)),
                    "duration": 30.0,
                    "is_ood": True,
                })
                count += 1
                logger.info(f"  [OOD] Processed {g} {count}/{target_per_genre}: {tid}")
            except Exception as e:
                logger.warning(f"Error processing {raw_path}: {e}")

    # 2. Add local external test uploads if present (folk.mp3, edm.mp3)
    user_music_dir = Path("/Users/dibyangshughosh/Downloads/music")
    if (user_music_dir / "folk.mp3").exists():
        try:
            y, sr = librosa.load(str(user_music_dir / "folk.mp3"), sr=22050, mono=True)
            target_len = int(min(len(y), 30.0 * 22050))
            y_clip = y[:target_len]
            max_amp = np.max(np.abs(y_clip))
            if max_amp > 0:
                y_clip = y_clip / max_amp
            folk_out = audio_dir / "folk" / "user_folk.wav"
            folk_out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(folk_out), y_clip, 22050)
            manifest_rows.append({
                "track_id": "user_folk_upload",
                "genre": "folk",
                "source": "User Download (folk.mp3)",
                "audio_path": str(folk_out.relative_to(PROJECT_ROOT)),
                "duration": len(y_clip) / 22050,
                "is_ood": True,
            })
            logger.info("  [OOD] Included user_folk.wav from Downloads")
        except Exception as e:
            logger.warning(f"Error processing user folk.mp3: {e}")

    if (user_music_dir / "edm.mp3").exists():
        try:
            y, sr = librosa.load(str(user_music_dir / "edm.mp3"), sr=22050, mono=True)
            target_len = int(min(len(y), 30.0 * 22050))
            y_clip = y[:target_len]
            max_amp = np.max(np.abs(y_clip))
            if max_amp > 0:
                y_clip = y_clip / max_amp
            edm_out = audio_dir / "electronic" / "user_edm.wav"
            edm_out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(edm_out), y_clip, 22050)
            manifest_rows.append({
                "track_id": "user_edm_upload",
                "genre": "electronic",
                "source": "User Download (edm.mp3)",
                "audio_path": str(edm_out.relative_to(PROJECT_ROOT)),
                "duration": len(y_clip) / 22050,
                "is_ood": True,
            })
            logger.info("  [OOD] Included user_edm.wav from Downloads")
        except Exception as e:
            logger.warning(f"Error processing user edm.mp3: {e}")

    # Write manifest
    manifest_csv = OOD_DATASET_DIR / "ood_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "genre", "source", "audio_path", "duration", "is_ood"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(f"\nSuccessfully assembled OOD dataset: {len(manifest_rows)} calibration tracks saved to {manifest_csv}")
    return manifest_rows


if __name__ == "__main__":
    assemble_ood_calibration_dataset(target_per_genre=10)

