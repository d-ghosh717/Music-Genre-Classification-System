"""
GTZAN Dataset Decontamination & Duplicate Auditing Module.

Performs:
1. Exact file byte SHA-256 hashing.
2. Canonicalized PCM sample SHA-256 hashing.
3. Near-duplicate audio alignment and cross-correlation detection.
4. Cross-partition split leakage analysis on baseline splits.
5. Label conflict investigation (e.g. metal.00058 vs rock.00016).
6. Comprehensive audit serialization to results/decontamination_audit.json.
"""

import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import librosa

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import dataset_config, path_config
from ml.data_splitter import split_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_duplicates")


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 of raw file bytes."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_audio_pcm_sha256(file_path: Path) -> Tuple[Optional[str], Optional[np.ndarray], int]:
    """
    Load raw audio, canonicalize to float32 mono array, and compute SHA-256 hash.
    Returns (sha256_hash, audio_samples, sample_rate).
    """
    try:
        y, sr = sf.read(str(file_path))
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        pcm_bytes = y.astype(np.float32).tobytes()
        return hashlib.sha256(pcm_bytes).hexdigest(), y, sr
    except Exception as e:
        logger.warning(f"Failed to read audio from {file_path}: {e}")
        return None, None, 0


def detect_exact_duplicates(dataset_dir: Path) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], List[Path]]:
    """
    Scan dataset directory for exact byte and exact PCM sample duplicates.
    """
    raw_hashes: Dict[str, List[Path]] = {}
    pcm_hashes: Dict[str, List[Path]] = {}
    invalid_files: List[Path] = []

    files = sorted(list(dataset_dir.glob("*/*.wav")))
    logger.info(f"Scanning {len(files)} tracks in {dataset_dir} for exact duplicates...")

    for f in files:
        raw_h = compute_file_sha256(f)
        pcm_h, y, sr = compute_audio_pcm_sha256(f)
        if pcm_h is None:
            invalid_files.append(f)
            continue
        raw_hashes.setdefault(raw_h, []).append(f)
        pcm_hashes.setdefault(pcm_h, []).append(f)

    exact_file_dups = {h: flist for h, flist in raw_hashes.items() if len(flist) > 1}
    exact_pcm_dups = {h: flist for h, flist in pcm_hashes.items() if len(flist) > 1}

    return exact_file_dups, exact_pcm_dups, invalid_files


def detect_near_duplicates(
    valid_files: List[Path],
    correlation_threshold: float = 0.98,
    sample_rate: int = 11025,
    max_duration: float = 15.0,
) -> List[Dict[str, Any]]:
    """
    Detect near-duplicate audio recordings (e.g. tiny time offsets, identical compositions).
    Uses downsampled time-domain cross-correlation on normalized prefixes.
    """
    logger.info(f"Computing near-duplicate candidates across {len(valid_files)} valid tracks...")
    near_dups: List[Dict[str, Any]] = []

    # First compute summary acoustic vectors (mean MFCC + Chroma) for fast candidate screening
    feature_vectors = []
    audios = []
    for f in valid_files:
        try:
            y, sr = librosa.load(str(f), sr=sample_rate, mono=True, duration=max_duration)
            y_norm = y / (np.max(np.abs(y)) + 1e-9)
            audios.append(y_norm)

            chroma = librosa.feature.chroma_stft(y=y_norm, sr=sr, n_chroma=12)
            mfcc = librosa.feature.mfcc(y=y_norm, sr=sr, n_mfcc=20)
            v = np.hstack([np.mean(chroma, axis=1), np.std(chroma, axis=1), np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            feature_vectors.append(v_norm)
        except Exception as e:
            audios.append(np.zeros(1, dtype=np.float32))
            feature_vectors.append(np.zeros(64, dtype=np.float32))

    feat_matrix = np.array(feature_vectors)
    sim_matrix = np.dot(feat_matrix, feat_matrix.T)

    # For candidates with feature similarity > 0.998, perform full waveform cross-correlation check
    checked_pairs = set()
    for i in range(len(valid_files)):
        for j in range(i + 1, len(valid_files)):
            if sim_matrix[i, j] >= 0.998:
                f1 = valid_files[i]
                f2 = valid_files[j]
                y1 = audios[i]
                y2 = audios[j]

                # Quick sample length check
                min_len = min(len(y1), len(y2))
                if min_len < sample_rate * 5:
                    continue

                # Cross-correlate first 10 seconds
                sub_len = min(min_len, sample_rate * 10)
                s1 = y1[:sub_len]
                s2 = y2[:sub_len]

                corr = np.correlate(s1, s2, mode="full")
                norm_denom = (np.linalg.norm(s1) * np.linalg.norm(s2)) + 1e-9
                max_corr = float(np.max(corr) / norm_denom)
                max_idx = int(np.argmax(corr))
                offset_samples = max_idx - (sub_len - 1)
                offset_seconds = float(offset_samples / sample_rate)

                if max_corr >= correlation_threshold:
                    near_dups.append({
                        "file1": f1.name,
                        "genre1": f1.parent.name,
                        "file2": f2.name,
                        "genre2": f2.parent.name,
                        "correlation": round(max_corr, 5),
                        "offset_seconds": round(offset_seconds, 4),
                        "is_exact_sample": bool(abs(max_corr - 1.0) < 1e-6 and offset_samples == 0),
                    })

    return near_dups


def run_comprehensive_audit(output_json: Path = None) -> Dict[str, Any]:
    """
    Run complete duplicate audit and return report dictionary.
    """
    dataset_dir = Path(dataset_config.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Map current baseline split (seed=42)
    baseline_splits = split_dataset(random_seed=42)
    split_map: Dict[str, str] = {}
    for s_name, tracks in baseline_splits.items():
        for path_str, _ in tracks:
            split_map[Path(path_str).name] = s_name

    # 1. Exact duplicates
    exact_file_dups, exact_pcm_dups, invalid_files = detect_exact_duplicates(dataset_dir)

    all_valid_files = sorted([
        f for f in dataset_dir.glob("*/*.wav") if f not in invalid_files
    ])

    exact_groups = []
    group_idx = 1
    leaking_groups_count = 0
    conflict_groups_count = 0

    for h, flist in exact_pcm_dups.items():
        names = [f.name for f in flist]
        genres = [f.parent.name for f in flist]
        partitions = [split_map.get(f.name, "unknown") for f in flist]
        unique_partitions = set(partitions)
        unique_genres = set(genres)

        is_split_leak = len(unique_partitions) > 1
        is_label_conflict = len(unique_genres) > 1

        if is_split_leak:
            leaking_groups_count += 1
        if is_label_conflict:
            conflict_groups_count += 1

        exact_groups.append({
            "group_id": group_idx,
            "sha256_pcm": h,
            "files": names,
            "genres": genres,
            "partitions": partitions,
            "has_split_leakage": is_split_leak,
            "has_label_conflict": is_label_conflict,
        })
        group_idx += 1

    # 2. Near duplicates
    near_dups = detect_near_duplicates(all_valid_files)

    # 3. Label conflict investigation
    # Specifically inspect metal.00058.wav and rock.00016.wav
    metal_rock_info = {
        "file_a": "metal.00058.wav",
        "genre_a": "metal",
        "file_b": "rock.00016.wav",
        "genre_b": "rock",
        "sha256_match": False,
        "max_sample_difference": 0.0,
        "song_identification": "Queen - 'Tie Your Mother Down' (A Day at the Races, 1976)",
        "mir_literature_reference": "Bob L. Sturm (2012, 2013) 'An analysis of the GTZAN music genre dataset' / 'The GTZAN dataset: Its contents, its faults'",
        "nature_of_error": (
            "George Tzanetakis sourced Queen tracks from commercial CDs and filed excerpts under both 'rock' and 'metal'. "
            "Because Queen's composition is Hard Rock / Classic Rock, the 'metal' label is historically and stylistically inaccurate, "
            "representing both a duplicate repetition and an erroneous genre assignment."
        ),
        "recommended_resolution": (
            "Exclude 'metal.00058.wav' from the decontaminated dataset. "
            "Retain 'rock.00016.wav' under its legitimate genre 'rock'. "
            "This eliminates contradictory supervisory signals (P(metal)=1 vs P(rock)=1 for identical audio) "
            "while maintaining Queen's proper historical genre attribution."
        )
    }

    f_metal = dataset_dir / "metal" / "metal.00058.wav"
    f_rock = dataset_dir / "rock" / "rock.00016.wav"
    if f_metal.exists() and f_rock.exists():
        h_m = compute_file_sha256(f_metal)
        h_r = compute_file_sha256(f_rock)
        metal_rock_info["sha256_match"] = (h_m == h_r)
        ym, _ = sf.read(str(f_metal))
        yr, _ = sf.read(str(f_rock))
        metal_rock_info["max_sample_difference"] = float(np.max(np.abs(ym - yr)))

    report = {
        "dataset_path": str(dataset_dir),
        "total_files": len(list(dataset_dir.glob("*/*.wav"))),
        "invalid_corrupted_files": [f.name for f in invalid_files],
        "total_valid_tracks": len(all_valid_files),
        "exact_duplicate_groups_count": len(exact_groups),
        "exact_duplicate_tracks_count": sum(len(g["files"]) for g in exact_groups),
        "groups_with_split_leakage_in_baseline": leaking_groups_count,
        "groups_with_label_conflict": conflict_groups_count,
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_pairs": near_dups,
        "contradictory_label_case_study": metal_rock_info,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Audit report saved to: {output_json}")

    return report


if __name__ == "__main__":
    out_path = Path(path_config.results_dir) / "decontamination_audit.json"
    rep = run_comprehensive_audit(out_path)
    print("\n" + "=" * 70)
    print("GTZAN DATASET DUPLICATE & INTEGRITY AUDIT")
    print("=" * 70)
    print(f"Total tracks scanned: {rep['total_files']}")
    print(f"Corrupted unreadable tracks: {rep['invalid_corrupted_files']}")
    print(f"Valid audio tracks: {rep['total_valid_tracks']}")
    print(f"Exact duplicate groups (identical PCM): {rep['exact_duplicate_groups_count']} groups ({rep['exact_duplicate_tracks_count']} tracks)")
    print(f"Duplicate groups leaking across partitions in baseline: {rep['groups_with_split_leakage_in_baseline']}")
    print(f"Duplicate groups with contradictory genre labels: {rep['groups_with_label_conflict']}")
    print("\nExact Duplicate Groups:")
    for g in rep["exact_duplicate_groups"]:
        leak_str = "[LEAK!]" if g["has_split_leakage"] else "[No leak]"
        conf_str = "[LABEL CONFLICT!]" if g["has_label_conflict"] else "[Same genre]"
        print(f"  Group {g['group_id']:2d}: {g['files']} | genres={g['genres']} | splits={g['partitions']} {leak_str} {conf_str}")

    print("\nNear-Duplicate Candidate Pairs (high correlation):")
    for nd in rep["near_duplicate_pairs"][:10]:
        print(f"  {nd['file1']} ({nd['genre1']}) <-> {nd['file2']} ({nd['genre2']}): corr={nd['correlation']:.4f}, offset={nd['offset_seconds']}s")

    print("\nContradictory Label Investigation:")
    cs = rep["contradictory_label_case_study"]
    print(f"  Files: {cs['file_a']} vs {cs['file_b']}")
    print(f"  Song Identity: {cs['song_identification']}")
    print(f"  Max sample difference: {cs['max_sample_difference']}")
    print(f"  Nature of error: {cs['nature_of_error']}")
    print(f"  Resolution: {cs['recommended_resolution']}")
    print("=" * 70 + "\n")
