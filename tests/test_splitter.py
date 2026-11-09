"""
Tests for dataset splitting and data leakage prevention.
"""
from typing import List, Tuple
import pytest
from ml.data_splitter import split_dataset


def test_split_dataset_no_leakage():
    """Verify that split sets are strictly mutually exclusive."""
    # Synthetic dataset of 100 tracks across 10 genres
    synthetic_files: List[Tuple[str, str]] = []
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    for g in genres:
        for i in range(10):
            synthetic_files.append((f"/fake/path/{g}.{i:05d}.wav", g))

    splits = split_dataset(synthetic_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    train_paths = {p for p, _ in splits["train"]}
    val_paths = {p for p, _ in splits["val"]}
    test_paths = {p for p, _ in splits["test"]}

    # Absolute disjointness
    assert len(train_paths & val_paths) == 0, "Data leakage detected between train and val!"
    assert len(train_paths & test_paths) == 0, "Data leakage detected between train and test!"
    assert len(val_paths & test_paths) == 0, "Data leakage detected between val and test!"

    # Total conservation
    assert len(train_paths) + len(val_paths) + len(test_paths) == len(synthetic_files)


def test_split_stratification_balance():
    """Verify that each genre is represented proportionally in every split."""
    synthetic_files: List[Tuple[str, str]] = []
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    for g in genres:
        for i in range(20):
            synthetic_files.append((f"/fake/path/{g}.{i:05d}.wav", g))

    splits = split_dataset(synthetic_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    for split_name in ["train", "val", "test"]:
        labels = [l for _, l in splits[split_name]]
        for g in genres:
            count = labels.count(g)
            assert count > 0, f"Genre {g} is missing from {split_name} split!"


def test_clean_split_no_leakage_and_duplicate_isolation():
    """
    CRITICAL: Verify that in the Phase 8 clean split:
    1. Sets are mutually exclusive.
    2. Zero duplicate groups cross partition boundaries.
    3. metal.00058.wav is excluded while rock.00016.wav is retained.
    """
    import json
    from pathlib import Path
    from ml.data_splitter import load_clean_split
    from ml.config import path_config

    splits = load_clean_split()
    train_names = {Path(p).name for p, _ in splits["train"]}
    val_names = {Path(p).name for p, _ in splits["val"]}
    test_names = {Path(p).name for p, _ in splits["test"]}

    # Mutual exclusivity
    assert len(train_names & val_names) == 0, "Train & Val overlap in clean split!"
    assert len(train_names & test_names) == 0, "Train & Test overlap in clean split!"
    assert len(val_names & test_names) == 0, "Val & Test overlap in clean split!"

    # Total clean dataset count
    total_tracks = len(train_names) + len(val_names) + len(test_names)
    assert total_tracks == 998, f"Expected 998 decontaminated tracks, got {total_tracks}"

    # Verify contradictory label resolution
    assert "metal.00058.wav" not in train_names
    assert "metal.00058.wav" not in val_names
    assert "metal.00058.wav" not in test_names
    assert "rock.00016.wav" in (train_names | val_names | test_names)

    # Load duplicate audit to check all duplicate groups
    audit_file = Path(path_config.results_dir) / "decontamination_audit.json"
    if audit_file.exists():
        with open(audit_file, "r") as f:
            audit = json.load(f)

        # Check exact duplicate groups
        for g in audit.get("exact_duplicate_groups", []):
            files = [fn for fn in g["files"] if fn != "metal.00058.wav"]
            in_train = any(fn in train_names for fn in files)
            in_val = any(fn in val_names for fn in files)
            in_test = any(fn in test_names for fn in files)
            partition_count = int(in_train) + int(in_val) + int(in_test)
            assert partition_count == 1, f"Exact duplicate group {files} leaked across {partition_count} partitions!"

        # Check near duplicate pairs
        for nd in audit.get("near_duplicate_pairs", []):
            f1, f2 = nd["file1"], nd["file2"]
            if f1 != "metal.00058.wav" and f2 != "metal.00058.wav":
                p1 = "train" if f1 in train_names else ("val" if f1 in val_names else ("test" if f1 in test_names else None))
                p2 = "train" if f2 in train_names else ("val" if f2 in val_names else ("test" if f2 in test_names else None))
                assert p1 == p2, f"Near-duplicate pair ({f1}, {f2}) leaked across partitions: {p1} != {p2}"
