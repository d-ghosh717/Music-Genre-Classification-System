"""
Stratified dataset splitting at the track level.
Ensures no data leakage: all segments of a single track stay in the same split.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.model_selection import train_test_split

from ml.config import dataset_config
from ml.dataset_validator import collect_audio_files

logger = logging.getLogger(__name__)


def split_dataset(
    files: List[Tuple[str, str]] = None,
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    random_seed: int = None,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Perform a stratified train/validation/test split at the track level.
    
    CRITICAL: This splits original tracks, not segments. All segments derived
    from a single track will inherit its split assignment.
    
    Args:
        files: List of (file_path, genre_label) tuples. If None, collects from dataset.
        train_ratio: Proportion for training. Defaults to dataset_config.train_ratio.
        val_ratio: Proportion for validation. Defaults to dataset_config.val_ratio.
        test_ratio: Proportion for testing. Defaults to dataset_config.test_ratio.
        random_seed: Random seed for reproducibility. Defaults to dataset_config.random_seed.
    
    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing a list of
        (file_path, genre_label) tuples.
    
    Raises:
        ValueError: If ratios don't sum to approximately 1.0.
    """
    train_ratio = train_ratio or dataset_config.train_ratio
    val_ratio = val_ratio or dataset_config.val_ratio
    test_ratio = test_ratio or dataset_config.test_ratio
    random_seed = random_seed if random_seed is not None else dataset_config.random_seed
    
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split ratios must sum to ~1.0, got {total:.3f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )
    
    if files is None:
        files = collect_audio_files()
    
    if len(files) == 0:
        raise ValueError("No audio files found. Cannot split an empty dataset.")
    
    file_paths = [f[0] for f in files]
    labels = [f[1] for f in files]
    
    # First split: separate test set
    # test_ratio relative to total
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths,
        labels,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labels,
    )
    
    # Second split: separate validation from training
    # val_ratio relative to (train + val)
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=relative_val_ratio,
        random_state=random_seed,
        stratify=train_val_labels,
    )
    
    splits = {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }
    
    # Log split statistics
    logger.info(f"Dataset split (seed={random_seed}):")
    for split_name, split_data in splits.items():
        genre_counts = {}
        for _, label in split_data:
            genre_counts[label] = genre_counts.get(label, 0) + 1
        logger.info(
            f"  {split_name:5s}: {len(split_data):4d} tracks — {dict(sorted(genre_counts.items()))}"
        )
    
    return splits


def print_split_summary(splits: Dict[str, List[Tuple[str, str]]]) -> None:
    """Print a readable summary of the dataset split."""
    print("\nDataset Split Summary")
    print("=" * 50)
    
    for split_name in ["train", "val", "test"]:
        data = splits[split_name]
        genre_counts = {}
        for _, label in data:
            genre_counts[label] = genre_counts.get(label, 0) + 1
        
        print(f"\n{split_name.upper()} SET: {len(data)} tracks")
        for genre in sorted(genre_counts.keys()):
            print(f"  {genre:12s}: {genre_counts[genre]:3d}")


def load_clean_split(manifest_path: Optional[Path] = None) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load the verified decontaminated grouped dataset split.
    If the manifest does not exist, it will be generated automatically.
    """
    manifest_path = manifest_path or (PROJECT_ROOT / "dataset" / "clean_splits.json")
    if not manifest_path.exists():
        from ml.create_clean_split import generate_clean_grouped_split, save_clean_split_manifest
        splits = generate_clean_grouped_split()
        save_clean_split_manifest(splits, manifest_path)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    splits: Dict[str, List[Tuple[str, str]]] = {}
    for s_name, items in manifest["splits"].items():
        splits[s_name] = [(item["path"], item["genre"]) for item in items]

    return splits
