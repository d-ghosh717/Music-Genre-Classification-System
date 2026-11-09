"""
GTZAN Decontaminated & Grouped Stratified Dataset Splitter.

Produces a scientifically defensible train/val/test split:
1. Excludes corrupted files ('jazz.00054.wav').
2. Resolves contradictory label ambiguity by excluding 'metal.00058.wav' (Queen's 'Tie Your Mother Down', Hard Rock mislabeled as Metal), retaining 'rock.00016.wav'.
3. Groups all exact and near-duplicate recordings into atomic connected components.
4. Performs stratified splitting on group level to preserve ~70/15/15 proportions across all 10 genres.
5. Verifies zero cross-partition leakage for all duplicate groups.
6. Serializes clean split to dataset/clean_splits.json.
"""

from collections import defaultdict, deque
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import dataset_config, path_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("create_clean_split")


def build_duplicate_connected_components(
    audit_json_path: Path,
    dataset_dir: Path,
    exclude_files: Optional[List[str]] = None,
) -> Tuple[List[List[Path]], List[str], Dict[str, str]]:
    """
    Build connected components of duplicate and near-duplicate tracks.
    Returns (components, component_genres, file_to_genre_map).
    """
    if exclude_files is None:
        exclude_files = ["jazz.00054.wav", "metal.00058.wav"]

    exclude_set = set(exclude_files)

    if not audit_json_path.exists():
        from ml.audit_duplicates import run_comprehensive_audit
        run_comprehensive_audit(audit_json_path)

    with open(audit_json_path, "r") as f:
        audit = json.load(f)

    all_wavs = sorted(list(dataset_dir.glob("*/*.wav")))
    valid_files = [f for f in all_wavs if f.name not in exclude_set]

    adj = defaultdict(set)
    file_to_genre = {}
    path_map = {}

    for f in valid_files:
        adj[f.name] = set()
        file_to_genre[f.name] = f.parent.name
        path_map[f.name] = f

    # Connect exact duplicates
    for g in audit.get("exact_duplicate_groups", []):
        files = [fname for fname in g["files"] if fname not in exclude_set]
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                adj[files[i]].add(files[j])
                adj[files[j]].add(files[i])

    # Connect near duplicates
    for nd in audit.get("near_duplicate_pairs", []):
        f1, f2 = nd["file1"], nd["file2"]
        if f1 not in exclude_set and f2 not in exclude_set:
            adj[f1].add(f2)
            adj[f2].add(f1)

    visited = set()
    components: List[List[Path]] = []
    component_genres: List[str] = []

    for f in valid_files:
        fname = f.name
        if fname not in visited:
            comp_names = []
            queue = deque([fname])
            visited.add(fname)
            while queue:
                curr = queue.popleft()
                comp_names.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            comp_paths = [path_map[fn] for fn in comp_names]
            components.append(comp_paths)
            component_genres.append(file_to_genre[comp_names[0]])

    return components, component_genres, file_to_genre


def generate_clean_grouped_split(
    audit_json_path: Path = None,
    dataset_dir: Path = None,
    random_seed: int = 42,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Generate grouped, stratified train/validation/test split.
    All duplicate and near-duplicate recordings are strictly confined to the same split.
    """
    audit_json_path = audit_json_path or (Path(path_config.results_dir) / "decontamination_audit.json")
    dataset_dir = dataset_dir or Path(dataset_config.dataset_dir)

    components, comp_genres, file_to_genre = build_duplicate_connected_components(
        audit_json_path, dataset_dir, exclude_files=["jazz.00054.wav", "metal.00058.wav"]
    )

    group_indices = np.arange(len(components))
    group_genres_arr = np.array(comp_genres)

    # 1. Stratified split for test set
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
    train_val_idx, test_idx = next(sss_test.split(group_indices, group_genres_arr))

    # 2. Stratified split for validation set
    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_ratio, random_state=random_seed)
    train_sub_idx, val_sub_idx = next(sss_val.split(train_val_idx, group_genres_arr[train_val_idx]))

    train_group_idx = train_val_idx[train_sub_idx]
    val_group_idx = train_val_idx[val_sub_idx]

    # Convert groups back to (path, genre) lists
    splits: Dict[str, List[Tuple[str, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for idx in train_group_idx:
        for p in components[idx]:
            splits["train"].append((str(p.resolve()), p.parent.name))

    for idx in val_group_idx:
        for p in components[idx]:
            splits["val"].append((str(p.resolve()), p.parent.name))

    for idx in test_idx:
        for p in components[idx]:
            splits["test"].append((str(p.resolve()), p.parent.name))

    # Verification: check for any leakage between partitions
    train_names = {Path(p).name for p, _ in splits["train"]}
    val_names = {Path(p).name for p, _ in splits["val"]}
    test_names = {Path(p).name for p, _ in splits["test"]}

    assert len(train_names & val_names) == 0, "Train and Val overlap!"
    assert len(train_names & test_names) == 0, "Train and Test overlap!"
    assert len(val_names & test_names) == 0, "Val and Test overlap!"

    # Verify duplicate groups are strictly atomic
    for comp in components:
        comp_fnames = {p.name for p in comp}
        in_train = len(comp_fnames & train_names) > 0
        in_val = len(comp_fnames & val_names) > 0
        in_test = len(comp_fnames & test_names) > 0
        assert (int(in_train) + int(in_val) + int(in_test)) == 1, (
            f"Leakage detected! Component {comp_fnames} crosses split partitions!"
        )

    logger.info("Verification passed: Zero duplicate or near-duplicate tracks cross partitions.")
    return splits


def save_clean_split_manifest(
    splits: Dict[str, List[Tuple[str, str]]],
    output_path: Path = None,
    random_seed: int = 42,
) -> None:
    """Save clean split manifest to JSON file."""
    output_path = output_path or (PROJECT_ROOT / "dataset" / "clean_splits.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "description": "GTZAN Decontaminated Grouped Stratified Split (Phase 8 Final Benchmark)",
        "random_seed": random_seed,
        "excluded_files": [
            {
                "filename": "jazz.00054.wav",
                "reason": "Corrupted unreadable audio file (0 bytes)."
            },
            {
                "filename": "metal.00058.wav",
                "reason": (
                    "Contradictory duplicate of rock.00016.wav (Queen - 'Tie Your Mother Down'). "
                    "Excluded from Metal category due to erroneous historical genre label."
                )
            }
        ],
        "retained_counterpart": {
            "filename": "rock.00016.wav",
            "genre": "rock",
            "reason": "Legitimate historical genre attribution for Queen."
        },
        "counts": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": sum(len(v) for v in splits.values())
        },
        "genre_counts": {},
        "splits": {
            split_name: [
                {"file": Path(p).name, "genre": g, "path": p}
                for p, g in tracks
            ]
            for split_name, tracks in splits.items()
        }
    }

    for s_name, tracks in splits.items():
        gc = defaultdict(int)
        for _, g in tracks:
            gc[g] += 1
        manifest["genre_counts"][s_name] = dict(sorted(gc.items()))

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Clean split manifest written to: {output_path}")


if __name__ == "__main__":
    splits = generate_clean_grouped_split()
    manifest_path = PROJECT_ROOT / "dataset" / "clean_splits.json"
    save_clean_split_manifest(splits, manifest_path)

    print("\n" + "=" * 65)
    print("GTZAN CLEAN GROUPED STRATIFIED SPLIT MANIFEST")
    print("=" * 65)
    for s_name, tracks in splits.items():
        gc = defaultdict(int)
        for _, g in tracks:
            gc[g] += 1
        print(f"\n{s_name.upper()} SET ({len(tracks)} tracks):")
        for g in sorted(gc.keys()):
            print(f"  {g:12s}: {gc[g]:3d} tracks")
    print("\n" + "=" * 65)

