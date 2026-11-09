"""
GTZAN dataset validation utility.
Checks directory structure, file counts, audio readability, and reports issues.
"""
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import dataset_config
from ml.preprocessing.audio_loader import validate_audio_file

logger = logging.getLogger(__name__)


@dataclass
class GenreReport:
    """Validation report for a single genre directory."""
    genre: str
    file_count: int = 0
    valid_files: int = 0
    invalid_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class DatasetReport:
    """Complete dataset validation report."""
    dataset_dir: str = ""
    exists: bool = False
    genres_found: List[str] = field(default_factory=list)
    genres_missing: List[str] = field(default_factory=list)
    genre_reports: Dict[str, GenreReport] = field(default_factory=dict)
    total_files: int = 0
    total_valid: int = 0
    total_invalid: int = 0
    is_valid: bool = False
    summary: str = ""
    
    def to_dict(self) -> dict:
        """Convert report to a JSON-serializable dictionary."""
        return {
            "dataset_dir": self.dataset_dir,
            "exists": self.exists,
            "genres_found": self.genres_found,
            "genres_missing": self.genres_missing,
            "genre_counts": {
                g: r.file_count for g, r in self.genre_reports.items()
            },
            "total_files": self.total_files,
            "total_valid": self.total_valid,
            "total_invalid": self.total_invalid,
            "invalid_files": {
                g: r.invalid_files for g, r in self.genre_reports.items()
                if r.invalid_files
            },
            "is_valid": self.is_valid,
            "summary": self.summary,
        }


def validate_dataset(
    dataset_dir: Optional[str] = None,
    check_audio: bool = True,
    verbose: bool = True,
) -> DatasetReport:
    """
    Validate the GTZAN dataset directory structure and file integrity.
    
    Args:
        dataset_dir: Path to the genres_original directory.
            Defaults to dataset_config.dataset_dir.
        check_audio: Whether to attempt loading each audio file to verify readability.
            Slower but catches corrupted files.
        verbose: Whether to print progress to stdout.
    
    Returns:
        DatasetReport with detailed validation results.
    """
    dataset_dir = dataset_dir or dataset_config.dataset_dir
    report = DatasetReport(dataset_dir=dataset_dir)
    
    dataset_path = Path(dataset_dir)
    
    # Check root directory
    if not dataset_path.exists():
        report.summary = (
            f"Dataset directory not found: {dataset_dir}\n"
            f"Please download the GTZAN dataset from Kaggle and place it at:\n"
            f"  {dataset_dir}\n"
            f"Expected structure:\n"
            f"  {dataset_dir}/blues/\n"
            f"  {dataset_dir}/classical/\n"
            f"  ... (10 genre folders)"
        )
        if verbose:
            print(report.summary)
        return report
    
    report.exists = True
    
    if not dataset_path.is_dir():
        report.summary = f"Dataset path exists but is not a directory: {dataset_dir}"
        if verbose:
            print(report.summary)
        return report
    
    # Identify genre directories
    expected_genres = set(dataset_config.genres)
    actual_dirs = {
        d.name for d in dataset_path.iterdir() if d.is_dir()
    }
    
    report.genres_found = sorted(actual_dirs & expected_genres)
    report.genres_missing = sorted(expected_genres - actual_dirs)
    
    # Warn about unexpected directories
    unexpected = actual_dirs - expected_genres
    if unexpected and verbose:
        print(f"  [INFO] Unexpected directories found: {sorted(unexpected)}")
    
    if report.genres_missing and verbose:
        print(f"  [WARNING] Missing genre directories: {report.genres_missing}")
    
    # Validate each genre directory
    extensions = set(dataset_config.audio_extensions)
    
    for genre in sorted(report.genres_found):
        genre_path = dataset_path / genre
        gr = GenreReport(genre=genre)
        
        audio_files = [
            f for f in genre_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        gr.file_count = len(audio_files)
        
        if check_audio:
            for af in audio_files:
                is_valid, msg = validate_audio_file(str(af))
                if is_valid:
                    gr.valid_files += 1
                else:
                    gr.invalid_files.append(str(af))
                    gr.errors.append(msg)
        else:
            gr.valid_files = gr.file_count
        
        report.genre_reports[genre] = gr
        report.total_files += gr.file_count
        report.total_valid += gr.valid_files
        report.total_invalid += len(gr.invalid_files)
        
        if verbose:
            status = "✓" if gr.valid_files == gr.file_count else "!"
            print(
                f"  [{status}] {genre:12s}: {gr.file_count:3d} files, "
                f"{gr.valid_files:3d} valid"
                + (f", {len(gr.invalid_files)} invalid" if gr.invalid_files else "")
            )
    
    # Determine overall validity
    # In standard GTZAN distributions, jazz.00054.wav is a well-known unreadable/corrupted file.
    # The dataset is valid for training if all 10 genres are present and >= 990 files are readable.
    report.is_valid = (
        report.exists
        and len(report.genres_missing) == 0
        and report.total_valid >= 990
    )
    
    # Generate summary
    lines = [
        f"Dataset directory: {dataset_dir}",
        f"Genres found: {len(report.genres_found)}/{len(expected_genres)}",
        f"Total audio files: {report.total_files}",
        f"Valid files: {report.total_valid}",
        f"Invalid files: {report.total_invalid}",
    ]
    if report.genres_missing:
        lines.append(f"Missing genres: {', '.join(report.genres_missing)}")
    
    if report.is_valid:
        if report.total_invalid > 0:
            lines.append(f"Status: VALID ✓ ({report.total_invalid} corrupted track(s) detected and safely excluded)")
        else:
            lines.append("Status: VALID ✓")
    else:
        lines.append("Status: ISSUES DETECTED — see details above")
    
    report.summary = "\n".join(lines)
    
    if verbose:
        print("\n" + report.summary)
    
    return report


def collect_audio_files(
    dataset_dir: Optional[str] = None,
    filter_invalid: bool = True,
) -> List[Tuple[str, str]]:
    """
    Collect all audio file paths and their genre labels.
    
    Args:
        dataset_dir: Path to the genres_original directory.
        filter_invalid: If True, validates readability and excludes corrupted files.
    
    Returns:
        List of (file_path, genre_label) tuples.
    """
    dataset_dir = dataset_dir or dataset_config.dataset_dir
    dataset_path = Path(dataset_dir)
    extensions = set(dataset_config.audio_extensions)
    
    files = []
    for genre in sorted(dataset_config.genres):
        genre_path = dataset_path / genre
        if not genre_path.exists():
            continue
        for f in sorted(genre_path.iterdir()):
            if f.is_file() and f.suffix.lower() in extensions:
                if filter_invalid:
                    is_valid, msg = validate_audio_file(str(f))
                    if not is_valid:
                        logger.warning(f"Excluding invalid file '{f}': {msg}")
                        continue
                files.append((str(f), genre))
    
    return files


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("GTZAN Dataset Validation")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Validate GTZAN dataset structure and integrity.")
    parser.add_argument("dataset_dir", nargs="?", default=None, help="Path to genres_original directory (optional)")
    parser.add_argument("--check-audio", action="store_true", help="Perform deep readability check by loading audio")
    
    args = parser.parse_args()
    
    report = validate_dataset(
        dataset_dir=args.dataset_dir,
        check_audio=args.check_audio,
    )
    
    sys.exit(0 if report.is_valid else 1)
