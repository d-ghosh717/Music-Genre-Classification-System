"""
Tests for GTZAN dataset validation and splitting.
"""
import os
import shutil
import tempfile
import unittest
import numpy as np
import soundfile as sf

from ml.config import dataset_config
from ml.dataset_validator import validate_dataset, collect_audio_files
from ml.data_splitter import split_dataset


class TestDatasetValidation(unittest.TestCase):
    """Test suite for dataset validation utilities."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataset_missing_dir(self):
        """Validation should flag nonexistent directories."""
        report = validate_dataset(dataset_dir="/nonexistent/path/to/dataset", verbose=False)
        self.assertFalse(report.exists)
        self.assertFalse(report.is_valid)
        self.assertIn("not found", report.summary)

    def test_dataset_missing_genres(self):
        """Validation should detect when genre directories are missing."""
        # Create only 2 genres instead of 10
        os.makedirs(os.path.join(self.temp_dir, "blues"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "classical"), exist_ok=True)

        report = validate_dataset(dataset_dir=self.temp_dir, verbose=False)
        self.assertTrue(report.exists)
        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.genres_found), 2)
        self.assertEqual(len(report.genres_missing), 8)

    def test_dataset_corrupted_file_detection(self):
        """Validation should detect corrupted audio files."""
        genre_dir = os.path.join(self.temp_dir, "blues")
        os.makedirs(genre_dir, exist_ok=True)
        # Create empty / invalid wav file
        corrupt_file = os.path.join(genre_dir, "blues.00000.wav")
        with open(corrupt_file, "wb") as f:
            f.write(b"not a valid wav")

        for g in dataset_config.genres:
            os.makedirs(os.path.join(self.temp_dir, g), exist_ok=True)

        report = validate_dataset(dataset_dir=self.temp_dir, check_audio=True, verbose=False)
        self.assertFalse(report.is_valid)
        self.assertGreater(report.total_invalid, 0)
        self.assertIn("blues", report.genre_reports)
        self.assertEqual(len(report.genre_reports["blues"].invalid_files), 1)

    def test_stratified_track_splitting_no_leakage(self):
        """Splitting must be track-level with zero overlap between splits."""
        splits = split_dataset()
        train_paths = set(f[0] for f in splits["train"])
        val_paths = set(f[0] for f in splits["val"])
        test_paths = set(f[0] for f in splits["test"])

        # No track may appear in multiple splits
        self.assertEqual(len(train_paths.intersection(val_paths)), 0)
        self.assertEqual(len(train_paths.intersection(test_paths)), 0)
        self.assertEqual(len(val_paths.intersection(test_paths)), 0)

        # Total count should be 999 (excluding the corrupt jazz.00054.wav)
        total_tracks = len(train_paths) + len(val_paths) + len(test_paths)
        self.assertEqual(total_tracks, 999)


if __name__ == "__main__":
    unittest.main()

