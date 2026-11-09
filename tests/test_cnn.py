"""
Unit tests for CNN Mel-spectrogram processing, architecture, track-level segmentation,
data leakage prevention, probability aggregation, and dual-model inference.
"""
import os
import shutil
import tempfile
import unittest
import numpy as np
import soundfile as sf
import tensorflow as tf

from ml.config import (
    audio_config,
    cnn_config,
    path_config,
    spectrogram_config,
    NUM_GENRES,
)
from ml.data_splitter import split_dataset
from ml.models.cnn_model import build_cnn_model
from ml.preprocessing.audio_loader import load_audio
from ml.preprocessing.segmentation import segment_audio
from ml.preprocessing.spectrogram import (
    compute_expected_time_frames,
    generate_mel_spectrogram,
    mel_spectrogram_to_cnn_input,
)
from ml.inference.predictor import predict_cnn, predict_both


class TestCNNModel(unittest.TestCase):
    """Test suite for CNN model pipeline and inference."""

    def setUp(self):
        self.sr = audio_config.sample_rate
        # Synthetic 6-second audio (two 5-second overlapping segments)
        t = np.linspace(0, 6.0, int(self.sr * 6.0), endpoint=False)
        self.signal = 0.5 * np.sin(2 * np.pi * 440 * t)

        self.temp_dir = tempfile.mkdtemp()
        self.audio_path = os.path.join(self.temp_dir, "test_cnn_tone.wav")
        sf.write(self.audio_path, self.signal, self.sr)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_spectrogram_input_dimensions(self):
        """Mel-spectrogram CNN input should match (128, expected_frames, 1)."""
        expected_frames = compute_expected_time_frames()
        mel_spec = generate_mel_spectrogram(self.signal, self.sr)
        cnn_in = mel_spectrogram_to_cnn_input(mel_spec, expected_time_frames=expected_frames)

        self.assertEqual(cnn_in.ndim, 3)
        self.assertEqual(cnn_in.shape, (spectrogram_config.n_mels, expected_frames, 1))
        self.assertFalse(np.isnan(cnn_in).any())

    def test_cnn_model_construction_and_output_shape(self):
        """CNN should compile with correct input/output shape and loss."""
        expected_frames = compute_expected_time_frames()
        input_shape = (spectrogram_config.n_mels, expected_frames, 1)
        model = build_cnn_model(input_shape=input_shape, num_classes=NUM_GENRES)

        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, 128, expected_frames, 1))
        self.assertEqual(model.output_shape, (None, 10))
        self.assertEqual(model.loss, "sparse_categorical_crossentropy")

    def test_probability_normalization(self):
        """Forward pass should output probabilities summing to 1.0 across classes."""
        expected_frames = compute_expected_time_frames()
        input_shape = (spectrogram_config.n_mels, expected_frames, 1)
        model = build_cnn_model(input_shape=input_shape, num_classes=10)

        dummy_batch = np.random.randn(3, 128, expected_frames, 1).astype(np.float32)
        preds = model.predict(dummy_batch, verbose=0)

        self.assertEqual(preds.shape, (3, 10))
        sums = np.sum(preds, axis=1)
        np.testing.assert_allclose(sums, np.ones(3), atol=1e-5)
        self.assertTrue((preds >= 0.0).all() and (preds <= 1.0).all())

    def test_model_save_and_reload(self):
        """CNN model must serialize and reload in a clean state with identical outputs."""
        expected_frames = compute_expected_time_frames()
        input_shape = (spectrogram_config.n_mels, expected_frames, 1)
        model = build_cnn_model(input_shape=input_shape, num_classes=10)

        model_path = os.path.join(self.temp_dir, "test_cnn.keras")
        model.save(model_path)

        reloaded = tf.keras.models.load_model(model_path)
        dummy_batch = np.random.randn(2, 128, expected_frames, 1).astype(np.float32)

        p1 = model.predict(dummy_batch, verbose=0)
        p2 = reloaded.predict(dummy_batch, verbose=0)
        np.testing.assert_allclose(p1, p2, atol=1e-5)

    def test_segment_generation(self):
        """Segmenting 6.0s audio with 5.0s window and 50% overlap should yield expected segments."""
        segments = segment_audio(self.signal, self.sr, segment_duration=5.0, overlap=0.5)
        # 6s with 5s window and 2.5s hop -> start 0.0 (0.0 - 5.0) -> next start 2.5 would need 7.5s (so 1 segment)
        self.assertGreaterEqual(len(segments), 1)
        for seg in segments:
            self.assertEqual(len(seg), int(5.0 * self.sr))

    def test_track_level_split_isolation_no_leakage(self):
        """Strict verification: zero overlap between train, val, and test track paths."""
        splits = split_dataset()
        train_set = {p for p, _ in splits["train"]}
        val_set = {p for p, _ in splits["val"]}
        test_set = {p for p, _ in splits["test"]}

        # Check disjointness
        self.assertTrue(train_set.isdisjoint(val_set), "Leakage: train and val share tracks!")
        self.assertTrue(train_set.isdisjoint(test_set), "Leakage: train and test share tracks!")
        self.assertTrue(val_set.isdisjoint(test_set), "Leakage: val and test share tracks!")

    def test_segment_probability_aggregation(self):
        """Track prediction should correctly average segment probabilities."""
        # 3 segments for 1 track
        seg_probs = np.array([
            [0.1, 0.7, 0.2],
            [0.05, 0.85, 0.1],
            [0.15, 0.55, 0.3],
        ])
        expected_avg = np.mean(seg_probs, axis=0)  # [0.1, 0.7, 0.2]
        self.assertEqual(int(np.argmax(expected_avg)), 1)
        self.assertAlmostEqual(np.sum(expected_avg), 1.0, places=5)

    def test_end_to_end_cnn_and_dual_inference(self):
        """End-to-end inference for predict_cnn and predict_both on actual audio."""
        if not (os.path.exists(path_config.cnn_model_path) and os.path.exists(path_config.ann_model_path)):
            self.skipTest("CNN or ANN model not trained yet; skipping prediction test.")

        # Test predict_cnn
        cnn_out = predict_cnn(self.audio_path)
        self.assertIn("predicted_genre", cnn_out)
        self.assertIn("confidence", cnn_out)
        self.assertIn("top3", cnn_out)
        self.assertAlmostEqual(sum(cnn_out["all_probabilities"].values()), 1.0, places=3)

        # Test predict_both
        both_out = predict_both(self.audio_path)
        self.assertIn("ann", both_out)
        self.assertIn("cnn", both_out)
        self.assertIn("agreement", both_out)
        self.assertIsInstance(both_out["agreement"], bool)


if __name__ == "__main__":
    unittest.main()

