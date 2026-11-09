"""
Unit tests for ANN model architecture, scaler persistence, inference, and feature consistency.
"""
import os
import shutil
import tempfile
import unittest
import numpy as np
import joblib
import soundfile as sf
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from ml.config import audio_config, ann_config, path_config, NUM_GENRES
from ml.models.ann_model import build_ann_model
from ml.preprocessing.audio_loader import load_audio
from ml.preprocessing.feature_extraction import extract_features
from ml.inference.predictor import predict_ann, GenrePredictor


class TestANNModel(unittest.TestCase):
    """Test suite for ANN model architecture and lifecycle."""

    def setUp(self):
        self.sr = audio_config.sample_rate
        # Create a synthetic 3-second audio wave (440Hz + harmonics)
        t = np.linspace(0, 3.0, int(self.sr * 3.0), endpoint=False)
        self.signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)

        self.temp_dir = tempfile.mkdtemp()
        self.audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        sf.write(self.audio_path, self.signal, self.sr)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ann_model_construction(self):
        """ANN model should build with correct input/output layers and compilation."""
        model = build_ann_model(input_dim=61, num_classes=10)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, 61))
        self.assertEqual(model.output_shape, (None, 10))
        self.assertEqual(model.loss, "sparse_categorical_crossentropy")

    def test_ann_forward_pass_and_probability_normalization(self):
        """Forward pass should produce probabilities summing to 1.0."""
        model = build_ann_model(input_dim=61, num_classes=10)
        dummy_input = np.random.randn(4, 61).astype(np.float32)
        preds = model.predict(dummy_input, verbose=0)

        self.assertEqual(preds.shape, (4, 10))
        # Each row should sum to 1.0 (Softmax)
        row_sums = np.sum(preds, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5)
        # All probabilities must be in [0, 1]
        self.assertTrue((preds >= 0.0).all() and (preds <= 1.0).all())

    def test_scaler_persistence_and_reload(self):
        """StandardScaler must persist and transform identically after reload."""
        train_features = np.random.randn(50, 61).astype(np.float32)
        scaler = StandardScaler()
        transformed_original = scaler.fit_transform(train_features)

        scaler_path = os.path.join(self.temp_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)

        reloaded_scaler = joblib.load(scaler_path)
        transformed_reloaded = reloaded_scaler.transform(train_features)

        np.testing.assert_allclose(transformed_original, transformed_reloaded, atol=1e-6)

    def test_model_persistence_and_reload(self):
        """Trained ANN model must save and reload with identical predictions."""
        model = build_ann_model(input_dim=61, num_classes=10)
        model_path = os.path.join(self.temp_dir, "ann_test.keras")
        model.save(model_path)

        reloaded_model = tf.keras.models.load_model(model_path)
        dummy_input = np.random.randn(2, 61).astype(np.float32)

        pred_orig = model.predict(dummy_input, verbose=0)
        pred_reload = reloaded_model.predict(dummy_input, verbose=0)

        np.testing.assert_allclose(pred_orig, pred_reload, atol=1e-5)

    def test_feature_extraction_consistency(self):
        """Feature extraction must yield identical vectors for the same audio."""
        y1, sr1 = load_audio(self.audio_path)
        y2, sr2 = load_audio(self.audio_path)

        feat1 = extract_features(y1, sr1)
        feat2 = extract_features(y2, sr2)

        self.assertEqual(len(feat1), len(feat2))
        self.assertIn(len(feat1), (61, 75, 127))
        np.testing.assert_allclose(feat1, feat2, atol=1e-7)

    def test_end_to_end_ann_prediction_pipeline(self):
        """End-to-end prediction should produce valid probabilities and top genres if model exists."""
        # Check if saved model exists
        if not (os.path.exists(path_config.ann_model_path) and os.path.exists(path_config.ann_scaler_path)):
            self.skipTest("Trained ANN model not yet saved; skipping prediction test.")

        result = predict_ann(self.audio_path)
        self.assertIn("predicted_genre", result)
        self.assertIn("confidence", result)
        self.assertIn("top3", result)
        self.assertEqual(len(result["top3"]), 3)
        self.assertAlmostEqual(sum(result["all_probabilities"].values()), 1.0, places=3)
        self.assertGreater(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()

