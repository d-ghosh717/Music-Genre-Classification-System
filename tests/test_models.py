"""
Tests for ANN and CNN architecture construction and forward inference pass.
"""
import numpy as np
import pytest

from ml.config import NUM_GENRES
from ml.models.ann_model import build_ann_model
from ml.models.cnn_model import build_cnn_model


def test_ann_model_architecture():
    """Verify ANN input/output layer dimensions and forward inference output."""
    input_dim = 61
    batch_size = 4
    model = build_ann_model(input_dim=input_dim, num_classes=NUM_GENRES)

    # Check input shape
    assert model.input_shape == (None, input_dim)
    assert model.output_shape == (None, NUM_GENRES)

    # Forward pass with random inputs
    dummy_input = np.random.randn(batch_size, input_dim).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)

    assert output.shape == (batch_size, NUM_GENRES)
    # Check softmax probability properties
    sums = np.sum(output, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)
    assert (output >= 0.0).all() and (output <= 1.0).all()


def test_cnn_model_architecture():
    """Verify CNN input/output layer dimensions and forward inference output."""
    input_shape = (128, 130, 1)
    batch_size = 2
    model = build_cnn_model(input_shape=input_shape, num_classes=NUM_GENRES)

    # Check shapes
    assert model.input_shape == (None, 128, 130, 1)
    assert model.output_shape == (None, NUM_GENRES)

    # Forward pass
    dummy_input = np.random.randn(batch_size, 128, 130, 1).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)

    assert output.shape == (batch_size, NUM_GENRES)
    sums = np.sum(output, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)
    assert (output >= 0.0).all() and (output <= 1.0).all()
