"""
CNN architecture for audio genre classification.
Learns spatio-temporal hierarchical features directly from log Mel-spectrogram patches.
"""
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ml.config import cnn_config, spectrogram_config, NUM_GENRES


from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ml.config import cnn_config, spectrogram_config, NUM_GENRES


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (128, 216, 1),
    num_classes: int = NUM_GENRES,
    conv_filters: Optional[List[int]] = None,
    kernel_size: Optional[Tuple[int, int]] = None,
    pool_size: Optional[Tuple[int, int]] = None,
    dense_units: Optional[int] = None,
    dropout_rate: Optional[float] = None,
    use_batch_norm: Optional[bool] = None,
    learning_rate: Optional[float] = None,
) -> keras.Model:
    """
    Construct the Convolutional Neural Network (CNN) architecture.

    Default Architecture:
        Input(input_shape) e.g., (128, 216, 1)
        Conv2D(32, (3,3), padding="same", activation="relu")
        BatchNormalization()
        MaxPooling2D((2,2))
        Conv2D(64, (3,3), padding="same", activation="relu")
        BatchNormalization()
        MaxPooling2D((2,2))
        Conv2D(128, (3,3), padding="same", activation="relu")
        BatchNormalization()
        MaxPooling2D((2,2))
        GlobalAveragePooling2D()
        Dense(128, activation="relu")
        Dropout(0.4)
        Dense(num_classes, activation="softmax")

    Args:
        input_shape: Shape of the input mel-spectrogram (n_mels, time_steps, channels).
        num_classes: Number of genre classes (default 10).
        conv_filters: List of filters for each conv layer (default: [32, 64, 128]).
        kernel_size: 2D tuple for conv filter window (default: (3, 3)).
        pool_size: 2D tuple for max pooling window (default: (2, 2)).
        dense_units: Units in penultimate dense layer (default: 128).
        dropout_rate: Dropout fraction before softmax (default: 0.4).
        use_batch_norm: Whether to apply batch normalization (default: True).
        learning_rate: Initial Adam learning rate.

    Returns:
        Compiled Keras Sequential model.
    """
    lr = learning_rate or cnn_config.learning_rate
    filters = conv_filters if conv_filters is not None else cnn_config.conv_filters
    k_size = kernel_size or cnn_config.kernel_size
    p_size = pool_size or cnn_config.pool_size
    d_units = dense_units or cnn_config.dense_units
    d_rate = dropout_rate if dropout_rate is not None else cnn_config.dropout_rate
    bn = use_batch_norm if use_batch_norm is not None else cnn_config.use_batch_norm

    model_layers = [layers.Input(shape=input_shape, name="spectrogram_input")]

    for i, f in enumerate(filters):
        model_layers.append(
            layers.Conv2D(f, kernel_size=k_size, padding="same", activation="relu", name=f"conv2d_{i+1}")
        )
        if bn:
            model_layers.append(layers.BatchNormalization(name=f"bn_{i+1}"))
        model_layers.append(layers.MaxPooling2D(pool_size=p_size, name=f"pool_{i+1}"))

    model_layers.append(layers.GlobalAveragePooling2D(name="gap"))
    model_layers.append(layers.Dense(d_units, activation="relu", name="dense_1"))
    if d_rate > 0:
        model_layers.append(layers.Dropout(d_rate, name="dropout_1"))
    model_layers.append(layers.Dense(num_classes, activation="softmax", name="genre_output"))

    model = keras.Sequential(model_layers, name="gtzan_cnn_classifier")

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
