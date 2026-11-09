"""
ANN (Multilayer Perceptron) architecture for audio genre classification.
Operates on tabular statistical audio feature vectors (MFCCs, Chroma, Spectral, etc.).
"""
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ml.config import ann_config, feature_config, NUM_GENRES


from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ml.config import ann_config, feature_config, NUM_GENRES


def build_ann_model(
    input_dim: int = 61,
    num_classes: int = NUM_GENRES,
    layer_sizes: Optional[List[int]] = None,
    dropout_rates: Optional[List[float]] = None,
    use_batch_norm: Optional[bool] = None,
    learning_rate: Optional[float] = None,
) -> keras.Model:
    """
    Construct the Multilayer Perceptron (ANN) architecture.
    
    Architecture (default):
        Input(input_dim)
        Dense(256, activation='relu')
        BatchNormalization()
        Dropout(0.3)
        Dense(128, activation='relu')
        BatchNormalization()
        Dropout(0.3)
        Dense(64, activation='relu')
        Dropout(0.2)
        Dense(num_classes, activation='softmax')

    Args:
        input_dim: Dimension of input feature vector (default: 61).
        num_classes: Number of genre categories (default: 10).
        layer_sizes: List of hidden dense layer unit counts.
        dropout_rates: List of dropout rates for each dense layer.
        use_batch_norm: Whether to apply batch normalization.
        learning_rate: Initial Adam learning rate.

    Returns:
        Compiled Keras Sequential model.
    """
    lr = learning_rate or ann_config.learning_rate
    sizes = layer_sizes if layer_sizes is not None else ann_config.layer_sizes
    dropouts = dropout_rates if dropout_rates is not None else ann_config.dropout_rates
    bn = use_batch_norm if use_batch_norm is not None else ann_config.use_batch_norm

    model_layers = [layers.Input(shape=(input_dim,), name="feature_input")]

    for i, (size, drop) in enumerate(zip(sizes, dropouts)):
        model_layers.append(layers.Dense(size, activation="relu", name=f"dense_{i + 1}"))
        if bn and i < len(sizes) - 1:
            model_layers.append(layers.BatchNormalization(name=f"bn_{i + 1}"))
        if drop > 0:
            model_layers.append(layers.Dropout(drop, name=f"dropout_{i + 1}"))

    model_layers.append(layers.Dense(num_classes, activation="softmax", name="genre_output"))

    model = keras.Sequential(model_layers, name="gtzan_ann_classifier")

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
