from typing import Any, List, Optional

import tensorflow as tf
from tensorflow import keras

from .base import BaseLayer


class Linear(BaseLayer):
    """Keras layer made of a single dense layer.
    Parameters
    ----------
    output_dim : int
        Output dimension.
    l1_kernel : float
        Intensity of the L1 regularization on the kernel.
    l1_activity : float
        Intensity of the L1 regularization on the activity.
    """

    def __init__(
        self,
        output_dim: int,
        l1_kernel: Optional[float] = 0,
        l2_kernel: Optional[float] = 0,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.update_config(dict(output_dim=output_dim))
        self.update_config(dict(l1_kernel=l1_kernel, l2_kernel=l2_kernel))

        self.dense = keras.layers.Dense(
            output_dim,
            kernel_regularizer=keras.regularizers.L1L2(l1_kernel, l2_kernel),
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            **kwargs
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.dense(inputs)


class MultiLayerPerceptron(BaseLayer):
    """Keras Layer of a multi layer perceptron.

    Parameters
    ----------
    hidden_units : List[int]
        Neumber of neurons in each hidden layer.
    activation : str
        Name of the activation function if all hidden layers.
    dropout_rate : float
        Rate of the drop-out to be applied in each hidden layer.
    l1_kernel: float
        Intensity of L1 kernel regularization.
    l2_kernel: float
        Intensity of L2 kernel regularization.
    """

    def __init__(
        self,
        hidden_units: List[int],
        activation: str = "relu",
        dropout_rate: Optional[float] = 0,
        l1_kernel: Optional[float] = 0,
        l2_kernel: Optional[float] = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.update_config(
            dict(
                hidden_units=hidden_units,
                activation=activation,
                dropout_rate=dropout_rate,
            )
        )

        self.dense_layers = [
            keras.layers.Dense(
                units=h,
                activation=activation,
                kernel_regularizer=keras.regularizers.L1L2(l1_kernel, l2_kernel),
            )
            for h in hidden_units
        ]
        self.dropout_layers = [keras.layers.Dropout(dropout_rate) for _ in hidden_units]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs

        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x)

        return x
