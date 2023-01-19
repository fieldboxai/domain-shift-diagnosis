from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .utils.layers.sparsity import DeepSparseGaussianDistribution
from .utils.types import (
    DeepGaussianDistributionParams,
    DeepSparseGaussianDistributionParams,
)
from .variational_auto_encoder import VAE


class SparseVAE(VAE):
    """Sparse VAE implementation from: https://arxiv.org/pdf/2110.10804v2.pdf"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_params: DeepGaussianDistributionParams,
        decoder_params: DeepSparseGaussianDistributionParams,
        beta: float,
        variance_type: str,
        **kwargs: Any
    ) -> None:

        super().__init__(
            input_dim,
            latent_dim,
            encoder_params,
            decoder_params,
            beta,
            variance_type,
            **kwargs
        )

        if "sigma_prior" in kwargs:
            self.X_logvar = tf.Variable(
                2
                * tf.math.log(
                    tf.convert_to_tensor(
                        kwargs.get["sigma_prior"], dtype=tf.float32  # type: ignore
                    )
                )
            )

        self.sigma_prior_df = kwargs.get("sigma_prior_df", 3)
        self.sigma_prior_scale = kwargs.get("sigma_prior_scale", 1)

    def _build_decoder(  # type: ignore
        self, decoder_params: DeepSparseGaussianDistributionParams
    ) -> keras.Model:
        decoder_params["sparse_mapping_params"]["output_dim"] = self.config["input_dim"]
        decoder_params["sparse_mapping_params"]["input_dim"] = self.config["latent_dim"]
        decoder_params = self._set_output_dim(decoder_params, 1)  # type: ignore
        # P(Xhat|Z)
        self.px_z = DeepSparseGaussianDistribution(**decoder_params)

        Z = keras.Input(shape=(self.config["latent_dim"]))
        X_mu, X_logvar = self.px_z(Z)

        return keras.Model(inputs=Z, outputs=[X_mu, X_logvar])

    def sigma_loss(self, batch_size: int) -> float:
        sigma_loss = (batch_size + self.sigma_prior_df + 2) * tf.reduce_sum(
            0.5 * self.X_logvar
        ) + 0.5 * self.sigma_prior_df * self.sigma_prior_scale * tf.reduce_sum(
            1 / tf.exp(self.X_logvar)
        )

        sigma_loss = sigma_loss / batch_size
        return sigma_loss

    def x_loss(  # type: ignore
        self, X: tf.Tensor, X_mu: tf.Tensor, X_logvar: tf.Tensor
    ) -> tf.Tensor:
        """Negative Log Likelihhood of X given X_mu and X_logvar"""
        X_var = tf.exp(X_logvar)

        log_unnormalized = -0.5 * tf.square(X - X_mu) / X_var

        if self.config["variance_type"] != "feature":
            log_normalization = 0.5 * (
                tf.constant(np.log(2.0 * np.pi), dtype=tf.float32) + X_logvar
            )
        else:
            batch_size = tf.cast(tf.shape(X)[0], tf.float32)
            log_normalization = self.sigma_loss(batch_size)
            self.add_metric(log_normalization, "sigma_loss")

        log_likelihood = tf.reduce_sum(log_unnormalized - log_normalization, axis=1)

        return -log_likelihood

    def get_factors_mapping(self) -> np.array:
        mask = self.px_z.get_W().T
        decoder_linear_weights = self.px_z.get_decoder_linear_weights().numpy()

        return mask * decoder_linear_weights
