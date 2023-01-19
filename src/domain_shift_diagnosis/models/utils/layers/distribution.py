from typing import Any, Tuple

import tensorflow as tf

from ..types import DeepRegressorParams
from .base import BaseLayer
from .regression import DeepParallelRegressors, DeepRegressor


class GaussianSampler(BaseLayer):
    """Layers that samples that takes parameters mu and logvar of a Gaussian
    and returns a sample.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, mu: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        eps = tf.random.normal(shape=tf.shape(mu))
        sigma = tf.exp(logvar * 0.5)
        return mu + eps * sigma


class DeepGaussianDistribution(BaseLayer):
    """Layers that computes the parametrises a multivariate diagonal gaussian
    distribution, with the output of multilayer perceptron.

    Parameters
    ----------
    mu_regressor_params : DeepRegressorParams
        Parameters of the mlp and linear layers used to compute mu.
    logvar_regressor_params : DeepRegressorParams
        Parameters of the mlp and linear layers used to compute logvar.
    """

    def __init__(
        self,
        mu_regressor_params: DeepRegressorParams,
        logvar_regressor_params: DeepRegressorParams,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.parameters_network = DeepParallelRegressors(
            [mu_regressor_params, logvar_regressor_params]
        )

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        outputs = self.parameters_network(inputs)
        mu = outputs[0]
        logvar = outputs[1]
        return mu, logvar

    def get_regressor(self, idx: int) -> DeepRegressor:
        return self.parameters_network.deep_regressors_list[idx]
