from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


class SourceDataSampler:
    def __init__(
        self,
        n_observable: int,
        n_latent: int,
        sparsity_intensity: float = 0.9,
        coeffs_range: List[float] = [0.1, 5],
        random_state: int = 0,
    ) -> None:
        self.n_observable = n_observable
        self.n_latent = n_latent
        self.sparsity_intensity = sparsity_intensity
        self.coeffs_range = coeffs_range
        self.random_state = random_state

        self.x_columns = [f"X{i}" for i in range(self.n_observable)]
        self.z_columns = [f"Z{i}" for i in range(self.n_latent)]

        self.W = pd.DataFrame(
            self.generate_sparse_mapping(), index=self.z_columns, columns=self.x_columns
        )
        self.C = pd.DataFrame(
            self.generate_loadings_matrix(),
            index=self.z_columns,
            columns=self.x_columns,
        )
        self.C_inv = pd.DataFrame(
            self.compute_inverse_mapping(), index=self.x_columns, columns=self.z_columns
        )

    def generate_sparse_mapping(self) -> np.array:
        """z to x mapping"""
        a = 1
        b = int(self.sparsity_intensity * self.n_observable)

        # Computing thetas where each theta(k,j) represents the probability that
        # the k-th factors interacts with the j-th feature.
        thetas = stats.beta(a, b).rvs(
            size=(self.n_latent, self.n_observable), random_state=self.random_state
        )

        # Make sure that all columns have at least a one = Each feature is linked
        # to at least one factor.
        for i in range(self.n_observable):
            thetas_ = thetas[:, i]
            idx_max = np.argmax(thetas_)
            thetas[idx_max, i] = 1

        # Bernoulli sampling from the probabilities thetas.
        W = stats.bernoulli(thetas).rvs(random_state=self.random_state)

        return W

    def generate_loadings_matrix(self) -> np.array:
        """Law X=F(Z) -> Coeffs of a linear transformation"""
        np.random.seed(self.random_state)
        abs_coeffs = (
            np.random.uniform(
                low=self.coeffs_range[0], high=self.coeffs_range[1], size=self.W.shape
            )
            * self.W.values
        )

        sign = np.random.choice([-1, 1], size=self.W.shape) * self.W.values

        # Generate random values for std prior
        self.x_std_true = np.random.uniform(low=0.5, high=1.5, size=self.n_observable)
        abs_coeffs = abs_coeffs / np.sum(abs_coeffs, axis=0) * self.x_std_true

        return sign * abs_coeffs

    def sample_latent_prior(self, n_samples: int) -> np.array:
        """Generating Z"""
        cov = np.eye(self.n_latent)
        return stats.multivariate_normal.rvs(
            mean=np.zeros(self.n_latent),
            cov=cov,
            size=n_samples,
            random_state=self.random_state,
        )

    def gaussian_noise(self, n_samples: int, gaussian_noise_std: float) -> np.array:
        np.random.seed(self.random_state)
        return np.random.normal(
            scale=gaussian_noise_std, size=(n_samples, self.n_observable)
        )

    def sample(
        self, n_samples: int, gaussian_noise_std: float = 0, random_state: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.random_state = random_state

        """Generating X"""
        Z = self.sample_latent_prior(n_samples)
        X = Z @ self.C

        noise = self.gaussian_noise(n_samples, gaussian_noise_std)
        X_noisy = X + noise

        Z = pd.DataFrame(Z, columns=self.z_columns)

        X = pd.DataFrame(
            X,
            columns=self.x_columns,
        )

        X_noisy = pd.DataFrame(
            X_noisy,
            columns=self.x_columns,
        )

        return Z, X, X_noisy

    def compute_inverse_mapping(self, n_samples: int = 1000) -> np.array:
        """Computes the inverse-mapping coefficients"""
        Z, _, X = self.sample(n_samples)
        linreg = LinearRegression().fit(X, Z)
        C_inv = linreg.coef_.T
        return C_inv
