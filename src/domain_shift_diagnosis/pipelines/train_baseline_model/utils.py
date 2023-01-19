from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler


def parse_n_components(X: np.array, latent_dim: int) -> int:
    if latent_dim == "input_dim":
        latent_dim = X.shape[1]
    return latent_dim


def compute_mask_from_weights(
    weights: np.array,
    compute_activity_mask: bool,
) -> Tuple[np.array, pd.DataFrame, int]:

    l1_weights = np.abs(weights)

    # Maximum loading
    load_max = weights.max(1)

    # L1 Loss
    l1_loss = l1_weights.sum(1)

    # Dispersion measure: Entropy
    weights_normalized = l1_weights / l1_loss.reshape(-1, 1)
    entropy = -np.sum(np.log(weights_normalized + 1e-6) * weights_normalized, axis=1)
    entropy[np.isnan(entropy)] = 0

    mask_features = np.stack([entropy, load_max], axis=-1)
    mask_features = MinMaxScaler().fit_transform(mask_features)

    mask_features_df = pd.DataFrame(
        mask_features, columns=["Entropy", "Maximum Loading"]
    )

    if compute_activity_mask:
        is_factor_active = AgglomerativeClustering(n_clusters=2).fit_predict(
            (load_max.reshape(-1, 1))
        )

        if (
            load_max[is_factor_active == 1].sum()
            < load_max[is_factor_active == 0].sum()
        ):
            is_factor_active = 1 - is_factor_active

        mask_active_factors = is_factor_active.astype(bool)
    else:
        mask_active_factors = np.ones(weights.shape[0], dtype=bool)

    latent_dim_pred = mask_active_factors.sum()
    mask_features_df["is_factor_active"] = mask_active_factors

    # The predicted mask is computed from the weights by setting to zero (=deactivate)
    # weights that contributes to less than 1% of the total factor l1 activity.
    W_pred = weights_normalized[mask_active_factors] > 0.01

    return W_pred, mask_features_df, latent_dim_pred
