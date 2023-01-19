import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def univariate_wasserstein_dist(X_source: np.array, X_target: np.array) -> np.array:

    # The source data is normalised to a 0 mean 1 std distribution.
    mean_source = X_source.mean(0).reshape(1, -1)
    scale_source = X_source.std(0).reshape(1, -1)

    X_source = (X_source - mean_source) / scale_source
    X_target = (X_target - mean_source) / scale_source

    output = np.zeros(X_source.shape[1])
    for i in range(X_source.shape[1]):
        output[i] = wasserstein_distance(X_source[:, i], X_target[:, i])
    return output


def shift_detection_score(X_shift: pd.DataFrame) -> pd.DataFrame:
    """Implementation of the Shift Detection Score"""

    # Sort all rows by descending order
    scores = X_shift.apply(
        lambda row: pd.Series(row.dropna().sort_values(ascending=False).values),
        axis=1,
    )

    # Normalize
    scores = scores / scores.sum(1).values.reshape(-1, 1)

    # Computes the crossentropy between each row and the theoretical
    # unitary shift profile: [1, 0, 0, ..., 0]
    scores = -np.log(scores.iloc[:, 0])

    # Mean aggregation and formatting
    scores = (
        scores.groupby(["N Samples", "Gaussian Noise Std"])
        .mean()
        .reset_index()
        .pivot_table(index="N Samples", columns="Gaussian Noise Std", values=0)
    )

    return scores
