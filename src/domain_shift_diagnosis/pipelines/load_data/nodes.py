from typing import Any, Dict, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from domain_shift_diagnosis.datasets import SourceDataSampler, UnitaryShiftSampler

from ..test_model.utils import univariate_wasserstein_dist


def load_source_data(
    source_data_params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SourceDataSampler]:
    sampler_params = source_data_params["sampler"]

    source_sampler = SourceDataSampler(**sampler_params)
    Z_source_true, X_source_true, X_source_noisy = source_sampler.sample(
        source_data_params["n_samples"], source_data_params["gaussian_noise_std"]
    )
    return Z_source_true, X_source_true, X_source_noisy, source_sampler


def load_target_data(
    source_data_params: Dict[str, Any], target_data_params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    source_sampler_params = source_data_params["sampler"]
    source_sampler = SourceDataSampler(**source_sampler_params)

    target_sampler = UnitaryShiftSampler(source_data_sampler=source_sampler)

    Z_target_true, X_target_true, X_target_noisy = target_sampler.sample(
        source_gaussian_noise_std=source_data_params["gaussian_noise_std"],
        shift_source_list=target_data_params["shift_source_list"],
        n_samples_list=target_data_params["n_samples_list"],
        shift_intensity_list=target_data_params["shift_intensity_list"],
        gaussian_noise_std_list=target_data_params["gaussian_noise_std_list"],
        random_state=target_data_params["random_state"],
    )

    return Z_target_true, X_target_noisy


def compute_univariate_shift(
    X_source: pd.DataFrame,
    X_target: pd.DataFrame,
) -> pd.DataFrame:

    return X_target.groupby(
        [
            "Shift Type",
            "Shift Source",
            "Shift Intensity",
            "Gaussian Noise Std",
            "N Samples",
            "shift_id",
        ]
    ).apply(
        lambda df: pd.Series(
            univariate_wasserstein_dist(X_source.values, df.values),
            index=X_source.columns,
        )
    )


def plot_true_mapping(source_sampler: SourceDataSampler) -> None:
    W_true = source_sampler.W
    plt.figure(figsize=(10, 4))
    g=sns.heatmap(W_true, square=True, cbar=False)
    plt.title("True Factors to Features Mapping", fontdict=dict(size=20))
    plt.xlabel("Features")
    plt.ylabel("Factors")
    plt.tight_layout()
    g.get_figure().savefig("notebooks/figures/W_true.pdf")