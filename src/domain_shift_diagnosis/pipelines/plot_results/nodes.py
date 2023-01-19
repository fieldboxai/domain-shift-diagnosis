import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def plot_model_output(
    C_pred: pd.DataFrame,
    W_pred: pd.DataFrame,
    Z_source_pred: pd.DataFrame,
    pairwise_mapping_jaccard_score: pd.DataFrame,
) -> Figure:

    C_pred_heatmap = plt.figure(figsize=(7, 6))
    p = sns.heatmap(C_pred, cmap="RdBu", center=0)
    p.set_xlabel("Features", fontsize=15)
    p.set_ylabel("Factors", fontsize=15)
    plt.title("Loading Matrix", fontsize=20)
    plt.tight_layout()

    W_pred_heatmap = plt.figure(figsize=(7, 6))
    p = sns.heatmap(W_pred, cmap="gray")
    p.set_xlabel("Features", fontsize=15)
    p.set_ylabel("Factors", fontsize=15)
    plt.title("Interactions Mapping Matrix", fontsize=20)
    plt.tight_layout()

    W_scores_heatmap = plt.figure(figsize=(7, 6))
    plt.title("Pairwise mapping smiliariy score", fontsize=20)
    p = sns.heatmap(pairwise_mapping_jaccard_score, cmap="Blues", vmin=0, vmax=1)
    p.set_xlabel("Pred", fontsize=15)
    p.set_ylabel("True", fontsize=15)
    plt.tight_layout()

    Z_source_pred_cov_heatmap = plt.figure(figsize=(7, 6))
    plt.title("Factors - Covariance Matrix", fontsize=20)
    sns.heatmap(Z_source_pred.cov(), cmap="RdBu", center=0)
    plt.tight_layout()

    return C_pred_heatmap, W_pred_heatmap, W_scores_heatmap, Z_source_pred_cov_heatmap


def plot_shift_detection_scores(
    covariate_shift_detection_scores: pd.DataFrame
) -> Figure:

    shift_detection_scores_heatmap = plt.figure(figsize=(7, 6))
    p=sns.heatmap(covariate_shift_detection_scores, cmap="binary", vmin=0)
    plt.title("Covariate Shift Dispersion Score", fontsize=20)
    p.set_xlabel("Gaussian Noise Std", fontsize = 15)
    p.set_ylabel("N Samples", fontsize = 15)
    plt.tight_layout()

    return shift_detection_scores_heatmap


def plot_mask_features(mask_features: pd.DataFrame) -> Figure:
    mask_features_scatter_plot = plt.figure()
    sns.scatterplot(
        data=mask_features, x="Entropy", y="Maximum Loading", hue="is_factor_active"
    )
    plt.title("mask_features")
    return mask_features_scatter_plot
