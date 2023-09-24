from typing import Any, Dict, List

import pandas as pd
from . import env, stats

# get logger
logger = env.get_logger()

clustering_metrics = {
    "wcss_euc": {
        "description": "Within Cluster Sum of Squares (Euclidean)",
        "is_higher_better": True,
        "func": lambda X, y: stats.intra_cluster_dispersion(X, y, "euclidean"),
    },
    "wcss_mah": {
        "description": "Within Cluster Sum of Squares (Mahalanobis)",
        "is_higher_better": True,
        "func": lambda X, y: stats.intra_cluster_dispersion(X, y, "mahalanobis"),
    },
    "aic": {
        "description": "Akaike Information Criterion",
        "is_higher_better": False,
        "func": lambda m, d: stats.AIC(m, d),
    },
    "bic": {
        "description": "Bayesian Information Criterion",
        "is_higher_better": False,
        "func": lambda m, d: stats.BIC(m, d),
    },
    "gap": {
        "description": "Gap Statistics",
        "is_higher_better": True,
        "func": lambda X, y: stats.gap_statistic(X, y)["gap"],
    },
    "sil_euc": {
        "description": "Silhouette Score (Euclidean)",
        "is_higher_better": True,
        "func": lambda X, y: stats.silhouette_score(X, y, metric="euclidean")["mean"],
    },
    "sil_mah": {
        "description": "Silhouette Score (Mahalanobis)",
        "is_higher_better": True,
        "func": lambda X, y: stats.silhouette_score(X, y, metric="mahalanobis")["mean"],
    },
    "dbs_euc": {
        "description": "Davies-Bouldin Score (Euclidean)",
        "is_higher_better": False,
        "func": lambda X, y: stats.davies_bouldin_score(X, y, metric="euclidean"),
    },
    "dbs_mah": {
        "description": "Davies-Bouldin Score (Mahalanobis)",
        "is_higher_better": False,
        "func": lambda X, y: stats.davies_bouldin_score(X, y, metric="mahalanobis"),
    },
    "chs_euc": {
        "description": "Calinski-Harabasz Score (Euclidean)",
        "is_higher_better": True,
        "func": lambda X, y: stats.calinski_harabasz_score(X, y, metric="euclidean"),
    },
    "chs_mah": {
        "description": "Calinski-Harabasz Score (Mahalanobis)",
        "is_higher_better": True,
        "func": lambda X, y: stats.calinski_harabasz_score(X, y, metric="mahalanobis"),
    },
    "ded_euc": {
        "description": "Density Distance Score (Euclidean)",
        "is_higher_better": True,
        "func": lambda X, y: stats.density_distance_score(X, y, metric="euclidean"),
    },
    "ded_mah": {
        "description": "Density Distance Score (Mahalanobis)",
        "is_higher_better": True,
        "func": lambda X, y: stats.density_distance_score(X, y, metric="mahalanobis"),
    },
    "dunn_euc": {
        "description": "Dunn Index (Euclidean)",
        "is_higher_better": True,
        "func": lambda X, y: stats.dunn_index(X, y, metric="euclidean"),
    },
    "dunn_mah": {
        "description": "Dunn Index (Mahalanobis)",
        "is_higher_better": True,
        "func": lambda X, y: stats.dunn_index(X, y, metric="mahalanobis"),
    },
}


def get_clustering_metrics(
    metric_names: List["str"] = None, verbose: bool = False
) -> Dict[str, Any]:
    """
    Retrieves a list of available clustering metrics.
    Optionally logs a human-readable description of each metric.

    Parameters:
    - metric_names (List[str], optional): A list of metric identifiers. If None, all available metrics are returned. Default is None.
    - verbose (bool, optional): If True, logs the description of each metric. Default is False.

    Returns:
    - List[str]: A list of available clustering metric identifiers.
    """
    if metric_names is None:
        metric_list = list(clustering_metrics.keys())
    else:
        metric_list = metric_names

    # get metrics
    metrics = {
        metric: clustering_metrics[metric]
        for metric in metric_list
        if metric in clustering_metrics
    }

    if verbose:
        logger.info("Available metrics:")
        for mm in metric_list:
            logger.info(f"  > {mm}: {clustering_metrics[mm]['description']}")

    return metrics.copy()


def align_clustering_scores(
    scores: pd.DataFrame, score_orientation: str
) -> pd.DataFrame:
    """
    Adjusts clustering scores in a DataFrame to be consistent with the is_higher_better flag.

    If a metric in the scores DataFrame is not consistent with the is_higher_better flag based on the
    global `clustering_metrics` dictionary, it will be multiplied by -1 to invert its orientation.

    Parameters:
    -----------
    scores : pd.DataFrame
        A DataFrame containing clustering scores. Each column represents a metric, and each row
        represents an observation or cluster.

    score_orientation : bool
        Either 'lower' or 'higher' indicating that whether higher scores or lower
        scores are better for the desired evaluation.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing adjusted clustering scores, consistent with the is_higher_better flag.

    Raises:
    -------
    ValueError
        If the score_orientation is not 'lower' or 'higher'.
    """
    # Check if the orientation is valid
    if score_orientation not in ["lower", "higher"]:
        raise ValueError(
            f"Invalid orientation '{score_orientation}'. Must be either 'lower' or 'higher'."
        )

    for metric in scores.columns:
        # Check if the metric is in the clustering_metrics dictionary
        if metric in clustering_metrics:
            # If the desired orientation is not consistent with clustering_metrics' value, invert the score

            if (
                score_orientation == "higher"
                and not clustering_metrics[metric]["is_higher_better"]
            ) or (
                score_orientation == "lower"
                and clustering_metrics[metric]["is_higher_better"]
            ):
                scores[metric] *= -1

    return scores


def normalize_clustering_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes clustering scores to a range of [0, 1].

    Parameters:
    ----------
    scores : pd.DataFrame
        A DataFrame containing clustering scores.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing normalized clustering scores.
    """

    # Identify valid metrics in the scores DataFrame based on clustering_metrics
    valid_metrics = [
        metric for metric in scores.columns if metric in clustering_metrics
    ]

    # Normalize scores for each valid metric
    for metric in valid_metrics:
        min_val = scores[metric].min()
        max_val = scores[metric].max()

        scores[metric] = (scores[metric] - min_val) / (max_val - min_val)

    return scores
