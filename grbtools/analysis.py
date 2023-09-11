import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import data as dt
from . import disp, env, stats, utils
from .gmm import GaussianMixtureModel

# get logger
logger = env.get_logger()


def detect_outliers(
    catalog: str, features: list, density_threshold: float, figsize=(6, 4)
):
    """
    Detects outliers in the given catalog using the given features.
    Plots the outliers and saves the results to the catalog.

    Args:
        catalog: A pandas dataframe containing the catalog.
        features: A list of features to be used for outlier detection.
        density_threshold: A float value between 0 and 1. The threshold density for outlier detection.

    Returns:
        None.
    """
    # read dataset
    dataset = dt.load_dataset(
        catalog, features, remove_invalids=True, remove_outliers=False, verbose=True
    )

    # get values
    X = dataset[features].values
    # reshape if 1D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    logger.info("")
    # detect outliers
    res = stats.detect_outliers(X, density_threshold, verbose=True)

    # save results to the dataset
    dataset["is_outlier"] = res["is_outlier"]
    dataset["log_dens"] = res["density"]

    # read catalog
    df_catalog = dt.load_dataset(
        catalog, remove_invalids=False, remove_outliers=False, verbose=False
    )
    # save flags to the catalog
    col_outlier = "is_outlier_" + "-".join(features)
    col_log_dens = "log_dens_" + "-".join(features)

    # first, reset the flags
    df_catalog[col_outlier] = np.nan
    df_catalog[col_log_dens] = np.nan
    # then, set the flags
    df_catalog.loc[dataset.index, col_outlier] = dataset["is_outlier"]
    df_catalog.loc[dataset.index, col_log_dens] = dataset["log_dens"]

    # save catalog
    dt.save_dataframe(df_catalog, catalog)

    # feature names for saving the plot
    feat_space_txt = "-".join(features)

    # plot results
    disp.plot_data(
        dataset,
        features,
        figsize=figsize,
        show_outliers=True,
        show_density_curve=True,
        title=f"{catalog.upper()} | {feat_space_txt} | Thr: {density_threshold}",
        save_kwargs=dict(
            filename=f"{catalog}_{feat_space_txt}_{density_threshold}",
            subdir="outlier_plots",
            fmt="pdf",
        ),
    )


def check_for_normality(data, with_outliers=True, feat_space=[]):
    # Extract inliers
    if with_outliers:
        data = data[data["is_outlier"] == False][feat_space].values.ravel()

    stats.normality_test_shapiro_wilkinson(data)
    print()
    stats.normality_test_ks(data, normalization=True)
    print()
    stats.normality_test_anderson(data)
    print()
    stats.normality_test_dagostino(data)


def perform_statistical_tests(df_data, feat_space, cat_name):
    predictions = model_operations.bring_all_predictions(
        cat_name=cat_name, feat_space=feat_space, data=df_data
    )

    models = model_operations.bring_all_models(cat_name=cat_name, feat_space=feat_space)

    scores = dict()
    # models = {k: v for k, v in models.items() if str(k) != "1"}
    # predictions = {k: v for k, v in predictions.items() if str(k) != "1"}

    # AIC
    scores["aic"] = {k: aic(data=df_data, model=model) for k, model in models.items()}

    # BIC
    scores["bic"] = {k: bic(data=df_data, model=model) for k, model in models.items()}

    # WCSS
    scores["wcss"] = {
        k: intra_cluster_dispersion(df_data.values, predictions[k])
        for k in models.keys()
    }

    # Silhouette score with Euclidean distance
    scores["sil_euc"] = {
        k: silhouette_score(df_data.values, predictions[k], metric="Euclidean")["mean"]
        for k in models.keys()
    }

    # Silhouette score with Mahalanobis distance
    scores["sil_mah"] = {
        k: silhouette_score(df_data.values, predictions[k], metric="Mahalanobis")[
            "mean"
        ]
        for k in models.keys()
    }
    """
    # Gap Statistic
    scores["gap"] = {
        k: gap_statistics(
            df_data.values,
            predictions[k],
            clusterer=models[k],
            n_repeat=100,
            random_state=None,
        )["gap"]
        for k in models.keys()
    }
    """
    scores["gap"] = {k: 0 for k in models.keys()}

    # Davies-Bouldin Index
    scores["dbs"] = {
        k: davies_bouldin_score(df_data, predictions[k]) for k in models.keys()
    }

    # Calinski-Harabasz Index
    scores["chs"] = {
        k: calinski_harabasz_score(df_data, predictions[k]) for k in models.keys()
    }

    df_scores = data_operations.merge_scores(scores)
    df_scores = df_scores.iloc[1:]

    normalized_scores = normalize_scores(df_scores)
    return df_scores, normalized_scores


def perform_wasserstein(feat_space, random_state=None, n_components=None):
    clusters = model_operations.make_all_clusters(
        feat_space=feat_space.copy(), n_components=n_components
    )
    batse_clusters = clusters["batse"]
    fermi_clusters = clusters["fermi"]
    swift_clusters = clusters["swift"]

    # Wasserstein distance
    distances_cluster = {
        "BATSE-FERMI": {"C" + str(k + 1): 0 for k in range(n_components)},
        "BATSE-SWIFT": {"C" + str(k + 1): 0 for k in range(n_components)},
        "FERMI-SWIFT": {"C" + str(k + 1): 0 for k in range(n_components)},
    }

    for k in range(n_components):
        dist = wasserstein_distance_bootstrap(
            batse_clusters[str(k)].values,
            fermi_clusters[str(k)].values,
            random_state=random_state,
            max_iter=100,
        )

        distances_cluster["BATSE-FERMI"]["C" + str(k + 1)] = (
            str(dist["mean"].round(2)) + "(" + str(dist["std"].round(2)) + ")"
        )

        dist = wasserstein_distance_bootstrap(
            batse_clusters[str(k)].values,
            swift_clusters[str(k)].values,
            random_state=random_state,
            max_iter=100,
        )

        distances_cluster["BATSE-SWIFT"]["C" + str(k + 1)] = (
            str(dist["mean"].round(2)) + "(" + str(dist["std"].round(2)) + ")"
        )

        dist = wasserstein_distance_bootstrap(
            fermi_clusters[str(k)].values,
            swift_clusters[str(k)].values,
            random_state=random_state,
            max_iter=100,
        )

        distances_cluster["FERMI-SWIFT"]["C" + str(k + 1)] = (
            str(dist["mean"].round(2)) + "(" + str(dist["std"].round(2)) + ")"
        )

    return pd.DataFrame(distances_cluster)


def perform_js(feat_space, random_state=None, n_components=None):
    clusters = model_operations.make_all_clusters(
        feat_space=feat_space.copy(), n_components=n_components
    )
    batse_clusters = clusters["batse"]
    fermi_clusters = clusters["fermi"]
    swift_clusters = clusters["swift"]

    # JS distance
    distances_cluster = {
        "BATSE-FERMI": {"C" + str(k + 1): 0 for k in range(n_components)},
        "BATSE-SWIFT": {"C" + str(k + 1): 0 for k in range(n_components)},
        "FERMI-SWIFT": {"C" + str(k + 1): 0 for k in range(n_components)},
    }

    for k in range(n_components):
        dist = jensen_shannon_distance_bootstrap(
            batse_clusters[str(k)].values,
            fermi_clusters[str(k)].values,
            random_state=random_state,
        )

        distances_cluster["BATSE-FERMI"]["C" + str(k + 1)] = (
            str(dist["mean"].round(2)) + "(" + str(dist["std"].round(2)) + ")"
        )

        dist = jensen_shannon_distance_bootstrap(
            batse_clusters[str(k)].values,
            swift_clusters[str(k)].values,
            random_state=random_state,
        )

        distances_cluster["BATSE-SWIFT"]["C" + str(k + 1)] = (
            str(dist["mean"].round(2)) + "(" + str(dist["std"].round(2)) + ")"
        )

        dist = jensen_shannon_distance_bootstrap(
            fermi_clusters[str(k)].values,
            swift_clusters[str(k)].values,
            random_state=random_state,
        )

        distances_cluster["FERMI-SWIFT"]["C" + str(k + 1)] = (
            str(dist["mean"].round(2)) + "(" + str(dist["std"].round(2)) + ")"
        )

    return pd.DataFrame(distances_cluster)


def perform_cross_catalogue_comparison(
    feat_space, random_state=None, n_components=None, plot_catalogues=False
):
    wasserstein = perform_wasserstein(
        feat_space, random_state=random_state, n_components=n_components
    )
    js = perform_js(feat_space, random_state=random_state, n_components=n_components)

    print("::: Wasserstein Distance :::")
    print(wasserstein)
    print("\n")
    print("::: Jensen-Shannon Distance :::")
    print(js)

    if plot_catalogues:
        disp_operations.plot_cross_catalogue_2D(feat_space, n_components=n_components)
