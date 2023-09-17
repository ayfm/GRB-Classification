import os
from copy import deepcopy
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import data as dt
from . import disp, env
from . import models as md
from . import stats, utils


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


def create_robust_gmm(df: pd.DataFrame, n_components: int, n_trials: int = 100) -> None:
    """
    Creates 'n_trials' number of GMM models and averages their parameters to
    obtain a single robust model.

    Parameters:
    - df (pd.DataFrame): Input dataset.
    - n_components (int): Number of Gaussian components for the model.
    - n_trials (int): Number of trials for GMM fitting. Default is 100.

    Returns:
    None. The function saves the plots and analysis results to disk.
    """

    # extract catalog name, features, and number of features
    catalog_name = df.index.name
    features = df.columns.tolist()
    n_features = len(features)

    # get model name
    model_name = md.get_model_name(catalog_name, features, n_components)

    # return if the model already exists
    if md.is_model_exists(model_name):
        logger.warning(f"Model '{model_name}' already created! Skipping...")
        return

    # Define base path for saving figures
    subdir = os.path.join(
        "models", catalog_name, "-".join(features), f"{n_components}G"
    )

    # Create models
    model_list = md.create_models(
        df=df,
        features=features,
        n_components=n_components,
        n_trials=n_trials,
        max_iter=10000,
        n_init=1,
        sort_clusters=True,
    )

    # Extract parameters of each model
    model_list_params = {
        model.model_name: model.get_component_params() for model in model_list
    }

    # Aggregate means of all Gaussian components from all models
    means_of_all_components = np.array(
        [
            model_list_params[mm][cid]["mean"]
            for mm in model_list_params
            for cid in model_list_params[mm].keys()
        ]
    ).reshape(-1, n_features)

    # Convert to DataFrame for easier manipulation and visualization
    df_means_of_all_components = pd.DataFrame(
        means_of_all_components, columns=[f"m{fid}" for fid in range(n_features)]
    )

    # Create subplots
    if n_features <= 2:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
    else:
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(111, projection="3d")

    # Display histogram of means based on feature count
    if n_features == 1:
        disp.histogram_1d(
            df=df_means_of_all_components,
            show_density_curve=False,
            n_bins=50,
            xlabel="Mean",
            title="Distribution of Component Masses",
            ax=ax0,
        )
    elif n_features == 2:
        disp.histogram_2d(
            df=df_means_of_all_components,
            n_bins=50,
            title="Distribution of Component Masses",
            xlabel="Mean-1",
            ylabel="Mean-2",
            ax=ax0,
        )

    # Apply k-means clustering to means of all components
    kmeans = KMeans(
        n_clusters=n_components, init="k-means++", n_init=10, max_iter=10000
    ).fit(means_of_all_components)

    # Assign cluster labels to each mean
    df_means_of_all_components["cluster"] = kmeans.predict(means_of_all_components)

    # Plot each cluster on the second subplot
    for cid in range(n_components):
        df_cluster = df_means_of_all_components[
            df_means_of_all_components["cluster"] == cid
        ]
        disp.plot_data(
            df=df_cluster,
            cols=df_cluster.columns[:-1],
            ax=ax1,
            return_ax=True,
            scatter=False,
            show_density_curve=False,
            color=disp.get_color(cid),
            n_bins="auto",
            title="Distribution of Component Masses (Clustered)",
        )

    # Save figure
    disp.save_figure(
        filename="mass_distribution",
        subdir=subdir,
    )

    # Compute the average parameters for each component
    new_component_params = {
        cid: {
            "mean": [],
            "covariance": [],
            "weight": [],
        }
        for cid in range(n_components)
    }

    # Use the means of each k-means cluster as reference means for GMM components
    ref_means = np.array(
        [
            df_means_of_all_components.groupby("cluster").mean().loc[cid].values
            for cid in range(n_components)
        ]
    ).reshape(n_components, n_features)

    # Average cluster parameters across models
    for model in model_list:
        component_params = model.get_component_params()
        component_means = np.array(
            [component_params[cid]["mean"] for cid in range(n_components)]
        ).reshape(n_components, n_features)
        mapping = utils.match_gaussian_components(component_means, ref_means)
        for orig_cid, matched_cid in mapping.items():
            new_component_params[orig_cid]["mean"].append(
                component_params[matched_cid]["mean"]
            )
            new_component_params[orig_cid]["covariance"].append(
                component_params[matched_cid]["covariance"]
            )
            new_component_params[orig_cid]["weight"].append(
                component_params[matched_cid]["weight"]
            )

    # Create a new averaged GMM model using the average parameters
    gmm_avg = deepcopy(model_list[0])
    for fid in range(n_components):
        gmm_avg.weights_[fid] = np.mean(new_component_params[fid]["weight"])
        gmm_avg.means_[fid] = np.mean(new_component_params[fid]["mean"], axis=0)
        covs = np.array(new_component_params[fid]["covariance"]).reshape(
            n_trials, n_features, n_features
        )
        gmm_avg.covariances_[fid] = utils.compute_avg_covariance(covs)
        gmm_avg.precisions_[fid] = np.linalg.inv(gmm_avg.covariances_[fid])
        gmm_avg.precisions_cholesky_[fid] = np.linalg.cholesky(gmm_avg.precisions_[fid])

    # Save and plot the new model
    gmm_avg.model_name = model_name
    gmm_avg.sort_clusters_by_means()
    md.save_model(gmm_avg, catalog_name, features, n_components)
    disp.plot_model(
        model=gmm_avg,
        df=df,
        cols=features,
        show_decision_boundary=True,
        show_cluster_centers=True,
        show_confidence_ellipses=True,
        show_confidence_ellipsoids=True,
        n_bins="auto",
        title=f"GMM (AVG) | {n_components}G ",
        save_kwargs=dict(filename=gmm_avg.model_name, subdir=subdir),
    )

    axes = disp.plot_models_as_grid(
        models=[gmm_avg] + np.random.choice(model_list, 8).tolist(),
        df=df,
        cols=features,
        show_decision_boundary=True,
        show_cluster_centers=True,
        show_confidence_ellipses=True,
        show_confidence_ellipsoids=True,
        return_axes=True,
    )
    axes[0].set_title(f"GMM (AVG) | {n_components}G ")
    disp.save_figure(filename="models", subdir=subdir)

    # clear all figures
    plt.close("all")


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
