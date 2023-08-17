#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib as mplot
import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from .env import DIR_FIGURES
import pickle
from grbtools import env
from grbtools import models as model_operations
from grbtools import data as data_operations
import seaborn as sns

sns.set(font_scale=4)


# customize matplotlib
mplot.style.use("seaborn-paper")
mplot.rc("text", usetex=False)
font = {"size": 30}
mplot.rc("font", **font)


################################


def histogram(
    data=None, col="", figsize=(6, 4), savefig=True, title="", filename="", ax=None
):
    if data is not None:
        if ax == None:
            ax = plt.figure(figsize=figsize)

        ax.histogram(data[col], bins=50, color="white", edgecolor="black")

        plt.xlabel(col)
        plt.ylabel("count")

        plt.title(title)

        if savefig:
            plt.savefig(os.path.join(env.DIR_FIGURES, "raw_data_plots/" + filename))
        return
    else:
        raise Exception("Data is not privided.")


def scatter2D(data=None, feats=[], figsize=(6, 4), savefig=True, title="", filename=""):
    if data is not None:
        plt.figure(figsize=figsize)

        plt.scatter(
            data[feats[0]],
            data[feats[1]],
            color="blue",
            marker=".",
            edgecolors="black",
            label="inlier",
            s=50,
        )

        plt.xlabel(feats[0])
        plt.ylabel(feats[1])

        plt.title(title)
        if savefig:
            plt.savefig(
                os.path.join(env.DIR_FIGURES, "raw_data_plots/" + filename),
                dpi=300,
                bbox_inches="tight",
            )

        return
    else:
        raise Exception("Data is not privided.")


def scatter3D(data=None, feats=[], figsize=(6, 4), savefig=True, title="", filename=""):
    if data is not None:
        plt.figure(figsize=figsize)

        ax = plt.axes(projection="3d")
        ax.scatter3D(
            data[feats[0]],
            data[feats[1]],
            data[feats[2]],
            color="blue",
            marker=".",
            edgecolors="black",
            label="inlier",
            s=50,
        )

        ax.set_xlabel(feats[0])
        ax.set_ylabel(feats[1])
        ax.set_zlabel(feats[2])

        ax.set_title(title)

        if savefig:
            plt.savefig(
                os.path.join(env.DIR_FIGURES, "raw_data_plots/" + filename),
                dpi=300,
                bbox_inches="tight",
            )
        return
    else:
        raise Exception("Data is not privided.")


def create_grid_space_2d(xmin, xmax, nGridX, ymin, ymax, nGridY):
    """
    creates a grid on a plane (parameter space)

    Parameters
    ----------
    nGridX : int
        how many points in x-dimension
    nGridY : int
        how many points in y-dimension

    Returns
    -------
    x-points of grid
    y-points of grid
    a matrix (array) with n_grid_points^2 rows and two columns
    """
    # assert nGridX*nGridY <= 5e+6, "!!! too much grid points. memory risk."

    x = np.linspace(xmin, xmax, nGridX)
    y = np.linspace(ymin, ymax, nGridY)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    return X, Y, XX


def draw_cluster_boundary(ax, model, n_grid_points=500, ax_range=None):
    # get axes ranges
    if ax_range is None:
        ax_range = None, None
    x_range, y_range = ax_range
    if x_range is None:
        x_range = ax.get_xlim()
    if y_range is None:
        y_range = ax.get_ylim()
    xmin, xmax = x_range
    ymin, ymax = y_range

    # create a grid on the parameter space
    _, _, GG = create_grid_space_2d(*x_range, n_grid_points, *y_range, n_grid_points)
    grid_shape = (n_grid_points, n_grid_points)

    Z3 = model.predict(GG).reshape(grid_shape)
    # color = plt.cm.binary(Z3)
    # ax.contour(Z3, origin='lower',extent=(xmin, xmax, ymin, ymax), cmap="black")
    ax.contour(Z3, origin="lower", extent=(xmin, xmax, ymin, ymax), colors="black")

    return ax


def colors(index):
    colors = [
        "firebrick",
        "olivedrab",
        "royalblue",
        "mediumorchid",
        "peru",
        "darkorange",
        "darkcyan",
        "darkslategray",
        "darkkhaki",
        "darkseagreen",
    ]
    return colors[index]


def markers(index):
    markers = ["o", "^", "x", "P", "p", "*", "D", "s", "v", "+"]
    return markers[index]


def clusterNames(index):
    clusters = ["Cluster " + str(i) for i in range(1, 11)]
    return clusters[index]


def extractCatalogueName(file_name):
    return file_name.split("_")[0]

    tokens = model_name.split("_")
    cov_token = tokens[-1]
    cov_type = cov_token.split(".")[0][1:]
    return cov_type


def scatter2DWithClusters(
    cat_name="",
    model_name="",
    feat_space=[],
    n_components=0,
    data=None,
    title="",
    xlabel="",
    ylabel="",
    figure_save=True,
    ax=None,
    legend=True,
):
    if data is None:
        raise Exception("Data is not provided.")

    if model_name != "":
        model_path = os.path.join(env.DIR_MODELS, cat_name, model_name)
        model = model_operations.loadModelbyName(model_path)
    else:
        model = model_operations.loadModelbyProperties(
            dataset_name=cat_name, feat_space=feat_space, n_components=n_components
        )

    data = data.drop("clusters", axis=1, errors="ignore")

    data["clusters"] = model.predict(data)
    # data = sortClusters2D(data, pd.unique(data['clusters']))

    grouped_data = data.groupby(data["clusters"])

    # create figure object
    if ax == None:
        fig = plt.figure()
        ax = plt.gca()

    # ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['clusters'], s=25, cmap='tab20b', alpha=0.8)
    for cluster in set(list(data["clusters"])):
        group = grouped_data.get_group(cluster)
        ax.scatter(
            group.iloc[:, 0],
            group.iloc[:, 1],
            c=colors(cluster),
            s=25,
            alpha=0.8,
            marker=markers(cluster),
            label=clusterNames(cluster),
        )

    ax = draw_cluster_boundary(ax, model)
    if legend:
        ax.legend(loc="lower right")
    # set figure options
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(color="lightgray")
    ax.set_title(title)

    if figure_save:
        figure_path = os.path.join(env.DIR_FIGURES, "model_plots", cat_name)
        figure_name = model_name.replace(".model", ".pdf")
        fig.savefig(
            os.path.join(figure_path, figure_name), dpi=300, bbox_inches="tight"
        )

    return ax


def plot_outliers(
    data=None,
    cat_name="",
    threshold_density=0.025,
    feat_space=[],
    save_plot=True,
    figsize=(6, 4),
):
    if data is not None:
        X_df = data_operations.get_values(data)

        df_outlier = data[data["is_outlier"] == True]
        df_inlier = data[data["is_outlier"] == False]
        n_outliers = len(df_outlier)
        n_inliers = len(df_inlier)

        plt.figure(figsize=figsize)

        if len(feat_space) == 1:  # if 1D
            plt.hist(
                X_df[:, 0],
                bins=50,
                color="lightsteelblue",
                edgecolor="black",
                alpha=0.8,
                density=True,
                label="Raw data",
            )
            plt.scatter(
                X_df[:, 0],
                np.exp(data["log_dens"]),
                lw=1,
                marker=".",
                s=10,
                color="steelblue",
                label="Log density",
            )
            plt.scatter(
                X_df[~data["is_outlier"], 0],
                -0.005 - 0.01 * np.random.random(n_inliers),
                marker=".",
                color="black",
                label="Inlier",
            )
            plt.scatter(
                X_df[data["is_outlier"], 0],
                -0.005 - 0.01 * np.random.random(n_outliers),
                marker="x",
                color="Red",
                label="Outlier",
            )

            plt.xlabel(feat_space[0])

        elif len(feat_space) == 2:  # if 2D
            plt.scatter(
                df_inlier.loc[:, feat_space[0]],
                df_inlier.loc[:, feat_space[1]],
                color="blue",
                marker=".",
                edgecolors="black",
                label="Inlier",
            )
            plt.scatter(
                df_outlier.loc[:, feat_space[0]],
                df_outlier.loc[:, feat_space[1]],
                color="red",
                marker="x",
                edgecolors="black",
                label="Outlier",
            )
            plt.xlabel(feat_space[0])
            plt.ylabel(feat_space[1])
            plt.grid(linestyle="--")

        elif len(feat_space) == 3:  # if 3D
            ax = plt.axes(projection="3d")
            ax.scatter3D(
                df_inlier.loc[:, feat_space[0]],
                df_inlier.loc[:, feat_space[1]],
                df_inlier.loc[:, feat_space[2]],
                color="blue",
                marker=".",
                edgecolors="black",
                label="Inlier",
                s=50,
            )
            ax.scatter3D(
                df_outlier.loc[:, feat_space[0]],
                df_outlier.loc[:, feat_space[1]],
                df_outlier.loc[:, feat_space[2]],
                color="red",
                marker="x",
                edgecolors="black",
                label="Outlier",
                s=100,
            )
            ax.set_xlabel(feat_space[0])
            ax.set_ylabel(feat_space[1])
            ax.set_zlabel(feat_space[2])
        else:
            raise Exception("Invalid data shape")

        feat_space_txt = "-".join(feat_space)

        plt.legend()
        plt.title(
            cat_name.upper()
            + " | "
            + feat_space_txt
            + " | Thr: "
            + str(threshold_density)
        )

        if save_plot:
            plt.savefig(
                os.path.join(
                    env.DIR_FIGURES,
                    (
                        "outlier_plots/"
                        + cat_name
                        + "_"
                        + feat_space_txt
                        + "_"
                        + str(threshold_density)
                        + ".pdf"
                    ),
                )
            )

        return
    else:
        raise Exception("Data is None")


def plot_models(cat_name="", data=None, model_name=""):
    if data is None:
        raise Exception("Data is None")

    if len(data.columns) == 2:  # Scatter 2D
        scatter2DWithClusters(
            cat_name=cat_name,
            model_name=model_name,
            data=data,
            title=cat_name.upper(),
            xlabel=data.columns[0],
            ylabel=data.columns[1],
            figure_save=True,
        )


def plot_scores(scores, title="", ax=None, xlabel="", ylabel=""):
    k_scores = tuple(scores.items())
    # sort scores by k
    k_scores = sorted(k_scores, key=lambda x: x[0])
    # extract k and score values
    k_values = [k for k, _ in k_scores]
    scores = [score for _, score in k_scores]

    if ax == None:
        ax = plt.figure(figsize=(6, 3))

    ax.plot(
        k_values,
        scores,
        marker="X",
        color="r",
        linestyle="dashed",
        linewidth=1,
        markersize=8,
    )
    ax.set_xticks(k_values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_scores_all(df_scores=None, ax=None):
    # Plot
    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 6))

    x = df_scores.index
    aic_ = df_scores["aic"].values
    bic_ = df_scores["bic"].values
    sil_ = df_scores["sil_euc"].values
    wcss_ = df_scores["wcss"].values
    gap_ = df_scores["gap"].values
    dbs_ = df_scores["dbs"].values
    chs_ = df_scores["chs"].values

    ax.plot(x, aic_, label="AIC", marker="o")
    ax.plot(x, bic_, label="BIC", marker="o")
    ax.plot(x, sil_, label="Silhouette Score", marker="o", linestyle="--")
    ax.plot(x, wcss_, label="WCSS", marker="o", linestyle="-.")
    ax.plot(x, gap_, label="Gap Statistics", marker="o", linestyle=":")
    ax.plot(x, dbs_, label="Davies-Bouldin Index", marker="o", linestyle="-")
    ax.plot(x, chs_, label="Calinski Harabasz Score", marker="o", linestyle="--")

    # For visual clarity
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Normalized Score")
    ax.legend(loc="best")
    ax.grid(True)
    plt.title("Normalized Clustering Metrics vs. Number of Components")
    plt.tight_layout()
    plt.show()


def plot_radar(df_scores_normalized=None):
    data = df_scores_normalized.to_dict()
    k_values = [f"K={k}" for k in df_scores_normalized.index]

    del data["wcss"]
    del data["sil_mah"]
    components = df_scores_normalized.shape[0]

    # Number of variables
    categories = list(data.keys())
    N = len(categories)

    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set1", len(data.keys()))

    # Create background
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    angles = [n / float(components) * 2 * np.pi for n in range(components)]
    # angles += angles[:1]  # This ensures the data is "closed" in the chart
    plt.xticks(angles, k_values)  # Using the component number for xtick labels

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(
        [0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=12
    )
    plt.ylim(0, 1)

    # Plot data
    for i, key in enumerate(data.keys()):
        values = list(data[key].values())
        # values += values[:1]  # This ensures the data is "closed" in the chart
        ax.plot(
            angles,
            values,
            linewidth=2,
            linestyle="solid",
            label=key,
            color=my_palette(i),
            marker="o",
        )
        # ax.fill(angles, values, color=my_palette(i), alpha=0.1)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.show()


def plot_parallel_coord(df_scores_normalized=None):
    df = df_scores_normalized.copy(deep=True)
    df["K"] = df.index.values
    df["K"] = df["K"].apply(lambda x: f"K={x}")
    df.drop(columns=["wcss", "sil_mah"], inplace=True)

    # Create parallel coordinates plot
    plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(
        df,
        class_column="K",
        colormap=plt.get_cmap("Set1"),
        linewidth=2,
        marker="x",
    )
    plt.title("Parallel Coordinates Plot for Clustering Scores")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend(title="Configurations", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def plot_heatmap(df_scores_normalized, cat_name):
    df = df_scores_normalized.copy(deep=True)
    df.drop(columns=["wcss", "sil_mah"], inplace=True)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    sns.heatmap(
        df,
        annot=True,
        cmap="gist_heat_r",
        cbar_kws={"label": "Normalized Score"},
    )
    plt.title(cat_name.upper() + " | Normalized Scores")
    plt.show()


def plot_models_as_grid(cat_name, feat_space, data):
    """
    Plot all models for a given dataset as a grid
    """
    # get all models
    models = [
        f
        for f in os.listdir(os.path.join(env.DIR_MODELS, cat_name))
        if f.endswith(".model") and f.find("_".join(feat_space)) != -1
    ]
    # sort models by number of components
    models = sorted(models, key=lambda x: int(x.split("_")[-2][1:]))
    models = models[1:7]
    # get number of rows and columns
    n_rows = 2
    n_cols = 3
    # create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

    # plot models
    for i, model_name in enumerate(models):
        # get row and column index
        row = i // n_cols
        col = i % n_cols

        # plot model
        ax = axes[row, col]

        if len(feat_space) == 1:
            histogram(
                cat_name=cat_name,
                model_name=model_name,
                feat_space=feat_space,
                n_components=i,
                data=data,
                title=model_name.replace(".model", ""),
                xlabel=feat_space[0],
                ylabel="count",
                figure_save=False,
                ax=ax,
            )

        if len(feat_space) == 2:
            scatter2DWithClusters(
                cat_name=cat_name,
                model_name=model_name,
                feat_space=feat_space,
                n_components=i,
                data=data,
                title=(model_name.replace(".model", "")).split("_")[-2][1:] + "G",
                xlabel="",
                ylabel="",
                figure_save=False,
                ax=ax,
                legend=False,
            )

    if len(feat_space) == 2:
        fig.text(0.5, 0.04, feat_space[0], ha="center")
        fig.text(0.04, 0.5, feat_space[1], va="center", rotation="vertical")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.97),
        fontsize=10,
    )

    fig.text(0.5, 1, cat_name.upper(), ha="center")

    # save figure
    fig.savefig(
        os.path.join(env.DIR_FIGURES, "model_plots", cat_name, "grid.pdf"),
        dpi=300,
        bbox_inches="tight",
    )


def plot_scores_as_grid(scores, cat_name):
    """
    Plot all scores as a grid
    """
    # get number of rows and columns
    n_rows = 2
    n_cols = 4
    # create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

    # plot models
    for i, model_name in enumerate(scores):
        # get row and column index
        row = i // n_cols
        col = i % n_cols

        # plot model
        ax = axes[row, col]
        plot_scores(scores[model_name], title=model_name.upper(), ax=ax)

    fig.text(0.5, 0.92, cat_name.upper(), ha="center")

    # save figure
    fig.savefig(
        os.path.join(env.DIR_FIGURES, "model_plots", "scores_grid.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
