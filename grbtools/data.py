"""
This file performs some data reading tasks.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy import stats
from grbtools import env
from grbtools import disp
from grbtools import stats


def check_catalogue_exists(cat_name):
    """
    checks if catalogue exists
    """
    if not os.path.exists(os.path.join(env.DIR_DATASETS, cat_name)):
        print("Catalogue {} does not exist.".format(cat_name))
        return False
    return True


def removeNaNs(dframe, subset=None, how="any"):
    """
    removes invalid values (nan, inf) from dataframe
    """
    dframe = dframe.replace((np.inf, -np.inf), np.nan)
    return dframe.dropna(subset=subset, how=how)


def replace_infs(dframe):
    """
    replaces inf with nans and drops nans
    """
    return dframe.replace([np.inf, -np.inf], np.nan).dropna()


def load(
    cat_name,
    feats=[],
    clean_nans=True,
    clean_infs=True,
    plot_data=False,
    without_outliers=False,
    verbose=False,
):
    """
    loads catalogue
    """
    cat_file_name = cat_name + ".xlsx"
    if not check_catalogue_exists(cat_file_name):
        return None
    else:
        path = os.path.join(env.DIR_DATASETS, cat_file_name)
        df = pd.read_excel(path, index_col=0)

        if len(feats) > 0:  # if features are specified
            if without_outliers:
                feats.append("is_outlier_" + "-".join(feats))
            df = df[feats]
            if clean_nans:
                df = removeNaNs(df, subset=feats)
            if clean_infs:
                df = replace_infs(df)

        if without_outliers:
            feats = feats[:-1]
            df = df[df["is_outlier_" + "-".join(feats)] == False]

        if plot_data:
            plot_filename = cat_name + "_" + "_".join(feats) + ".pdf"
            if len(feats) == 1:
                disp.histogram(
                    data=df,
                    col=feats[0],
                    title="Histogram of " + cat_name + " " + feats[0],
                    figsize=(8, 6),
                    filename=plot_filename,
                )
            elif len(feats) == 2:
                disp.scatter2D(
                    data=df,
                    feats=feats,
                    figsize=(6, 4),
                    title="Plot of " + cat_name.upper() + " " + " ".join(feats),
                    filename=plot_filename,
                )

            elif len(feats) == 3:
                disp.scatter3D(
                    data=df,
                    figsize=(6, 4),
                    feats=feats,
                    savefig=False,
                    title="Plot of " + cat_name.upper() + " " + " ".join(feats),
                    filename=plot_filename,
                )

        if without_outliers:
            df = df[df["is_outlier_" + "-".join(feats)] == False]
            df = df.drop(["is_outlier_" + "-".join(feats)], axis=1)

        if verbose:
            print(df.describe())
            """
            print("--------------------------")
            print("Catalogue {} loaded.".format(cat_name.upper()))
            print("Number of GRBs: {}".format(len(df)))
            print("Number of features: {}".format(len(df.columns)))
            print("Features: {}".format(df.columns.tolist()))
            print("--------------------------")
            """

        return df


def get_values(df):
    """
    returns values of dataframe
    """
    if df.shape[1] == 1:  # if 1D
        return df.values.reshape(-1, 1)
    else:  # if more than 1D
        return df.values


def save_data_to_file(df, cat_name):
    """
    saves data to the file
    """
    path = os.path.join(env.DIR_DATASETS, cat_name + ".xlsx")
    df.to_excel(path)


def find_outliers(
    data,
    threshold_density=0.025,
    cat_name="",
    save_data=True,
    feat_space=[],
    plot_result=False,
    save_plot=False,
    figsize=(10, 8),
    verbose=True,
):
    data_values = get_values(data)
    is_outlier, log_dens = stats.detect_outliers(data_values, threshold_density)

    data["is_outlier"] = is_outlier
    data["log_dens"] = log_dens

    n_outliers = data["is_outlier"].sum()
    n_inliers = len(data) - n_outliers
    if verbose:
        print("Total number of GRBs: {}".format(len(data)))
        print("Number of outliers: {}".format(n_outliers))
        print("Number of inliers: {}".format(n_inliers))

    if save_data:
        df = load(cat_name)
        feat_space_txt = "-".join(feat_space)

        df.loc[df.index.isin(data.index), "is_outlier_" + feat_space_txt] = is_outlier
        df.loc[df.index.isin(data.index), "log_dens_" + feat_space_txt] = log_dens

        save_data_to_file(df, cat_name)

    if plot_result:
        disp.plot_outliers(
            data=data,
            cat_name=cat_name,
            threshold_density=threshold_density,
            feat_space=feat_space,
            figsize=figsize,
        )

    return data


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


def merge_scores(scores={}):
    n_clusters = list(scores.keys())
    n_clusters.sort()
    scores = {i: scores[i] for i in n_clusters}

    df_scores = pd.DataFrame(scores).round(2)
    df_scores["k"] = range(1, len(df_scores) + 1)
    df_scores.set_index("k", inplace=True)

    return df_scores


def normalize(x, inverse=False):
    # normalize the array considering the inf and nan values
    x = np.array(x)
    if inverse:
        x = -x

    min_ = np.nanmin(x)
    max_ = np.nanmax(x)
    x = (x - min_) / (max_ - min_)

    return x


def normalize_scores(df_scores=None):
    df_scores_normalized = df_scores.copy(deep=True)

    df_scores_normalized["aic"] = normalize(df_scores_normalized["aic"].values)
    df_scores_normalized["bic"] = normalize(df_scores_normalized["bic"].values)
    df_scores_normalized["wcss"] = normalize(df_scores_normalized["wcss"].values)
    df_scores_normalized["sil_euc"] = normalize(df_scores_normalized["sil_euc"].values)
    df_scores_normalized["sil_mah"] = normalize(df_scores_normalized["sil_mah"].values)
    df_scores_normalized["gap"] = normalize(df_scores_normalized["gap"].values)
    df_scores_normalized["dbs"] = normalize(df_scores_normalized["dbs"].values)
    df_scores_normalized["chs"] = normalize(df_scores_normalized["chs"].values)

    df_scores_normalized = df_scores_normalized.round(2)

    return df_scores_normalized
