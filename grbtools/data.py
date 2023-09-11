"""
Module for data reading and preprocessing tasks.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd

from . import env, utils, disp

# get logger
logger = env.get_logger()


def get_catalogue_path(cat_name: str) -> str:
    """
    returns path to catalogue
    """
    return os.path.join(env.DIR_CATALOGS, cat_name + ".xlsx")


def get_dataset_path(cat_name: str) -> str:
    """
    returns path to dataset
    """
    return os.path.join(env.DIR_DATASETS, cat_name + ".xlsx")


def remove_invalid_values(
    dataframe: pd.DataFrame, subset=None, method="any"
) -> pd.DataFrame:
    """Remove NaN and inf values from the DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): Input DataFrame.
    - subset (list): List of columns to consider for NaN removal.
    - method (str): If 'any', drop rows containing any NaN value, if 'all', drop rows where all values are NaN.
    - infs (bool): If True, drop inf values as well.
    Returns:
    - pd.DataFrame: DataFrame without NaN and inf values.
    """

    # replace inf values with NaN
    dataframe = dataframe.replace((np.inf, -np.inf), np.nan)

    return dataframe.dropna(subset=subset, how=method)


def load_dataset(
    catalog_name: str,
    features: list = [],
    remove_invalids: bool = True,
    remove_outliers: bool = True,
    plot_data: bool = False,
    plot_kwargs: Dict = dict(),
    verbose=True,
) -> pd.DataFrame:
    """
    Load a given catalog, optionally removing outliers and invalid values, and plotting data.

    Parameters:
    - catalog_name (str): Name of the catalog file without extension.
    - features (list): List of features/columns to retain.
    - remove_invalids (bool): Whether to drop rows containing inf and nan values.
    - remove_outliers (bool): Whether to exclude rows marked as outliers.
    - plot_kwargs (dict, optional): Keyword arguments for plotting method. If None, data is not plotted. Default is None.
    - verbose (bool): Whether to print additional info.

    Returns:
    - pd.DataFrame: Loaded and processed DataFrame.
    """

    # get dataset path
    path = get_dataset_path(catalog_name)

    # return if dataset does not exist
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset does not exist. Path: '{path}'")

    # load dataset
    df = pd.read_excel(path, index_col=0, sheet_name="data")

    # get only the specified features
    if features:
        # get outlier column name
        outlier_col = "is_outlier_" + "-".join(features)

        # get outlier flags
        is_outlier = None
        # check if outlier column exists
        if outlier_col in df.columns:
            is_outlier = df[outlier_col].values == True
        # if not, disable outlier removal
        elif remove_outliers:
            logger.warning(
                f"Outlier column '{outlier_col}' does not exist. Skipping outlier removal."
            )
            remove_outliers = False

        # get only specified features
        df = df[features]

        # remove outliers
        if remove_outliers:
            df = df.loc[~is_outlier, :]

        # otherwise, add outlier column
        elif is_outlier is not None:
            df["is_outlier"] = is_outlier

    # remove invalid values
    if remove_invalids:
        df = remove_invalid_values(df, subset=features)

    # set the index name
    df.rename_axis(catalog_name)

    # print info
    if verbose:
        logger.info(
            f">>> Dataset {catalog_name.upper()} loaded with features: {features}"
        )
        logger.info(f"  > Number of GRBs: {len(df)}")
        if "is_outlier" in df.columns:
            logger.info(f"  > Number of outliers: {df['is_outlier'].sum()}")
        # logger.info("")
        # logger.info(f">>> Descriptive stats: \n{df.describe().round(2)}")

    # plot data if arg is set to true
    if plot_data:
        disp.plot_data(df, features, **plot_kwargs)

    return df


def save_dataframe(df: pd.DataFrame, catalog_name: str):
    """
    saves dataframe as excel file
    """
    # create directory if it doesn't exist
    utils.create_directory(env.DIR_DATASETS)
    # set the index name
    df.rename_axis(catalog_name, inplace=True)
    # get path
    path = os.path.join(env.DIR_DATASETS, catalog_name + ".xlsx")
    # save as excel file
    df.to_excel(path, sheet_name="data", freeze_panes=(1, 0), engine="xlsxwriter")
