import glob
import os
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from . import env, stats, utils, metrics as mets
from .gmm import GaussianMixtureModel

# get logger
logger = env.get_logger()


def get_model_path(
    catalog_name: str,
    features: List[str],
    n_components: int,
    subdir: Optional[str] = None,
) -> str:
    """
    Construct the path where the model should be located.

    Parameters:
    - model_name (str): Name of the model.
    - catalog_name (str): Catalog name
    - features (List of string): List of features
    - n_components (int): Number of clusters in the model.
    - subdir (str, optional): A subdirectory under the main models directory.

    Returns:
    - str: Complete path to the model.
    """

    # features as text
    features_txt = "-".join(features)

    # get model name
    model_name = get_model_name(catalog_name, features, n_components)

    # create path
    model_path = os.path.join(
        env.DIR_MODELS,
        catalog_name,
        features_txt,
        f"{n_components}G",
        subdir or "",
        model_name,
    )

    # add the extension if not exists
    if not model_path.endswith(".model"):
        model_path += ".model"

    return model_path


def get_model_name(catalog_name: str, features: List[str], n_components: int) -> str:
    """
    Generates a model name based on provided parameters.

    Parameters:
    - catalog_name (str): Name of the catalog.
    - features (List[str]): List of features.
    - n_components (int): Number of components.

    Returns:
    - str: A string representing the model name.
    """
    feature_string = "-".join(features)
    return f"{catalog_name}_{feature_string}_{n_components}G"


def parse_model_name(model_name: str) -> dict:
    """
    Parse model name to retrieve its components.

    Parameters:
    - model_name (str): Name of the model.

    Returns:
    - dict: Dictionary containing the model's properties.
    """
    model_name = model_name.rstrip(".model")
    tokens = model_name.split("_")

    return {
        "catalog_name": tokens[0],
        "features": tokens[1].split("-"),
        "n_components": int(tokens[2][:-1]),
    }


def is_model_exists(model_name: str) -> bool:
    """
    Check if a model exists on disk.

    Parameters:
    - model_name (str): Name of the model.

    Returns:
    - bool: True if the model exists, False otherwise.
    """
    # get model path
    path = get_model_path(**parse_model_name(model_name))

    return os.path.exists(path)


def get_model_params(model: GaussianMixtureModel) -> dict:
    """
    Get the parameters of the given model.

    Parameters:
    - model (GaussianMixtureModel): The model to get the parameters from.

    Returns:
    - dict: A dictionary containing the parameters of the model.
    """
    # raise exception if the model is not fitted
    if not model.converged_:
        raise ValueError("The model is not fitted yet.")

    return {
        "n_components": model.n_components,
        "max_iter": model.max_iter,
        "n_init": model.n_init,
        "init_params": model.init_params,
        "weights_init": model.weights_init,
        "means_init": model.means_init,
        "precisions_init": model.precisions_init,
        "random_state": model.random_state,
        "warm_start": model.warm_start,
        "verbose": model.verbose,
        "verbose_interval": model.verbose_interval,
        "sort_clusters": model.sort_clusters,
    }


def save_model(
    model: GaussianMixtureModel,
    catalog_name: str,
    features: List[str],
    n_components: int,
):
    """
    Save the model to disk.

    Parameters:
    - model (GaussianMixtureModel): The model to save.
    - catalog_name (str): Name of the catalog.
    - features (List[str]): List of features.
    - n_components (int): Number of components.
    """

    # get model name
    model_name = get_model_name(
        catalog_name=catalog_name,
        features=features,
        n_components=n_components,
    )

    # set model name (for any case)
    model.set_name(model_name)

    # get model path
    path = get_model_path(catalog_name, features, n_components)

    # before saving, check if the directory exists
    utils.create_directory(os.path.dirname(path))

    # save model
    model.save(path)


def load_model(
    catalog_name: str,
    features: List[str],
    n_components: int,
) -> GaussianMixtureModel:
    """
    Load a specific model from disk.

    Parameters:
    - catalog_name (str): Name of the catalog.
    - features (List[str]): List of features.
    - n_components (int): Number of components.

    Returns:
    - GaussianMixtureModel: Loaded model.
    """

    # get model path
    path = get_model_path(catalog_name, features, n_components)

    return GaussianMixtureModel.load(path)


def load_all_models(
    catalog_name: Optional[str] = None,
    features: Optional[List[str]] = None,
    n_components: Optional[int] = None,
) -> List[GaussianMixtureModel]:
    """
    Load all Gaussian Mixture Models (GMMs) based on filter criteria from a specified directory.

    Parameters:
    - catalog_name (str, optional): Filter models based on the catalog name.
    - features (list[str], optional): Filter models that were trained using the given features.
    - n_components (int, optional): Filter models based on the number of components used in the GMM.

    Returns:
    - list[GaussianMixtureModel]: A list of loaded GMMs.
    """
    # Initialize features if not provided
    if features is None:
        features = []

    # Construct the search path for models
    search_path = os.path.join(env.DIR_MODELS, "**", "*.model")
    # Get all the .model file paths
    files = [f for f in glob.glob(search_path, recursive=True) if os.path.isfile(f)]
    # remove experimental models
    files = [f for f in files if not os.path.dirname(f).endswith("experiments")]

    # parse all model names
    filtered_model_dicts = [parse_model_name(os.path.basename(f)) for f in files]

    # filter models by catalog name if provided
    if catalog_name is not None:
        filtered_model_dicts = [
            m for m in filtered_model_dicts if m["catalog_name"] == catalog_name
        ]

    # filter models by features if provided
    if len(features) > 0:
        filtered_model_dicts = [
            m for m in filtered_model_dicts if set(m["features"]) == set(features)
        ]

    # filter models by n_components if provided
    if n_components is not None:
        filtered_model_dicts = [
            m for m in filtered_model_dicts if m["n_components"] == n_components
        ]

    # Sort list based on catalog name, features, n_components
    sorted_model_dicts = sorted(
        filtered_model_dicts,
        key=lambda x: (
            x["catalog_name"],
            x["features"],
            x["n_components"],
        ),
    )

    # Load all the sorted models
    loaded_models = [
        load_model(
            catalog_name=model_dict["catalog_name"],
            features=model_dict["features"],
            n_components=model_dict["n_components"],
        )
        for model_dict in sorted_model_dicts
    ]

    return loaded_models


def create_models(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_components: Union[int, List[int]] = 2,
    max_iter: int = 10000,
    n_init: int = 100,
    n_trials: int = 1,
    sort_clusters: bool = True,
) -> List[GaussianMixtureModel]:
    """
    Create Gaussian Mixture Models based on the provided dataset and parameters.

    Parameters:
    - df (pd.DataFrame): The input data for training the models.
    - features (List[str], optional): Features from the DataFrame to be used. If not provided, all columns except 'is_outlier' will be used.
    - n_components (Union[int, List[int]]): The number of components for the GMM. Can be a single integer or a list.
    - max_iter (int): Maximum number of iterations.
    - n_init (int): Number of times the algorithm will be run with different centroid seeds.
    - n_trials (int): Number of trials for creating the model.
    - sort_clusters (bool): Whether to sort the clusters.

    Returns:
    - List[GaussianMixtureModel]: List of created models.
    """

    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("Provided dataframe is empty.")

    # Check for outliers
    if "is_outlier" in df.columns:
        logger.warning("Outliers are not removed from the dataset.")

    # Determine features
    if not features:
        # get features to be used
        features = [col for col in df.columns if not "is_outlier" in col]
    else:
        # check if all features exist
        for f in features:
            if f not in df.columns:
                raise ValueError(f"Feature '{f}' does not exist.")

    # Extract data for modeling
    X = df[features].values

    # if the data is 1-dimensional, reshape it
    if len(X.shape) == 1:
        X = X.flatten()

    # Standardize n_components into a list
    if isinstance(n_components, int):
        n_components = [n_components]
    else:
        n_components = sorted(list(n_components))

    # Get catalog name
    if not df.index.name:
        raise ValueError(
            "Make sure the dataframe has an index name corresponding to the catalog name."
        )
    catalog_name = df.index.name

    # create models
    models = []
    for n_component in n_components:
        for n_trial in range(1, n_trials + 1):
            # create custom name for GMM
            custom_name = f"[GMM {catalog_name} ({', '.join(features)}) (C={n_component}) (N={n_trial}))]"

            # create model
            model = GaussianMixtureModel(
                n_components=n_component,
                max_iter=max_iter,
                n_init=n_init,
                sort_clusters=sort_clusters,
                model_name=custom_name,
            ).fit(X)

            models.append(model)

    # return created models
    return models


def get_model_scores(
    catalog_name: Optional[str] = None,
    features: Optional[List[str]] = None,
    n_components: Optional[int] = None,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetches the scores of models based on provided filters.

    Parameters:
    - catalog_name (Optional[str]): Filter the scores based on the catalog name.
    - features (Optional[List[str]]): Filter the scores based on a list of features.
    - n_components (Optional[int]): Filter the scores based on the number of components.
    - metrics (Optional[List[str]]): If provided, filter columns to show only these metrics.

    Returns:
    - pd.DataFrame: A dataframe containing model scores based on provided filters.
    """

    # default columns in the dataframe
    meta_cols = ["catalog", "features", "n_components"]
    # get all available metric names
    metric_cols = list(mets.get_clustering_metrics().keys())
    # insert 'gap_err' right after 'gap'
    if "gap" in metric_cols:
        metric_cols.insert(metric_cols.index("gap") + 1, "gap_err")

    # get metrics to be shown
    if metrics is None:
        metrics = list(metric_cols)
    # insert 'gap_err' right after 'gap'
    if "gap" in metrics and not "gap_err" in metrics:
        metrics.insert(metrics.index("gap") + 1, "gap_err")

    # Define path to the scores file
    path = os.path.join(env.DIR_MODELS, "model_scores.xlsx")
    # Initialize DataFrame from the file or create a new one if the file doesn't exist
    if os.path.exists(path):
        df = pd.read_excel(path, sheet_name="scores", index_col=0)
    else:
        df = pd.DataFrame(columns=meta_cols + metric_cols)
        df.to_excel(path, sheet_name="scores", index=True, header=True)

    # Filter operations
    if catalog_name:
        df = df[df["catalog"] == catalog_name]
    if features:
        df = df[df["features"] == "-".join(features)]
    if n_components:
        df = df[df["n_components"] == n_components]

    return df[meta_cols + metrics]


def get_model_scores_by_name(
    model_name: str,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetches the scores of models based on provided filters.

    Parameters:
    - model_name (str): Model name.
    - metrics (Optional[List[str]]): If provided, filter columns to show only these metrics.

    Returns:
    - pd.DataFrame: A dataframe containing model scores based on provided filters.
    """

    # get model attrbs
    model_attrs = parse_model_name(model_name)

    return get_model_scores(
        catalog_name=model_attrs["catalog_name"],
        features=model_attrs["features"],
        n_components=model_attrs["n_components"],
        metrics=metrics,
    )


def save_model_scores(model_name: str, scores: Dict[str, float]) -> None:
    """
    Save the evaluation scores of a given model to an excel file.

    This function saves the scores of a model in an existing excel file, updating or appending rows as necessary.
    It also includes metadata parsed from the model name, like catalog, features and number of components.

    Parameters:
    -----------
    model_name : str
        The name of the model, which will be parsed to extract meta-information like catalog, features, etc.

    scores : Dict[str, float]
        A dictionary containing evaluation metrics as keys and their respective scores as values.

    Returns:
    --------
    None

    """

    if not model_name:
        raise ValueError("Model name is not provided.")

    # Fetch existing scores.
    df = get_model_scores()
    # parse model name
    model_attrs = parse_model_name(model_name)
    df.loc[model_name, "catalog"] = model_attrs["catalog_name"]
    df.loc[model_name, "features"] = "-".join(model_attrs["features"])
    df.loc[model_name, "n_components"] = model_attrs["n_components"]

    # If a new metric (column) is being introduced, add it to the DataFrame.
    for key in scores.keys():
        if key not in df.columns:
            df[key] = np.nan

    # Update scores and model attributes in the dataframe.
    for key, value in scores.items():
        df.loc[model_name, key] = value

    # Get the path where scores are saved.
    path = os.path.join(env.DIR_MODELS, "model_scores.xlsx")

    # Ensure the directory structure exists.
    utils.create_directory(os.path.dirname(path))

    # Sort rows based on catalog name, features, n_components.
    df.sort_values(by=["catalog", "features", "n_components"], inplace=True)

    # Save the dataframe as an excel file.
    df.round(4).to_excel(
        path,
        sheet_name="scores",
        index=True,
        header=True,
        freeze_panes=(1, 0),
        engine="xlsxwriter",
    )


def evaluate_model(
    model: GaussianMixtureModel,
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    save_scores: bool = False,
) -> Dict[str, float]:
    """
    Evaluates a Gaussian Mixture Model using specified metrics and dataset.

    Parameters:
    - model (GaussianMixtureModel): The model to be evaluated.
    - df (pd.DataFrame): The data frame containing the features.
    - features (Optional[List[str]]): List of feature columns to use. Defaults to all columns except 'is_outlier'.
    - metrics (Optional[List[str]]): List of metrics to be used for evaluation. If None, all metrics will be used.
    - save_scores (bool): If True, saves the scores to a file. Default is True.

    Returns:
    - Dict[str, float]: Dictionary of scores for each metric.

    Raises:
    - ValueError: If the model is not fitted or if an invalid metric is provided.
    """
    # Check if the model is fitted
    if not model.converged_:
        raise ValueError("Model is not fitted.")

    # Extract features from DataFrame
    features = features or [col for col in df.columns if "is_outlier" not in col]
    X = df[features].values

    # get model attributes
    model_attrs = parse_model_name(model.model_name)

    # ensure that model is created using the same catalog
    if df.index.name and model_attrs["catalog_name"] != df.index.name:
        raise ValueError(
            f"Model is created using catalog '{model_attrs['catalog_name']}' but '{df.index.name}' is provided."
        )

    # ensure that the model is created using the same features
    if set(model_attrs["features"]) != set(features):
        raise ValueError(
            f"Model is created using features '{model_attrs['features']}' but '{features}' is provided."
        )

    # Predict clusters
    y_pred = model.predict(X)

    # get requested metrics
    metric_defs = mets.get_clustering_metrics(metric_names=metrics, verbose=False)

    if "gap" in metric_defs:
        # Special handling for gap statistic due to multiple return values
        def gap_statistic():
            # create a copy of the model
            model_copy = deepcopy(model)
            # set some params
            model_copy.n_init = 1

            result = stats.gap_statistics(X, y_pred, clusterer=model_copy, n_repeat=100)
            return result["gap"], result["err"]

        metric_defs["gap"]["func"] = gap_statistic

    # Create a dictionary to store the scores
    scores = {}
    # Compute scores
    for metric in metric_defs:
        try:
            if metric == "gap":
                result = metric_defs[metric]["func"]()
                scores[metric], scores[metric + "_err"] = result
            elif metric in ("aic", "bic"):
                scores[metric] = metric_defs[metric]["func"](model, X)
            else:
                scores[metric] = metric_defs[metric]["func"](X, y_pred)
        except KeyError:
            logger.error(f"Invalid metric: '{metric}'")

    # Save scores if needed
    if save_scores:
        try:
            save_model_scores(model.model_name, scores)
        except Exception as e:
            logger.error(f"Failed to save model scores due to: {e}")

    return scores
