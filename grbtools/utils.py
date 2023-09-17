import os
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from pyriemann.utils.mean import mean_riemann


def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Parameters:
    - path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def match_gaussian_components(
    mean_list_1st: np.ndarray, mean_list_2nd: np.ndarray
) -> Dict[int, int]:
    """
    Match Gaussian components from two models based on the absolute distance between their values.

    The function computes a distance matrix between the two sets of components and then
    uses the Hungarian algorithm to solve the assignment problem, ensuring an optimal
    matching between the components.

    Parameters:
    - mean_list_1st (np.ndarray): An array of values representing the means of components of 1st group.
    - mean_list_2nd (np.ndarray): An array of values representing the means of components of 2nd group.

    Returns:
    - Dict[int, int]: A dictionary where keys represent indices from the first group and
                      values represent the corresponding matched indices from the second group.
    """

    # Ensure both component groups are of the same size
    assert len(mean_list_1st) == len(
        mean_list_2nd
    ), "Both component groups should have the same size."

    # Calculate the distance matrix between the two sets of components
    distance_matrix = np.array(
        [[np.linalg.norm(s - f) for f in mean_list_1st] for s in mean_list_2nd]
    )
    distance_matrix = distance_matrix.reshape(len(mean_list_1st), len(mean_list_2nd))

    # Use the Hungarian algorithm to find the optimal assignment between components
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Return the matched component pairs as a dictionary
    return dict(zip(row_indices, col_indices))


def compute_avg_covariance(cov_matrix_list: np.ndarray) -> np.ndarray:
    """
    Compute the Riemannian mean of a list of covariance matrices using the pyriemann module.

    Parameters:
    - cov_matrix_list (np.ndarray): A list or array of covariance matrices.
                                   Expected shape is (n_matrices, n_channels, n_channels).

    Returns:
    - np.ndarray: The Riemannian mean of the provided covariance matrices.
    """
    return mean_riemann(cov_matrix_list)


def merge_scores(scores={}):
    n_clusters = list(scores.keys())
    n_clusters.sort()
    scores = {i: scores[i] for i in n_clusters}

    df_scores = pd.DataFrame(scores).round(2)
    df_scores["k"] = range(1, len(df_scores) + 1)
    df_scores.set_index("k", inplace=True)

    return df_scores
