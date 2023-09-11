import os
import numpy as np
import pandas as pd


def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Parameters:
    - path (str): Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def merge_scores(scores={}):
    n_clusters = list(scores.keys())
    n_clusters.sort()
    scores = {i: scores[i] for i in n_clusters}

    df_scores = pd.DataFrame(scores).round(2)
    df_scores["k"] = range(1, len(df_scores) + 1)
    df_scores.set_index("k", inplace=True)

    return df_scores
