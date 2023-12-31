import os
import pickle
from typing import Any, Literal, Dict, Union

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike
from sklearn.mixture import GaussianMixture as _GMM

from . import env

# get logger
logger = env.get_logger()


class GaussianMixtureModel(_GMM):
    """
    Inherited version of sklearn.mixture.GaussianMixture that allows for re-mapping of cluster
    indices. This is useful for when you want to re-assign the label of each cluster.
    If the sort_clusters parameter is set to True, then the clusters are sorted based on the means of the components,
    right after the model is fitted.
    """

    def __init__(
        self,
        n_components: int = 1,
        *,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        tol: float = 0.001,
        reg_covar: float = 0.000001,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: Literal[
            "kmeans", "k-means++", "random", "random_from_data"
        ] = "kmeans",
        weights_init: Union[ArrayLike, None] = None,
        means_init: Union[ArrayLike, None] = None,
        precisions_init: Union[ArrayLike, None] = None,
        random_state: Union[int, RandomState, None] = None,
        warm_start: bool = False,
        verbose: int = 0,
        verbose_interval: int = 10,
        model_name: str = None,
        sort_clusters: bool = True,
    ) -> None:
        super().__init__(
            n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        # flag to sort clusters based on the means of the components
        self.sort_clusters = sort_clusters

        # set the model name
        if not model_name:
            self.model_name = f"C{n_components}"
        else:
            self.model_name = model_name

    def __str__(self) -> str:
        return f"<GMM {self.model_name}>"

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        return self.__str__()

    def set_name(self, name: str) -> None:
        """
        Set the name of the model.
        """
        self.model_name = name

    def remap_labels(self, label_map: Dict[int, int]) -> None:
        """
        Remaps the cluster labels based on a given label map.

        Args:
            label_map (Dict[int, int]): A dictionary mapping existing cluster labels to new cluster labels.

        Raises:
            ValueError: If the label_map does not contain a mapping for each cluster.
            ValueError: If the model is not fitted yet.

        Notes:
            - This method updates the parameters of the Gaussian Mixture Model (GMM) instance
            to reflect the new cluster labels.
            - The clusters are identified by integer labels starting from 0 to n_components-1,
            where n_components is the total number of clusters in the GMM.
            - The label_map should provide a mapping for each existing cluster label,
            and the new cluster labels should also cover all the clusters.

        Example:
            label_map = {0: 1, 1: 2, 2: 0}
            gmm.remap_labels(label_map)
        """

        # Raise an exception if the model is not fitted
        if not self.converged_:
            raise ValueError("The model is not fitted yet.")

        # get the set of labels
        label_set = set(range(self.n_components))

        # Check if label_map contains a mapping for each cluster
        if set(label_map.keys()) != label_set:
            raise ValueError(
                "label_map must contain a mapping for each existing cluster"
            )

        # Check if label_map contains a mapping for each cluster
        if set(label_map.values()) != label_set:
            raise ValueError("label_map must contain a mapping for each new cluster")

        # Create a tuple from the label_map
        label_map_tuple = sorted(label_map.items(), key=lambda x: x[1])
        # Get the list of the new cluster labels
        new_cluster_labels = [x[0] for x in label_map_tuple]

        # Update the GMM parameters to reflect the new cluster labels
        self.weights_ = self.weights_[new_cluster_labels]
        self.means_ = self.means_[new_cluster_labels]

        # update the matrices if the covariance type is not tied
        # i.e. if the covariance type is tied, then the covariance matrix is the same for all clusters
        if self.covariance_type != "tied":
            self.covariances_ = self.covariances_[new_cluster_labels]
            self.precisions_ = self.precisions_[new_cluster_labels]
            self.precisions_cholesky_ = self.precisions_cholesky_[new_cluster_labels]

    def sort_clusters_by_means(self):
        """
        Sorts the clusters of a Gaussian Mixture Model (GMM) based on the means of the components.

        Args:
            gmm_model: A trained instance of Gaussian Mixture Model (sklearn.mixture.GaussianMixture).

        Returns:
            None. Modifies the labels of the GMM model in-place.
        """
        logger.debug("Sorting clusters by means.")

        # Raise an exception if the model is not fitted
        if not self.converged_:
            raise ValueError("The model is not fitted yet.")

        # Get means and their corresponding component indices
        means = [(i, mean) for i, mean in enumerate(self.means_)]

        # Sort the means based on the first dimension of each mean vector
        means.sort(key=lambda x: x[1][0])

        # Create a label mapping dictionary using the sorted means
        label_map = {means[i][0]: i for i in range(len(means))}

        # Remap the labels of the GMM model using the label mapping
        self.remap_labels(label_map)

    def fit(self, X: ArrayLike, y: Any = None) -> "GaussianMixtureModel":
        """
        Fits a Gaussian Mixture Model (GMM) to the given data.

        Args:
            X (ArrayLike): The data to fit the GMM to.
            y (Any, optional): Ignored. Defaults to None.

        Returns:
            GaussianMixtureModel: The fitted GMM model.

        Notes:
            - This method overrides the fit method of the sklearn.mixture.GaussianMixture class.
            - If the sort_clusters parameter is set to True, then the clusters are sorted based on the means of the components.
        """

        # print log
        logger.debug(f"Fitting model: {self.__str__()}")
        # fit the model
        super().fit(X, y)
        # then, sort the clusters if required
        if self.sort_clusters:
            self.sort_clusters_by_means()

        return self

    def save(self, file_path: str):
        """
        Save the GaussianMixtureModel to a file using pickle.

        Parameters:
        - file_path (str): Path to the file where the model will be saved.

        Raises:
        - ValueError: If the model hasn't been fitted yet.
        """

        # Check if the model has been fitted
        if not self.converged_:
            raise ValueError("The model is not fitted yet.")

        # Check if the file already exists
        file_exists = os.path.isfile(file_path)

        # Save the model
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

        # Log information about the operation
        if file_exists:
            logger.warning(f"Model overwritten at: {file_path}")
        else:
            logger.info(f"Model saved to: {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "GaussianMixtureModel":
        """
        Load a GaussianMixtureModel from a file.

        Parameters:
        - file_path (str): Path to the file from which the model will be loaded.

        Returns:
        - GaussianMixtureModel: Loaded model.

        Raises:
        - FileNotFoundError: If the specified file doesn't exist.
        """

        # Ensure the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at: '{file_path}'")

        # Load the model
        with open(file_path, "rb") as file:
            loaded_model = pickle.load(file)

        return loaded_model

    def get_component_params(self) -> dict:
        """ """
        # for each clusters
        component_params = {}
        for i in range(self.n_components):
            component_params[i] = {
                "weight": self.weights_[i],
                "mean": self.means_[i],
            }
            if self.covariance_type == "spherical":
                component_params[i]["covariance"] = np.diag(
                    self.covariances_[i], self.covariances_[i]
                )
            elif self.covariance_type == "tied":
                component_params[i]["covariance"] = self.covariances_.copy()

            elif self.covariance_type == "diag":
                component_params[i]["covariance"] = np.diag(self.covariances_[i])

            elif self.covariance_type == "full":
                component_params[i]["covariance"] = self.covariances_[i].copy()

        return component_params
