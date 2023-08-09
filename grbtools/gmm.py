from typing import Any, Literal, Dict, Union

from numpy.random import RandomState
from numpy.typing import ArrayLike
from sklearn.mixture import GaussianMixture


class GaussianMixtureModel(GaussianMixture):
    """
    Inherited version of sklearn.mixture.GaussianMixture that allows for
    re-mapping of cluster indices. This is useful for when you want to re-assign the label of each cluster.
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
        verbose_interval: int = 10
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
            raise ValueError("label_map must contain a mapping for each existing cluster")
    
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
        if self.covariance_type != 'tied':
            self.covariances_ = self.covariances_[new_cluster_labels]
            self.precisions_ = self.precisions_[new_cluster_labels]
            self.precisions_cholesky_ = self.precisions_cholesky_[new_cluster_labels]

