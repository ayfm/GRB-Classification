from typing import Dict, Iterable, Literal, Optional, Union
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score as dbs, calinski_harabasz_score as chs
from copy import deepcopy 
from sklearn.neighbors import NearestNeighbors

Matrixlike = Union[np.ndarray, np.matrix, Iterable[Iterable[float]]]



def compute_mahalanobis_distance(
    X: ArrayLike,
    mean: Union[ArrayLike, None] = None,
    covar: Union[ArrayLike, None] = None,
) -> ArrayLike:
    """
    Calculates the Mahalanobis distance between data points and means.

    Args:
        X (ArrayLike): Input data of shape (N, d), where N is the number of data points and d is the number of dimensions.
        mean (Union[ArrayLike, None], optional): Mean values to use for computing the distance. If None, the mean will be calculated from the input data. Defaults to None.
        covar (Union[ArrayLike, None], optional): Covariance matrix to use for computing the distance. If None, the covariance matrix will be calculated from the input data. Defaults to None.

    Returns:
        np.ndarray: Array of shape (1, N) containing the Mahalanobis distances between each data point and the mean.

    Raises:
        AssertionError: If the shape of the mean or covariance matrix is not correct.

    Notes:
        - The Mahalanobis distance is a measure of the distance between a data point and a distribution, taking into account the covariance structure of the distribution.
        - If mean is None, it is calculated as the mean of the input data along the specified axis.
        - If covar is None, it is calculated as the covariance matrix of the input data.
    """

    # check if mean is given
    if mean is None:
        mean = np.mean(X, axis=0)
    # check if covar is given
    if covar is None:
        covar = np.cov(X.T)

    # get size of the data
    N, d = X.shape

    # check dimensions
    assert mean.shape in ((1, d), (d,)), "Mean shape is not correct"
    assert covar.shape == (d, d), "Covariance shape is not correct"

    # reshape mean
    mean = mean.reshape(1, -1)

    # get inverse covariance matrix
    icovar = np.linalg.inv(covar)
    # calculate mahalanobis distance of each data point
    mah_dist = cdist(X, mean, VI=icovar, metric="mahalanobis").reshape(1, -1)

    # return distance
    return mah_dist


def silhouette_samples_mahalanobis(
    X: ArrayLike,
    labels: ArrayLike,
    means: Union[ArrayLike, None] = None,
    covars: Union[ArrayLike, None] = None,
) -> ArrayLike:
    """
    Calculate the Mahalanobis-based silhouette coefficients for clustered data.

    Parameters:
        X (ArrayLike): Input data array of shape (N, d), where N is the number of samples and d is the number of features.
        labels (ArrayLike): Array of cluster assignments for each data point, of shape (N,).
        means (Union[ArrayLike, None], optional): Cluster mean vectors. If None, they will be calculated from the data. 
            Must have shape (n_clusters, d), where n_clusters is the number of clusters.
        covars (Union[ArrayLike, None], optional): Cluster covariance matrices. If None, they will be calculated from the data. 
            Must have shape (n_clusters, d, d), where n_clusters is the number of clusters.

    Returns:
        np.ndarray: Array of silhouette coefficients for each data point, of shape (N, 1).

    Raises:
        AssertionError: If the dimensions of means or covars are incorrect.

    """

    # get unique labels
    cluster_labels = np.sort(np.unique(labels))
    # how many clusters?
    n_clusters = len(cluster_labels)

    # if there is only one cluster, we cannot calculate silhouette coefficients
    if n_clusters == 1:
        raise ValueError("Cannot calculate silhouette coefficients for one cluster")

    # check if cluster means' are given
    if means is None:
        means = np.array([np.mean(X[labels == i], axis=0) for i in cluster_labels])
    # check if cluster covariances' are given
    if covars is None:
        covars = np.array([np.cov(X[labels == i].T) for i in cluster_labels])

    # get size of the data
    N, d = X.shape

    # check dimensions of means
    # The dimension must be (n_clusters, d)
    assert means.shape == (n_clusters, d), "Means shape is not correct"

    # check dimensions of covars
    # The dimension must be (n_clusters, d, d)
    assert covars.shape == (n_clusters, d, d), "Covariances shape is not correct"

    # create distance array
    mah_dist_arr = np.zeros((N, n_clusters))

    # calculate mahalanobis distance for each cluster
    for i in range(n_clusters):
        # calculate mahalanobis distance based on the mean and covariance matrix
        mah_dist_arr[:, i] = compute_mahalanobis_distance(
            X, mean=means[i], covar=covars[i]
        )

    # calculate intra-cluster distance (a(x))
    a_x = np.array([mah_dist_arr[i, labels[i]] for i in range(N)]).reshape(-1, 1)

    # Replace Mahalanobis distances for own clusters with infinity
    for i in range(N):
        mah_dist_arr[i, labels[i]] = np.inf
        
    # calculate inter-cluster distance (b(x)) as minimum distance to other clusters
    b_x = mah_dist_arr.min(axis=1).reshape(-1, 1)

    # calculate silhouette coeffs
    sil_coeffs = (b_x - a_x) / np.hstack((b_x, a_x)).max(axis=1).reshape(-1, 1)

    # return silhouette coeffs
    return sil_coeffs



def silhouette_score(X: ArrayLike, labels: ArrayLike, metric: Literal["Euclidean", "Mahalanobis"] = None):
    """
    Calculates the silhouette score for a clustering.

    The silhouette score measures how well each sample in a cluster is assigned to its own cluster compared to other clusters.
    It provides an indication of the quality of the clustering results.

    Args:
        X (ArrayLike): The input data samples.
        labels (ArrayLike): The cluster labels assigned to each sample.
        metric ({"Euclidean", "Mahalanobis"}, optional): The distance metric to be used. Defaults to "Euclidean".

    Returns:
        Dict: A dict containing the mean silhouette coefficient and an array of sample silhouette coefficients.

    Raises:
        ValueError: If the metric is not one of {"Euclidean", "Mahalanobis"}.

    """
    # how many clusters?
    n_clusters = len(set(labels))

    # if there is only one cluster, we cannot calculate silhouette score
    if n_clusters == 1:
        return {'mean': np.nan, 'coeffs': np.nan}

    # default metric is Euclidean
    if metric is None:
        metric = "Euclidean"

    if metric == "Euclidean":
        sample_coeffs = silhouette_samples(X, labels, metric="euclidean")

    elif metric == "Mahalanobis":
        sample_coeffs = silhouette_samples_mahalanobis(X, labels)

    else:
        raise ValueError("Metric must be either 'Euclidean' or 'Mahalanobis'")

    # calculate mean score
    mean_coeff = np.mean(sample_coeffs)

    return {'mean': mean_coeff, 'coeffs': sample_coeffs}



def dispersion(X: ArrayLike, labels: ArrayLike):
    clusters = np.unique(labels)
    dispersion = 0
    for cluster in clusters:
        cluster_points = X[labels == cluster]
        centroid = cluster_points.mean(axis=0)
        dispersion += ((cluster_points - centroid) ** 2).sum()
    return dispersion

def gap_statistics(X: ArrayLike, labels:ArrayLike, clusterer=None, n_repeat=10, random_state=None):
    # if clusterer is not specified, use KMeans with the same number of clusters as the labels
    if clusterer is None:
        clusterer = KMeans(n_clusters=len(np.unique(labels)))
    # otherwise, work with a clone of the clusterer
    else:
        clusterer = deepcopy(clusterer)
    

    # Compute the gap statistic
    ref_disps = np.zeros(n_repeat)

    # set the random state
    np.random.seed(random_state)

    for i in range(n_repeat):
        # Generate a reference dataset
        ref_X = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), X.shape)

        # Fit the model to the reference data and get the labels
        ref_labels = clusterer.fit_predict(ref_X)

        # Compute the dispersion for the reference data
        ref_disp = dispersion(ref_X, ref_labels)
        ref_disps[i] = np.log(ref_disp)

    # Compute the dispersion for the original data
    orig_disp = dispersion(X, labels)
    orig_disp = np.log(orig_disp)

    # Compute the gap statistic
    gap = np.mean(ref_disps) - orig_disp
    gap_err = np.std(ref_disps) * np.sqrt(1 + 1/n_repeat)
    
    return {'gap': gap, 'gap_err': gap_err}


def davies_bouldin_score(X: ArrayLike, labels: ArrayLike) -> float:
    # how many clusters?
    n_clusters = len(set(labels))
    # if there is only one cluster, we cannot calculate Davies-Bouldin score
    if n_clusters == 1:
        return np.nan
    
    return dbs(
        X, labels
    )

def calinski_harabasz_score(X: ArrayLike, labels: ArrayLike) -> float:
    # how many clusters?
    n_clusters = len(set(labels))
    # if there is only one cluster, we cannot calculate Calinski-Harabasz score
    if n_clusters == 1:
        return np.nan
    
    return chs(
        X, labels
    )

def hopkins_statistic(X: ArrayLike, sample_ratio: float = 0.05, random_state = None) -> float:
    """
    Calculates the Hopkins statistic - a statistic which indicates the cluster tendency of data.

    Parameters:
    X (ArrayLike): The dataset.
    sample_ratio (float): The ratio of datapoints to sample for calculating the statistic.
    random_state (int): The random seed for sampling.

    Returns:
    H (float): The Hopkins statistic, between 0 and 1. A value near 1 tends to indicate the data is highly clusterable.
    
    Notes:

       - If the value of H is close to 1, then the data is highly clusterable, and not uniformly distributed. This means it's a good candidate for clustering.
       - If the value of H is around 0.5, then it's not clear whether the data is uniformly distributed or whether it has clusters. It's generally a borderline case.
       - If the value of H is close to 0, then the data is likely uniformly distributed, and so is probably not a good candidate for clustering.

       - If H > 0.75, the data is considered to have a high tendency to cluster.
       - If 0.5 < H < 0.75, the data may have a tendency to cluster, but it's not clear.
       - If H < 0.5, the data is unlikely to have a meaningful cluster structure.

    
    """
    
    # reshape the data if it's 1-dimensional
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # get the number of datapoints and dimension
    N, d = X.shape

    # number of samples to take    
    n = int(sample_ratio * N)

    # set the random seed
    np.random.seed(random_state)

    # randomly sample n datapoints
    samples = X[np.random.choice(N, size=n, replace=False)]

    # Fit a nearest neighbor model to the data
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(X)
    
    # Calculate distance to nearest neighbor for each sample point
    sum_d = nbrs.kneighbors(samples, return_distance=True)[0].sum()

    # Generate n random points in the same dimensional space and calculate their distance to nearest neighbor
    random_points = np.array([np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0)) for _ in range(n)])
    sum_u = nbrs.kneighbors(random_points, return_distance=True)[0].sum()

    # Hopkins statistic
    H = sum_u / (sum_u + sum_d)

    return H
 