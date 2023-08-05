from copy import deepcopy
from typing import Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import ot
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from scipy.stats import entropy, gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score as chs
from sklearn.metrics import davies_bouldin_score as dbs
from sklearn.metrics import pairwise_distances, silhouette_samples
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


def silhouette_score(
    X: ArrayLike, labels: ArrayLike, metric: Literal["Euclidean", "Mahalanobis"] = None
):
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
        return {"mean": np.nan, "coeffs": np.nan}

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

    return {"mean": mean_coeff, "coeffs": sample_coeffs}


def dispersion(X: ArrayLike, labels: ArrayLike):
    clusters = np.unique(labels)
    dispersion = 0
    for cluster in clusters:
        cluster_points = X[labels == cluster]
        centroid = cluster_points.mean(axis=0)
        dispersion += ((cluster_points - centroid) ** 2).sum()
    return dispersion


def gap_statistics(
    X: ArrayLike, labels: ArrayLike, clusterer=None, n_repeat=10, random_state=None
):
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
    gap_err = np.std(ref_disps) * np.sqrt(1 + 1 / n_repeat)

    return {"gap": gap, "gap_err": gap_err}


def davies_bouldin_score(X: ArrayLike, labels: ArrayLike) -> float:
    # how many clusters?
    n_clusters = len(set(labels))
    # if there is only one cluster, we cannot calculate Davies-Bouldin score
    if n_clusters == 1:
        return np.nan

    return dbs(X, labels)


def calinski_harabasz_score(X: ArrayLike, labels: ArrayLike) -> float:
    # how many clusters?
    n_clusters = len(set(labels))
    # if there is only one cluster, we cannot calculate Calinski-Harabasz score
    if n_clusters == 1:
        return np.nan

    return chs(X, labels)


def hopkins_statistic(
    X: ArrayLike, sample_ratio: float = 0.05, random_state=None
) -> float:
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
    samples = sample(X, size=n, replace=False)

    # Fit a nearest neighbor model to the data
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="brute").fit(X)

    # Calculate distance to nearest neighbor for each sample point
    sum_d = nbrs.kneighbors(samples, return_distance=True)[0].sum()

    # Generate n random points in the same dimensional space and calculate their distance to nearest neighbor
    random_points = np.array(
        [np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0)) for _ in range(n)]
    )
    sum_u = nbrs.kneighbors(random_points, return_distance=True)[0].sum()

    # Hopkins statistic
    H = sum_u / (sum_u + sum_d)

    return H


def sample(
    X: np.ndarray,
    size: int,
    weights: Optional[np.ndarray] = None,
    replace: bool = True,
    random_state: int = None,
) -> np.ndarray:
    """
    Sample from an array of values with or without replacement.

    Parameters
    ----------
    X : np.ndarray
        Array of values to sample from.
    size : int
        Number of samples to draw.
    weights : np.ndarray, optional
        Weights associated with each value in `X`. If None, all values are equally likely to be drawn.
    replace : bool, optional
        Whether to sample with replacement or not. Default is True.
    random_state : int, optional
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        Array of samples.

    Raises
    ------
    ValueError
        If `size` is greater than the length of `X` and `replace` is False.
    """

    # get the length of X
    N = len(X)

    # Validate inputs
    if size > N and not replace:
        raise ValueError(
            "Cannot draw more samples than exist in X without replacement."
        )

    if weights is not None and len(weights) != N:
        raise ValueError("`weights` must be the same length as `X`.")

    # If weights is not provided, use uniform distribution
    if weights is None:
        weights = np.ones(shape=(N,)) / N

    # Normalize weights
    weights /= np.sum(weights)

    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Sample indices
    idx = np.random.choice(N, size=size, p=weights, replace=replace)

    # Get samples
    samples = X[idx]

    return samples


def js_divergence(pdf1: np.ndarray, pdf2: np.ndarray, base=2) -> float:
    """
    Compute the Jensen-Shannon divergence of two probability density functions (PDFs).

    Parameters
    ----------
    pdf1 : np.ndarray
        First probability density function.
    pdf2 : np.ndarray
        Second probability density function.
    base : int, optional
        The logarithmic base to use, defaults to `2`.

    Returns
    -------
    float
        The Jensen-Shannon divergence.
    """

    # create a copy of the arrays
    pdf1 = pdf1.copy()
    pdf2 = pdf2.copy()

    # Make sure PDFs sum to 1
    pdf1 /= np.sum(pdf1)
    pdf2 /= np.sum(pdf2)

    # Compute the average distribution
    pdf_avg = 0.5 * (pdf1 + pdf2)

    # Compute Jensen-Shannon divergence
    js_divergence = 0.5 * (
        entropy(pdf1, pdf_avg, base=base) + entropy(pdf2, pdf_avg, base=base)
    )

    return js_divergence


def jensen_shannon_distance(
    X1: np.ndarray,
    X2: np.ndarray,
    base: int = 2,
    bandwidth_method: str = "scott",
    grid_size: int = 100,
    n_repeat: int = 1000,
    sample_ratio: float = 1.0,
    weights1: Optional[np.ndarray] = None,
    weights2: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute the Jensen-Shannon distance (JSD) between two data sets with bootstrapping.

    The JSD is a symmetrical and finite measure of the similarity between two probability distributions.
    It is derived from the Kullback-Leibler Divergence (KLD), a measure of how one probability distribution
    diverges from a second, expected probability distribution. Unlike KLD, JSD is symmetrical, giving the
    same value for JSD(P||Q) and JSD(Q||P), where P and Q are the two distributions being compared.

    A JSD value of 0 indicates that the distributions are identical. Higher values indicate that the
    distributions are more different from each other. The maximum value of JSD is log2 (for base=2),
    which occurs when the two distributions are mutually singular.

    Parameters
    ----------
    X1 : np.ndarray
        First data set.
    X2 : np.ndarray
        Second data set.
    base : int, optional
        The logarithmic base to use, defaults to `2`.
    bandwidth_method : str, optional
        The method used to calculate the bandwidth for the KDE, defaults to 'scott'.
    grid_size : int, optional
        Number of points where the PDFs are evaluated, defaults to 100.
    n_repeat : int, optional
        Number of bootstrap samples to generate, defaults to 1000.
    sample_ratio : float, optional
        The ratio of the number of samples to the size of the data set, defaults to 1.0.
    weights1 : np.ndarray, optional
        Weights for the first data set.
    weights2 : np.ndarray, optional
        Weights for the second data set.
    random_state : int, optional
        Seed for the random number generator.

    Returns
    -------
    tuple of float
        The mean and standard deviation of the Jensen-Shannon distances between `X1` and `X2`.
    """

    # reshape the data if it's 1-dimensional
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)

    # get size of each array
    N1 = len(X1)
    N2 = len(X2)

    # Determine the number of bootstrap samples to generate for each array
    n_samples1 = int(sample_ratio * N1)
    n_samples2 = int(sample_ratio * N2)

    # Create a range over which to evaluate the PDFs
    x_range = np.linspace(
        min(np.min(X1), np.min(X2)), max(np.max(X1), np.max(X2)), num=grid_size
    )

    # set random state
    np.random.seed(random_state)

    # Store the calculated distances
    distances = []

    # Generate bootstrap samples and calculate JSD for each
    for _ in range(n_repeat):
        # Sample from each array with replacement
        sample1 = sample(X1, size=n_samples1, weights=weights1)
        sample2 = sample(X2, size=n_samples2, weights=weights2)

        # Estimate PDFs of samples
        pdf1 = gaussian_kde(sample1.ravel(), bw_method=bandwidth_method)(x_range)
        pdf2 = gaussian_kde(sample2.ravel(), bw_method=bandwidth_method)(x_range)

        # Calculate JS-Divergence
        js_div = js_divergence(pdf1, pdf2, base=base)
        # Calculate JS-Distance
        js_dist = np.sqrt(js_div)
        # store the distance
        distances.append(js_dist)

    # Calculate the mean and standard deviation of the distances
    jsd_mean = np.mean(distances)
    jsd_std = np.std(distances)

    return jsd_mean, jsd_std


def wasserstein_distance(
    X1: ArrayLike, X2: ArrayLike, random_state=None, max_iter: int = 1000000
) -> float:
    """
    Compute the Wasserstein distance between two multi-dimensional distributions.

    This function relies on the Python Optimal Transport (POT) library.

    Parameters
    ----------
    X1, X2 : array-like, shape = [n_samples, n_dimensions]
        Input data. Each row corresponds to a sample, and each column corresponds to a dimension of the sample.

    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation.
        Pass an int for reproducible output across multiple function calls.

    max_iter : int, default=None
        Maximum number of iterations for optimization algorithm. The default value is 1000000.

    Returns
    -------
    distance : float
        The computed square root of the Wasserstein distance (i.e., the actual Wasserstein distance)
        between the two input distributions.

    Notes
    -----
    The Wasserstein distance is a distance measure between probability distributions. It's defined as the minimum
    cost that is enough to transform one distribution into the other. Cost is measured in the amount of distribution
    weight that must be moved and the distance it has to be moved.

    The Wasserstein distance takes values from 0 to +inf. The value is 0 if and only if the distributions are equal.
    Larger values indicate that more cost is required to transform one distribution into the other, meaning the distributions
    are more different.

    This function computes the Wasserstein distance in a multidimensional space. It uses the Python Optimal Transport (POT)
    library to do so.
    """

    # if X1 and X2 are 1-dimensional, reshape them to 2-dimensional
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)

    # make sure that X1 and X2 have the same number of dimensions
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same number of dimensions.")

    # set the random seed
    np.random.seed(random_state)

    # Compute a uniform distribution over the samples
    a, b = np.ones((X1.shape[0],)) / X1.shape[0], np.ones((X2.shape[0],)) / X2.shape[0]

    # Compute the cost matrix (Euclidean distance in this case)
    M = ot.dist(X1, X2, metric="sqeuclidean")

    # Compute the Wasserstein distance
    wasserstein_distance = ot.emd2(a, b, M, numItermax=max_iter)

    # Compute the square root of the Wasserstein distance to get the actual Wasserstein distance
    wasserstein_distance = np.sqrt(wasserstein_distance)

    return wasserstein_distance


def normalize_wasserstein_distance(d: float, scale: float = 1.0) -> float:
    """
    Normalize a Wasserstein distance to a value between 0 and 1 using a sigmoid function.

    Parameters
    ----------
    d : float
        The Wasserstein distance to normalize.

    scale : float, default=1.0
        The scale parameter for the sigmoid function. Larger values will squash
        the output closer to 0 or 1.

    Returns
    -------
    distance : float
        The normalized distance.
    """
    return 1 / (1 + np.exp(-d * scale))
