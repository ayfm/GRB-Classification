from copy import deepcopy
from typing import Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import ot
from scipy.spatial.distance import cdist
from scipy.stats import anderson, entropy, gaussian_kde, kstest, normaltest, shapiro
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score as chs
from sklearn.metrics import davies_bouldin_score as dbs
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, NearestNeighbors
from grbtools import env

# get logger
logger = env.get_logger()


def _set_seed(seed: Union[int, None]) -> None:
    """
    Set the seed for numpy's random number generator.

    Args:
        seed (int or None): Seed to use. If None, no seed is set.
    """

    # set seed if given
    if seed is not None:
        np.random.seed(seed)


def AIC(model: GaussianMixture, data: np.ndarray) -> float:
    """
    Compute the Akaike Information Criterion (AIC) for a given GaussianMixtureModel and data.

    Parameters:
    -----------
    model : GaussianMixture
        The Gaussian mixture model instance.

    data : np.ndarray
        The dataset to compute the AIC for.

    Returns:
    --------
    float
        The computed AIC value.
    """
    return model.aic(data)


def BIC(model: GaussianMixture, data: np.ndarray) -> float:
    """
    Compute the Bayesian Information Criterion (BIC) for a given GaussianMixtureModel and data.

    Parameters:
    -----------
    model : GaussianMixture
        The Gaussian mixture model instance.

    data : np.ndarray
        The dataset to compute the BIC for.

    Returns:
    --------
    float
        The computed BIC value.
    """
    return model.bic(data)


def compute_mahalanobis_distance(
    X: np.ndarray,
    mean: Union[np.ndarray, None] = None,
    covar: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """
    Calculates the Mahalanobis distance between data points and means.

    Args:
        X (np.ndarray): Input data of shape (N, d), where N is the number of data points and d is the number of dimensions.
        mean (Union[np.ndarray, None], optional): Mean values to use for computing the distance. If None, the mean will be calculated from the input data. Defaults to None.
        covar (Union[np.ndarray, None], optional): Covariance matrix to use for computing the distance. If None, the covariance matrix will be calculated from the input data. Defaults to None.

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
    X: np.ndarray,
    labels: np.ndarray,
    means: Union[np.ndarray, None] = None,
    covars: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """
    Calculate the Mahalanobis-based silhouette coefficients for clustered data.

    Parameters:
        X (np.ndarray): Input data array of shape (N, d), where N is the number of samples and d is the number of features.
        labels (np.ndarray): Array of cluster assignments for each data point, of shape (N,).
        means (Union[np.ndarray, None], optional): Cluster mean vectors. If None, they will be calculated from the data.
            Must have shape (n_clusters, d), where n_clusters is the number of clusters.
        covars (Union[np.ndarray, None], optional): Cluster covariance matrices. If None, they will be calculated from the data.
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
    X: np.ndarray,
    labels: np.ndarray,
    metric: Literal["Euclidean", "Mahalanobis"] = None,
) -> Dict:
    """
    Calculates the silhouette score for a clustering.

    The silhouette score measures how well each sample in a cluster is assigned to its own cluster compared to other clusters.
    It provides an indication of the quality of the clustering results.

    Args:
        X (np.ndarray): The input data samples.
        labels (np.ndarray): The cluster labels assigned to each sample.
        metric ({"Euclidean", "Mahalanobis"}, optional): The distance metric to be used. Defaults to "Euclidean".

    Returns:
        Dict:
            'mean: the mean silhouette coefficient over all samples.
            'coeffs': is an array of sample silhouette coefficients.

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


def intra_cluster_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the total intra-cluster dispersion (within-cluster sum of squares) for the given data and labels.

    Parameters
    ----------
    X : np.ndarray, shape = [n_samples, n_features]
        The input samples. Each row corresponds to a sample, and each column corresponds to a feature of the sample.

    labels : np.ndarray, shape = [n_samples]
        The labels predicting the cluster each sample belongs to. This should align with the samples in `X`.

    Returns
    -------
    float:
        The total intra-cluster dispersion for the given data and labels. This is the sum of the squared distances of each
        point to the centroid of its assigned cluster.

    """
    clusters = np.unique(labels)
    dispersion = 0
    for cluster in clusters:
        cluster_points = X[labels == cluster]
        centroid = cluster_points.mean(axis=0)
        dispersion += ((cluster_points - centroid) ** 2).sum()

    return dispersion


def gap_statistics(
    X: np.ndarray,
    labels: np.ndarray,
    clusterer=None,
    n_repeat: int = 100,
    random_state=None,
) -> Dict:
    """
    Compute the gap statistic for the given data and labels.

    The gap statistic compares the total intra-cluster dispersion of the input data to that of a reference dataset
    generated from a uniform distribution with the same range as the input data.

    A higher gap value indicates that the clustering structure in the input data is stronger relative to a
    random distribution, while a lower (including negative) gap value suggests that the clustering structure in the
    input data is not significantly different from a random distribution.

    The gap value theoretically ranges from negative infinity (when the clustering structure of the input data is significantly
    worse than the random reference data) to positive infinity (when the clustering structure of the input data is
    significantly better than the random reference data). In practice, a negative gap value would typically suggest
    no meaningful clustering structure in the input data.

    Parameters
    ----------
    X : np.ndarray, shape = [n_samples, n_features]
        The input samples. Each row corresponds to a sample, and each column corresponds to a feature of the sample.

    labels : np.ndarray, shape = [n_samples]
        The labels predicting the cluster each sample belongs to. This should align with the samples in `X`.

    clusterer : estimator object implementing 'fit_predict'
        The clusterer to use for the data. If `None`, `KMeans` with the same number of clusters as the labels will be used.

    n_repeat : int, optional
        Number of times to generate a reference dataset and compute its dispersion. Default is 10.

    random_state : int or None, optional
        Determines random number generation for dataset creation. Pass an int for reproducible output across multiple
        function calls. If None, the random number generator is the RandomState instance used by np.random. Default is None.

    Returns
    -------
    Dict : dict
        'gap': (float) The log of the average dispersion of the reference datasets minus the log dispersion of the input data.
        'err': (float) The standard deviation of the gap statistic, scaled by a factor sqrt(1 + 1/n_repeat).
    """

    # if clusterer is not specified, use KMeans with the same number of clusters as the labels
    if clusterer is None:
        clusterer = KMeans(n_clusters=len(np.unique(labels)))
    # otherwise, work with a clone of the clusterer
    else:
        clusterer = deepcopy(clusterer)

    # Compute the gap statistic
    ref_disps = np.zeros(n_repeat)

    # set the random state
    _set_seed(random_state)

    for i in range(n_repeat):
        # Generate a reference dataset
        ref_X = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), X.shape)

        # Fit the model to the reference data and get the labels
        ref_labels = clusterer.fit_predict(ref_X)

        # Compute the dispersion for the reference data
        ref_disp = intra_cluster_dispersion(ref_X, ref_labels)
        ref_disps[i] = np.log(ref_disp)

    # Compute the dispersion for the original data
    orig_disp = intra_cluster_dispersion(X, labels)
    orig_disp = np.log(orig_disp)

    # Compute the gap statistic
    gap = np.mean(ref_disps) - orig_disp
    gap_err = np.std(ref_disps) * np.sqrt(1 + 1 / n_repeat)

    return {"gap": gap, "err": gap_err}


def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Davies-Bouldin score for a clustering result.

    The Davies-Bouldin index (DBI) is a metric of internal cluster validation that measures the average 'similarity'
    between clusters, where the similarity is a ratio of within-cluster distances to between-cluster distances.
    Thus, clusters which are farther apart and less dispersed will result in a better score.

    The minimum score is 0, with smaller values indicating better clustering.
    The maximum score is unbounded, with larger values indicating worse clustering.
    The DBI is undefined for a single cluster.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_dimensions]
        Input data. Each row corresponds to a sample, and each column corresponds to a dimension of the sample.

    labels : array-like, shape = [n_samples]
        Cluster labels for each sample in the input data.

    Returns
    -------
    davies_bouldin : float
        The Davies-Bouldin score for the input clustering.
    """
    # how many clusters?
    n_clusters = len(set(labels))
    # if there is only one cluster, we cannot calculate Davies-Bouldin score
    if n_clusters == 1:
        return np.nan

    return dbs(X, labels)


def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Calinski-Harabasz score for a clustering result.

    The Calinski-Harabasz index (CHI) is a metric of internal cluster validation that measures the ratio of
    between-cluster dispersion to within-cluster dispersion. Higher values of the CHI indicate better clustering.
    The CHI is undefined for a single cluster.

    Theoretically, the Calinski-Harabasz index can be infinitely large but in practice it's usually within a
    finite range. The score is higher when clusters are dense and well separated.
    The lower bound is 0, with lower values indicating worse clustering.
    The upper bound is unbounded, with larger values indicating better clustering.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_dimensions]
        Input data. Each row corresponds to a sample, and each column corresponds to a dimension of the sample.

    labels : array-like, shape = [n_samples]
        Cluster labels for each sample in the input data.

    Returns
    -------
    calinski_harabasz : float
        The Calinski-Harabasz score for the input clustering.
    """
    # how many clusters?
    n_clusters = len(set(labels))
    # if there is only one cluster, we cannot calculate Calinski-Harabasz score
    if n_clusters == 1:
        return np.nan

    return chs(X, labels)


def hopkins_statistic(
    X: np.ndarray, sample_ratio: float = 0.05, random_state=None
) -> float:
    """
    Calculates the Hopkins statistic - a statistic which indicates the cluster tendency of data.

    Parameters:
    X (np.ndarray): The dataset.
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

    # set the random state
    _set_seed(random_state)

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
    size: Optional[int] = None,
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
        Number of samples to draw. If None, an array of len(X) is returned.
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

    # if size is None, update it to be the length of X
    if size is None:
        size = N

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

    # set the random state
    _set_seed(random_state)

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
    bandwidth_method: Literal["scott", "silverman"] = "scott",
    grid_size: int = 100,
):
    """
    Compute the Jensen-Shannon distance (JSD) between two data sets.

    The JSD is a symmetrical and finite measure of the similarity between two probability distributions.
    It is derived from the Kullback-Leibler Divergence (KLD), a measure of how one probability distribution
    diverges from a second, expected probability distribution. Unlike KLD, JSD is symmetrical, giving the
    same value for JSD(P||Q) and JSD(Q||P), where P and Q are the two distributions being compared.

    A JSD value of 0 indicates that the distributions are identical. Higher values indicate that the
    distributions are more different from each other. The maximum value of JSD is log2 (for base=2),
    which occurs when the two distributions are mutually singular.

    Parameters
    ----------
    X1 : array-like
        First data set.
    X2 : array-like
        Second data set.
    base : float, optional
        The logarithmic base to use, defaults to `e` (natural logarithm).
    bandwidth_method : str, optional
        The method used to calculate the bandwidth for the KDE, defaults to 'scott'.
    grid_size : int, optional
        Number of points where the PDFs are evaluated, defaults to 100.

    Returns
    -------
    jsd : float
        The Jensen-Shannon distance between `X1` and `X2`.
    """

    # reshape the data if it's 1-dimensional
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)

    # get size of each array
    N1 = len(X1)
    N2 = len(X2)

    # Create a range over which to evaluate the PDFs
    x_range = np.linspace(
        min(np.min(X1), np.min(X2)), max(np.max(X1), np.max(X2)), num=grid_size
    )

    # Estimate PDFs of X1 and X2
    pdf1 = gaussian_kde(X1.ravel(), bw_method=bandwidth_method)(x_range)
    pdf2 = gaussian_kde(X2.ravel(), bw_method=bandwidth_method)(x_range)

    # Compute Jensen-Shannon divergence
    js_div = js_divergence(pdf1, pdf2, base=base)

    # Compute Jensen-Shannon distance
    js_distance = np.sqrt(js_div)

    return js_distance


def jensen_shannon_distance_bootstrap(
    X1: np.ndarray,
    X2: np.ndarray,
    base: int = 2,
    bandwidth_method: Literal["scott", "silverman"] = "scott",
    grid_size: int = 100,
    n_repeat: int = 1000,
    sample_ratio: float = 1.0,
    weights1: Optional[np.ndarray] = None,
    weights2: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> Dict:
    """
    Compute the Jensen-Shannon distance (JSD) between two data sets with bootstrapping.

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
    Dict:
        'mean': float - The mean of the Jensen-Shannon distances between `X1` and `X2`.
        'std': float - The standard deviation of the Jensen-Shannon distances between `X1` and `X2`.
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

    # set the random state
    _set_seed(random_state)

    # Store the calculated distances
    distances = []

    # Generate bootstrap samples and calculate JSD for each
    for _ in range(n_repeat):
        # Sample from each array with replacement
        sample1 = sample(X1, size=n_samples1, replace=True, weights=weights1)
        sample2 = sample(X2, size=n_samples2, replace=True, weights=weights2)

        # Calculate JS-Distance
        js_dist = jensen_shannon_distance(
            sample1,
            sample2,
            base=base,
            bandwidth_method=bandwidth_method,
            grid_size=grid_size,
        )
        # store the distance
        distances.append(js_dist)

    # Calculate the mean and standard deviation of the distances
    jsd_mean = np.mean(distances)
    jsd_std = np.std(distances)

    return {"mean": jsd_mean, "std": jsd_std}


def wasserstein_distance(
    X1: np.ndarray, X2: np.ndarray, max_iter: int = 1000000, random_state=None
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

    # set the random state
    _set_seed(random_state)

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


def wasserstein_distance_bootstrap(
    X1: np.ndarray,
    X2: np.ndarray,
    max_iter: int = 1000000,
    n_repeat: int = 100,
    sample_ratio: float = 1.0,
    weights1: Optional[np.ndarray] = None,
    weights2: Optional[np.ndarray] = None,
    random_state: int = None,
) -> Dict:
    """
    Compute the Wasserstein distance between two multi-dimensional distributions with bootstrapping.

    Parameters
    ----------
    X1, X2 : np.ndarray, shape = [n_samples, n_dimensions]
        Input data. Each row corresponds to a sample, and each column corresponds to a dimension of the sample.
    n_repeat : int, optional
        Number of bootstrap samples to generate. Default is 100.
    max_iter : int, default=None
        Maximum number of iterations for optimization algorithm. The default value is 1000000.
    n_repeat : int, optional
        Number of bootstrap samples to generate, defaults to 1000.
    sample_ratio : float, optional
        The ratio of the number of samples to the size of the data set, defaults to 1.0.
    weights1 : np.ndarray, optional
        Weights for the first data set.
    weights2 : np.ndarray, optional
        Weights for the second data set.
    random_state : int or None, optional
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used by np.random. Default is None.

    Returns
    -------
    Dict:
        'mean': (float) The mean Wasserstein distance calculated over the bootstrap samples.
        'std' : (float) The standard deviation of the Wasserstein distances calculated over the bootstrap samples.
    """

    # Ensure the arrays are 2D
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)

    # make sure that X1 and X2 have the same number of dimensions
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same number of dimensions.")

    # get size of each array
    N1 = len(X1)
    N2 = len(X2)

    # Determine the number of bootstrap samples to generate for each array
    n_samples1 = int(sample_ratio * N1)
    n_samples2 = int(sample_ratio * N2)

    # set the random seed
    _set_seed(random_state)

    distances = []
    # Generate bootstrap samples and calculate distance for each
    for _ in range(n_repeat):
        # Sample from each array with replacement
        sample_X1 = sample(X1, n_samples1, replace=True, weights=weights1)
        sample_X2 = sample(X2, n_samples2, replace=True, weights=weights2)

        # calculate the distance between the two samples
        distance = wasserstein_distance(sample_X1, sample_X2, max_iter=max_iter)

        # add the distance to the list
        distances.append(distance)

    # compute the mean and standard deviation of the distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    return {"mean": mean_distance, "std": std_distance}


def normality_test_shapiro_wilkinson(
    X: np.ndarray, alpha: float = 0.05, verbose: bool = True
) -> Dict[str, float]:
    """
    Perform a Shapiro-Wilkinson test for normality on the input data.

    This test checks the null hypothesis that the data was drawn from a normal distribution. It is a good choice
    for testing normality when the sample size is small (n <= 50) as it has been shown to have good power performance
    for such sample sizes. However, for larger sample sizes (n > 2000), the test may be too sensitive.

    HO: the sample is drawn from a normal distribution.
    if p-value < alpha: the null hypothesis is rejected.

    Parameters
    ----------
    X : np.ndarray
        The array containing the sample to be tested.

    alpha : float, default=0.05
        Significance level for the test.

    verbose : bool, default=True
        If True, print the results of the test.

    Returns
    -------
    dict
        A dictionary with keys 'stat' and 'p'.
        'stat' is the calculated test statistic and 'p' is the associated p-value from the test.

    Notes
    ----
    - Works well for small sample sizes (n < 50), but can also handle larger sample sizes.
    - It has been found to have the best power for a given significance, effectively determining whether the data being tested are normally distributed.
    - Shapiro-Wilk has the limitation that it is designed specifically for testing normality.
    """

    stat, p = shapiro(X)

    if verbose:
        logger.info("::: Shapiro-Wilkinson Normality Test :::")
        logger.info(f"  > Statistics={stat:.3f}, p={p:.3f}")
        if p > alpha:
            logger.info("  > Sample looks Gaussian (fail to reject H0)")
        else:
            logger.info("  > Sample does not look Gaussian (reject H0)")

    return {"stat": stat, "p": p}


def normality_test_ks(
    X: np.ndarray, alpha: float = 0.05, normalization: bool = False, verbose: bool = True
) -> Dict[str, float]:
    """
    Perform a Kolmogorov-Smirnov test for normality on the input data.

    This test checks the null hypothesis that the data was drawn from a normal distribution. The K-S test has the advantage
    of making no assumption about the distribution of data.

    However, one should be cautious while using K-S Test. The D statistic is sensitive towards the centre of the distribution and
    may not detect deviations at the tails. Moreover, compared to other tests, it may have less power, i.e., it is less likely
    to reject the null hypothesis when it is false (Type II error).

    HO: the sample is drawn from a normal distribution.
    if p-value < alpha: the null hypothesis is rejected.

    Parameters
    ----------
    X : np.ndarray
        The array containing the sample to be tested.

    alpha : float, default=0.05
        Significance level for the test. If the p-value is less than alpha, the null hypothesis is rejected.

    normalization : bool, default=False
        If True, the data is normalized before the test is performed. This is recommended when the data is not normally distributed.
    
    verbose : bool, default=True
        If True, print the results of the test.

    Returns
    -------
    dict
        A dictionary with keys 'stat' and 'p'.
        'stat' is the calculated test statistic and 'p' is the associated p-value from the test.

    Notes
    ----
    - Use when comparing a sample with a reference probability distribution (one-sample K-S test), or when comparing two samples (two-sample K-S test).
    - It is a non-parametric test which can be applied to any continuous distribution.
    - Has the advantage of making no assumption about the distribution of data.
    - Less sensitive around the mean as compared to the tails.
    """

    # normalize the data if required
    if normalization:
        X_ = (X - np.mean(X)) / np.std(X)
    else:
        X_ = X.copy()

    # test
    stat, p = kstest(X_, "norm")

    if verbose:
        logger.info("::: Kolmogorov-Smirnov Normality Test :::")
        logger.info(f"  > Statistics={stat:.3f}, p={p:.3f}")
        if p > alpha:
            logger.info("  > Sample looks Gaussian (fail to reject H0)")
        else:
            logger.info("  > Sample does not look Gaussian (reject H0)")

    return {"stat": stat, "p": p}


def normality_test_anderson(X: np.ndarray, verbose: bool = True) -> Dict:
    """
    Perform the Anderson-Darling test for normality on the input data.

    The Anderson-Darling test is a modification of the Kolmogorov-Smirnov test `kstest` for the null hypothesis
    that a sample is drawn from a population that follows a particular distribution. It gives more weight to
    the tails than does the `kstest`.

    For the Anderson-Darling test, the critical values depend on which distribution is being tested against.
    This function works for normal distributions.

    HO: the sample is drawn from a normal distribution
    if test statistic > critical value for a given significance level: reject H0

    Parameters
    ----------
    X : np.ndarray
        The array containing the sample to be tested.
    verbose : bool, default=True
        If True, print the results of the test.

    Returns
    -------
    Dict:
        'stat': (float) The test statistic.
        'critical_values': (list) The critical values for this distribution.
        'significance_level': (list) The significance levels for the corresponding critical values in `critical_values`.

    Notes
    ----
    - Can be used with large sample sizes.
    - More sensitive to departures from normality at the tails.
    - Unlike the KS-Test and Shapiro-Wilk Test, the Anderson-Darling test is a modification of the KS-Test which gives more weight to the tails.
    - Has the advantage of testing against different distribution types (exponential, logistic, etc), not just normal.
    """

    result = anderson(X)

    if verbose:
        logger.info("::: Anderson-Darling Normality Test :::")
        logger.info(f"  > Statistics={result.statistic:.3f}")
        logger.info(f"  > Critical values: {result.critical_values}")
        logger.info(f"  > Significance levels: {result.significance_level}")

        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < cv:
                logger.info(f"  > Sample looks Gaussian (fail to reject H0) at the {sl}% level")
            else:
                logger.info(f"  > Sample does not look Gaussian (reject H0) at the {sl}% level")

    return {
        "stat": result.statistic,
        "critical_values": result.critical_values,
        "significance_level": result.significance_level,
    }


def normality_test_dagostino(X: np.ndarray, alpha: float = 0.05, verbose: bool = True) -> Dict:
    """
    Perform D'Agostino's K^2 test for normality on the input data.

    D'Agostino's K^2 test is a goodness-of-fit normality test based on combined measures of skewness and kurtosis.
    This test gives a combined measure of skewness and kurtosis, two parameters of the normal distribution.
    A combined test increases the chances of rejecting a null hypothesis. In other words, it's a more conservative
    test and has more power to reject the null hypothesis if it's not true.

    HO: the sample is drawn from a normal distribution
    if p-value < alpha: reject H0

    Parameters
    ----------
    X : np.ndarray
        The array containing the sample to be tested.
    alpha : float, optional
        The significance level at which to test. The default is 0.05.
    verbose : bool, default=True
        If True, print the results of the test.

    Returns
    -------
    Dict:
        'stat': (float) The test statistic.
        'p' : (float) The p-value for the test.

    Notes
    ----
    - Tests for skewness and kurtosis, as well as omnibus test for normality.
    - Useful for larger sample sizes, as skewness and kurtosis become more meaningful for larger samples.
    - Like the Shapiro-Wilk Test, it is most powerful against departures from normality, without being heavily influenced by sample size.
    """

    stat, p = normaltest(X)

    if verbose:
        logger.info("::: D'Agostino's K^2 Normality Test :::")
        logger.info(f"  > Statistics={stat:.3f}, p={p:.3f}")
        if p > alpha:
            logger.info("  > Sample looks Gaussian (fail to reject H0)")
        else:
            logger.info("  > Sample does not look Gaussian (reject H0)")

    return {"stat": stat, "p": p}


def detect_outliers(
    X: np.ndarray,
    density_threshold: float,
    bandwidth: Union[float, str] = "scott",
    kernel: str = "gaussian",
    verbose: bool = True,
) -> Dict:
    """
    Perform outlier detection using Kernel Density Estimation.

    This function uses Kernel Density Estimation to model the probability density of the input data
    and identifies outliers as points that have a density below a certain threshold.

    Parameters
    ----------
    X : np.ndarray
        Input data.

    density_threshold : float
        Density threshold for identifying outliers. Points with a density below this value are considered outliers.

    bandwidth : Union[float, str], optional
        The bandwidth of the kernel. Can be specified as a float or one of the following strings:
        'scott', 'silverman'. The default is 'scott'. If a string is passed, the bandwidth is estimated using the
        corresponding rule.

    kernel : str, optional
        The kernel to be used in the density estimation. Should be one of the following:
        'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'. The default is 'gaussian'.

    verbose : bool, optional
        If True, print the number of outliers detected.

    Returns
    -------
    Dict:
        'is_outlier': np.ndarray
            Boolean array of the same shape as X. If the ith element of 'is_outlier' is True,
            then the ith element of X is considered an outlier.

        'density': np.ndarray
            Array of the same shape as X representing the estimated density of each point.

    Notes
    -----
     - This function treats outliers as points that are unlikely under the estimated density model of the input data.
     - It should be noted that this outlier detection method is unsupervised,
    meaning it doesn't require labeled data and it doesn't make any assumptions about the distribution of outliers.
     - This method may be problematic for high-dimensional data.
    """

    # reshape the data if it's 1-dimensional
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # Fit the KernelDensity estimator to the data
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)

    # Estimate the density of the data
    log_dens = kde.score_samples(X)
    dens = np.exp(log_dens)

    # Identify outliers as points with a density below the threshold
    is_outlier = dens < density_threshold

    # Print the number of outliers detected
    if verbose:
        # how many data points?
        n_X = len(X)
        # how many outliers were detected
        n_outliers = np.sum(is_outlier)
        # how many inliers were detected
        n_inliers = n_X - n_outliers
        # calculate the percentage of outliers
        outlier_percentage = int(np.round((n_outliers / n_X) * 100))
        # calculate the percentage of inliers
        inlier_percentage = 100 - outlier_percentage

        logger.info(f">>> Total data points  : {len(X)}")
        logger.info(f"  > Number of outliers : {n_outliers} [%{outlier_percentage}]")
        logger.info(f"  > Number of inliers  : {n_inliers} [%{inlier_percentage}]")

    return {"is_outlier": is_outlier, "density": dens}


def normalize(x: np.ndarray, invert: bool = False):
    """
    Normalize an array to [0, 1] range, considering NaN and inf values.

    Parameters:
    -----------
    x : array-like
        The input data to be normalized. Can be a list or numpy array.

    invert : bool, optional (default=False)
        If True, invert the values before normalization. This does not mean
        the resulting values will be in [1, 0] range, it simply inverts the
        sign of each value.

    Returns:
    --------
    np.ndarray
        The normalized array.
    """

    # Convert input to numpy array
    x = np.array(x)

    # If inverse flag is True, invert the values
    if invert:
        x = -x

    # Compute the minimum and maximum, ignoring NaN values
    min_ = np.nanmin(x)
    max_ = np.nanmax(x)

    # Normalize the array
    x = (x - min_) / (max_ - min_)

    return x
