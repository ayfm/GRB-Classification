from sklearn.neighbors import KernelDensity
import numpy as np

def outliersKernelDensity1D(data1d, threshold_density = 0.025):

    kde = KernelDensity(
        bandwidth="scott",
        kernel='gaussian', 
    ).fit(data1d.reshape(-1, 1))

    log_dens = kde.score_samples(data1d.reshape(-1, 1))
    dens = np.exp(log_dens)

    is_outlier = dens < threshold_density
    return is_outlier, log_dens
    # data1d['is_outlier'] = is_outlier