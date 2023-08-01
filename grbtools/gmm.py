# This file performs GMM to a given set of data.

from sklearn.mixture import GaussianMixture
from grbtools import env, models
import pickle

    
def createGMMs(dataset_name, data, cov_type, verbose = True, max_iter=10000, n_init=100):
    """
     PARAMTERS:
    # n_components   : The number of mixture components.
    # covariance_type: String describing the type of covariance parameters to use. Must be one of 'full', 'tied', 'diag', 'spherical'     (default: full)
    # max_iter       : The number of EM iterations to perform (default 10)
    # n_init         : The number of initializations to perform. The best results are kept (default 1).
    # warm_start     : If ‘warm_start’ is True, the solution of the last fitting is used as initialization for the next call of fit(). 
    #                  This can speed up convergence when fit is called several times on similar problems.
    # 
    """
    
    n_components = [1, 2, 3, 4, 5]
    
    for n_component in n_components:
        gmm_model = GaussianMixture(n_components=n_component, random_state=42, 
                                covariance_type=cov_type, max_iter=max_iter, 
                                n_init = n_init).fit(data)
        models.saveModel(gmm_model, dataset_name, '_'.join(list(data.columns)), n_component, cov_type)
        if verbose:
            print("Model with %d components has been saved." %(n_component))
    


