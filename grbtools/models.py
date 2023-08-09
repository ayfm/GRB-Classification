import pickle
import numpy as np
import pandas as pd
from grbtools import env
from grbtools import gmm
from grbtools import disp
import os
import pickle

def checkModelFiles(model_path):
    return os.path.isfile(model_path)

def saveModel(model, dataset_name, model_name):
    model_file = os.path.join(env.DIR_MODELS, dataset_name, model_name + ".model")

    if checkModelFiles(model_file):
        print("Overwriting on the existing model...")
    pickle.dump(model, open(model_file, "wb"))
    print("Model is saved to {}".format(model_file))

def loadModelbyName(model_path):

    if checkModelFiles(model_path):
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model
    else:
        print("Model file {} does not exist.".format(model_path))
        return None

def loadModelbyProperties(dataset_name, 
                          feat_space, 
                          n_components, 
                          cov_type = "full"):
    model_name = dataset_name + '_' + "_".join(feat_space) + '_N' + str(n_components) + '_C' + cov_type + '.model'
    model_path = os.path.join(env.DIR_MODELS, dataset_name, model_name)
    return loadModelbyName(model_path)
        


def sort_clusters(gmm_model):
    """
    Sorts the clusters of a Gaussian Mixture Model (GMM) based on the means of the components.

    Args:
        gmm_model: A trained instance of Gaussian Mixture Model (sklearn.mixture.GaussianMixture).

    Returns:
        None. Modifies the labels of the GMM model in-place.
    """

    # Get means and their corresponding component indices
    means = [(i, mean) for i, mean in enumerate(gmm_model.means_)]

    # Sort the means based on the first dimension of each mean vector
    means.sort(key=lambda x: x[1][0])

    # Create a label mapping dictionary using the sorted means
    label_map = {i: means[i][0] for i in range(len(means))}

    # Remap the labels of the GMM model using the label mapping
    gmm_model.remap_labels(label_map)

    
def createGMMs(dataset_name = "", 
               data = None, 
               cov_type = "full", 
               n_clusters_max = 5, 
               verbose = True, 
               max_iter=10000, 
               n_init=100, 
               sorting_clusters: bool = True,
               plot_model: bool = False):
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
    if data is None:
        raise Exception("Data is not provided.")
    
    n_components = [i for i in range(1, n_clusters_max+1)]
    
    for n_component in n_components:
        gmm_model = gmm.GaussianMixtureModel(n_components=n_component, random_state=42, 
                                covariance_type=cov_type, max_iter=max_iter, 
                                n_init = n_init).fit(data)
 
        if sorting_clusters:
            sort_clusters(gmm_model)

        model_name = dataset_name + '_' + '_'.join(list(data.columns)) + '_N' + str(n_component) + '_C' + cov_type
        saveModel(gmm_model, dataset_name, model_name)
        
        if verbose:
            print("Model with %d components has been saved." %(n_component))
    
        if plot_model:
            disp.plot_models(cat_name=dataset_name,
                             model_name=model_name+".model",
                             data=data)

def extract_cluster_count(model_name):
    tokens = model_name.split("_")
    return int(tokens[-2][1:])


def sort_dict_by_key(d):
    return {k: d[k] for k in sorted(d.keys())}

def bring_all_models(cat_name = "",
                     feat_space = []):
    models_path = os.path.join(env.DIR_MODELS, cat_name)
    models_all = os.listdir(models_path)
    
    models = dict()
    for model_name in models_all:
        if model_name.find("_".join(feat_space)) != -1:
            models[extract_cluster_count(model_name)] = loadModelbyName(os.path.join(models_path, model_name))
    
    return sort_dict_by_key(models)        

def bring_all_predictions(cat_name = "",
                          feat_space = [],
                          data = None):
    models = bring_all_models(cat_name, feat_space)
    predictions = dict()
    for n_component in models.keys():
        predictions[n_component] = models[n_component].predict(data)
    
    return sort_dict_by_key(predictions)      