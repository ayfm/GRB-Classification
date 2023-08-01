import pickle
import numpy as np
import pandas as pd
from grbtools import env, check_files

def saveModel(model, dataset_name, feat_space, n_components, cov_type):
    path_models = env.DIR_MODELS
    model_file = path_models + '/' + dataset_name + '_' + feat_space + '_N' + str(n_components) + '_C' + cov_type + '.model'
    if check_files.checkModelFiles(model_file):
        print("Overwriting on the existing model...")
    pickle.dump(model, open(model_file, "wb"))
    
def loadModelbyProperties(dataset_name, feat_space, n_components, cov_type):
    path_models = env.DIR_MODELS
    model_file = path_models + '/' + dataset_name + '_' + feat_space + '_N' + str(n_components) + '_C' + cov_type + '.model'
    
    try:
        if check_files.checkModelFiles(model_file):
            with open(model_file, "rb") as f:
                loaded_model = pickle.load(f)
                return loaded_model
    except:
        raise Exception("Model file is not found.")
        
def loadModelbyName(model_path):
    try:
        if check_files.checkModelFiles(model_path):
            with open(model_path, "rb") as f:
                loaded_model = pickle.load(f)
                return loaded_model
    except:
        raise Exception("Model file is not found.")