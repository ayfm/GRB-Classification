#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ayf

methods for data operations
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy import stats
from .env import dir_catalogs, dir_datasets

# catalog directories
path_cat_batse = os.path.join(dir_catalogs, "batse_catalog.xlsx")
path_cat_fermi = os.path.join(dir_catalogs, "fermi_catalog.xlsx")
path_cat_swift = os.path.join(dir_catalogs, "swift_catalog.xlsx")
path_cat_grb_ee = os.path.join(dir_catalogs, "grb_ee.xlsx")
path_cat_batse_redshift = os.path.join(dir_catalogs, "batse_redshift.xlsx")
path_cat_swift_magnetars = os.path.join(dir_catalogs, "swift_magnetars.xlsx")


def _createDatasetDirectory(subdir=None):
    """ 
    creates dataset directory if not exists
    """
    
    # if base dir not exists, create
    if not os.path.exists(dir_datasets):
        os.makedirs(dir_datasets)
    
    # if sub directories are not exists, create
    if not subdir is None:
        refdir = os.path.join(dir_datasets, subdir)
        if not os.path.exists(refdir):
            os.makedirs(refdir)
    

def createDatasetName(catalog, t90=False, t90i=False, hrd=False, lum=False):
    """ 
    create path for grb dataset  specified with arguments
    """
    # ee is always True
    ee = True
    
    fname = str(catalog).lower().strip()
    if t90:
        fname += "_t90"
    if t90i:
        fname += "_t90i"
    if hrd:
        fname += "_hrd"
    if lum:
        fname += "_lum"
    
    if any([t90, t90i, hrd, lum]):
        if ee:
            fname += "_yEE"
        else:
            fname += "_nEE"
    # add file extension
    # fname += ".csv"
    return fname

def getDatasetPath(catalog, dataset_name, fmt="csv"):
    """ 
    returns path of the dataset
    """
    return os.path.join(dir_datasets, catalog, dataset_name+"."+fmt)

def isExist(catalog, dataset_name, fmt="csv"):
    """ 
    checks if specified data set is exists 
    """
    fpath = getDatasetPath(catalog, dataset_name, fmt)
    return os.path.exists(fpath)

def loadFromPath(dataset_path, fmt="csv", sheet_name=None):
    """ 
    loads dataset from path
    """
    if fmt == "csv":
        return pd.read_csv(dataset_path, index_col=0) 
    if fmt == "xlsx":
        return pd.read_excel(dataset_path, sheet_name=sheet_name)
    raise Exception("invalid format: {}".format(fmt))

def load(catalog, t90=False, t90i=False, hrd=False, lum=False,
         fmt="csv", sheet_name=None):
    """
    load grb dataset specified with arguments
    """
    # ee is always True
    ee = True
    
    # get corresponding dataset file name
    dataset_name = createDatasetName(catalog, t90, t90i, hrd, lum)
    # get dataset path
    dataset_path = getDatasetPath(catalog, dataset_name, fmt)
    # check if dataset exists
    assert isExist(catalog, dataset_name, fmt), "No such dataset: {}".format(dataset_path)
    # load dataset
    return loadFromPath(dataset_path, fmt, sheet_name)  
    

def save(frame, dataset_name, catalog, force=False, fmt="csv", sheet_name=None, 
         freez_panes=True):
    """ 
    save dataset to the directory \n
    dataset is pandas dataframe
    """
    # create dataset dir if not exists
    _createDatasetDirectory(subdir=catalog)
    
    # get dataset path
    dataset_path = getDatasetPath(catalog, dataset_name, fmt)
    
    # check if dataset is already created
    if not force and isExist(catalog, dataset_name, fmt):
        raise Exception("!!! dataset exists: {}".format(dataset_path))
    
    # save dataframe
    if fmt == "csv":
        frame.to_csv(dataset_path)
    elif fmt == "xlsx":
        if sheet_name is None:
            sheet_name = "data"
        if freez_panes:
            freeze_panes=(1,0)
        frame.to_excel(dataset_path, sheet_name=sheet_name, engine="xlsxwriter",
                       freeze_panes=freeze_panes)
    else:
        raise Exception("invalid file format: {}".format(fmt))

def getFeaturesFromFlags(t90, t90i, hrd, lum, log=False, normalized=False):
    """ 
    returns feature list specified with flags
    """
    features = list()
    if not log:
        if t90  : features.append("t90")
        if t90i : features.append("t90i")
        if hrd  : features.append("hrd")
        if lum  : features.append("lum")
    else:
        if t90  : features.append("lgT90")
        if t90i : features.append("lgT90i")
        if hrd  : features.append("lgHrd")
        if lum  : features.append("lgLum")
        
        if normalized:
            features = [f+"_N" for f in features]
        
    return features

def getFlagsFromFeatures(features, log=False, normalized=False):
    """ 
    returns flags (t90, t90i, hrd, lum) based on features \n
    features are array-like contains features "t90", "t90i", "hrd", "lum"
    """
    # normalized is always False
    normalized = False
    
    if not log:
        t90  = "t90"  in features
        t90i = "t90i" in features 
        hrd  = "hrd"  in features
        lum  = "lum"  in features
    else:
        if not normalized:
            t90  = "lgT90"  in features
            t90i = "lgT90i" in features 
            hrd  = "lgHrd"  in features
            lum  = "lgLum"  in features
        else:
            t90  = "lgT90_N"  in features
            t90i = "lgT90i_N" in features 
            hrd  = "lgHrd_N"  in features
            lum  = "lgLum_N"  in features
            
    # return flags in order
    return t90, t90i, hrd, lum



def removeNaNs(dframe, subset=None, how="any"):
    """ 
    removes invalid values (nan, inf) from dataframe
    """
    dframe = dframe.replace((np.inf, -np.inf), np.nan)
    return dframe.dropna(subset=subset, how=how)
    

def detectOutliersSigma(data, sigma=3):
    raise Exception("check it")

    # create outliers frame
    outliers = pd.DataFrame(index=data.index, columns=data.columns)
    outliers.fillna(False, inplace=True)
    
    # for each column
    for col in data.columns:
        # calculate zscore
        zscore = stats.zscore(data.loc[:, col])
        outliers.loc[:, col] = abs(zscore) > sigma
        print(">>> Total outlier in {} column is {}".format(col, outliers.loc[:, col].sum()))
    
    return outliers

def detectOutliersIQR(data, whisker=1.5):
    raise Exception("check it")
    
    # create outliers frame
    outliers = pd.DataFrame(index=data.index, columns=data.columns)
    outliers.fillna(False, inplace=True)
    
    # for each column
    for col in data.columns:
        # calculate quartiles
        q1 = data.loc[:, col].quantile(0.25)
        q3 = data.loc[:, col].quantile(0.75)
        IQR = q3-q1
        # calculate lower and upper range
        lower_range = q1 - (whisker * IQR)
        upper_range = q3 + (whisker * IQR)
        outliers.loc[:, col] = data.loc[:, col].apply(lambda x: 
                                                      not (lower_range <= x <= upper_range))
        print(">>> Total outlier in {} column is {}".format(col, outliers.loc[:, col].sum()))
    
    return outliers

def detectOutliersDBSCAN(data, cols, min_samples, eps):
    """ 
    detects outliers in data \n
    data is pandas DataFrame \n
    cols is array of columns to be examined \n
    min_samples and eps are DBSCAN parameters \n
    a new column "outlier" is added to the dataframe
    """
    # create DBSCAN object
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=6)
    # fit and get labels
    labels = db.fit_predict(data[cols].values)
    # add outliers to the dataframe
    data["outlier"] = (labels==-1)
    # return (labels==-1).reshape(-1,1)
    # return dataframe
    return data
 
