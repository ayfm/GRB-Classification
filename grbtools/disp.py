#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ayf

methods for creating figures (matplotlib)
"""

import os
import matplotlib as mplot
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from .env import DIR_FIGURES
import pickle
from grbtools import models, env



# customize matplotlib
mplot.style.use('seaborn-paper')
mplot.rc('text', usetex=False)
font = {#'family' : 'sans-serif',
        #'sans-serif' : ['Helvetica'],
        'size'   : 22}
mplot.rc('font', **font)

    
################################

def sortClusters2D(data, group_list):
    """
    This function sorts the clusters that are randomly created by the model. 
    The one with the lowest mean in the x-axis is labeled as 0 and the rest are sorted
    in the same way. (i.e. 0 -- 1 -- 2 etc.)
    """
    group_list = np.sort(group_list)
    grouped_data = data.groupby(data['clusters'])
    
    group_means = pd.DataFrame(columns=['group_no', 'means'])
    group_means['group_no'] = group_list
    means = []
    for cluster in group_list:
        group = grouped_data.get_group(cluster)
        means.append(np.mean(group.iloc[:, 0]))
    group_means['means'] = means
    group_means = group_means.sort_values(by=['means'])
    
    no = group_means['group_no'].tolist() # no = new order
    
    if len(group_list) == 2:
        data = data.replace({'clusters':{0:no.index(0), 1:no.index(1)}})
    elif len(group_list) == 3:
        data = data.replace({'clusters':{0:no.index(0), 1:no.index(1), 2:no.index(2)}})
    elif len(group_list) == 4:
        data = data.replace({'clusters':{0:no.index(0), 1:no.index(1), 2:no.index(2), 3:no.index(3)}})
    elif len(group_list) == 5:
        data = data.replace({'clusters':{0:no.index(0), 1:no.index(1), 2:no.index(2), 3:no.index(3), 4:no.index(4)}})
    
    return data


def scatter2D(data, xcol, ycol, outliers=False, magnetars=None, title="", ax=None, 
              color=None, ref_line=None, ref_line_ax="x", legend_label=None,
              fname=None, subdir=None, fmt="png", figsize=(10,8), force=False):
    """ 
    scatter 2-D data \n
    data is pandas DataFrame \n
    xcol and ycol indicate that which columns will be considered \n
    outliers arg can be either True or a pandas Series. If it is true, 
        then outliers will be read from dataframe.  \n
    magnetars arg can be either True or a pandas Series. If it is true, 
        then magnetars will be read from dataframe \n
    if add_tline is True, then T=2 sec line will be drawn
    """
    
    # get data values
    dvalues = data[[xcol, ycol]].values
    # set default values
    ovalues = np.repeat(False, repeats=dvalues.shape[0])
    mvalues = np.repeat(False, repeats=dvalues.shape[0])
    
    
    # check if outliers is specified
    if isinstance(outliers, bool) and outliers:
        assert "outlier" in data.columns, "Outlier column could not be found in dataframe"
        ovalues = data[["outlier"]].values.flatten()
    elif isinstance(outliers, (pd.DataFrame, pd.Series)):
        ovalues = outliers.loc[data.index].values.flatten()
    # check if magnetars is specified
    if isinstance(magnetars, bool) and magnetars:
        assert "magnetar" in data.columns, "Magnetar column could not be found in dataframe"
        mvalues = data[["magnetar"]].values.flatten()
    elif isinstance(magnetars, (pd.DataFrame, pd.Series)):
        mvalues = magnetars.loc[data.index].values.flatten()
    
    # create figure object
    if ax is None:
        fig = plt.figure()
        # set figsize
        if not figsize is None:
            fig.set_size_inches(figsize) 
        ax = plt.gca()
    
    # scatter non-outlier values 
    ax.scatter(dvalues[ovalues==False,0], dvalues[ovalues==False,1], 
               marker="o", color=color, alpha=0.7, label=legend_label)
    # scatter outlier data
    if any(ovalues):
        ax.scatter(dvalues[ovalues==True,0], dvalues[ovalues==True,1], 
                   marker="x", color="red", alpha=0.7, label="outliers")
    # scatter magnetars
    if any(mvalues):
        ax.scatter(dvalues[mvalues==True,0], dvalues[mvalues==True,1],
                   marker = '*', s=100, facecolor="gray", edgecolor="black", linewidth=1.,
                   label="magnetars")
         
    # draw T=2sec line
    if not ref_line is None:
        draw_ref_line(ax, ref_line, ref_line_ax)

    # set ax props
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    
    # save figure
    if not fname is None:
        save_figure(fname, subdir, fmt, force=force)
    
    # return ax
    return ax


def create_grid_space_2d(xmin, xmax, nGridX, ymin, ymax, nGridY):
    """
    creates a grid on a plane (parameter space)
    
    Parameters
    ----------
    nGridX : int
        how many points in x-dimension
    nGridY : int 
        how many points in y-dimension
    
    Returns
    -------
    x-points of grid
    y-points of grid
    a matrix (array) with n_grid_points^2 rows and two columns
    """
    # assert nGridX*nGridY <= 5e+6, "!!! too much grid points. memory risk."

    x = np.linspace(xmin, xmax, nGridX)
    y = np.linspace(ymin, ymax, nGridY)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    return X, Y, XX
    
    
def draw_cluster_boundary(ax, model, n_grid_points=500, ax_range=None):
   
    # get axes ranges
    if ax_range is None: ax_range = None, None
    x_range, y_range = ax_range
    if x_range is None:
        x_range = ax.get_xlim()
    if y_range is None:
        y_range = ax.get_ylim()
    xmin, xmax = x_range
    ymin, ymax = y_range
    
    # create a grid on the parameter space        
    _, _, GG = create_grid_space_2d(*x_range, n_grid_points, *y_range, n_grid_points)
    grid_shape = (n_grid_points, n_grid_points)
    
    Z3 = model.predict(GG).reshape(grid_shape)
    # color = plt.cm.binary(Z3)
    # ax.contour(Z3, origin='lower',extent=(xmin, xmax, ymin, ymax), cmap="black")
    ax.contour(Z3, origin='lower',extent=(xmin, xmax, ymin, ymax), colors="black")

    return ax

def colors(index):
    colors = ['firebrick', 'olivedrab', 'royalblue', 'mediumorchid', 'peru']
    return colors[index]

def markers(index):
    markers = ['o','^','x','P','p']
    return markers[index]

def clusterNames(index):
    clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
    return clusters[index]

def extractCatalogueName(file_name):
    return file_name.split('_')[0]
    
def extractCovType(model_name):
    tokens = model_name.split("_")
    cov_token = tokens[-1]
    cov_type = cov_token.split(".")[0][1:]
    return cov_type
    
def scatter2DWithClusters(model_name, data, title="", xlabel="", ylabel="", figure_save=True):
    model_path = env.DIR_MODELS + '/' + model_name
    model = models.loadModelbyName(model_path)
    
    data = data.drop('clusters', axis=1, errors='ignore')
    
    
    data['clusters'] = model.predict(data)
    data = sortClusters2D(data, pd.unique(data['clusters']))
    
    grouped_data = data.groupby(data['clusters'])
    
    
    # create figure object
    fig = plt.figure()
    ax = plt.gca()
    
    #ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['clusters'], s=25, cmap='tab20b', alpha=0.8)
    for cluster in set(list(data['clusters'])):
        group = grouped_data.get_group(cluster)
        ax.scatter(group.iloc[:, 0], group.iloc[:, 1], c=colors(cluster), s=25, alpha=0.8, marker=markers(cluster), label = clusterNames(cluster))
    
    
    ax = draw_cluster_boundary(ax, model)
    ax.legend(loc='lower right')
        # set figure options
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(color='lightgray')
    ax.set_title(title + " | " + extractCovType(model_name).upper())
    
    
    if figure_save:
        figure_path = env.DIR_FIGURES + "/" + extractCatalogueName(model_name) + "/"
        figure_name = model_name.replace('.model', '.pdf')
        fig.savefig(figure_path + figure_name)

    
    #ax.show()
    