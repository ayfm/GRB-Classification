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


# customize matplotlib
mplot.style.use('seaborn-paper')
mplot.rc('text', usetex=False)
font = {#'family' : 'sans-serif',
        #'sans-serif' : ['Helvetica'],
        'size'   : 22}
mplot.rc('font', **font)


def save_figure(filename, subdir=None, fmt="png", force=False):
    """ 
    save matplotlib figure in given formats. \n
    fmt is comma seperated formats. i.e. fmt="png,pdf,svg" \n
    """
    # update subdir
    if subdir is None:
        subdir = ""
    if subdir.startswith("/"):
        subdir = subdir[1:]
    
    # check if directory exists 
    fdir = os.path.join(DIR_FIGURES, subdir)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    
    # split comma seperated file formats
    formats = fmt.strip().lower().split(",")
    
    # for each format
    for ext in formats:
        # if force is not disabled, check if file exists
        if not force and is_exists(filename, subdir, fmt):
            print("!!! Figure exists {}".format(filename))
            return
        # create file path
        fpath = os.path.join(fdir, filename+"."+ext)
        # save figure
        plt.savefig(fpath, format=ext,  bbox_inches='tight')

def is_exists(filename, subdir=None, fmt="png"):
    """ 
    checks if the figure file exists
    """
    # update subdir
    if subdir is None:
        subdir = ""
    if subdir.startswith("/"):
        subdir = subdir[1:]
    
    # create file path
    fpath = os.path.join(DIR_FIGURES, subdir, filename+"."+fmt)    
    return os.path.exists(fpath)
    

def get_color_cycle():
    """ 
    returns default color cylce of matplotlib
    """
    return plt.rcParams['axes.prop_cycle'].by_key()['color']

def maximize_plot():
    """
    maximizes the current plot window
    """
    try: # Option 1 : QT backend
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        return
    except: 
        pass
    
    try: # Option 2 : TkAgg backend
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        return
    except: 
        pass

    try: # Option 3 : WX backend
        manager = plt.get_current_fig_manager()
        manager.frame.Maximize(True)
        return
    except: 
        pass
    
    print("Matplotlib backend: {}".format(mplot.get_backend()))  
    raise Exception("cannot maximize window")  

def meshgridND(*arrs):
    """ 
    creates n-dimensional mesh grid
    """
    arrs = tuple(arrs) # tuple(reversed(arrs))  #edit
    lens = list(map(len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    return tuple(ans)

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

def create_grid_space_3d(xmin, xmax, nGridX, ymin, ymax, nGridY, zmin, zmax, nGridZ):
    """
    creates a grid on a plane (parameter space)
    
    Parameters
    ----------
    nGridX : int
        how many points in x-dimension
    nGridY : int 
        how many points in y-dimension
    nGridZ : int 
        how many points in z-dimension
    
    Returns
    -------
    x-points of grid
    y-points of grid
    z-points of grid
    a matrix (3-D array) with n_grid_points^3 rows and three columns
    """
    assert nGridX*nGridY <= 5e+6, "!!! too much grid points. memory risk."

    x = np.linspace(xmin, xmax, nGridX)
    y = np.linspace(ymin, ymax, nGridY)
    z = np.linspace(zmin, zmax, nGridZ)
    
    X, Y, Z = meshgridND(x, y, z)
    XXX = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
    return X, Y, Z, XXX

def draw_ellipse(ax, mean, cov, ell_std, color):
    # calculate eigenvectors and eigenvalues
    eig_vec,eig_val,_ = np.linalg.svd(cov)
    # Make sure 0th eigenvector has positive x-coordinate
    if eig_vec[0][0] < 0: eig_vec[0] *= -1
    # calculate major and minor axes length
    majorLength = 2*ell_std*np.sqrt(eig_val[0])
    minorLength = 2*ell_std*np.sqrt(eig_val[1])
    # calculate rotation angle
    u = eig_vec[0] / np.linalg.norm(eig_vec[0])
    angle = np.arctan(u[1]/u[0])
    angle = 180.0*angle/np.pi
    # create ellipse
    ell = mplot.patches.Ellipse(mean, majorLength, minorLength, angle, 
                              linestyle="dotted", linewidth=1.0,
                              edgecolor=color, facecolor="none")
    ell.set_clip_box(ax.bbox)
    ax.add_artist(ell)

def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3, n_points=100):
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.

    Plot on your favourite 3d axis. 
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
    """
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z

def draw_ellipsoid(ax, mean, cov, ell_std, color):
    # add ellipsoid
    X,Y,Z = get_cov_ellipsoid(cov, mean, nstd=ell_std)
    ax.plot_wireframe(X,Y,Z, color="black", rstride=10, cstride=10, alpha=0.3, linewidth=0.5)
    ax.plot_surface(X,Y,Z, color=color, rstride=10, cstride=10, alpha=0.2)

def draw_cluster_boundary(ax, model, n_grid_points, ax_range=None):
   
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


def draw_ref_line(ax, ref_line, ref_line_ax="x"):
    """ 
    draws reference line
    """
    if ref_line_ax == "x":
        ax.axvline(x=ref_line, ymin=0, ymax=1., color="black", linestyle="dashed")
    elif ref_line_ax == "y":
        ax.axhline(y=ref_line, xmin=0, xmax=1., color="black", linestyle="dashed")
    else:
        raise Exception("invalid ax")

def knn_histogram(data, cols, neighbors=5, title=None, fname=None, 
                  subdir=None, fmt="png", force=False):
    """ 
    displays KNN histogram for 1-D or 2-D data. \n
    data is pandas DataFrame. \n
    cols will be considered \n
    if fname is specified, the figure will be saved.  \n
    if subdir is subdirectory of the file that will be saved.  \n
    """
    if title is None:
        title = "KNN Histogram"
    
    # get values from dataframe
    dvalues = data.loc[:, cols].values
    
    # create nearest neighbors object
    nn = NearestNeighbors(n_neighbors=neighbors)
    # fit data
    nn.fit(dvalues)
    # get distances of nearest neighbors in the data
    distances, _ = nn.kneighbors(dvalues)
    distances = distances[:, -1]
    # sort distances
    distances.sort()
    # plot distances
    plt.figure()
    plt.plot(distances, linewidth=2)
    plt.grid()
    plt.title(title)
    plt.xlabel(" - ".join(cols))
    plt.ylabel("Distance")
    
    # save figure
    if not fname is None:
        save_figure(fname, subdir, fmt, force=force)
    

def compute_bin_size(data, method="freedman"):
    """ 
    computes efficient number of bins for histogram \n
    data is numpy array \n
    method is either "freedman" or "sturge"
    """
    n_data = len(data)
    
    if method=="freedman":
        IQR = stats.iqr(data)
        bin_width = 2*IQR*n_data**(-1/3)
        n_bins = (np.max(data) - np.min(data))/bin_width
        n_bins = np.round(n_bins).astype(int)
        return n_bins
    if method == "sturge":
        n_bins = 1 + 3.322*np.log10(n_data) 
        n_bins = np.round(n_bins).astype(int)
        return n_bins
    raise Exception("invalid method")
   

def histogram(data, col, ax=None, outliers=False, density=False, n_bins=None, 
              include_outliers=True, scatter=False, ref_line=None,
              title="", fname=None, subdir=None, fmt="png", figsize=(8,10), force=False):
    """ 
    display histogram of the 1-D data \n
    data is pandas DataFrame \n
    col is the data column to be considered \n
    outliers arg can be either True or a pandas Series. If it is true, 
        then outliers will be read from dataframe.  \n
    if include_outliers is True, then all data will be considered. Else
        outliers will be removed at first \n
    if scatter is True, then scatter figure will also be displayed (if ax is None)
    """
    
    # get data values
    dvalues = data[col].values
    # set default values for outliers
    ovalues = np.repeat(False, repeats=dvalues.shape[0])
    
    # check if outliers is specified
    if isinstance(outliers, bool) and outliers:
        assert "outlier" in data.columns, "Outlier column could not be found in dataframe"
        ovalues = data[["outlier"]].values.flatten()
    elif isinstance(outliers, (pd.DataFrame, pd.Series)):
        ovalues = outliers.loc[data.index].values.flatten()
    
    # determine outliers data
    histdata = dvalues
    # check if outliers will be included
    if not include_outliers:
        histdata = dvalues[ovalues==False]
    
    # how much bins ?
    if n_bins is None: 
        n_bins = compute_bin_size(histdata)
        # print(n_bins)
    
    # if not specified, create figure object
    if ax is None:
        # if scatter is True, then create a two-figure object
        if scatter:
            fig, (ax0, ax1) = plt.subplots(2, gridspec_kw={'height_ratios': [4, 0.5]}, 
                                           figsize=figsize)
        else:
            plt.figure()
            ax0 = plt.gca()
    else:
        scatter=False
        ax0 = ax
        
    # display histogram
    ax0.hist(histdata, bins=n_bins, density=density, histtype="stepfilled", 
             edgecolor="black", facecolor="none", alpha=1., linewidth=1.)
    # if scatter is True, display scatter figure
    if scatter:
        scatter1D(data, col=col, ax=ax1, outliers=outliers, color="blue", title=None)
        # set xlim
        xlim = ax1.get_xlim()
        ax0.set_xlim(xlim)
        ax0.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
    
    # draw reference line
    if not ref_line is None:
        draw_ref_line(ax, ref_line, ref_line_ax="x")
    
    # set y-axis label
    if density:
        ax0.set_ylabel("Density")
    else:
        ax0.set_ylabel("Frequency")
    
    # if scatter is not specified, then set x-label
    if not scatter:
        ax0.set_xlabel(col)
    ax0.set_title(title)
    
    # save figure
    if not fname is None:
        save_figure(fname, subdir, fmt, force=force)
    
    if scatter:
        return (ax0,ax1)
    return (ax0,)


def histogram_cluster(mbox, n_bins=None, outliers=False, scatter=True, ax=None,
                      decision_line=True, density_curve=True,cluster_curves=False, 
                      ref_line=None, title="", show_legend=True, fname=None, 
                      subdir=None, fmt="png", figsize=(8,10), force=False):
    """ 
    displays histogram of 1-D fitted data \n
    mbox is modelbox object \n
    if ax is set, then scatter will be ignored !!!
    """
    
    # get feature
    col = mbox.features[0]
    # get 1-D data with outliers column
    data = mbox.getFitData(include_all=True)
    # if outliers will be included;
    if outliers:
        # get outliers
        data = pd.concat((data, mbox.getOutlierData(include_all=True)))
    
    # determine bin size
    if n_bins is None: 
        n_bins = compute_bin_size(data[col].values)
    
    # set title
    if not title:
        title = mbox.catalog.upper()
        title += " (with EE)" if mbox.ee else " (without EE)"
        title += " {}".format(col)
    
    
    # display histogram
    axes = histogram(data, col=col, ax=ax, outliers=outliers, density=True, n_bins=n_bins, 
                     include_outliers=False, scatter=scatter, 
                     title="", figsize=figsize)
    ax = axes[0]
    
    # create space for plotting curvs
    xmin, xmax = ax.get_xlim()
    xspace = np.linspace(xmin, xmax, num=1000).reshape(-1,1)
    
    # plot density curve
    if density_curve:
        # get mixture density of space
        likelihoods = np.exp(mbox.model.score_samples(xspace))
        ax.plot(xspace, likelihoods, color="black", linewidth=2.)
    
    # plot cluster curves
    if cluster_curves:
        # for each clusters
        for cid in mbox.getClusterLabels():
            # get cluster params
            weight = mbox.clusters.getClusterWeight(cid)
            mean = mbox.clusters.getClusterMean(cid)
            cov = mbox.clusters.getClusterCov(cid)
            std = cov[0,0]**0.5
            # create dspace
            dspace = stats.norm.pdf(xspace, loc=mean, scale=std)
            dspace *= weight
            # if legends will be shown;
            legend_label = None
            if show_legend:
                legend_label = "cluster-{}".format(cid)
            # plot
            ax.plot(xspace, dspace, linestyle="dashed", linewidth=1.3, label=legend_label)
    
    # display decision lines
    if decision_line:
        # if there is more then one active cluster
        if mbox.clusters.nActiveClusters() > 1:
            # get predictions
            preds = mbox.model.predict(xspace)
            grad = np.gradient(preds)
            flag = grad != 0.
            dpoints = np.where(flag==True)[0]
            # remove sequantial indices
            xpoints = []
            for i in dpoints:
                if not i+1 in dpoints:
                    xpoints.append(xspace[i])
            # xpoints = xspace[dpoints]
            for xpoint in xpoints:
                ax.axvline(x=xpoint, ymin=0, ymax=1., color="gray", linestyle="dotted")
            # for dpoint in dpoints:
            #     ax.vlines(x=dpoint, color="black", linestyle="dashed")
    
    # draw reference line
    if not ref_line is None:
        draw_ref_line(ax, ref_line, ref_line_ax="x")
    
    # show legends
    if show_legend  and cluster_curves:
        ax.legend(loc="best")
    
    # set title
    ax.set_title(title)
    
    # save figure
    if not fname is None:
        save_figure(fname, subdir, fmt, force=force)


def scatter1D(data, col, ax=None, outliers=False, title="",  color=None, legend_label=None,
              fname=None, subdir=None, fmt="png", figsize=None, force=False):
    """ 
    scatter 1-D data \n
    data is pandas DataFrame \n
    col indicates that which column will be considered \n
    outliers arg can be either True or a pandas Series. If it is true, 
        then outliers will be read from dataframe.  \n
    """
    
    # get data values
    dvalues = data[col].values
    # set default values for outliers
    ovalues = np.repeat(False, repeats=dvalues.shape[0])
    
    # check if outliers is specified
    if isinstance(outliers, bool) and outliers:
        assert "outlier" in data.columns, "Outlier column could not be found in dataframe"
        ovalues = data[["outlier"]].values.flatten()
    elif isinstance(outliers, (pd.DataFrame, pd.Series)):
        ovalues = outliers.loc[data.index].values.flatten()
    
    # if outliers is specified, update data
    data_outlier = dvalues[ovalues==True]
    data_no_outlier = dvalues[ovalues==False]
    
    # set y-values for 1-D data
    y_outlier = [0]*len(data_outlier)
    y_no_outlier = [0]*len(data_no_outlier)
    
    # if not specified, create a figure object
    if ax is None:
        fig = plt.figure()
        # set figsize
        if not figsize is None:
            fig.set_size_inches(figsize) 
        ax = plt.gca()
        
    # scatter non-outlier data
    ax.scatter(data_no_outlier, y_no_outlier, marker=".", color=color, label=legend_label)
    # scatter outlier data
    ax.scatter(data_outlier, y_outlier, marker="x", color="red", alpha=0.8)
    # set axes props
    ax.set_xlabel(col)
    ax.set_title(title)
    
    # save figure
    if not fname is None:
        save_figure(fname, subdir, fmt, force=force)
        
    # return ax
    return ax

def scatter1D_cluster(mbox, outliers=False, decision_boundary=True, 
                      ref_line=None, ax=None, title="", show_legend=True, 
                      fname=None, subdir=None, fmt="png", figsize=(16,9), force=False):
    """ 
    scatter 1-D data \n
    data is pandas DataFrame \n
    col indicates that which column will be considered \n
    outliers arg can be either True or a pandas Series. If it is true, 
        then outliers will be read from dataframe.  \n
    """
    # get features
    col = mbox.features[0]
    # get 1-D data with outliers column
    data = mbox.getFitData(include_all=True)
    # if outliers will be included;
    if outliers:
        # get outliers
        data = pd.concat((data, mbox.getOutlierData(include_all=True)))
    
    if not title:
        title = mbox.catalog.upper()
        title += " (with EE)" if mbox.ee else " (without EE)"
        title += " {}".format(col)
    
    # display scatter plot seperately for each cluster
    for cid in mbox.getClusterLabels():
        # get cluster data
        cdata = mbox.getClusterData(cid, True)
        # set legend
        legend_label=None
        if show_legend:
            legend_label = "cluster-{}".format(cid)
        # scatter data
        ax = scatter1D(cdata, col, ax=ax, outliers=outliers, title="", 
                       legend_label=legend_label, figsize=figsize)
    
    # draw decision boundary
    if decision_boundary:
        # if there is more then one active cluster
        if mbox.clusters.nActiveClusters() > 1:
            # create space for plotting curvs
            xmin, xmax = ax.get_xlim()
            xspace = np.linspace(xmin, xmax, num=1000).reshape(-1,1)
            # get predictions
            preds = mbox.model.predict(xspace)
            grad = np.gradient(preds)
            flag = grad != 0.
            dpoints = np.where(flag==True)[0]
            # remove sequantial indices
            xpoints = []
            for i in dpoints:
                if not i+1 in dpoints:
                    xpoints.append(xspace[i])
            # xpoints = xspace[dpoints]
            for xpoint in xpoints:
                ax.axvline(x=xpoint, ymin=0, ymax=1., color="gray", linestyle="dotted")
            # for dpoint in dpoints:
            #     ax.vlines(x=dpoint, color="black", linestyle="dashed")
       
    # draw reference line (T=2sec)
    if not ref_line is None:
        draw_ref_line(ax, ref_line, "x")
    
    # show legends
    if show_legend:
        ax.legend(loc="best")
    
    # set title
    ax.set_title(title)
    
    # save figure
    if not fname is None:
        save_figure(fname, subdir, fmt, force=force)
        
    # return ax
    return ax   


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


def scatter2D_cluster(mbox, outliers=False, decision_boundary=True, cluster_boundary=True,
                      ell_std=2.0, magnetars=True, ref_line=None, ref_line_ax="x", ax=None,
                      title="", show_legend=True, fname=None, subdir=None, fmt="png", 
                      figsize=(10,8), force=False):
    """ 
    scatter 2D data by clusters
    """
    # get features
    xcol, ycol = mbox.features
    # get 2-D data with outliers column
    data = mbox.getFitData(include_all=True)
    # if outliers will be included;
    if outliers:
        # get outliers
        data = pd.concat((data, mbox.getOutlierData(include_all=True)))
    
    # check if magnetars is in the dataframe
    if magnetars:
        magnetars = "magnetar" in data.columns
    
    # set title
    if not title:
        title = mbox.catalog.upper()
        title += " (with EE)" if mbox.ee else " (without EE)"
        title += " {}".format(" - ".join(mbox.features))
    
    # display scatter plot seperately for each cluster
    for cid in mbox.getClusterLabels():
        # get cluster data
        cdata = mbox.getClusterData(cid, True)
        # set legend
        legend_label=None
        if show_legend:
            legend_label = "cluster-{}".format(cid)
        # scatter data
        ax = scatter2D(cdata, xcol, ycol, outliers=outliers, magnetars=magnetars, 
                       ref_line=ref_line, ref_line_ax=ref_line_ax, ax=ax, 
                       legend_label=legend_label, title="", figsize=figsize)
        
    # draw confidence ellipses
    if cluster_boundary:
        for cid in mbox.getClusterLabels():
            mean = mbox.clusters.getClusterMean(cid)
            cov = mbox.clusters.getClusterCov(cid)
            draw_ellipse(ax, mean, cov, ell_std, color="black")
    
    # draw decision boundary
    if decision_boundary:
        # if there is more then one active cluster
        if mbox.clusters.nActiveClusters() > 1:
            draw_cluster_boundary(ax, mbox.model, n_grid_points=500)
       
    # draw reference line (T=2sec)
    if not ref_line is None:
        draw_ref_line(ax, ref_line, ref_line_ax)
    
    # show legends
    if show_legend:
        ax.legend(loc="best")
    
    # set title
    ax.set_title(title)
    
    # save figure
    if not fname is None:
        save_figure(fname, subdir, fmt, force=force)
        
    return ax

def scatter3D(data, xcol, ycol, zcol, outliers=False, magnetars=None, title="", ax=None, 
              color=None, legend_label=None,
              fname=None, subdir=None, fmt="png", figsize=(16,9), force=False):
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
    dvalues = data[[xcol, ycol, zcol]].values
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
        ax = fig.add_subplot(111, projection='3d')
    
    # scatter non-outlier values 
    ax.scatter(dvalues[ovalues==False,0], dvalues[ovalues==False,1], dvalues[ovalues==False,2], 
               marker="o", edgecolor="black", facecolor=color, alpha=0.7, label=legend_label)
    # scatter outlier data
    if any(ovalues):
        ax.scatter(dvalues[ovalues==True,0], dvalues[ovalues==True,1], dvalues[ovalues==True,2],
                   marker="x", color="red", alpha=0.7, label="outliers")
    # scatter magnetars
    if any(mvalues):
        ax.scatter(dvalues[mvalues==True,0], dvalues[mvalues==True,1], dvalues[mvalues==True,2],
                   marker = '*', s=100, facecolor="gray", edgecolor="black", linewidth=1.,
                   label="magnetars")
 
    # set ax props
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_zlabel(zcol)
    
    # save figure
    if not fname is None:
        save_figure(fname, subdir, fmt, force=force)
    
    # return ax
    return ax

def scatter3D_cluster(mbox, outliers=False, cluster_boundary=True,
                      ell_std=2.0, magnetars=True, ax=None, title="", show_legend=True, 
                      fname=None, subdir=None, fmt="png", figsize=(16,9), force=False):
    """ 
    scatter 3-D data by clusters\n
    if cluster_boundary is True, then ellipsoids will be drawn
    """
    # get features
    xcol, ycol,zcol = mbox.features
    # get 3-D data with outliers column
    data = mbox.getFitData(include_all=True)
    # if outliers will be included;
    if outliers:
        # get outliers
        data = pd.concat((data, mbox.getOutlierData(include_all=True)))
    
    # check if magnetars is in the dataframe
    if magnetars:
        magnetars = "magnetar" in data.columns
    
    # set title
    if not title:
        title = mbox.catalog.upper()
        title += " (with EE)" if mbox.ee else " (without EE)"
        title += " {}".format(" - ".join(mbox.features))
    
    # get colors
    colors = get_color_cycle()
    
    # display scatter plot seperately for each cluster
    for cid in mbox.getClusterLabels():
        # get cluster data
        cdata = mbox.getClusterData(cid, True)
        # set legend
        legend_label=None
        if show_legend:
            legend_label = "cluster-{}".format(cid)
        
        # get color
        color = colors[cid]
        # scatter data
        ax = scatter3D(cdata, xcol, ycol, zcol, color=colors[cid], outliers=outliers, 
                       magnetars=magnetars, legend_label=legend_label, title="",  
                       ax=ax, figsize=figsize)
        # draw ellipsoid
        if cluster_boundary:
            mean = mbox.clusters.getClusterMean(cid)
            cov = mbox.clusters.getClusterCov(cid)
            draw_ellipsoid(ax, mean, cov, ell_std, color=color)
    
    # show legends
    if show_legend:
        ax.legend(loc="best")
    
    # set title
    ax.set_title(title)
    
    # save figure
    if not fname is None:
        save_figure(fname, subdir, fmt, force=force)
    
    # return ax
    return ax    


def plotModelScores(mboxes, score, marksize=60, ax=None, title="", xlabel="", ylabel="",
                    xticklabels=None, mark=None, **kwargs):
    """ 
    plots scores of multiple models \n
    if mark is highest/lowest, then highest/lowest score will be marked \n
    score shoud be one of {"aic", "bic," logL, "avg_logL", "avg_sil", "avg_sil2"} \n
    kwargs are arguments for plt.plot() function
    """
    
    # check if there is more than zero model
    if len(mboxes) == 0:
        raise Exception("there is no model!!!")
    
    # check if score is valid
    if not score in vars(mboxes[0].scores).keys() :
        raise Exception("invalid score: {}".format(score))
        
    # get scores
    scores = [vars(mbox.scores)[score] for mbox in mboxes]
    
    # check if there is invalid value in scores
    # if (None in scores) or (np.nan in scores) or (np.inf in scores) or (-np.inf in scores):
    #     raise Exception("there is invalid value in scores")
    
    # set xtick labels
    if xticklabels is None:
        # set xlabels as the model name
        xticklabels = [mbox.name for mbox in mboxes]
    # set xtick locations
    xticklocs = list(range(1, len(mboxes)+1))
    
    # set x-label
    if not xlabel:
        xlabel = "Models"
    # set y-label
    if not ylabel:
        ylabel = "Score"
    # set title
    if not title:
        title = "Model Scores"
    
    # create figure if not created 
    if ax is None:
        plt.figure()
        ax = plt.gca()
        
    # plot scores
    ax.plot(xticklocs, scores, label=score, **kwargs)
    # if mark highest or lowest is selected
    if not mark is None:
        # check mark is correct
        mark = mark.lower().strip()
        assert mark in {"highest", "lowest"}, "invalid mark option"
        if mark == "highest":
            idx = np.argmax(scores)
        else:
            idx = np.argmin(scores)
        xmark = xticklocs[idx]
        ymark = scores[idx]
        # marke score
        ax.scatter(xmark, ymark, s=marksize, marker="o", 
                    edgecolor="black", facecolor="gray", alpha=0.5, linewidth=1.2)
        
    # set figure options
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid()
    
    ax.tick_params(axis = 'both', which = 'major', labelsize=10)
    ax.set_xticks(list(xticklocs))
    ax.set_xticklabels(labels=xticklabels, rotation='vertical', fontsize=10)
    # set title
    ax.set_title(title, fontsize=12)
    # set legends
    leg = ax.legend(loc="best")
    leg.get_frame().set_facecolor('whitesmoke')
    
    
def plotModelScoresAll(mboxes, title="", marksize=60, xticklabels=None, figsize=(16,9),
                       **kwargs):
    """ 
    plots all scores of multiple models \n
    kwargs are arguments for plt.plot() function
    """
    
    # check if there is more than zero model
    if len(mboxes) == 0:
        raise Exception("there is no model!!!")

    # how many figures will be created ?
    num_plots = 6

    # create subplots
    fig, axes = plt.subplots(ncols=2, nrows=int(num_plots/2), constrained_layout=True)
    # get axes
    if num_plots == 4:
        (ax1, ax2), (ax3, ax4) = axes
    else:
        (ax1, ax2), (ax3, ax4), (ax5, ax6 )= axes
    
    # set figsize
    fig.set_size_inches(figsize)    
    
    # plot AIC scores
    plotModelScores(mboxes, "aic", ax=ax1, title="AIC", xlabel=" ", ylabel=" ", 
                    xticklabels=xticklabels, mark="lowest", marksize=marksize, **kwargs)
    # plot BIC scores
    plotModelScores(mboxes, "bic", ax=ax2, title="BIC", xlabel=" ", ylabel=" ", 
                    xticklabels=xticklabels, mark="lowest", marksize=marksize, **kwargs)
    # plot Log Likelihoods
    plotModelScores(mboxes, "logL", ax=ax3, title="Log Likelihood", xlabel=" ", ylabel=" ", 
                    xticklabels=xticklabels, mark="highest", marksize=marksize, **kwargs)
    # plot Average Log Likelihoods
    plotModelScores(mboxes, "avg_logL", ax=ax4, title="AVG Log Likelihood", 
                    xlabel=" ", ylabel=" ", xticklabels=xticklabels, mark="highest", 
                    marksize=marksize, **kwargs)
    # if num_plots == 6:
    # plot Average Silhouette Coeffs (Euclidean)
    plotModelScores(mboxes, "avg_sil", ax=ax5, title="AVG Silhouette Coeffs (Euclidean)", 
                    xlabel=" ", ylabel=" ",  xticklabels=xticklabels, mark="highest", 
                    marksize=marksize, **kwargs)
    # plot Average Silhouette Coeffs (Mahalanobis)
    plotModelScores(mboxes, "avg_sil2", ax=ax6, title="AVG Silhouette Coeffs (Mahalanobis)", 
                    xlabel=" ", ylabel=" ", xticklabels=xticklabels, mark="highest", 
                    marksize=marksize, **kwargs)
    
    # set title
    fig.suptitle(title, fontsize=16)
    
    
