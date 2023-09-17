import os
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib as mplot
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as st

from . import env, stats, utils, metrics as mets
from .gmm import GaussianMixtureModel

# get logger
logger = env.get_logger()

# customize seaborn
sns.set(font_scale=4)
# customize matplotlib
mplot.style.use("seaborn-paper")
mplot.rc("text", usetex=False)
font = {"size": 30}
mplot.rc("font", **font)

# Some constants
_FIGSIZE_HIST = (5, 3)
_FIGSIZE_SCATTER_2D = (5, 3)
_FIGSIZE_SCATTER_3D = (8, 8)


def _check_ax(
    ax: plt.Axes, figsize: Tuple[int, int], projection: str = "2d"
) -> plt.Axes:
    """
    Checks if the axes object is provided, and creates a new one if it isn't.

    Parameters:
    - ax (plt.Axes): The axes object to check.
    - figsize (tuple): The size of the figure to create if ax is None.
    - projection (str): The projection of the axes object. Default is "2d".

    Returns:
    - plt.Axes: The axes object.
    """
    if ax:
        return ax

    if projection == "2d":
        _, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    return ax


def _check_model_convergence(model: GaussianMixtureModel) -> None:
    """
    Checks if the model is fitted, and raises a ValueError if it isn't.

    Parameters:
    - model (GaussianMixtureModel): The model to check.

    Raises:
    - ValueError: If the model is not fitted.
    """

    if hasattr(model, "converged_") and not model.converged_:
        raise ValueError("Model is not fitted yet.")


def _check_dataframe_columns(
    df: pd.DataFrame, columns: Optional[List[str]] = None, n_cols: int = 1
) -> List[str]:
    """
    Check and validate provided columns against the dataframe. If columns are not provided,
    it returns the first n_cols columns from the dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe to check.
    - columns (List[str], optional): List of columns to validate. If not provided,
      the first n_cols from the dataframe will be returned.
    - n_cols (int, optional): Number of columns expected or to return if columns are not provided.
      Default is 1.

    Returns:
    - List[str]: List of valid columns.

    Raises:
    - ValueError: If columns length does not match n_cols or any of the provided columns
      don't exist in the dataframe or there aren't enough columns in the dataframe.
    """

    # If columns are provided, validate them.
    if columns is not None:
        # Check if the provided columns match the expected number of columns.
        if len(columns) != n_cols:
            raise ValueError(
                f"Expected {n_cols} columns, but {len(columns)} columns were provided."
            )

        # Check if the provided columns exist in the dataframe.
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"The following columns are not found in the dataframe: {', '.join(missing_columns)}"
            )

    # If columns are not provided, use the first n_cols from the dataframe.
    else:
        if len(df.columns) < n_cols:
            raise ValueError(
                f"The dataframe only has {len(df.columns)} columns, but {n_cols} were expected."
            )
        columns = df.columns[:n_cols]

    return columns


def _check_outlier_column(df: pd.DataFrame) -> None:
    """
    Check if the dataframe has an 'is_outlier' column. If it does, it raises a warning.
    """
    if any([col for col in df.columns if "is_outlier" in col]):
        logger.warning("Outliers detected but not removed!")


def get_color(index):
    colors = [
        "firebrick",
        "olivedrab",
        "royalblue",
        "mediumorchid",
        "peru",
        "darkorange",
        "darkcyan",
        "darkslategray",
        "darkkhaki",
        "darkseagreen",
    ]
    return colors[index]


def get_marker(index):
    markers = ["o", "^", "x", "P", "p", "*", "D", "s", "v", "+"]
    return markers[index]


def save_figure(
    filename: str = None, subdir: Optional[str] = None, fmt: str = "pdf"
) -> None:
    """
    Save the current active Matplotlib figure to a specified file format.

    Parameters:
    - filename (Optional[str]): The name of the file without the extension. If None, the figure will not be saved.
    - subdir (Optional[str]): Subdirectory under the base directory where the figure should be saved.
                             If None, the base directory will be used. Default is None.
    - fmt (str): Desired format for the figure. Supported formats are 'png', 'pdf', and 'svg'.
                 If multiple formats are desired, they can be comma-separated e.g., "png,pdf".
                 Default is 'pdf'.

    Notes:
    - The function saves the current active Matplotlib figure. Ensure that the figure you want to save is active when calling this function.
    """

    # If filename is None, don't save the figure
    if filename is None:
        return

    # Get base directory for saving figures
    dir_figure = env.DIR_FIGURES

    # If a subdirectory is provided, append it to the base directory, ensuring there's no leading '/'
    if subdir:
        subdir = subdir.lstrip("/")
        dir_figure = os.path.join(dir_figure, subdir)

    # Create the directory (and any intermediate directories) if they don't exist
    utils.create_directory(dir_figure)

    # Iterate over each format provided in 'fmt' and save the figure
    for format_ in fmt.split(","):
        # Construct the full path to save the figure
        full_path = os.path.join(dir_figure, filename)

        # If filename doesn't have the current format as its extension, append the format
        if not full_path.endswith(f".{format_}"):
            full_path += f".{format_}"

        # Save the current active Matplotlib figure to the constructed path
        plt.savefig(full_path, format=format_, bbox_inches="tight", dpi=300)


def histogram_1d(
    df: pd.DataFrame,
    column: str = None,
    n_bins: int = None,
    show_density_curve: bool = True,
    scatter: bool = True,
    show_outliers: bool = False,
    vertical_lines: list = [],
    title: str = "",
    xlabel: str = None,
    legend_label: str = None,
    color: str = None,
    figsize: Tuple[int, int] = _FIGSIZE_HIST,
    ax: plt.Axes = None,
    return_ax: bool = False,
    hist_kwargs: dict = dict(),
    save_kwargs: dict = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """
    Display a histogram of a specified column in a dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - column (str, optional): The column name to plot. If not specified, the first column in the dataframe will be used. Default is None.
    - n_bins (int, optional): Number of bins for the histogram. If None, the number of bins is automatically determined. Default is None.
    - show_density_curve (bool, optional): If True, displays a density curve. Default is True.
    - scatter (bool, optional): If True, displays a 1-D scatter plot. Default is True.
    - show_outliers (bool, optional): If True, displays outliers in the scatter plot. Outliers must be specified in the dataframe in "is_outlier" column. Default is False.
    - vertical_lines (list, optional): List of x-values where vertical lines should be displayed.
    - title (str, optional): Title of the plot.
    - xlabel (str, optional): Label for the x-axis.
    - legend_label (str): Label for the data in the legend.
    - figsize (Tuple[int, int]): Size of the plot.
    - ax (matplotlib.Axes, optional): Axes object to plot on. If not specified, a new figure object is created.
    - return_ax (bool, optional): If True, returns the Axes object. Default is False.
    - hist_kwargs (dict, optional): Arbitrary keyword arguments to be passed to sns.histplot function. Default is None.
    - save_kwargs (dict, optional): Arbitrary keyword arguments to be passed to save_figure function. Default is None.

    Returns:
    - plt.Axes: Matplotlib Axes object with the plot.

    Raises:
    - ValueError: If the specified column doesn't exist in the dataframe.
    """

    # check if column is provided
    column = _check_dataframe_columns(
        df, columns=[column] if column else None, n_cols=1
    )[0]

    # get X values
    X = df[column].values.flatten()

    # get flags for outliers
    is_outlier = np.repeat(False, repeats=X.shape[0])
    if "is_outlier" in df.columns:
        is_outlier = df["is_outlier"].astype(bool).values.flatten()
    elif show_outliers:
        logger.warning(
            "Cannot display outliers. 'is_outlier' column not found in dataframe"
        )
        show_outliers = False

    # Create a new figure if ax is not provided
    ax = _check_ax(ax, figsize)

    # Plot the histogram
    if show_outliers:
        # determine the number of bins
        if n_bins is None:
            n_bins = stats.compute_bin_size(X)
        sns.histplot(
            X,
            bins=n_bins,
            kde=show_density_curve,
            ax=ax,
            **hist_kwargs,
        )
    else:
        # determine the number of bins
        if n_bins is None:
            n_bins = stats.compute_bin_size(X[~is_outlier])
        sns.histplot(
            X[~is_outlier],
            bins=n_bins,
            kde=show_density_curve,
            ax=ax,
            **hist_kwargs,
        )

    # how many inliners and outliers
    n_inliers = np.sum(~is_outlier)
    n_outliers = np.sum(is_outlier)

    # Add scatter points if specified
    if scatter:
        ax.scatter(
            X[~is_outlier],
            -0.005 - 0.01 * np.random.random(n_inliers),
            marker=".",
            color=color or "black",
            label=legend_label or "Inlier",
        )

        if show_outliers:
            ax.scatter(
                X[is_outlier],
                -0.005 - 0.01 * np.random.random(n_outliers),
                marker="x",
                color="Red",
                label="Outlier",
            )

    # Add vertical lines if specified
    for vline in vertical_lines:
        ax.axvline(vline, linestyle="--", color="black")

    # Set title and labels
    ax.set_title(title or "")
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel("Frequency")

    # display legend if scatter is True and show_outliers is True
    if scatter and show_outliers:
        ax.legend(loc="best")

    # Save the plot if save_kwargs is provided
    if save_kwargs:
        save_figure(**save_kwargs)

    # Return the axis if specified
    if return_ax:
        return ax


def histogram_1d_with_clusters(
    model: GaussianMixtureModel,
    df: pd.DataFrame,
    column: str = None,
    n_bins: int = None,
    scatter: bool = True,
    show_decision_boundary: bool = False,
    title: str = None,
    xlabel: str = None,
    figsize: Tuple[int, int] = _FIGSIZE_HIST,
    ax: plt.Axes = None,
    return_ax: bool = False,
    save_kwargs: Optional[Dict] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """
    Plots a histogram with Gaussian Mixture Model clusters.

    Parameters:
    - model (GaussianMixtureModel): Fitted GMM model.
    - df (pd.DataFrame): Data source.
    - column (str): Column name from the DataFrame to plot.
    - n_bins (int): Number of bins for the histogram.
    - scatter (bool): If True, data points will be plotted as scatter below the histogram.
    - show_decision_boundary (bool): If True, decision boundaries will be plotted.
    - title (str): Title for the plot.
    - xlabel (str): Label for the x-axis.
    - figsize (Tuple[int, int]): Size of the plot.
    - ax (plt.Axes): Axes to plot on.
    - return_ax (bool): If True, returns the plot axes.
    - save_kwargs (Optional[Dict]): If provided, saves the plot using the arguments from this dictionary.

    Returns:
    - plt.Axes: Axes with the plotted histogram and clusters.
    """

    # Check if model is fitted
    _check_model_convergence(model)

    # check if column is provided
    column = _check_dataframe_columns(df, columns=[column], n_cols=1)[0]

    # check if outliers are present
    _check_outlier_column(df)

    # Data extraction
    X = df[column].values.reshape(-1, 1)

    # Create a new figure if ax is not provided
    ax = _check_ax(ax, figsize)

    # plot the histogram
    ax = histogram_1d(
        df=df,
        column=column,
        n_bins=n_bins,
        scatter=False,
        title=title,
        xlabel=xlabel,
        show_outliers=False,
        show_density_curve=False,
        ax=ax,
        return_ax=True,
        hist_kwargs={
            "stat": "density",
            "element": "step",
            "fill": False,
            "color": "black",
        },
    )

    # Plot clusters
    cluster_params = model.get_component_params()
    xmin, xmax = ax.get_xlim()
    xspace = np.linspace(xmin, xmax, num=1000).reshape(-1, 1)

    y_pred = model.predict(X)

    for cluster_id in range(model.n_components):
        # get the cluster parameters
        mean = cluster_params[cluster_id]["mean"]
        cov = cluster_params[cluster_id]["covariance"]
        weight = cluster_params[cluster_id]["weight"]
        std = np.sqrt(cov[0, 0])

        density = st.norm.pdf(xspace, loc=mean, scale=std) * weight
        # get legend label
        legend_label = "Cluster-{}".format(cluster_id + 1)
        # get color
        color = get_color(cluster_id)
        ax.plot(
            xspace,
            density,
            linestyle="solid",
            linewidth=1.3,
            label=legend_label,
            color=color,
        )

        if scatter:
            n_points = np.sum(y_pred == cluster_id)
            if n_points > 0:
                ax.scatter(
                    X[y_pred == cluster_id],
                    -0.005 - 0.01 * np.random.random(n_points),
                    marker=".",
                    color=color,
                )

    # if specified is True, plot the decision boundary
    if show_decision_boundary:
        transitions = np.where(np.diff(model.predict(xspace)))[0]
        for t in transitions:
            ax.axvline(xspace[t], color="black", linestyle="dotted")

    # Set plot title and x-label
    ax.set_title(title or "")
    ax.set_xlabel(xlabel or column)
    # display legends
    ax.legend(loc="best")

    if save_kwargs:
        save_figure(**save_kwargs)

    if return_ax:
        return ax


def histogram_2d(
    df: pd.DataFrame,
    columns: Tuple[str, str] = None,
    n_bins: Tuple[int, int] = 10,
    cbar_label: str = "Count",
    title: str = "",
    xlabel: str = None,
    ylabel: str = None,
    color_map: str = "viridis",
    figsize: Tuple[int, int] = (10, 7),
    ax: plt.Axes = None,
    return_ax: bool = False,
    hist2d_kwargs: dict = dict(),
    save_kwargs: dict = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """
    Display a 2D histogram of two specified columns in a dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - columns (Tuple[str, str]): The column names to plot.
    - n_bins (Tuple[int, int], optional): Number of bins for the histograms. Defaults are None.
    - cbar_label (str): Label for the colorbar.
    - title (str, optional): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - color_map (str): Colormap to be used.
    - figsize (Tuple[int, int]): Size of the plot.
    - ax (matplotlib.Axes, optional): Axes object to plot on. If not specified, a new figure object is created.
    - return_ax (bool, optional): If True, returns the Axes object. Default is False.
    - hist2d_kwargs (dict, optional): Arbitrary keyword arguments to be passed to ax.hist2d function. Default is None.
    - save_kwargs (dict, optional): Arbitrary keyword arguments to be passed to save_figure function. Default is None.

    Returns:
    - plt.Axes: Matplotlib Axes object with the plot.

    Raises:
    - ValueError: If any of the specified columns don't exist in the dataframe.
    """

    # check if columns are provided
    columns = _check_dataframe_columns(df, columns, n_cols=2)

    # check if outliers are present
    _check_outlier_column(df)

    # Extract column names
    xcol, ycol = columns

    # Extract data
    X = df[xcol].values
    Y = df[ycol].values

    # Create a new figure if ax is not provided
    ax = _check_ax(ax, figsize)

    # Plot the 2D histogram
    cax = ax.hist2d(X, Y, bins=n_bins, cmap=color_map, **hist2d_kwargs)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel or columns[0])
    ax.set_ylabel(ylabel or columns[1])

    # Adding a colorbar
    cbar = plt.colorbar(cax[3], ax=ax)
    cbar.set_label(cbar_label)

    # Save the plot if save_kwargs is provided
    if save_kwargs:
        save_figure(**save_kwargs)

    # Return the axis if specified
    if return_ax:
        return ax


def scatter_2d(
    df: pd.DataFrame,
    columns: Tuple[str, str] = None,
    show_outliers: bool = False,
    title: str = "",
    xlabel: str = None,
    ylabel: str = None,
    legend_label: str = None,
    color: str = "blue",
    alpha: float = 0.8,
    marker: str = ".",
    marker_size: int = 20,
    figsize: Tuple[int, int] = _FIGSIZE_SCATTER_2D,
    ax: plt.Axes = None,
    return_ax: bool = False,
    save_kwargs: dict = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """
    Display a scatter plot of two columns from a dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - columns (tuple, optional): The columns to be plotted. If specified, the tuple length must be 2. If not
                                 specified, the first 2 columns in the dataframe will be used. Default is None.
    - show_outliers (bool): Flag to show outliers in the plot.
    - title (str): Title of the scatter plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - legend_label (str): Label for the data in the legend.
    - color (str): Color of the data points.
    - alpha (float): Transparency of the data points.
    - marker (str): Marker style.
    - marker_size (int): Size of the marker.
    - figsize (tuple): Size of the figure (only used if ax is None).
    - ax (plt.Axes, optional): Axes object to plot on. If None, a new figure will be created.
    - return_ax (bool, optional): If True, returns the Axes object. Default is False.
    - save_kwargs (dict, optional): Arguments to pass to a save function if you want to save the figure.

    Returns:
    - plt.Axes: Matplotlib Axes object containing the scatter plot.

    Raises:
    - ValueError: If the specified column(s) doesn't exist in the dataframe.
    """

    # check if columns are provided
    columns = _check_dataframe_columns(df, columns, n_cols=2)

    # Extract column names
    xcol, ycol = columns

    # Extract data
    X = df[xcol].values
    Y = df[ycol].values

    # Handle outliers
    outlier_mask = np.array([False] * len(X))
    if "is_outlier" in df.columns:
        outlier_mask = df["is_outlier"].values
    elif show_outliers:
        logger.warning(
            "Cannot display outliers. 'is_outlier' column not found in dataframe"
        )
        show_outliers = False

    # Create a new figure if ax is not provided
    ax = _check_ax(ax, figsize)

    # Plot outliers
    if show_outliers:
        ax.scatter(
            X[outlier_mask],
            Y[outlier_mask],
            color="red",
            marker="x",
            s=marker_size,
            label="Outlier",
            alpha=alpha,
        )

    # Plot main data
    ax.scatter(
        X[~outlier_mask],
        Y[~outlier_mask],
        color=color,
        marker=marker,
        label=legend_label or "Inlier",
        alpha=alpha,
        s=marker_size,
    )

    # Set labels and title
    ax.set_xlabel(xlabel or xcol)
    ax.set_ylabel(ylabel or ycol)
    ax.set_title(title)

    # Display legend
    if legend_label or show_outliers:
        ax.legend(loc="best")

    # Save the plot if save_kwargs are provided
    if save_kwargs:
        save_figure(**save_kwargs)

    # Return the axis if specified
    if return_ax:
        return ax


def scatter_2d_with_clusters(
    model: GaussianMixtureModel,
    df: pd.DataFrame,
    columns: Optional[Tuple[str, str]] = None,
    show_decision_boundary: bool = False,
    show_cluster_centers: bool = False,
    show_confidence_ellipses: bool = False,
    confidence_level: float = 0.95,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    alpha: float = 0.8,
    marker_size: int = 20,
    figsize: Tuple[int, int] = _FIGSIZE_SCATTER_2D,
    ax: Optional[plt.Axes] = None,
    return_ax: bool = False,
    save_kwargs: Optional[Dict] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """
    Scatter plot the data in 2D while highlighting clusters using the GaussianMixtureModel.

    Parameters:
    - model: Trained GaussianMixtureModel.
    - df: Dataframe containing the data to be plotted.
    - columns: Tuple of column names to be plotted. If None, the first two columns of df are used.
    - show_decision_boundary: Whether to display the decision boundary.
    - show_cluster_centers: Whether to display the cluster centers.
    - show_confidence_ellipses: Whether to display confidence ellipses for clusters.
    - confidence_level: Confidence level for the ellipses.
    - title: Title for the plot.
    - xlabel: X-axis label.
    - ylabel: Y-axis label.
    - marker_size: Size of the scatter plot markers.
    - alpha: Transparency level for scatter points.
    - figsize: Size of the figure (if a new one is created).
    - ax: Existing matplotlib axis object to plot on. If None, a new figure and axis are created.
    - return_ax: Whether to return the axis object.
    - save_kwargs: Keyword arguments for save_figure function if you wish to save the plot.
    - **kwargs: Other keyword arguments.

    Returns:
    - ax: Matplotlib axis object if return_ax is True, else None.
    """

    # Check if model is fitted
    _check_model_convergence(model)

    # check if columns are provided
    columns = _check_dataframe_columns(df, columns, n_cols=2)

    # check if outliers are present
    _check_outlier_column(df)

    # Extract column data
    xcol, ycol = columns
    X = df[xcol].values
    Y = df[ycol].values
    XY = np.column_stack((X, Y))

    # Create a new figure if ax is not provided
    ax = _check_ax(ax, figsize)

    # get predictions
    y_pred = model.predict(XY)

    # plot clusters
    for cluster_id in range(model.n_components):
        # get color for the cluster
        color = get_color(cluster_id)
        # get marker for the cluster
        marker = get_marker(cluster_id)
        # get legend label
        legend_label = "Cluster-{}".format(cluster_id + 1)
        # scatter plot for the cluster
        ax = scatter_2d(
            df.loc[y_pred == cluster_id, columns],
            ax=ax,
            legend_label=legend_label,
            color=color,
            alpha=alpha,
            marker=marker,
            marker_size=marker_size,
            return_ax=True,
        )

    # show decision boundaries
    if show_decision_boundary:
        draw_cluster_boundary(model=model, ax=ax)

    # get cluster params
    cluster_params = model.get_component_params()

    # if specified, display gaussians
    if show_confidence_ellipses:
        for cluster_id in range(model.n_components):
            # get cluster params
            mean = cluster_params[cluster_id]["mean"]
            cov = cluster_params[cluster_id]["covariance"]
            draw_confidence_ellipse_2d(
                mean=mean,
                cov=cov,
                confidence_level=confidence_level,
                ax=ax,
                color="black",
                linestyle="dotted",
                linewidth=1.5,
            )

    # if specified, display cluster centers
    if show_cluster_centers:
        for cluster_id in range(model.n_components):
            # get cluster params
            mean = cluster_params[cluster_id]["mean"]
            ax.scatter(mean[0], mean[1], marker="*", color="black", s=100)

    # Set plot title and x-label
    ax.set_title(title or "")
    ax.set_xlabel(xlabel or xcol)
    ax.set_ylabel(ylabel or ycol)
    # display legends
    ax.legend(loc="best")

    if save_kwargs:
        save_figure(**save_kwargs)

    if return_ax:
        return ax


def scatter_3d(
    df: pd.DataFrame,
    columns: Tuple[str, str, str] = None,
    show_outliers: bool = False,
    title: str = "",
    xlabel: str = None,
    ylabel: str = None,
    zlabel: str = None,
    legend_label: str = None,
    color: str = "blue",
    alpha: float = 0.8,
    marker: str = ".",
    marker_size: int = 10,
    figsize: Tuple[int, int] = _FIGSIZE_SCATTER_3D,
    ax: plt.Axes = None,
    return_ax: bool = False,
    save_kwargs: dict = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """
    Display a scatter plot of three columns from a dataframe in 3D.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - columns (tuple, optional): The columns to be plotted. If specified, the tuple length must be 3. If not
                                 specified, the first 3 columns in the dataframe will be used. Default is None.
    - show_outliers (bool): Flag to show outliers in the plot.
    - title (str): Title of the scatter plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - zlabel (str): Label for the z-axis.
    - legend_label (str): Label for the data in the legend.
    - color (str): Color of the data points.
    - alpha (float): Transparency of the data points.
    - marker (str): Marker style.
    - marker_size (int): Size of the marker.
    - figsize (tuple): Size of the figure (only used if ax is None).
    - ax (optional): Axes3D object to plot on. If None, a new figure will be created.
    - return_ax (bool, optional): If True, returns the Axes object. Default is False.
    - save_kwargs (dict, optional): Arguments to pass to a save function if you want to save the figure.

    Returns:
    - plt.Axes: Matplotlib Axes3D object containing the scatter plot.

    Raises:
    - ValueError: If the specified column(s) don't exist in the dataframe.
    """

    # check if columns are provided
    columns = _check_dataframe_columns(df, columns, n_cols=3)

    # Extract column names
    xcol, ycol, zcol = columns

    # Extract data
    X = df[xcol].values
    Y = df[ycol].values
    Z = df[zcol].values

    # Handle outliers
    outlier_mask = np.array([False] * len(X))
    if "is_outlier" in df.columns:
        outlier_mask = df["is_outlier"].values
    elif show_outliers:
        logger.warning(
            "Cannot display outliers. 'is_outlier' column not found in dataframe"
        )
        show_outliers = False

    # Create a new figure if ax is not provided
    ax = _check_ax(ax, figsize, "3d")

    # Plot outliers
    if show_outliers:
        ax.scatter(
            X[outlier_mask],
            Y[outlier_mask],
            Z[outlier_mask],
            color="red",
            marker="x",
            label="Outlier",
        )

    # Plot main data
    ax.scatter(
        X[~outlier_mask],
        Y[~outlier_mask],
        Z[~outlier_mask],
        color=color,
        alpha=alpha,
        marker=marker,
        s=marker_size,
        label=legend_label or "Inlier",
    )

    # Set labels and title
    ax.set_xlabel(xlabel or xcol)
    ax.set_ylabel(ylabel or ycol)
    ax.set_zlabel(zlabel or zcol)
    ax.set_title(title)

    # Display legend
    if legend_label or show_outliers:
        ax.legend(loc="best")

    # Save the plot if save_kwargs are provided
    if save_kwargs:
        save_figure(**save_kwargs)

    # Return the axis if specified
    if return_ax:
        return ax


def scatter_3d_with_clusters(
    model: GaussianMixtureModel,
    df: pd.DataFrame,
    columns: Optional[Tuple[str, str, str]] = None,
    show_cluster_centers: bool = False,
    show_confidence_ellipsoids: bool = False,
    confidence_level: float = 0.95,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    alpha: float = 0.8,
    marker_size: int = 5,
    figsize: Tuple[int, int] = _FIGSIZE_SCATTER_3D,
    ax: Optional[plt.Axes] = None,
    return_ax: bool = False,
    save_kwargs: Optional[Dict] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """
    Scatter plot the data in 2D while highlighting clusters using the GaussianMixtureModel.

    Parameters:
    - model: Trained GaussianMixtureModel.
    - df: Dataframe containing the data to be plotted.
    - columns: Tuple of column names to be plotted. If None, the first two columns of df are used.
    - show_cluster_centers: Whether to display the cluster centers.
    - show_confidence_ellipsoids: Whether to display confidence ellipsoids for clusters.
    - confidence_level: Confidence level for the ellipses.
    - title: Title for the plot.
    - xlabel: X-axis label.
    - ylabel: Y-axis label.
    - zlabel: Z-axis label.
    - alpha: Transparency level for scatter points.
    - marker_size: Size of the scatter plot markers.
    - figsize: Size of the figure (if a new one is created).
    - ax: Existing matplotlib axis object to plot on. If None, a new figure and axis are created.
    - return_ax: Whether to return the axis object.
    - save_kwargs: Keyword arguments for save_figure function if you wish to save the plot.
    - **kwargs: Other keyword arguments.

    Returns:
    - ax: Matplotlib axis object if return_ax is True, else None.
    """

    # Check if model is fitted
    _check_model_convergence(model)

    # check if columns are provided
    columns = _check_dataframe_columns(df, columns, n_cols=3)

    # check if outliers are present
    _check_outlier_column(df)

    # Extract column data
    xcol, ycol, zcol = columns
    X = df[xcol].values
    Y = df[ycol].values
    Z = df[zcol].values
    XYZ = np.column_stack((X, Y, Z))

    # Create a new figure if ax is not provided
    ax = _check_ax(ax, figsize, "3d")

    # get predictions
    y_pred = model.predict(XYZ)

    # plot clusters
    for cluster_id in range(model.n_components):
        # get color for the cluster
        color = get_color(cluster_id)
        # get marker for the cluster
        marker = get_marker(cluster_id)
        # get legend label
        legend_label = "Cluster-{}".format(cluster_id + 1)
        # scatter plot for the cluster
        ax = scatter_3d(
            df.loc[y_pred == cluster_id, columns],
            ax=ax,
            legend_label=legend_label,
            color=color,
            alpha=alpha,
            marker=marker,
            marker_size=marker_size,
            return_ax=True,
        )

    # get cluster params
    cluster_params = model.get_component_params()

    # if specified, display gaussians
    if show_confidence_ellipsoids:
        for cluster_id in range(model.n_components):
            # get cluster params
            mean = cluster_params[cluster_id]["mean"]
            cov = cluster_params[cluster_id]["covariance"]
            draw_confidence_ellipsoid_3d(
                mean=mean,
                cov=cov,
                confidence_level=confidence_level,
                ax=ax,
                color=get_color(cluster_id),
                linestyle="dotted",
                linewidth=0.1,
                alpha1=0.3,
                alpha2=0.4,
            )

    # if specified, display cluster centers
    if show_cluster_centers:
        for cluster_id in range(model.n_components):
            # get cluster params
            mean = cluster_params[cluster_id]["mean"]
            ax.scatter(mean[0], mean[1], mean[2], marker="*", color="black", s=40)

    # Set plot title and x-label
    ax.set_title(title or "")
    ax.set_xlabel(xlabel or xcol)
    ax.set_ylabel(ylabel or ycol)
    ax.set_zlabel(zlabel or zcol)
    # display legends
    ax.legend(loc="best")

    if save_kwargs:
        save_figure(**save_kwargs)

    if return_ax:
        return ax


def plot_data(
    df: pd.DataFrame, cols: list, ax: plt.Axes = None, return_ax: bool = False, **kwargs
) -> Optional[plt.Axes]:
    """
    Plots the data from a given DataFrame based on the provided columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be plotted.
    - cols (list): The columns to be plotted. The list length determines the type of plot:
        1 column  -> Histogram
        2 columns -> 2D Scatter Plot
        3 columns -> 3D Scatter Plot
    - ax (plt.Axes, optional): The axis on which to plot the data. If None, a new figure and axis will be created.
    - return_ax (bool, optional): If True, returns the Axes object. Default is False.
    - **kwargs: Additional arguments passed to the plotting functions.

    Returns:
    - plt.Axes: The axis containing the plot.

    Raises:
    - ValueError: If the length of cols is neither 1, 2, nor 3.
    """

    num_cols = len(cols)

    # Plot based on the number of columns provided
    if num_cols == 1:
        return histogram_1d(df, cols[0], ax=ax, return_ax=return_ax, **kwargs)
    elif num_cols == 2:
        return scatter_2d(df, cols, ax=ax, return_ax=return_ax, **kwargs)
    elif num_cols == 3:
        return scatter_3d(df, cols, ax=ax, return_ax=return_ax, **kwargs)
    else:
        raise ValueError("The length of 'cols' should be 1, 2, or 3.")


def plot_model(
    model: GaussianMixtureModel,
    df: pd.DataFrame,
    cols: list,
    ax: plt.Axes = None,
    return_ax: bool = False,
    **kwargs,
) -> Optional[plt.Axes]:
    """
    Plots the data and clusters from the GaussianMixtureModel based on the provided columns.

    Parameters:
    - model (GaussianMixtureModel): The Gaussian mixture model used to predict clusters.
    - df (pd.DataFrame): The DataFrame containing the data to be plotted.
    - cols (list): The columns to be plotted. The list length determines the type of plot:
        1 column  -> Histogram with clusters
        2 columns -> 2D Scatter Plot with clusters
        3 columns -> 3D Scatter Plot with clusters
    - ax (plt.Axes, optional): The axis on which to plot the data. If None, a new figure and axis will be created.
    - return_ax (bool, optional): If True, returns the Axes object. Default is False.
    - **kwargs: Additional arguments passed to the plotting functions.

    Returns:
    - plt.Axes: The axis containing the plot.

    Raises:
    - ValueError: If the length of cols is neither 1, 2, nor 3.
    """

    num_cols = len(cols)

    # Plot based on the number of columns provided
    if num_cols == 1:
        return histogram_1d_with_clusters(
            model, df, cols[0], ax=ax, return_ax=return_ax, **kwargs
        )
    elif num_cols == 2:
        return scatter_2d_with_clusters(
            model, df, cols, ax=ax, return_ax=return_ax, **kwargs
        )
    elif num_cols == 3:
        return scatter_3d_with_clusters(
            model, df, cols, ax=ax, return_ax=return_ax, **kwargs
        )
    else:
        raise ValueError("The length of 'cols' should be 1, 2, or 3.")


def plot_models_as_grid(
    models: List[GaussianMixtureModel],
    df: pd.DataFrame,
    cols: List[str],
    n_ax_columns: int = 3,
    title: str = None,
    figsize: Tuple[int, int] = (15, 8),
    wspace: float = 0.02,
    hspace: float = 0.02,
    return_axes: bool = False,
    save_kwargs: Optional[Dict] = None,
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Plot Gaussian Mixture Models in a grid layout.

    Parameters:
    -----------
    models : List[GaussianMixtureModel]
        A list of Gaussian Mixture Models to be plotted.

    df : pd.DataFrame
        The DataFrame containing the data to be plotted.

    cols : List[str]
        The columns in the DataFrame to be plotted. The number of columns determines the type of plot.

    figsize : Tuple[int, int], optional, default=(15, 8)
        The size of the entire grid figure.

    wspace : float, default=0.02
        Vertical space between axes.

    hspace : float, default=0.02
        Horizontal space between axes.

    n_ax_columns : int, optional, default=3
        The number of columns in the grid.

    return_axes : bool, optional, default=False
        If True, the function returns the numpy array of axes used in the grid.

    save_kwargs : Dict, optional, default=None
        Keyword arguments for save_figure function if you wish to save the plot.

    **kwargs :
        Additional keyword arguments to be passed to the `plot_model` function.

    Returns:
    --------
    np.ndarray, optional
        A numpy array of Axes objects. Returned only if `return_axes` is True.

    """

    # Calculate the number of rows needed in the grid based on the number of models and desired columns.
    n_rows = int(np.ceil(len(models) / n_ax_columns))

    # Create a grid of subplots. All subplots share the same x and y axes for consistency.
    if len(cols) <= 2:
        fig, axes = plt.subplots(
            n_rows, n_ax_columns, figsize=figsize, sharex=True, sharey=True
        )

    # If the number of columns is 3, we need to use 3D axes.x
    else:
        fig = plt.figure(figsize=figsize)
        axes = np.empty((n_rows, n_ax_columns), dtype=object)
        for i in range(n_rows):
            for j in range(n_ax_columns):
                ax = fig.add_subplot(
                    n_rows, n_ax_columns, i * n_ax_columns + j + 1, projection="3d"
                )
                axes[i, j] = ax

    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Flatten the 2D array of axes for ease of iteration.
    axes = axes.ravel()

    # Plot each model on its respective axis.
    for idx, (model, ax) in enumerate(zip(models, axes)):
        plot_model(df=df, model=model, cols=cols, ax=ax, **kwargs)

    # Hide any remaining unused axes.
    for ax in axes[len(models) :]:
        ax.axis("off")

    # Set the title of the figure.
    if title:
        fig.suptitle(title, fontsize=25)

    # Save the plot if save_kwargs is provided
    if save_kwargs:
        save_figure(**save_kwargs)

    # If the caller wants the axes objects returned, do so.
    if return_axes:
        return axes


def meshgrid_nd(*arrays) -> List[np.ndarray]:
    """
    Create an N-dimensional mesh grid.

    This function creates coordinate matrices from coordinate vectors. It behaves
    similarly to `numpy.meshgrid` with some minor modifications to handle N-dimensional grids.

    Parameters:
    *arrays : array_like
        One or more arrays defining the grid edges.

    Returns:
    list of ndarray
        N-D coordinate matrices.

    Examples:
    >>> x = [1, 2, 3]
    >>> y = [4, 5]
    >>> meshgrid_nd(x, y)
    [array([[1, 1],
            [2, 2],
            [3, 3]]),
     array([[4, 5],
            [4, 5],
            [4, 5]])]
    """

    # Ensure the input is in tuple format
    arrays = tuple(arrays)
    dimensions = len(arrays)

    # Calculate the size of the output arrays
    shape = [len(arr) for arr in arrays]

    # List to store the resultant meshgrids
    meshgrids = []

    for i, arr in enumerate(arrays):
        # Create a shape filled with ones, then modify the current axis with the size of the current array
        repeat_shape = np.ones(dimensions, dtype=int)
        repeat_shape[i] = len(arr)

        # Create an array with the right shape to be repeated in other dimensions
        temp_array = np.asarray(arr).reshape(repeat_shape)

        # Calculate repetition factors for the current array
        repeat_factors = shape.copy()
        repeat_factors[i] = 1

        # Create the meshgrid by repeating the current array as required
        grid = temp_array.repeat(repeat_factors, axis=0)

        # Append the meshgrid to our list
        meshgrids.append(grid)

    return meshgrids


def create_grid_space(
    x_range: Tuple[float, float],
    x_points: int,
    y_range: Tuple[float, float],
    y_points: int,
    z_range: Optional[Tuple[float, float]] = None,
    zpoints: Optional[int] = None,
):
    """
    Create a 2D/3D grid space for given range and points.

    Parameters:
    - x_range (tuple of floats): X-axis range.
    - x_points (int): Number of grid points for the X-axis.
    - y_range (tuple of floats): Y-axis range.
    - y_points (int): Number of grid points for the Y-axis.
    - z_range (tuple of floats, Optional): Z-axis range.
    - z_points (int, Optional): Number of grid points for the Z-axis.


    Returns:
        - if z_range and z_points are provided:
            - X, Y, Z, grid_space (np.array, np.array, np.array, np.array): X, Y, Z coordinates and grid_space.
        - else:
            - X, Y, grid_space (np.array, np.array, np.array): X, Y coordinates and grid_space.
    """
    xmin, xmax = x_range
    x = np.linspace(xmin, xmax, x_points)

    ymin, ymax = y_range
    y = np.linspace(ymin, ymax, y_points)

    if z_range is not None and zpoints is not None:
        zmin, zmax = z_range
        z = np.linspace(zmin, zmax, zpoints)

        X, Y, Z = meshgrid_nd(x, y, z)
        grid_space = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T

        return X, Y, Z, grid_space

    else:
        X, Y = np.meshgrid(x, y)
        grid_space = np.array([X.ravel(), Y.ravel()]).T

        return X, Y, grid_space


def draw_cluster_boundary(
    model: GaussianMixtureModel, ax: plt.Axes, n_grid_points: int = 500, ax_range=None
):
    """
    Draw decision boundary of a model on the provided axes.

    Parameters:
    - model (object): Trained clustering model with predict method.
    - ax (plt.Axes): Axis to plot on.
    - n_grid_points (int, default 500): Number of grid points to create in each dimension.
    - ax_range (tuple of tuples, optional): Specify axis range in format ((xmin, xmax), (ymin, ymax)).
                                            If None, it will be inferred from ax.

    Returns:
    - ax (plt.Axes): Axes with the plotted decision boundary.
    """

    # Get axes ranges
    x_range, y_range = ax_range or (ax.get_xlim(), ax.get_ylim())

    # Create grid space
    _, _, grid_space = create_grid_space(x_range, n_grid_points, y_range, n_grid_points)

    # Predict cluster assignment for each grid point
    predictions = model.predict(grid_space).reshape(n_grid_points, n_grid_points)

    # Draw contour of decision boundary on the provided axis
    ax.contour(predictions, origin="lower", extent=(*x_range, *y_range), colors="black")

    return ax


def draw_confidence_ellipse_2d(
    mean: Union[np.ndarray, list, tuple],
    cov: np.ndarray,
    confidence_level: float = 0.95,
    ax: plt.Axes = None,
    color: str = "black",
    linestyle: str = "dotted",
    linewidth: float = 1.0,
) -> plt.Axes:
    """
    Draw a 2D Gaussian ellipse on a given axis.

    Parameters:
    - mean : array-like of shape (2,)
        The mean of the Gaussian distribution.
    - cov : array-like of shape (2, 2)
        The covariance matrix of the Gaussian distribution.
    - confidence_level : float, default=0.95
        The desired confidence level for the ellipse. Common values:
        - 0.68 corresponds to 1 standard deviation in a Gaussian distribution (~68% of data points).
        - 0.95 corresponds to 2 standard deviations (~95% of data points).
        - 0.99 corresponds to 3 standard deviations (~99% of data points).
    - ax : matplotlib axis
        The axis on which the ellipse will be drawn.
    - color : str, default="black"
        Color of the ellipse edge.
    - linestyle : str, default="dotted"
        Style of the ellipse edge.
    - linewidth : float, default=1.0
        Width of the ellipse edge.

    Returns:
    - Ellipse artist added to the axis.
    """

    # Convert confidence level to equivalent chi-squared value
    chi2_val = st.chi2.ppf(confidence_level, df=2)

    # Calculate eigenvalues and eigenvectors
    eig_val, eig_vec = np.linalg.eigh(cov)
    # Sort eigenvectors by eigenvalues in descending order
    order = eig_val.argsort()[::-1]
    eig_val, eig_vec = eig_val[order], eig_vec[:, order]

    # Calculate ellipse parameters
    width, height = 2 * np.sqrt(chi2_val * eig_val)
    angle = np.arctan2(*eig_vec[:, 0][::-1])

    # Draw ellipse
    ell = mplot.patches.Ellipse(
        mean,
        width,
        height,
        np.degrees(angle),
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        linewidth=linewidth,
    )

    # create a new figure if ax is not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.add_patch(ell)
    return ax


def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3, n_points=100):
    """
    Generate the 3D points representing the ellipsoid defined by a covariance matrix.

    Parameters:
    - cov (ndarray): 3x3 covariance matrix.
    - mu (ndarray, optional): 3D vector representing the ellipsoid's center. Default is [0, 0, 0].
    - nstd (float, optional): Number of standard deviations to determine the radii of the ellipsoid.
    - n_points (int, optional): Number of points to generate for the ellipsoid.

    Returns:
    - tuple: X, Y, Z arrays representing the ellipsoid's 3D coordinates.

    Example:
    - X, Y, Z = get_cov_ellipsoid(cov_matrix, mu_vector)
    """

    # Ensure covariance matrix shape is correct
    assert cov.shape == (3, 3), "Covariance matrix should be of shape (3, 3)."

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Spherical angles for ellipsoid
    theta = np.linspace(0, 2 * np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Ellipsoid radii corresponding to the eigenvalues (scaled by number of standard deviations)
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Ellipsoid points in its standard position
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Flatten for rotation
    old_shape = X.shape
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

    # Rotate ellipsoid using eigenvectors
    X, Y, Z = np.dot(eigvecs, np.array([X, Y, Z]))
    X, Y, Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)

    # Translate ellipsoid to its mean
    X += mu[0]
    Y += mu[1]
    Z += mu[2]

    return X, Y, Z


def draw_confidence_ellipsoid_3d(
    mean: np.ndarray,
    cov: np.ndarray,
    confidence_level: float = 0.95,
    ax: plt.Axes = None,
    color: str = "black",
    linestyle: str = "dotted",
    linewidth: float = 0.5,
    alpha1: float = 0.3,
    alpha2: float = 0.2,
) -> plt.Axes:
    """
    Draw a confidence ellipsoid on a 3D axis.

    Parameters:
    - mean (ndarray): 3D vector indicating the ellipsoid's center.
    - cov (ndarray): 3x3 covariance matrix.
    - confidence_level (float): Confidence level used to scale the ellipsoid.
    - ax : matplotlib 3D axis object.
    - color (str): Color for the ellipsoid surface.
    - linestyle (str): Linestyle for the ellipsoid surface.
    - linewidth (float): Linewidth for the ellipsoid surface.
    - alpha1 (float): Transparency for the ellipsoid surface.
    - alpha2 (float): Transparency for the ellipsoid wireframe.
    """

    # Convert confidence level to number of standard deviations
    nstd = stats.confidence_to_nstd(confidence_level)

    # Calculate ellipsoid 3D points
    X, Y, Z = get_cov_ellipsoid(cov, mean, nstd=nstd)

    # Create a new figure if ax is not provided
    ax = _check_ax(ax, figsize=(6, 6), projection="3d")

    # Plot wireframe and surface
    ax.plot_wireframe(
        X,
        Y,
        Z,
        color="black",
        rstride=10,
        cstride=10,
        alpha=alpha1,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.plot_surface(
        X,
        Y,
        Z,
        color=color,
        rstride=10,
        cstride=10,
        alpha=alpha2,
        linewidth=linewidth,
        linestyle=linestyle,
    )

    return ax


def plot_scores(
    model_names: np.ndarray,
    scores: np.ndarray,
    highlight_extreme: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = "Models",
    ylabel: Optional[str] = "Score",
    score_legend: Optional[str] = None,
    linestyle: str = "dashed",
    linewidth: float = 1.0,
    marker: str = "o",
    marker_size: int = 5,
    extreme_marker_size: int = 35,
    xtick_rotation: str = "horizontal",
    color: str = "red",
    alpha: float = 0.8,
    figsize: Tuple[int, int] = (6, 3),
    ax: Optional[plt.Axes] = None,
    return_ax: bool = False,
) -> Optional[plt.Axes]:
    """
    Plot model scores with optional highlighting of the highest or lowest score.

    Parameters:
    ----------
    model_names : np.ndarray
        Array of model names.
    scores : np.ndarray
        Array of corresponding scores.
    highlight_extreme : str, optional
        It is either 'highest' or 'lowest'. Default is None (no highlighting).
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label. Default is 'Models'.
    ylabel : str, optional
        Y-axis label. Default is 'Score'.
    score_legend : str, optional
        Legend for the scores.

    return_ax : bool, optional
        Whether to return the Axes object. Default is False.

    Returns:
    -------
    ax : plt.Axes, optional
        If `return_ax` is True, return the Axes object.
    """

    assert len(model_names) == len(
        scores
    ), "model_names and scores should be of the same length"

    xticklocs = np.arange(len(model_names))

    ax = ax or plt.figure(figsize=figsize).add_subplot()

    ax.plot(
        xticklocs,
        scores,
        marker=marker,
        markersize=marker_size,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=score_legend,
        alpha=alpha,
    )

    if highlight_extreme is not None:
        idx = (
            np.nanargmax(scores)
            if highlight_extreme == "highest"
            else np.nanargmin(scores)
        )
        ax.scatter(
            xticklocs[idx],
            scores[idx],
            s=extreme_marker_size,
            marker="o",
            edgecolor="black",
            facecolor="black",
            alpha=1,
            linewidth=1.2,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_xticks(xticklocs)
    ax.set_xticklabels(labels=model_names, rotation=xtick_rotation, fontsize=10)
    ax.set_title(title, fontsize=12)

    if score_legend:
        leg = ax.legend(loc="best")
        leg.get_frame().set_facecolor("whitesmoke")

    if return_ax:
        return ax


def _check_scores_xvalues(
    df_scores: pd.DataFrame, xvalues: Optional[List[Any]] = None
) -> np.ndarray:
    """
    Check if xvalues are provided and return them.

    Parameters:
    -----------

    df_scores : pd.DataFrame
        DataFrame containing clustering scores.

    xvalues : Optional[List[Any]], default=None
        X-values to be plotted. If not provided, will use the index of `df_scores`.

    Returns:
    --------
    xvalues : np.ndarray
        X-values to be plotted.
    """
    # check if xvalues are provided
    if xvalues is None:
        # if so, use the index of df_scores
        xvalues = df_scores.index
    else:
        # otherwise, check if xvalues lenght is the same as the number of rows in df_scores
        assert len(xvalues) == len(
            df_scores
        ), "x-values should be of the same length as the number of rows in df_scores"

    return np.array(xvalues)


def _check_scores_columns(
    df_scores: pd.DataFrame, scores: Optional[List[str]] = None
) -> List[str]:
    """
    Check if scores are provided and return them.

    Parameters:
    -----------

    df_scores : pd.DataFrame
        DataFrame containing clustering scores.

    scores : Optional[List[str]], default=None
        List of scores (columns from df_scores) to be plotted. If not provided, all columns

    Returns:
    --------
    scores : List[str]
        List of scores to be plotted.

    """
    # check if scores are provided
    if scores is None:
        # if not, use all columns in df_scores except the metadata
        scores = [
            col
            for col in df_scores.columns
            if col not in ["catalog", "features", "n_components", "n_trial"]
        ]
        # also, skip columns with 'err' suffix
        scores = [score for score in scores if not score.endswith("_err")]
        # also, skip wcss_mah (EXCEPTION)
        scores = [score for score in scores if score != "wcss_mah"]

    # othewise
    else:
        # check if scores are in df_scores
        assert all(
            score in df_scores.columns for score in scores
        ), "'scores' should all be in dataframe"

    return scores


def _modify_scores_dataframe(
    df_scores: pd.DataFrame,
    align_scores: Optional[str] = None,
    normalize_scores: bool = False,
) -> pd.DataFrame:
    """
    Modify scores if requested.

    Parameters:
    -----------

    df_scores : pd.DataFrame
        DataFrame containing clustering scores.

    align_scores : Optional[str], default=None
        If specified, aligns clustering scores. Accepted values are 'higher' or 'lower'.

    normalize_scores : bool, default=False
        Whether to normalize the scores to the range [0, 1].

    Returns:
    --------
    df_scores : pd.DataFrame
        Modified DataFrame containing clustering scores.
    """
    # create a copy of dataframe
    df_scores = df_scores.copy(deep=True)

    # if align_scores is provided
    if align_scores:
        df_scores = mets.align_clustering_scores(
            df_scores, score_orientation=align_scores
        )

    # normalize scores if requested
    if normalize_scores:
        df_scores = mets.normalize_clustering_scores(df_scores)

    return df_scores


def plot_scores_as_grid(
    df_scores: pd.DataFrame,
    xvalues: Optional[List[str]] = None,
    scores: Optional[List[str]] = None,
    align_scores: Optional[str] = None,
    normalize_scores: bool = False,
    highlight_best: bool = True,
    n_ax_columns: int = 4,
    title: str = None,
    figsize: Tuple[int, int] = (15, 10),
    wspace: float = 0.3,
    hspace: float = 0.2,
    save_kwargs: Optional[Dict] = None,
    return_scores: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Display clustering scores as a grid of plots.

    Parameters:
    ----------
    df_scores : pd.DataFrame
        DataFrame containing clustering scores.

    xvalues : Optional[List[str]], default=None
        X-values to be plotted. If not provided, will use the index of `df_scores`.

    scores : Optional[List[str]], default=None
        List of scores (columns from df_scores) to be plotted. If not provided, all columns
        except some predefined metadata columns will be used.

    align_scores : Optional[str], default=None
        If specified, aligns clustering scores. Accepted values are 'higher' or 'lower'.

    normalize_scores : bool, default=False
        Whether to normalize the scores to the range [0, 1].

    highlight_best : bool, default=True
        Whether to highlight the best score on each subplot.

    n_ax_columns : int, default=3
        Number of columns for the grid layout.

    title : str, default=None
        Title for the entire figure.

    figsize : Tuple[int, int], default=(15, 8)
        Size of the entire figure.

    wspace : float, default=0.2
        Width space between subplots.

    hspace : float, default=0.3
        Height space between subplots.

    save_kwargs : Optional[Dict], default=None
        If provided, the plot will be saved using the options provided in this dictionary.
        The dictionary can contain keys such as 'filename', 'dpi', etc. that are valid for plt.savefig().

    return_scores : bool, default=False
        If True, the possibly modified df_scores will be returned.

    **kwargs
        Other arguments to pass to the individual score plotting function.

    Returns:
    --------
    pd.DataFrame (optional)
        If `return_scores` is set to True, the modified DataFrame is returned.

    Notes:
    ------
    It's important to ensure that the `scores` passed are present in the `df_scores` DataFrame.
    """
    # check if xvalues are provided
    xvalues = _check_scores_xvalues(df_scores, xvalues)

    # check if scores are provided
    scores = _check_scores_columns(df_scores, scores)

    # create a copy of dataframe
    df_scores = _modify_scores_dataframe(df_scores, align_scores, normalize_scores)

    # Calculate the number of rows needed in the grid based on the number of models and desired columns.
    n_rows = int(np.ceil(len(scores) / n_ax_columns))

    # Create a grid of subplots. All subplots share the same x.
    fig, axes = plt.subplots(n_rows, n_ax_columns, figsize=figsize, sharex=False)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Flatten the 2D array of axes for ease of iteration.
    axes = axes.ravel()

    # get metric descriptions
    metric_descriptions = mets.get_clustering_metrics(verbose=False)

    # should we highlight the highest or lowest score or not highlighting?
    highlight_extreme = None

    # Plot each model on its respective axis.
    for idx, (score, ax) in enumerate(zip(scores, axes)):
        # get metric info
        metric_info = metric_descriptions[score]
        # get metric characteristic
        is_higher_better = metric_info["is_higher_better"]

        # score title that will be used in the plot
        score_title = score
        # if modified,
        if align_scores:
            if is_higher_better and align_scores == "lower":
                score_title += " (inv)"
            elif not is_higher_better and align_scores == "higher":
                score_title += " (inv)"

        # if aligned, best scores are all highest or lowest ones.
        if highlight_best:
            if align_scores:
                highlight_extreme = "highest" if align_scores == "higher" else "lowest"
            else:
                highlight_extreme = "highest" if is_higher_better else "lowest"

        # skip if scores all nan
        if df_scores[score].isna().all():
            continue

        plot_scores(
            xvalues,
            df_scores[score].values,
            highlight_extreme=highlight_extreme,
            xlabel=" ",
            ylabel=" ",
            ax=ax,
            title=score_title,
            **kwargs,
        )

    # Hide any remaining unused axes.
    for ax in axes[len(scores) :]:
        ax.axis("off")

    # Set the title of the figure.
    if title:
        fig.suptitle(title, fontsize=25)

    # Save the plot if save_kwargs is provided
    if save_kwargs:
        save_figure(**save_kwargs)

    if return_scores:
        return df_scores


def plot_scores_as_heatmap(
    df_scores: pd.DataFrame,
    xvalues: Optional[List[str]] = None,
    scores: Optional[List[str]] = None,
    align_scores: Optional[str] = "higher",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "seagreen",
    xtick_orientation: str = "vertical",
    save_kwargs: Optional[Dict] = None,
    return_scores: bool = False,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """
    Plot clustering scores as a heatmap.

    Parameters:
    - df_scores (pd.DataFrame): DataFrame containing the scores.
    - xvalues (Optional[List[str]]): Names of the models, if not provided, use df_scores index.
    - scores (Optional[List[str]]): Specific columns in df_scores to be plotted.
    - align_scores (str, optional): Aligns scores. Defaults to 'higher'.
    - title (Optional[str]): Title of the plot.
    - figsize (Tuple[int, int], optional): Size of the figure. Defaults to (10, 8).
    - cmap (str, optional): Color map for the heatmap. Defaults to 'seagreen'.
    - xtick_orientation (str, optional): X-axis tick orientation. Defaults to 'vertical'.
    - save_kwargs (Optional[Dict]): Arguments to save the figure.
    - return_scores (bool, optional): Return the modified scores dataframe. Defaults to False.
    - **kwargs: Additional keyword arguments for sns.heatmap().

    Returns:
    - Optional[pd.DataFrame]: If return_scores is True, returns the modified scores dataframe.
    """

    # Ensure xvalues are provided or default to the index of df_scores
    xvalues = _check_scores_xvalues(df_scores, xvalues)

    # Ensure scores columns are provided or default to excluding metadata columns
    scores = _check_scores_columns(df_scores, scores)

    # Adjust scores based on alignment preference and normalization
    df_scores = _modify_scores_dataframe(df_scores, align_scores, normalize_scores=True)

    # Remove metadata columns for visualization
    df_scores = df_scores.drop(
        columns=["catalog", "features", "n_components", "n_trial"]
    )

    # Extract desired scores for plotting
    df = df_scores[scores]
    df.index = xvalues

    # Initialize the plot
    fig = plt.figure(figsize=figsize)
    sns.set(font_scale=1)
    cmap = sns.light_palette(cmap, as_cmap=True)

    # Plot the heatmap
    g = sns.heatmap(
        df.round(2),
        annot=True,
        cmap=cmap,
        cbar_kws={"label": "Normalized Clustering Scores"},
        **kwargs,
    )

    # Adjust xtick orientation for clarity
    g.set_xticklabels(
        g.get_xticklabels(),
        rotation=0 if xtick_orientation == "horizontal" else 90,
        horizontalalignment="right",
    )

    # Optional title for the plot
    if title:
        fig.suptitle(title, fontsize=25)

    # Optional save functionality for the plot
    if save_kwargs:
        plt.savefig(**save_kwargs)

    # Optionally return the scores dataframe
    if return_scores:
        return df_scores
