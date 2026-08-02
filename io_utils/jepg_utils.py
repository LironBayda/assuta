import os
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt


def save_two_series_plot(
        path: str,
        x1: Sequence[float],
        y1: Sequence[float],
        x2: Sequence[float],
        y2: Sequence[float],
        title: str,
        x_label: str,
        y_label: str,
        marker1: str = "ro"
):
    """
    Save a plot containing two data series to disk.

    This function produces a figure with two sets of x-y data,
    applies consistent formatting, and saves the output image
    as a PNG file to the specified directory.

    Parameters
    ----------
    path : str
        Directory where the image file will be saved.
    x1, y1 : sequence of floats
        First data series.
    x2, y2 : sequence of floats
        Second data series.
    title : str
        Title text for the figure.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    marker1 : str, optional
        Matplotlib format string for the first dataset.

    Raises
    ------
    ValueError
        If input arrays are mismatched in length.
    OSError
        If the output directory does not exist.
    """

    # -------------------------------------
    # Input validation
    # -------------------------------------
    if not os.path.isdir(path):
        raise OSError(f"Output directory does not exist: {path}")

    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    if x1.size != y1.size:
        raise ValueError("x1 and y1 must have the same length.")
    if x2.size != y2.size:
        raise ValueError("x2 and y2 must have the same length.")

    # -------------------------------------
    # Create and configure figure
    # -------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(x1, y1, marker1, label="Series 1")
    ax.plot(x2, y2, label="Series 2")

    ax.set_title(title, fontsize=18)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    ax.legend()
    ax.grid(True)

    # -------------------------------------
    # Save figure
    # -------------------------------------
    output_path = os.path.join(path, f"{title}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def save_single_series_plot(
        path: str,
        x: Sequence[float],
        y: Sequence[float],
        title: str,
        x_label: str,
        y_label: str
):
    """
    Save a plot containing a single data series to disk.

    Parameters
    ----------
    path : str
        Directory where the image file will be saved.
    x, y : sequence of floats
        Data series to plot.
    title : str
        Name of the output file (and figure title).
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.

    Raises
    ------
    ValueError
        If inputs are mismatched in length.
    OSError
        If directory does not exist.
    """

    if not os.path.isdir(path):
        raise OSError(f"Output directory does not exist: {path}")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(x, y)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.grid(True)

    output_path = os.path.join(path, f"{title}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
