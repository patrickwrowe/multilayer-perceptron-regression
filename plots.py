import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_correlations(correlations: pd.DataFrame, colormap: str = "seismic"):
    """
    Plots the result of calling pandas.DataFrame.corr() as a heatmap.
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(correlations, cmap=colormap)

    # We want to show all ticks...
    ax.set_xticks(range(len(correlations.columns)))
    ax.set_yticks(range(len(correlations.columns)))

    # ... and label them with the respective list entries
    ax.set_xticklabels(correlations.columns)
    ax.set_yticklabels(correlations.columns)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(correlations.columns)):
        for j in range(len(correlations.columns)):
            text = ax.text(
                j,
                i,
                f"{correlations.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="w",
                size=10,
                weight="bold",
            )

    ax.set_title("Diabetes Correlations")
    fig.tight_layout()

    return fig, ax

def plot_two_feature_correlation(
    feature_1: pd.Series, feature_2: pd.Series, title: str = "Feature Correlation"
):
    """
    Plot the correlation between two features. 
    """

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(feature_1, feature_2)

    ax.set_title(title)
    ax.set_xlabel(feature_1.name)
    ax.set_ylabel(feature_2.name)

    return fig, ax


def plot_umap(umap_coords: np.ndarray, color_by: np.ndarray, colormap: str = "viridis", title: str = "UMAP"):
    """
    Plot the UMAP embedding of the dataset.
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=color_by, cmap=colormap)

    ax.set_title(title)

    # No markers on axes
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    return fig, ax


    