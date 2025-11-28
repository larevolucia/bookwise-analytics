"""Utility functions for EDA module"""
import pandas as pd


def create_volume_bins(series, bin_edges, labels):
    """
    Flexible binning with zero-handling

    Parameters:
    - series: pandas Series to bin
    - bin_edges: list of bin boundaries
    - labels: list of category names
    """
    # Check if data contains zeros
    has_zeros = (series == 0).any()

    # Adjust first bin edge if needed
    if has_zeros and bin_edges[0] == 0:
        bin_edges = [-0.001] + bin_edges[1:]

    return pd.cut(series, bins=bin_edges, labels=labels)
