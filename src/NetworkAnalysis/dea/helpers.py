#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   helpers.py
@Time    :   2024/06/14 08:47:37
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Functions that help analyse the Volcano, Pi and Scatter plots
'''

import pandas as pd
import numpy as np


def toggle_legend(fig: dict, kept_traces=[]):
    """
    Toggles the visibility of legend items in a plotly figure.

    Args:
        fig (dict): The plotly figure dictionary.
        kept_traces (list, optional): A list of trace names to keep visible in the legend. Defaults to [].

    Returns:
        dict: The updated plotly figure dictionary with toggled legend visibility.
    """
    for trace in fig["data"]:
        if trace.name in kept_traces:
            continue
        trace.visible = "legendonly"
    return fig


def filter_points_in_rectangle(df: pd.DataFrame, rectangle: dict):
    """
    Filter points in a DataFrame that fall within a given rectangle.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the points.
    - rectangle (dict): A dictionary representing the rectangle with keys 'x0', 'x1', 'y0', 'y1'.

    Returns:
    - pd.DataFrame: The filtered DataFrame containing the points within the rectangle.
    """
    return df[(df["x"] >= rectangle["x0"]) & (df["x"] <= rectangle["x1"]) & (df["y"] >= rectangle["y0"]) & (df["y"] <= rectangle["y1"])]


def distance_from_ref(df: pd.DataFrame, ref: tuple):
    """
    Calculate the Euclidean distance between each point in the DataFrame `df` and a reference point `ref`.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the points.
        ref (tuple): The coordinates of the reference point (x, y).
        
    Returns:
        np.ndarray: An array of distances between each point and the reference point.
    """
    x1, y1 = ref[0], ref[1]  # Coordinates of the reference point
    return np.sqrt((df["x"] - x1) ** 2 + (df["y"] - y1) ** 2)

