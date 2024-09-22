#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   scatter_plot.py
@Time    :   2022/07/11 09:52:17
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Helper function for scatter plot
'''

import pandas as pd
import numpy as np
from os import path
import plotly.express as px
import plotly.graph_objects as go



def significant_genes (row, labels):
   if row['fold_change'] < 1.0 and row['fold_change'] > -1.0:
      return 'Non-significant'
   elif row["fold_change"] > 1.0:
       return "Significant {}".format(labels[0])
   else: 
        return "Significant {}".format(labels[1])

def centroid_genes():
    centroid_genes = {}
    centroid_genes["Small Ba/Sq"] = ['IGKJ1', 'IGKV3-20', 'IGLC2', 'IGKV2-28', 'IGKJ2', 'IGHJ4', 'IGLC1', 'IGKV4-1', 'H19', 'CD14', 'IGHG1', 'C1QB', 'IGHJ5', 'POSTN', 'HTRA3', 'C1QC', 'IGKV1D-39', 'CTHRC1', 'IGHG2', 'AEBP1', 'IGHG3', 'HLA-DQA1', 'IGLC3', 'NNMT', 'CCN2', 'IGKJ4', 'MMP11', 'CTSK']
    centroid_genes["Large Ba/Sq"] = ['KRT5', 'DSP', 'S100A2', 'H19', 'LY6D', 'CDH3', 'AQP3', 'MALL', 'KRT16', 'S100A8', 'SERPINB5', 'ANXA8L1', 'SLPI', 'COL17A1', 'KRT14', 'TNS4', 'S100A9']
    centroid_genes["LumP"] = ['S100A2', 'PSCA', 'H19', 'LY6D', 'UPK2', 'AQP3', 'ABCC3', 'KRT13', 'IGLC2', 'GPX2', 'CDC42EP5', 'VSIG2', 'DHRS2', 'SPINK1', 'TMPRSS4']
    centroid_genes["Mixed"] = ['IGHG1', 'IGLC2', 'CDKN2A']
    centroid_genes["LumInf"] = ['IGKV3-20', 'IGLC2', 'ACTG2', 'IGHJ4', 'IGLC1', 'S100A9', 'H19', 'MUC20', 'IGHG1', 'C1QB', 'POSTN', 'GPX2', 'IGHA1', 'PSCA', 'UPK2', 'AEBP1', 'IGLC3', 'NNMT', 'KRT23', 'CTSK']

    return centroid_genes

def create_custom_traces(selected_genes, label):
    custom_traces = []

    if selected_genes is not None:
        custom_traces.append({"genes":selected_genes, "title": label})
    
    luminal_markers = ["KRT20", "PPARG", "FOXA1", "GATA3", "SNX31", "UPK1A", "UPK2", "FGFR3"]
    basal_markers = ["CD44", "KRT6A", "KRT5", "KRT14", "COL17A1"]
    squamos_markers = ["DSC3", "GSDMC", "TCGM1", "PI3", "TP63"]
    immune_markers = ["CD274", "PDCD1LG2", "IDO1", "CXCL11", "L1CAM", "SAA1"]
    neural_diff = ["MSI1", "PLEKHG4B", "GNG4", "PEG10", "RND2", "APLP1", "SOX2", "TUBB2B"]

    # TCGA
    custom_traces.append({"genes":luminal_markers, "title": "TCGA_luminal"})
    custom_traces.append({"genes":basal_markers, "title": "TCGA_basal"})
    custom_traces.append({"genes":immune_markers, "title": "TCGA_immune"})
    custom_traces.append({"genes":squamos_markers, "title": "TCGA_squamos"})
    custom_traces.append({"genes":neural_diff, "title": "TCGA_neuroendocrine"})
    
    for key, value in centroid_genes().items():
        custom_traces.append({"genes": value, "title": "Highest in {}".format(key)})
    
    return custom_traces

#### Scatter Plot updated #### 
# This contains the code for scatter plots after v3, as the .tsv contain all the required information
def draw_scatter(filename, base_path, selected_genes=[], known_markers=False):
    """
    Initiates all the function calls to draw the scatter plot

    Args:
        df (DataFrame): The data to be plotted
        selected_genes (list): Genes selected on volcano plots

    Returns:
        dict: The scatter figure
    """
    full_path = base_path + filename
    if not path.exists(full_path):
        raise FileExistsError("{} at {} doesn't exists ".format(filename, full_path))
    
    df = pd.read_csv(full_path, sep="\t")

    df = add_sig(df)
    selected_points = df[df["genes"].isin(selected_genes)]

    fig = plt_scatter(df, selected_points, known_markers=known_markers)
    fig.update_layout(clickmode='event+select')

    return fig


def add_sig(df):
    """ 
    Functions that classifies the genes if are significant to one of the groups compared in the DEA.

    Args:
        df (DataFrame): The data used

    Returns:
        DataFrame: The dataframe with the extra-col w/ indificating the type of significance
    """
    labels = df.columns.values[-2:]
    df["significance"] = df.apply(lambda row: significant_genes(row, labels), axis=1)
    return df

def plt_fc_lines(fig, df, fc_values):
    """

    On the given figure and dataframe, it shows the lines for the given fold change values

    Args:
        fig (dict): Plotly figure
        df (DataFrame): Data used on the figure
        fc_values (list): The list of the fold changes to display the fold change lines

    Returns:
        dict: The figure with the resultant lines
    """

    # get the maximum values across fold_change and the cluster groups
    med_cols = [col for col in df.columns if "med" in col and "fold" not in col]
    groups = ["fold_change"] + list(med_cols)
    max_value = round(df[groups].max().max())

    colours = ["Black", "Green", "goldenrod", "Red", "orange", "Green", "Purple"]
    line_type = [None, None, "dot", "dash", "dash"]
    for value in fc_values:

        if value:
            color = colours[value]
        else:
            color = "Black"

        line_dict = dict(color=color, width=3,  dash=line_type[value])

        fig.add_shape(type='line',
                x0=value, y0=0,
                x1=max_value, y1=max_value-value,
                line=line_dict,
                xref='x',yref='y', name="log2(FC)={}".format(value))

        if value:
            color = colours[value]
        else:
            color = "Black"

        line_dict = dict(color=color, width=3,  dash=line_type[value])

        fig.add_shape(type='line',
                x0=0, y0=value,
                x1=max_value-value+1, y1=max_value+1,
                line=line_dict,
                xref='x',yref='y', name="log2(FC)={}".format(value))

    return fig

def plt_scatter(df, selected_points, known_markers=False):
    """
    Creates the scatter plot

    Args:
        df (DataFrame): Thee data to be plotted
        selected_points (string): The points selected from the volcano plot
        known_markers (bool, optional): To display or not the custom traces. Defaults to False.

    Returns:
        dict: The Scatter figure
    """

    # select the columns with fold change based on the median
    clusters = [col for col in df.columns if ("med" in col) and ("fold" not in col)]
    fig = px.scatter(df, x=clusters[0], y=clusters[1], hover_data=df.columns, color="significance",  color_discrete_sequence=['#636EFA', "grey", '#EF553B'])

    fig.update_traces(marker=dict(size=10, opacity=0.4), selector=dict(mode='markers'))
    
    fig = plt_fc_lines(fig, df, fc_values = [0, 1, 2, 4])

    if not selected_points.empty:
        fig.add_trace(go.Scatter(x=selected_points[clusters[0]], y=selected_points[clusters[1]], mode="markers+text", text=selected_points["genes"], hoverinfo='all', textposition="top right", name="Selected Points"))

    if known_markers:
        custom_traces = create_custom_traces(selected_genes=None, label=None)
        colors =  px.colors.qualitative.Vivid + px.colors.qualitative.Bold
        marker_size = 12
        for idx, trace in enumerate(custom_traces): 
            selected_df = df[df["genes"].isin(trace["genes"])]

            markers = {"size": marker_size, "color": selected_df.shape[0] * [colors[idx]], "symbol": "x"}
            trace = dict(type='scatter', x=selected_df[clusters[0]], y=selected_df[clusters[1]],  showlegend=True, marker=markers, text=selected_df["genes"], mode="markers+text" , name=trace["title"],  textposition="top right")

            fig.add_trace(trace)

    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = np.log2([1, 10, 100, 1000, 10000]),
            ticktext = ["0", "1", "10", "100", "1000", "10000"]
        ),
        yaxis = dict(
            tickmode = 'array',
            tickvals = np.log2([1, 10, 100, 1000, 10000]),
            ticktext = ["0", "1", "10", "100", "1000", "10000"]
        )
    )

    return fig 


   

## Old functions to deal with the Scatter plot, files with <v3. In these, there is no TPM
def prc_fc_comp_old(tcga_tpm_df, sleuth_results, exp, mapping_cols, drop_outliers = False):

    info_path = "/Users/vlad/Documents/Code/York/visualisation/visualisation/data/ScatterPlot/infoFiles/"

    # get the tpms
    pd_for_diff = pd.read_csv(info_path + exp + ".info", sep="\t")

    ##### Apply pre-processing by Robertson et al. #####
    dummy_df = pd.concat([pd.DataFrame(tcga_tpm_df["genes"]), pd.DataFrame(tcga_tpm_df.iloc[:, 1:])], axis=1)

    df = dummy_df[dummy_df["genes"].isin(sleuth_results["genes"])]

    df.rename(columns=mapping_cols, inplace=True)

    df = df[["genes"] +  list(pd_for_diff["sample"].values)]
    df = df.set_index("genes").transpose()
    df.index.names = ["sample"]
    df["cluster"] = pd_for_diff.set_index("sample")["express"]

    print("Are arrays in sync? {}".format(np.array_equal(df.iloc[:, -1].reset_index(), pd_for_diff[["sample", "express"]])) )

    fold_change = pd.DataFrame(df.columns[:-1], columns=["genes"])
    cluster_labels = pd_for_diff["express"].unique()
    new_labels = []
    for label in cluster_labels:
        new_labels.append("cluster_{}".format(label))
        fold_change[new_labels[-1]] = df[df["cluster"] == label].iloc[:, :-1].median().values

    fold_change[new_labels] = np.log2(fold_change[new_labels]+1)
    outliers = []
    if drop_outliers:
        outliers = fold_change.sort_values(by=new_labels, ascending=False).iloc[0:10]["genes"].values 
        fold_change = fold_change[~fold_change["genes"].isin(outliers)]

    fold_change["FC"] = fold_change[new_labels[0]] - fold_change[new_labels[1]]
    fold_change["sig"] = fold_change.apply(lambda row: significant_genes(row, new_labels), axis=1 )
    return fold_change, outliers
