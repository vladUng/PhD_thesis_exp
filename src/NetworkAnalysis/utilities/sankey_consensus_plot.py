#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sankey.py
@Time    :   2022/07/04 18:10:23
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   None
@Details :   This files computes and plots (w/ plotly) Sankey plots. It can received either a DataFrame or a path to .tsv containing a DataFrame
"""


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# load data, aggregate by subtypes across columns and create counts columnn
base_path = "/Users/vlad/Documents/Code/York/BU_clustering/results/"


def genSankey(df, cat_cols=[], value_cols="", title="Sankey Diagram", labels=None):
    """Generate the Sankey plot for a givent DataFrame

    Args:
        df (DataFrame): The DataFramed passed from which the plot is generated
        cat_cols (list, optional): The Sankey categories. Defaults to [].
        value_cols (str, optional): Index of the dataframe. Defaults to ''.
        title (str, optional): Title of the plit. Defaults to 'Sankey Diagram'.
        labels (dict, optional): Labels to be matched for each of the groups. Defaults to None.

    Returns:
        dict: Plotly figure
    """
    colorPalette = px.colors.qualitative.Vivid + px.colors.qualitative.Prism

    labelList = []
    displayLabels = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp = list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp
        oldLabels = [labels.get(key) for key in labelListTemp]
        displayLabels = displayLabels + oldLabels

    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))

    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]] * colorNum

    # transform df into a source-target pair
    for i in range(len(cat_cols) - 1):
        if i == 0:
            sourceTargetDf = df[[cat_cols[i], cat_cols[i + 1], value_cols]]
            sourceTargetDf.columns = ["source", "target", "count"]
        else:
            tempDf = df[[cat_cols[i], cat_cols[i + 1], value_cols]]
            tempDf.columns = ["source", "target", "count"]
            sourceTargetDf = pd.concat([sourceTargetDf, tempDf])
        sourceTargetDf = sourceTargetDf.groupby(["source", "target"]).agg({"count": "sum"}).reset_index()

    # add index for source-target pair
    sourceTargetDf["sourceID"] = sourceTargetDf["source"].apply(lambda x: labelList.index(x))
    sourceTargetDf["targetID"] = sourceTargetDf["target"].apply(lambda x: labelList.index(x))

    if labels == None:
        labels = displayLabels
    else:
        labels = list(labels.keys())

    # creating the sankey diagram
    data = dict(
        arrangement="perpendicular",
        type="sankey",
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=displayLabels, color=colorList),
        link=dict(source=sourceTargetDf["sourceID"], target=sourceTargetDf["targetID"], value=sourceTargetDf["count"], color="#bdbdbd"),
    )

    layout = dict(
        title=title,
        font=dict(size=16, color="#003366"),
        height=600,
        # paper_bgcolor="rgba(0,0,0,0)",
    )

    fig = dict(data=[data], layout=layout)
    return fig, (sourceTargetDf, displayLabels)


#### More descriptive functions


# Assuming figs is a list of Plotly figures and titles is a list of their titles
def prep_sankey_description(sky_meta: tuple, input_df: pd.DataFrame, sel_cols: list):
    source_target_df, labels = sky_meta[0].copy(deep=True), sky_meta[1]

    # below are the mapped labels of the groups used in Sankey (k) and the unique number given by the algorithm
    mapped_values = {k: v for k, v in enumerate(labels)}
    source_target_df["SourceLabel"] = source_target_df["sourceID"].map(mapped_values)
    source_target_df["TargetLabel"] = source_target_df["targetID"].map(mapped_values)

    # Mapp back the classifier label
    classifier_map = []
    offset = 0
    for col in sel_cols:
        for i in range(0, len(input_df[col].unique())):
            classifier_map.append((i + offset, col))
        offset += i + 1

    classifier_map = dict(classifier_map)
    # Add the information for each classifier and the size
    for source in source_target_df["source"].unique():
        source_target_df.loc[source_target_df["source"] == source, "Classifier"] = classifier_map[source]
        source_label = source_target_df.loc[source_target_df["source"] == source]["SourceLabel"].values[0]
        group_size = input_df.loc[input_df[classifier_map[source]] == source_label].shape[0]

        source_target_df.loc[source_target_df["source"] == source, "source_size"] = int(group_size)

    source_target_df["prct_chg"] = round(source_target_df["count"] / source_target_df["source_size"], 4)
    source_target_df["SourceLabel"] = source_target_df["SourceLabel"].astype(str) + "(" + source_target_df["source_size"].astype(int).astype(str) + ")"

    source_target_df.rename(columns={"count": "#Samples"}, inplace=True)
    return source_target_df


def plot_sankey_description(config: dict, sky_stats: pd.DataFrame):
    figs, titles = [], []

    max_clusters = max([len(sky_stats[sky_stats["Classifier"] == elem]["SourceLabel"].unique()) for elem in sky_stats["Classifier"].unique()])

    titles = [["" for _ in range(max_clusters)] for _ in sky_stats["Classifier"].unique()]
    for idx, classifier in enumerate(sky_stats["Classifier"].unique()):
        sel_df = sky_stats.loc[sky_stats["Classifier"] == classifier]
        sel_df["TargetLabel"] = sel_df["TargetLabel"].astype(str)
        fig = px.bar(sel_df, x="TargetLabel", y="#Samples", facet_col="SourceLabel", text_auto=True)
        figs.append(fig)
        # titles.append(f"{classifier}")
        for label_idx, label in enumerate(sel_df["SourceLabel"].unique()):
            titles[idx][label_idx] = label

    titles = np.array(titles).flatten()

    if config is None:
        config = {
            "shared_x": False,
            "shared_y": False,
            "h_spacing": 0.02,
            "v_spacing": 0.15,
            "main_title": "Subplots",
            "height": 700,
            "width": None,
            "y_title": "Y-axis",
            "x_title": "X-axis",
            "specs": None,
        }

    # Determine the total number of rows and columns
    num_rows = len(figs)
    num_cols = max(len(fig.data) for fig in figs)  # Assuming each fig has the same number of columns

    # Create a subplot grid
    subplot_titles = [""] * (num_rows * num_cols)

    # Set the title for the first subplot in each row
    for i in range(num_rows):
        subplot_titles[i * num_cols + round(num_cols / 2)] = titles[i]

    # Create a subplot grid with titles
    subplot = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=titles,
        shared_xaxes=config["shared_x"],
        shared_yaxes=config["shared_y"],
        horizontal_spacing=config["h_spacing"],
        vertical_spacing=config["v_spacing"],
    )

    # Add each trace to the correct position
    for i, single_fig in enumerate(figs):
        for j, trace in enumerate(single_fig.data):
            subplot.add_trace(trace, row=i + 1, col=j + 1)

    # Update layout if necessary (e.g., titles, axes labels)
    layout = go.Layout(title_text=config["main_title"])
    subplot = subplot.update_layout(layout, height=config["height"], width=config["width"], showlegend=False)
    # subplot = subplot.update_xaxes(title_text=config["x_title"])
    # subplot = subplot.update_yaxes(title_text=config["y_title"])
    return subplot


def main(df=None, filename="TCGA_classifier_labels.tsv", title="TCGA MIBC mRNA subtypes", drop_cols=None, reorder_cols=None, retMeta=False):
    """Function which process the dataframe and get it ready to generate a Saneku plot

    Args:
        df (DataFrame, optional): The DataFrame from where the Saneky is generated. Defaults to None.
        filename (str, optional): The filePath to a .tsv file from which the Saneky is generated.. Defaults to "TCGA_classifier_labels.tsv".
        title (str, optional): Title of the plot. Defaults to "TCGA MIBC mRNA subtypes".
        drop_cols (_type_, optional): Remove the specified groups from the Sankey. Defaults to None.
        reorder_cols (_type_, optional): Re-arrange the groups in the Sankey. Defaults to None.
    """

    if df is None:
        classDF = pd.read_csv(base_path + filename, sep="\t", index_col="Sample")
    else:
        classDF = df

    if drop_cols != None:
        classDF.drop(columns=drop_cols, axis=1, inplace=True)

    if reorder_cols != None:
        classDF = classDF[reorder_cols].copy(deep=True)

    oricols = list(classDF.columns)
    uniq_list = []
    remap_labels = {}
    # keeps track of the number of groups in the previous columns
    init = 0
    # go through each column, and the values in that column (2nd loop) and make sure that each label has an unique value
    for col in list(classDF.columns):
        col_uniq = sorted(list(classDF[col].unique()))
        uniq_list += col_uniq
        col_rep = list(classDF[col])
        idx = 0
        for label in col_uniq:
            new_idx = init + idx
            # There are some cases where a new value for the label might be with something that is already in the column; This will break the code where we do re-labeling.
            if new_idx in col_uniq and label != new_idx:
                new_idx += 1
                idx += 1

            remap_labels[new_idx] = label
            col_rep = [new_idx if x == label else x for x in col_rep]
            idx += 1

        classDF[col + "_idx"] = col_rep
        init += idx

    # remove the originals and groupby rows to generate counts
    classDFidx = classDF.drop(oricols, axis=1)
    classDFidxagg = classDFidx.groupby(list(classDFidx.columns)).size().reset_index(name="counts")

    # if show_labels:
    fig_dict, meta = genSankey(classDFidxagg, cat_cols=list(classDFidx.columns), value_cols="counts", title=title, labels=remap_labels)
    fig = go.Figure(fig_dict)

    # add it column names
    for x_coordinate, col_name in enumerate(classDFidx.columns):
        col_name = "_".join(col_name.split("_")[:-1])
        fig.add_annotation(
            x=x_coordinate,
            y=1.07,
            xref="x",
            yref="paper",
            text=col_name,
            showarrow=False,
            font=dict(size=18, color="#003366"),
            align="left",
        )

    fig = fig.update_layout(
        xaxis={
            "showgrid": False,  # thin lines in the background
            "zeroline": False,  # thick line at x=0
            "visible": False,  # numbers below
        },
        yaxis={
            "showgrid": False,  # thin lines in the background
            "zeroline": False,  # thick line at x=0
            "visible": False,  # numbers below
        },
        plot_bgcolor="rgba(0,0,0,0)",
    )

    if retMeta:
        return meta, fig
    else:
        fig.show()

    # pio.write_image(fig, base_path + "TCGA_classifications_no-ori.pdf"
