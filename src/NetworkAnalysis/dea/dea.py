import dash_bio as dashbio
import plotly.express as px
import plotly.graph_objects as go

from os import path
import time

import numpy as np
import pandas as pd
from . import scatter_plot as sp

# import utilities.scatter_plot as sp
from . import scatter_plot as sp


def import_data(fullPath):
    start_time = time.time()
    if path.exists(fullPath):
        ret_dict = {}
        ret_dict["data"] = pd.read_csv(fullPath, delimiter="\t")
        print("Finished loading the data in {}".format(time.time() - start_time))
        return ret_dict
    else:
        return None


def volcano(filename, base_path, selected_genes=None, label=None, known_markers=False, **kwargs):

    full_path = base_path + filename
    if not path.exists(full_path):
        raise FileExistsError("{} at {} doesn't exists ".format(filename, base_path))

    data_dict = import_data(full_path)
    relevant_aread_fc = [-1, 1]  # change this to include more points in the relevant area (i.e. red points)
    volcano_fig = draw_volcano(data_dict["data"], relevant_aread_fc, selected_genes, label, known_markers=known_markers, **kwargs)

    return volcano_fig


def draw_volcano(df, fold_changes, selected_genes, label, known_markers=False, **kwargs):
    """Drawing the volcano plot

    # There is something weird going on with indexes because we're setting customData to df['genes'].
    # It can get confusing as the data in df['genes'] will be put in alphabetical order and not correlated with the [x,y] points
    #  Thus the real value of the genes will be different and is passed through selected_genes

    Args:
        df (dataframe): _description_
        fold_changes (dataframe): _description_
        selected_genes (list): the list of selected genes

    Returns:
        fig: figure object from plotly
    """
    fig = dashbio.VolcanoPlot(
        dataframe=df,
        effect_size="fold_change",
        gene="genes",
        snp=None,
        p="q",
        genomewideline_value=2.5,
        logp=True,
        effect_size_line=fold_changes,
        ylabel="-log10(q)",
        xlabel="log2(FC)",
        col="#2A3F5F",
        point_size=8,
        effect_size_line_width=4,
        genomewideline_width=2,
        highlight=True,
        annotation="group",
    )
    
    fig = fig.update_traces(marker=dict(size=12, opacity=0.2), selector=dict(mode="markers"))
    fig = show_selected_genes_vulcano(df, fig, selected_genes, label, known_markers=known_markers, **kwargs)

    fig.update_traces(customdata=df["genes"], textposition="bottom left", selector=dict(mode="markers"))
    fig.update_layout(height=700)
    return fig


def show_selected_genes_vulcano(df, fig, selected_genes, label, known_markers=False, **kwargs):
    custom_traces = create_custom_traces(selected_genes, label, known_markers=known_markers, **kwargs)
    colors =  px.colors.qualitative.G10 + px.colors.qualitative.Bold + px.colors.qualitative.Vivid + px.colors.qualitative.Dark24
    for idx, trace in enumerate(custom_traces):
        fig.add_trace(create_gene_trace(df, trace["genes"], name=trace["title"], marker_color=colors[idx], marker_size=12))

    return fig


def create_custom_traces(selected_genes, label, known_markers=False, **kwargs):
    custom_traces = []

    if selected_genes is not None:
        custom_traces.append({"genes": selected_genes, "title": label})

    if known_markers:
        custom_traces = add_known_markers(custom_traces=custom_traces)
        
    if "markers" in kwargs.keys():
        markers = kwargs.pop("markers")
        for name, genes in markers.items():
            custom_traces.append({"genes": genes, "title": name})

    
    return custom_traces


def add_known_markers(custom_traces: list):
    base_path = '/Users/vlad/Documents/Code/York/iNet_v2/data/ref_markers/'
    sel_tfs = pd.read_csv(f'{base_path}/tf_ctrl.csv', index_col='gene')
    custom_traces.append({"genes": list(sel_tfs.index), "title": "Selected TF"})


    all_markers = pd.read_csv(f'{base_path}/known_markers.tsv', sep='\t')
    for col in all_markers.columns:
        genes = all_markers[col].dropna()
        custom_traces.append({'genes': list(genes), 'title': col})

    return custom_traces


def create_gene_trace(df, genes, name="custom genes", marker_color="yellow", marker_size=12, df_2=None):

    x, y, gene_txt = [], [], []
    # top_pi = [] 
    if df_2 is None:

        selected_df = df[df["genes"].isin(genes)]
        y = -np.log10(selected_df["q"])
        x = selected_df["fold_change"]
        gene_txt = selected_df["genes"]
        # Below is an attempt to show hust the top genes on either right/left
        # For volcano - genes is a separate column
        # selected_df = df[df["genes"].isin(genes)]
        # num = 10
        # top_fold_change_left = df.sort_values(by='fold_change', ascending=True)['genes'].values[:num]
        # top_fold_change_right = df.sort_values(by='fold_change', ascending=False)['genes'].values[:num]
        # top_pi = df.sort_values(by='-log10(q)', ascending=False)['genes'].values[:num]

        # # top_left = set(top_fold_change_left) & set(top_pi)
        # # top_right = set(top_fold_change_right) & set(top_pi)
        # genes = list(set( list(top_pi) + list(top_fold_change_right) + list(top_fold_change_left)))
        # selected_pi = df[df["genes"].isin(genes)]
        # y_pi = -np.log10(selected_pi["q"])
        # x_pi = selected_pi["fold_change"]
        # gene_txt = selected_pi["genes"]

    else:
        # for pi plot - genes are in the index
        selected_df = df[df.index.isin(genes)]
        selected_df_2 = df_2[df_2.index.isin(genes)]

        selected_df["pi_x"] = -np.log10(selected_df["q"]) * selected_df["fold_change"]
        selected_df_2["pi_y"] = -np.log10(selected_df_2["q"]) * selected_df_2["fold_change"]
        dmy_df = pd.concat([selected_df_2, selected_df], axis=1)
        x = dmy_df["pi_x"]
        y = dmy_df["pi_y"]

        gene_txt = dmy_df.index
        # if dmy_df.shape[0] != len(genes):
        #     print("\n\n### Genes not found for PI ", list(set(genes) - set(gene_txt)))

    markers = {"size": marker_size, "color": selected_df.shape[0] * [marker_color], "symbol": "x", 'opacity': 1} # "line": dict(color='black', width=2) }

    if name == 'text':
        trace = dict(type="scatter", x=x, y=y, showlegend=False, marker=markers, text=gene_txt, mode="text", name=name, textposition="top center", opacity=1)
    else:
        trace = dict(type="scatter", x=x, y=y, showlegend=True, marker=markers, text=gene_txt, mode="markers+text", name=name, textposition="top center", opacity=1)

        # if len(top_pi):
        #     trace = dict(type="scatter", x=x_pi, y=y_pi+2, showlegend=False, marker=markers, text=gene_txt, mode="text", name=name, textposition="top center", opacity=1)


    return trace


#### Pi Plot ####


def plotPi(file_1, file_2, base_path, selected_genes=None, label=None, known_markers=False, **kwargs):

    full_path_1 = base_path + file_1
    full_path_2 = base_path + file_2
    if not path.exists(full_path_1) or not path.exists(full_path_1):
        raise FileExistsError("{}  or {} at {} doesn't exists ".format(file_1, file_2, base_path))

    df_1 = pd.read_csv(full_path_1, sep="\t")
    df_2 = pd.read_csv(full_path_2, sep="\t")

    figure, df = draw_pi_plot(df_1, df_2, file_1, file_2, selected_genes, label, known_markers=known_markers, **kwargs)

    figure.update_layout(
        font_size=14,
        xaxis=dict( tickfont=dict(size=16), title_font_size=22),
        yaxis=dict( tickfont=dict(size=16), title_font_size=22),
    )

    height = None
    if "height" in kwargs.keys():
        height = kwargs.pop("height")

    # Prettify
    if known_markers:
        figure.update_layout(
            legend=dict(
                orientation="h",
                # title = '',
                yanchor="bottom",
                xanchor="center",
                y=1,
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=16, color="#003366"),
            ),
            template="ggplot2",
            # paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickmode="linear", tick0=0, dtick=25, tickfont=dict(size=20)),
            yaxis=dict(tickmode="linear", tick0=0, dtick=25, tickfont=dict(size=20)),
            height=height,
        )
    return figure, df


# Just for the selective edge pruning experiments
def rename_sel_tfs_groups(a_text):
    match_val = {"small-basal": "basal-5", "large-basal":"basal-4","mes-like": "mes_3", "lum-inf":"lum-12", "large-luminal":"lum-13", "large_basq": "basal-4", "lumInf":"lum-12", "small_basq": 'basal-5', "luminf":"lum-12", "lump":"lum-13",}

    prcs_txt = a_text.lower()
    changed = False
    for old_label, new_label in match_val.items():
        if old_label in prcs_txt:
            prcs_txt = prcs_txt.replace(old_label, new_label)
            changed = True

    if changed:
        return prcs_txt.replace("_", " ")
    else:
        return a_text

def draw_pi_plot(df_1, df_2, file_1, file_2, selected_genes, label, known_markers=False, **kwargs):

    x_title, y_title = "", ""
    # Decide which is bigger, the number of genes may differ
    if df_1.shape[0] < df_2.shape[0]:
        first_df = df_2.copy(deep=True).set_index("genes")
        second_df = df_1.copy(deep=True).set_index("genes")
            
        x_title, y_title = file_2.replace("_v1_vulcano_labels.tsv", ""), file_1.replace("v1_vulcano_labels.tsv", "")
    else:
        first_df = df_1.copy(deep=True).set_index("genes")
        second_df = df_2.copy(deep=True).set_index("genes")
        x_title, y_title = file_1.replace("_v1_vulcano_labels.tsv", ""), file_2.replace("v1_vulcano_labels.tsv", "")

    # x_title = rename_sel_tfs_groups(x_title)
    # y_title = rename_sel_tfs_groups(y_title)

    # compute the values
    first_df["x"] = -np.log10(first_df["q"]) * first_df["fold_change"]
    second_df["y"] = -np.log10(second_df["q"]) * second_df["fold_change"]

    # Note the genes number may differ and we're setting the index of the DataFrame which has the most genes (i.e. rows)
    #  However, there might be some genes in the second df which are not in the first one. Regardless, we set the nan values to 0 (points will be added to the center)
    first_df.rename(columns={"group": "comp_1"}, inplace=True)
    second_df.rename(columns={"group": "comp_2"}, inplace=True)

    dummy_df = pd.concat([first_df[["x", "comp_1"]], second_df[["y", "comp_2"]]], axis=1).fillna(0).reset_index().rename(columns={"index": "genes"})

    dummy_df["main"] = "PI_plot"
    fig = px.scatter(dummy_df, x="x", y="y", hover_data=["genes", "comp_1", "comp_2"], color="main")
    fig.update_traces(marker=dict(size=12, opacity=0.2), selector=dict(mode="markers"))

    fig = add_anottations(first_df, second_df, dummy_df, fig)
    fig = show_selected_genes_pi(first_df, second_df, fig, selected_genes, label, known_markers=known_markers, **kwargs)

    fig.update_layout(height=700, xaxis_title=x_title, yaxis_title=y_title)
    return fig, dummy_df.set_index("genes")


def show_selected_genes_pi(df_1, df_2, fig, selected_genes, label, known_markers=False, **kwargs):
    custom_traces = create_custom_traces(selected_genes, label, known_markers, **kwargs)
    # colors = px.colors.qualitative.Bold + px.colors.qualitative.Vivid  + px.colors.qualitative.G10 + px.colors.qualitative.Alphabet
    plotly_qual = px.colors.qualitative
    colors = plotly_qual.Vivid + plotly_qual.Prism + plotly_qual.Bold + plotly_qual.Pastel + plotly_qual.Alphabet
    # colors = plotly_qual.Vivid_r + plotly_qual.Prism + plotly_qual.Bold +  plotly_qual.Vivid + plotly_qual.Pastel + plotly_qual.Alphabet

    for idx, trace in enumerate(custom_traces):
        fig.add_trace(create_gene_trace(df_1, trace["genes"], name=trace["title"], marker_color=colors[idx], df_2=df_2))

    return fig


def add_anottations(first_df, second_df, dummy_df, fig):
    offset = 4
    fig.add_shape(type="line", x0=dummy_df["x"].min() * 1.5, y0=0, x1=dummy_df["x"].max() * 1.5, y1=0, line=dict(color="Black", width=1), xref="x", yref="y")

    fig.add_shape(type="line", x0=0, y0=dummy_df["y"].min() * 1.5, x1=0, y1=dummy_df["y"].max() * 1.5, line=dict(color="Black", width=1), xref="x", yref="y")

    first_df['comp_1'] = first_df['comp_1'].astype(str)
    second_df['comp_2'] = second_df['comp_2'].astype(str)

    q1_text = first_df.loc[first_df["x"] == first_df["x"].max()]["comp_1"].values[0]
    q2_text = first_df.loc[first_df["x"] == first_df["x"].min()]["comp_1"].values[0]
    q3_text = second_df.loc[second_df["y"] == second_df["y"].max()]["comp_2"].values[0]
    q4_text = second_df.loc[second_df["y"] == second_df["y"].min()]["comp_2"].values[0]

    # q1_text = rename_sel_tfs_groups(q1_text).replace("cluster", "")
    # q2_text = rename_sel_tfs_groups(q2_text).replace("cluster", "")
    # q3_text = rename_sel_tfs_groups(q3_text).replace("cluster", "")
    # q4_text = rename_sel_tfs_groups(q4_text).replace("cluster", "")

    font = dict(size=18,  color="#003366")
    if 0:
        fig.add_annotation(
            showarrow=True,
            arrowhead=1,
            align="right",
            x=first_df["x"].max() + offset,
            y=0,
            ay=10,
            text=q1_text,
            font=font,
            opacity=0.7,
        )

        fig.add_annotation(
            showarrow=True,
            arrowhead=1,
            align="right",
            x=first_df["x"].min() - offset,
            y=0,
            text=q2_text,
            font=font,
            opacity=0.7,
        )

        fig.add_annotation(
            showarrow=True,
            arrowhead=1,
            align="right",
            y=second_df["y"].max() + offset + 5,
            x=0,
            text=q3_text,
            font=font,
            opacity=0.7,
        )

        fig.add_annotation(
            showarrow=True,
            arrowhead=1,
            align="left",
            y=second_df["y"].min() - offset - 5,
            x=0,
            ay=0,
            text=q4_text,
            font=font,
            opacity=0.7,
        )

    return fig


# Add a box around a core defined by the user
def add_box_core(fig: go.Figure, df: pd.DataFrame, subset: pd.DataFrame, th=0.05, display=True):
    """
    Adds a rectangular box to a given plot figure and returns the figure along with the core genes within the box.

    Parameters:
    - fig (go.Figure): The plot figure to which the box will be added.
    - df (pd.DataFrame): The dataframe containing the x and y coordinates.
    - subset (pd.DataFrame): The subset of the dataframe to consider for finding core genes.
    - th (float, optional): The threshold value used to determine the size of the box. Default is 0.05.
    - display (bool, optional): Whether to display the box on the figure. Default is True.

    Returns:
    - fig (go.Figure): The updated plot figure with the added box.
    - core_genes (pd.Index): The index of the core genes within the box.
    """

    min_x, max_x = df["x"].min(), df["x"].max()
    min_y, max_y = df["y"].min(), df["y"].max()

    x_neg, x_pos = min_x * th, max_x * th
    y_neg, y_pos = min_y * th, max_y * th

    dmy = df.loc[subset.index]
    core_genes = dmy[(dmy["x"] >= x_neg) & (dmy["x"] <= x_pos) & (dmy["y"] >= y_neg) & (dmy["y"] <= y_pos)].index

    if display:
        fig = fig.add_shape(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=x_neg,
                y0=y_neg,
                x1=x_pos,
                y1=y_pos,  # x0, y0 bottom left and x1, y1 top right
                line=dict(
                    color="red",
                    width=4,
                ),
                opacity=0.4,
                # layer='below'
            )
        )

    return fig, core_genes


#### Scatter Plot old ####
# The below functions were used when the Volcano plots weren't cotaining all the information required for scatter plot (up until v3.x). After that I changed the pre-processing for Volcano to contain the fold change and other required information.
# Kept the code in case I need for further work
def draw_scatter_old(filename, tcga_tpm_df, filtered_genes, mapping_cols, path, selected_genes=None, label=None):
    exp = "_".join(filename.split("_")[:-2])

    # path = "/Users/vlad/Documents/Code/York/visualisation/visualisation/data/VolcanoPlots/"
    data_dict = import_data(path + filename)

    fold_change, _ = sp.prc_fc_comp(tcga_tpm_df, data_dict["data"], exp, mapping_cols, drop_outliers=False)

    fig = plt_scatter_old(fold_change, filtered_genes, selected_genes, label, known_markers=True)

    return fig


def plt_scatter_old(df, filetered_genes, selected_genes, label, known_markers=False):

    clusters = df.columns[1:]
    fig = px.scatter(df, x=clusters[0], y=clusters[1], hover_data=["genes", "fold_change"], color="sig", color_discrete_sequence=["#636EFA", "grey", "#EF553B"])

    fig = sp.plt_fc_lines(fig, df, fc_values=[0, 1, 2, 4])
    fig.update_traces(marker=dict(size=10, opacity=0.4), selector=dict(mode="markers"))

    if len(filetered_genes):
        dmy_df = df[df["genes"].isin(filetered_genes)]
        fig.add_trace(
            go.Scatter(
                x=dmy_df[clusters[0]],
                y=dmy_df[clusters[1]],
                mode="markers",
                text=dmy_df["genes"],
                hoverinfo="all",
                textposition="top right",
                name="Clustered genes",
            )
        )

    if known_markers:
        custom_traces = create_custom_traces(selected_genes, label)
        colors = px.colors.qualitative.Vivid + px.colors.qualitative.Bold
        marker_size = 12
        for idx, trace in enumerate(custom_traces):
            selected_df = df[df["genes"].isin(trace["genes"])]

            markers = {"size": marker_size, "color": selected_df.shape[0] * [colors[idx]], "symbol": "x"}
            trace = dict(
                type="scatter",
                x=selected_df[clusters[0]],
                y=selected_df[clusters[1]],
                showlegend=True,
                marker=markers,
                text=selected_df["genes"],
                mode="markers+text",
                name=trace["title"],
                textposition="top right",
            )

            fig.add_trace(trace)

    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=np.log2([1, 10, 100, 1000, 10000]), ticktext=["0", "1", "10", "100", "1000", "10000"]),
        yaxis=dict(tickmode="array", tickvals=np.log2([1, 10, 100, 1000, 10000]), ticktext=["0", "1", "10", "100", "1000", "10000"]),
    )

    return fig
