#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   gsea.py
@Time    :   2024/06/14 08:59:00
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Helps running the GSEA
"""

import gseapy as gp
from gseapy.plot import dualplot
from gseapy import gseaplot2
import pandas as pd
import os
from .helpers import *
from .dea import plotPi
from ..utilities.helpers import save_fig


def plot_enrichr_summary(res2d: pd.DataFrame, label="", savePath="./"):
    """
    Plot the enrichment summary using Enrichr results.

    Args:
        res2d (pd.DataFrame): DataFrame containing the Enrichr results.
        label (str, optional): Label for the plot. Defaults to "".
        savePath (str, optional): Path to save the plot. Defaults to "./".

    Returns:
        ax_dot (matplotlib.axes.Axes): The matplotlib Axes object containing the plot.
    """

    title_dot = f"enrichment {label}"
    top_term = 10
    cutoff = 0.05
    figsize = (7, 6)

    res2d["Ratio"] = res2d["Gene %"].str.split("%", expand=True)[0].astype(float) / 100

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    ax_dot = dualplot(
        res2d,
        column="FDR q-val",
        columnBar="Ratio",
        x="Gene_set",  # set x axis, so you could do a multi-sample/library comparsion
        size=12,
        top_term=top_term,
        figsize=figsize,
        title=title_dot,
        xticklabels_rot=45,  # rotate xtick labels
        show_ring=True,  # set to False to revmove outer ring
        marker="o",
        cutoff=cutoff,
        cutoffBar=1.0,  # needs to be able to show up to 100% match
        colorBar=["darkred", "darkblue"],
        group="Gene_set",
        ofname=f"{savePath}/{label}_dualPlot.png" if savePath else None,
    )

    return ax_dot


def plot_top_gsea(res: gp.Prerank, path: str, num=10, label="tst", man_terms=[], database=None):
    """
    Plot the top Gene Set Enrichment Analysis (GSEA) results.

    Args:
        res (gp.Prerank): The GSEA results object.
        path (str): The path to save the plot.
        num (int, optional): The number of top terms to plot. Defaults to 10.
        label (str, optional): The label for the plot. Defaults to "tst".
        man_terms (list, optional): A list of manually specified terms to plot. Defaults to an empty list.
        database (str, optional): The database name. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The generated plot.
        list: The list of terms used for plotting.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    if path:
        path = f"{path}/{label}_{num}_top"

    terms = res.res2d.Term[0:num]
    if len(man_terms):
        terms = man_terms
        path = f"{path}_manTerms"

    hits = [res.results[t]["hits"] for t in terms]
    runes = [res.results[t]["RES"] for t in terms]

    if database != None:
        new_terms = []
        for t in terms:
            t = t.replace(f"{database.upper()}_", "").replace("_", " ")
            if len(t) > 55:
                t = t[:55] + "..."
            new_terms.append(t)
        terms = new_terms

    fig = gseaplot2(
        terms=terms,
        RESs=runes,
        hits=hits,
        rank_metric=res.ranking,
        legend_kws={"loc": (0.56, 0.57)},  # set the legend loc
        figsize=(20, 14),
        ofname=path,
    )

    return fig, terms


# For global analysis
def rank_pi_vals(label: str, config: dict, dea_path: str, custom_points: dict, show_known_markers=False, th=500):
    """
    Rank the pi values and plot the results.

    Args:
        label (str): The label for the plot.
        config (dict): A dictionary containing configuration parameters.
        dea_path (str): The base path for DEA (Differential Expression Analysis).
        custom_points (dict): A dictionary of custom points to be marked on the plot.

    Returns:
        tuple: A tuple containing the plot object and a DataFrame with the rankings.
    """

    height = 900
    pi, pi_df = plotPi(config["file_1"], config["file_2"], base_path=dea_path, known_markers=False, markers={}, height=height)
    # -- max/mins
    min_x, max_x = pi_df["x"].min(), pi_df["x"].max()
    min_y, max_y = pi_df["y"].min(), pi_df["y"].max()

    # Marking the corner ref
    ref_point = [min_x, min_y]
    if config["ref_x"] == "max":
        ref_point[0] = max_x
    if config["ref_y"] == "max":
        ref_point[1] = max_y

    pi_df["dist"] = distance_from_ref(pi_df, ref_point)
    rank = pi_df["dist"].rank(ascending=False, method="first")

    # Re-draw with the rankings
    top_th = rank[rank > (rank.shape[0] - th)].index.values
    used_markers = {f"top_{th}": top_th}
    used_markers.update(custom_points)
    pi, pi_df = plotPi(config["file_1"], config["file_2"], base_path=dea_path, known_markers=show_known_markers, markers=used_markers, height=height)
    pi_df["rank"] = rank

    pi.add_scatter(
        x=[ref_point[0]],
        y=[ref_point[1]],
        marker_symbol="x-thin",
        marker_line_color="lightskyblue",
        marker_line_width=4,
        marker_size=15,
        mode="markers+text",
        showlegend=False,
        text=["Ref point"],
        textposition="bottom center",
    )

    if "path" in config.keys():
        if not os.path.exists(config["path"]):
            os.makedirs(config["path"])
        save_fig(f"pi_{label}", pi, height=height, width=1400, base_path=config["path"])

    return pi, pi_df


def run_gsea(subtype: str, config: dict, databases: dict):
    """
    Run Gene Set Enrichment Analysis (GSEA) for a given subtype using the provided configuration and databases.

    Args:
        subtype (str): The subtype for which GSEA is being performed.
        config (dict): A dictionary containing the configuration settings.
        databases (dict): A dictionary containing the databases to be used for GSEA.

    Returns:
        dict: A dictionary containing the GSEA results for each database.
    """

    results = {}
    df = config["pi_df"].reset_index()[["genes", "rank"]]
    base_path = config["path"]
    for key, value in databases.items():
        # database
        print(f"--> {key}")
        gsea_path = f"{base_path}/{key}/"
        rank_gsea = gp.prerank(
            rnk=df,
            gene_sets=value["path"],
            permutation_num=1000,
            outdir=gsea_path,
            format="png",
            no_plot=False,
            ascending=False,
            seed=20,
            graph_num=20,
            threads=12,
        )  # Change graph_num if you want to produce enrichment plots

        value["res"] = rank_gsea

        # Plot summaries plot
        _ = plot_enrichr_summary(rank_gsea.res2d, label=key, savePath=gsea_path)
        _, _ = plot_top_gsea(res=rank_gsea, path=base_path, num=10, label=f"{subtype}_{key}", database=key)

        results[key] = rank_gsea

    return results


def gene_match(gene_list: list, target_list: list):
    """
    Matches genes from the gene_list with the target_list and calculates the matching ratio.

    Args:
        gene_list (list): A list of genes to be matched.
        target_list (list): A list of target genes to match against.

    Returns:
        tuple: A tuple containing the following elements:
            - matched_count (int): The number of genes that matched between gene_list and target_list.
            - total (int): The total number of genes in gene_list.
            - ratio (float): The ratio of matched genes to the total number of genes in gene_list.
            - matched (list): A list of genes that matched between gene_list and target_list.
            - matched_str (str): A string representation of the matched genes, joined by commas.
    """
    matched = list(set(gene_list) & set(target_list))
    matched_count = len(matched)
    total = len(gene_list)
    ratio = matched_count / total if total > 0 else 0  # Avoid division by zero
    return matched_count, total, ratio, matched, ", ".join(matched)


def process_gsea(config: dict, database="hallmark", th=5000):
    """
    Process Gene Set Enrichment Analysis (GSEA) using the given configuration.

    Args:
        config (dict): A dictionary containing the GSEA configuration.
        database (str, optional): The database to use for GSEA. Defaults to "hallmark".
        th (int, optional): The threshold for selecting top genes. Defaults to 5000.

    Returns:
        pandas.DataFrame: The processed GSEA results dataframe sorted by matched lead ratio.
    """

    res_df = config[database].res2d
    top_sel = config[database].ranking[:th].index

    res_df["Lead_genes_list"] = res_df["Lead_genes"].str.split(";", expand=False).rename("Lead_genes_list")

    dmy = res_df["Lead_genes_list"].apply(gene_match, target_list=top_sel)
    res_df[["#matched_lead_genes", "#lead_genes", "matched_lead_ratio", "matched_list", "lead_genes_matched"]] = pd.DataFrame(dmy.tolist(), index=res_df.index)

    return res_df.sort_values("matched_lead_ratio", ascending=False)
