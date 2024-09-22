#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   pgcna_processing.py
@Time    :   2023/01/24 10:28:02
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Functions used to analyse the output from PGCNA
'''

import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scipy.stats import percentileofscore, zscore

# PGCNA scores
def get_ModCon(leiden_best, tpm_values, meta_df):
    modCons = {}

    all_genes = []
    for mod_class in leiden_best["Modularity Class"].unique():
        # 1. Get the genes in the module
        genes = leiden_best.loc[leiden_best["Modularity Class"] == mod_class]["Id"].values
        all_genes.extend(genes)

        # 2. Find the TPMs and compute the corr matrix
        df = tpm_values.loc[tpm_values.index.isin(genes)]
        df_corr = df.transpose().corr(method="spearman")
        # 3. Summ all the values to get the connectivity value
        connectivity = df_corr.sum(axis=0)

        # 4. Create the DataFrame with the gene metrics (across dataset(s)) and the connectivity (info at the module level)
        working_df = pd.concat([meta_df.loc[meta_df["genes"].isin(genes)].set_index("genes"), connectivity], axis=1).rename(columns={0:"connectivity"})
        # 5. Workout the ModCon and save it
        working_df["ModCon"] = (working_df["connectivity"] ** 2) * working_df["q2E"] * working_df["varWithin"] * (100 - working_df["varAcross"]) / 100
        modCons[mod_class] = working_df.sort_values(by="ModCon", ascending=False)
    return modCons, set(all_genes)



# Stats
def compute_graph_stats(graph_obj):
    graph_stats = pd.DataFrame(index = graph_obj.vs[:]["name"])
    graph_stats["betwenees"] = graph_obj.betweenness()
    graph_stats["closeness"] = graph_obj.closeness()
    graph_stats["degree"] = graph_obj.degree()
    graph_stats["strength"] = graph_obj.strength()
    graph_stats["pageRank"] = graph_obj.pagerank()
    graph_stats["hubScore"] = graph_obj.hub_score()

    return graph_stats

def top_genes(graph_stats):
    top_num, display_num = 100, 10
    common = {}

    print("### Top {} for ".format(display_num))
    for col in graph_stats.columns:
        metric = graph_stats.sort_values(by=[col], ascending=False).iloc[:top_num]
        genes = list(metric.index)
        print(" {}: {}".format(col, genes[:display_num]))
        if len(common):
            common = common & set(genes)
        else:
            common = set(genes)

    print(" shared genes ({}) : {} ".format(len(common), common))

def plot_metrics(graph_obj, graph_stats, exp_name):

    # node stats
    graph_stats.hist(graph_stats.columns, figsize=(25, 10), bins=200)


# plot stats
def plot_stats(modCons, metric_x="ModCon", metric_y="degree", height=1500):
    fig = go.Figure()

    traces, titles = [], []
    for key, value in modCons.items():
        # indexes are the genes
        trace = go.Scatter(x=value[metric_x], y=value[metric_y], mode="markers", text=value.index, hoverinfo='all')
        if metric_x != "count":
            titles.append("Com_{}".format(key))
        else:
            # 1 > mut, > 5 more and >10
            prct_mut = 1 - (modCons[key]["count"].values == 0).sum() / modCons[key].shape[0]
            prct_mut_5 = (modCons[key]["count"].values >= 5).sum() / modCons[key].shape[0]
            prct_mut_10 = (modCons[key]["count"].values >= 10).sum() / modCons[key].shape[0]

            titles.append("Com_{}. {:.2f} (>1), {:.2f} (>=5), {:.2f} (>=10)".format(key, prct_mut, prct_mut_5, prct_mut_10))

        traces.append(trace)

    # configure subplots
    num_plots = len(modCons.keys())
    num_cols = 3
    num_rows = int(np.ceil(num_plots/num_cols))

    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=titles, shared_yaxes=False, horizontal_spacing=0.05, vertical_spacing=0.07)

    idx_row, idx_col = 1, 1
    for trace in traces:
        fig.add_trace(trace, row=idx_row, col=idx_col)
        # What we do here, we increment the column and row idxs. This means we increment the column at each iteration and reset it when it is divisible with the max number of columnbs. After that we increment the row idx
        if idx_col % num_cols == 0:
            idx_col = 0
            idx_row += 1
        idx_col += 1

    layout = go.Layout(title_text="Genes: {} vs {}".format(metric_x, metric_y))

    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='Black')), selector=dict(mode='markers'))
    fig.update_layout(layout, height=height)
    fig.update_xaxes(title_text=metric_x)
    fig.update_yaxes(title_text=metric_y)
    
    return fig

def plot_mev(df, height=1500):
    traces, titles = [], []
    metric_x = "mut_count"
    metric_y = "MEV"
    for col in df.columns[:-1]:
        trace = go.Scatter(x=df[metric_x], y=df[col], mode="markers", text=df.index, hoverinfo='all')
        traces.append(trace)
        titles.append("{} vs Mut".format(col))

    # configure subplots
    num_plots = len(df.columns[:-1])
    num_cols = 3
    num_rows = int(np.ceil(num_plots/num_cols))

    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=titles, shared_yaxes=False, horizontal_spacing=0.05, vertical_spacing=0.07)

    idx_row, idx_col = 1, 1
    for trace in traces:
        fig.add_trace(trace, row=idx_row, col=idx_col)
        # What we do here, we increment the column and row idxs. This means we increment the column at each iteration and reset it when it is divisible with the max number of columnbs. After that we increment the row idx
        if idx_col % num_cols == 0:
            idx_col = 0
            idx_row += 1
        idx_col += 1

    layout = go.Layout(title_text="Samples: {} vs {}".format(metric_x, metric_y))

    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='Black')), selector=dict(mode='markers'))
    fig.update_layout(layout, height=height)
    fig.update_xaxes(title_text=metric_x)
    fig.update_yaxes(title_text=metric_y)

    return fig

# 
def mut_count_corr(modCons, metrics_1 = "count"):
    metrics = modCons[1].columns[4:-1].values
    corrs_pd = pd.DataFrame()
    for metric in metrics:
        corrs = []
        for key, value in modCons.items():
            corrs.append(modCons[key][[metrics_1, metric]].corr("spearman").values[0][1])

        tst = pd.DataFrame(corrs, columns=["corr"])
        tst["metric"] = metric

        corrs_pd = pd.concat([tst, corrs_pd])

    fig = px.scatter(corrs_pd, y="corr", facet_col="metric", facet_col_wrap=4, title="{} vs Graphs stats".format(metrics_1))
    return fig
