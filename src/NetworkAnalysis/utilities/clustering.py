#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   clustering.py
@Time    :   2023/06/12 14:10:48
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Relevant clustering functions for Network project. The functions were taken from the BU/Subtyping project.
"""
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px

# plotting
import plotly.graph_objects as go

# from fcmeans import FCM
from plotly.subplots import make_subplots
from sklearn import cluster, metrics, mixture
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

colours_pool = (
    px.colors.qualitative.Bold
    + px.colors.qualitative.D3
    + px.colors.qualitative.G10
    + px.colors.qualitative.T10
    + px.colors.qualitative.Pastel
    + px.colors.qualitative.Prism
    + px.colors.qualitative.Vivid
    + px.colors.qualitative.Alphabet
)


# Functions from utilities on clustering


def clustering_methods(datasets, default_base, samples, selected_clusters=None):
    """This function is the core of applying different clustering methods to a dataset. It can run different experiments with different datasets and configuration for the algorithms. It's a modification of the scikit-learn [blogpost](https://scikit-learn.org/stable/modules/clustering.html). The following clusters are supported (name, id):

        1. Kmeans - RawKMeans"
        2. Mini Batch KMeans - MiniBatchKMeans
        3. Ward - Ward
        4. Birch - Birch
        5. Gaussian Mixture Models - GaussianMixture
        6. Affinity Propagation - AffinityPropagation
        7. SpectralClustering - Spectral Clustering
        8. DBSCAN - DBSCAN
        9. OPTICS - OPTICS
        10. Hierarchical Clustering - Hierarchical Clustering

    Args:
        datasets ([dic]): List of the touples which containes the datasets, which is a DataFrame and the cluster models parameters in the form of a dictionary that needs to be override. See default_base of the parameters that can be override.
        default_base ([dict]): The configurations to be override for an experiments. Defaults to 3 clusters and birch_th 1.7. In case it needs to be override, below are the acceptable parameters and defualt values:
                    {'quantile': .2,
                    'eps': .3,
                    'damping': .5,
                    'preference': 5,
                    'n_neighbors': 5,
                    'n_clusters': 3,
                    'min_samples': 5,
                    'xi': 0.05,
                    'min_cluster_size': 0.1,
                    "birch_th": 1.7,
                    'name': "Name of the experiment" }
        samples ([String]): The list of samples
        selected_clusters ([String], optional): List of the strings that are the cluster models supported. Defaults to None, which means that all the available cluster models will be used.

    Returns:
        [list]: List of touples which contains the following data (in  this order): the experiment name, cluster model output and the cluster model object itself.
    """

    # for backwards compatibility
    if "fuzziness" != default_base:
        default_base["fuzziness"] = 1.15

    # array which contains a list with the name of dataset, the cluster models and the output labels
    ret_val = []
    for dataset, algo_params in datasets:
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X = dataset.values
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=params["n_neighbors"], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # ============
        # Create cluster objects
        # ============
        two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"])
        kmeans = cluster.KMeans(n_clusters=params["n_clusters"], max_iter=1000, random_state=10, n_init=10)
        ward = cluster.AgglomerativeClustering(n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity)
        birch = cluster.Birch(n_clusters=params["n_clusters"], threshold=params["birch_th"])
        affinity_propagation = cluster.AffinityPropagation(damping=params["damping"], preference=params["preference"])
        # we know from prev experiments this is a good configuration
        gmm = mixture.GaussianMixture(n_components=params["n_clusters"], covariance_type="diag", max_iter=500)
        spectral = cluster.SpectralClustering(n_clusters=params["n_clusters"], eigen_solver="arpack", affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        optics = cluster.OPTICS(min_samples=params["min_samples"], xi=params["xi"], min_cluster_size=params["min_cluster_size"])
        hierarchical_clutering = cluster.AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage="average", connectivity=connectivity)

        average_linkage = cluster.AgglomerativeClustering(linkage="average", metric=params["affinity"], n_clusters=params["n_clusters"])

        # fcm = FCM(n_clusters=params["n_clusters"], m=params["fuzziness"])

        # average_linake_2 = cluster.AgglomerativeClustering(linkage="average", n_clusters=params["n_clusters"]);

        clustering_algorithms = (
            ("RawKMeans", kmeans),
            # ("FuzzyCMeans", fcm),
            ("MiniBatchKMeans", two_means),
            ("Ward", ward),
            ("Birch", birch),
            ("GaussianMixture", gmm),
            ("AffinityPropagation", affinity_propagation),
            ("SpectralClustering", spectral),
            ("Avg", average_linkage),
            # ('AggCluster', average_linkage),
            ("DBSCAN", dbscan),
            ("OPTICS", optics),
            ("Hierarchical Clustering", hierarchical_clutering),
        )

        # run only the ones we're interested
        clustering_algorithms = [clustering_algorithm for clustering_algorithm in clustering_algorithms if clustering_algorithm[0] in selected_clusters]

        output_algorithms = pd.DataFrame(samples, columns=["samples"])
        output_models = []

        for name, algorithm in clustering_algorithms:
            # selected_clusters is None means that all are selected
            if selected_clusters is not None:
                if name not in selected_clusters:
                    continue  # skip if it's not in the loop
            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the "
                    + "connectivity matrix is [0-9]{1,2}"
                    + " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore", message="Graph is not fully connected, spectral embedding" + " may not work as expected.", category=UserWarning
                )
                algorithm.fit(X)

            if hasattr(algorithm, "labels_"):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(X)

            output_algorithms[name] = y_pred
            output_models.append((name, algorithm))

        # this need refactoring to returning a dictionary with key, pairs as it will be clearer
        ret_val.append((params["name"], output_algorithms, output_models))
    return ret_val


def compare_exp(
    data, rob_comp=None, n_clusters=None, default_base=None, selected_clusters=None, show_figures=True, show_consensus=True, pca_data=False, n_comp=2
):
    data = data.sort_index()
    # Process results
    samples = list(data.index.values)
    # sel_cols = data.columns.values

    # Show output
    pca_df, pca_plot = apply_pca(df=data, pca_components=2, transpose=True, samples_col="Sample", genes_col="Genes")

    pca_model = {}
    if pca_data:
        pca = PCA(n_components=n_comp)
        initial_data = data
        data = pd.DataFrame(pca.fit_transform(data), index=initial_data.index)
        pca_model["pca"] = pca
        pca_model["data"] = data
        print("PC var {} with a total of {:.02f}%".format(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_) * 100))
        print("PCA score ", pca.score(initial_data.values))

    datasets = []
    if n_clusters is None:
        for cluster_no in range(2, 15):
            datasets.append((data, {"name": f"CS_{cluster_no}", "n_clusters": cluster_no}))
    else:
        datasets = [(data, {"name": f"CS_{n_clusters}", "n_clusters": n_clusters})]

    if selected_clusters is None:
        selected_clusters = ["Ward", "Birch", "SpectralClustering", "RawKMeans", "GaussianMixture", "Avg"]

    if default_base is None:
        default_base = {
            "quantile": 0.2,
            "eps": 0.3,
            "damping": 0.5,
            "preference": 5,
            "n_neighbors": 5,
            "n_clusters": 3,
            "min_samples": 5,
            "xi": 0.05,
            "min_cluster_size": 0.1,
            "name": "",
            "birch_th": 1.7,
            "affinity": "cosine",
            "fuzziness": 1.2,
        }

    all_outputs = pd.DataFrame(index=samples)

    # Run the experiments
    results = clustering_methods(datasets, default_base, samples, selected_clusters)

    models = []
    all_metrics = pd.DataFrame()
    for exp_name, output, cluster_models in results:
        # added the aditional data
        algorithm_names = output.iloc[:, 1:].columns

        if exp_name not in output.columns[1]:
            new_cols = {col_name: col_name + "_" + exp_name for col_name in algorithm_names}
            new_cols["samples"] = "Samples"
            output.rename(columns=new_cols, inplace=True)

        models.append((exp_name, cluster_models))
        all_outputs = pd.concat([all_outputs, output.set_index("Samples")], axis=1)

        # plot metrics
        metrics_df = compute_cluster_metrics(cluster_models, data)
        if 0:
            plot_cluster_metrics(metrics_df, exp_name)

        metrics_df["Exp"] = exp_name
        metrics_df["Cluster"] = metrics_df["Cluster"] + "_" + exp_name
        all_metrics = pd.concat([all_metrics, metrics_df])

    # make the biggest cluster is 0
    # pca_bu_df = pca_bu_df.sort_values(by="Sample").reset_index(drop=True)
    pca_df = pd.concat([pca_df.set_index("Sample"), all_outputs], axis=1)
    all_outputs.reset_index().rename(columns={"index": "Samples"})

    for col in all_outputs.columns:
        pca_df[col] = order_labels_size(pca_df[col])
        all_outputs[col] = order_labels_size(pca_df[col])

    all_outputs = pca_df.copy(deep=True)

    if show_figures:
        pca_df = pca_df.reset_index().rename(columns={"index": "Sample"})
        # add_metadata(pca_bu_df, dummy_meta, type="TCGA")
        plot_cols = list(pca_df.columns[2:-7])

        if show_consensus:
            plot_cols = ["TCGA408_classifier"] + plot_cols  # "consenus" add consensus
            pca_df["consenus"] = pd.factorize(pca_df["2019_consensus_classifier"])[0]
            pca_df["TCGA408_classifier"] = pd.factorize(pca_df["TCGA408_classifier"])[0]
            pca_df["consenus"] = order_labels_size(pca_df["consenus"])
            pca_df["TCGA408_classifier"] = order_labels_size(pca_df["TCGA408_classifier"])

        if rob_comp is not None:
            # transform labels to codes
            for col in rob_comp.columns:
                rob_comp[col] = order_labels_size(rob_comp[col])
                plot_cols.append(col)
            pca_df = pd.concat([pca_df, rob_comp], axis=1)

        if len(plot_cols) / 9 > 1:
            no_plots = int(len(plot_cols) / 9)

            for no_plot in range(0, no_plots + 1):
                fig = plot_clusters(
                    pca_df,
                    plot_cols[no_plot * 9 : (no_plot + 1) * 9],
                    x_label="PC_1",
                    y_label="PC_2",
                    exp_name="Various cluster methods I",
                    hover_data=list(pca_df.columns[-7:]),
                    pca_variance=pca_plot.explained_variance_ratio_,
                )
                fig.show()
        else:
            fig = plot_clusters(
                pca_df,
                plot_cols,
                x_label="PC_1",
                y_label="PC_2",
                exp_name="Various cluster methods",
                hover_data=list(pca_df.columns[-7:]),
                pca_variance=pca_plot.explained_variance_ratio_,
            )
            fig.show()

    # add metadata
    all_outputs = all_outputs.reset_index().rename(columns={"index": "Sample"})

    return all_outputs, models, all_metrics, pca_model


def compute_cluster_metrics(cluster_models, data, id=""):
    """Calculates the ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’ metrics for a given set of cluster models

    Args:
        cluster_models ([list]): List of cluster models such as Kmeans, Birch, Ward etc.
        data ([DataFrame]): The data used to trained the cluster models
        id ([String]): The id to append to the cluster names. This is useful for a set of experiments. Defaults to ""

    Returns:
        [DataFrame]: DataFrame used for storing the cluster metrics
    """

    clusters_metrics, cluster_names, sample_metrics = [], [], []
    for name, cluster_model in cluster_models:
        if name not in ["GaussianMixture", "FuzzyCMeans"]:
            labels = cluster_model.labels_
        else:
            labels = cluster_model.predict(data.values)

        # only if there is more than 1 label
        if len(set(labels)) > 1:
            # ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’

            sample_sillhouette_cosine = metrics.silhouette_samples(data, labels, metric="cosine")

            silhoute_metric = np.mean(sample_sillhouette_cosine)
            silhoute_metric_2 = metrics.silhouette_score(data, labels, metric="manhattan")
            silhoute_metric_3 = metrics.silhouette_score(data, labels, metric="cosine")
            calinski_harabasz_metric = metrics.calinski_harabasz_score(data, labels)
            davies_bouldin_metric = metrics.davies_bouldin_score(data, labels)

            sample_sillhouette_cosine = metrics.silhouette_samples(data, labels, metric="cosine")
            clusters_metrics.append([silhoute_metric, silhoute_metric_2, silhoute_metric_3, calinski_harabasz_metric, davies_bouldin_metric])
            sample_metrics.append((name + id, sample_sillhouette_cosine))
        else:
            clusters_metrics.append([0, 0, 0, 0, 0])
        cluster_names.append(name + id)

    metrics_names = ["Silhoute_euclidean", "Silhoute_manhattan", "Silhoute_cosine", "Calinski_habrasz", "Davies_bouldin"]
    cluster_metrics = pd.DataFrame(np.around(clusters_metrics, decimals=5), columns=metrics_names)
    cluster_metrics.insert(0, "Cluster", cluster_names)

    cluster_metrics["Samples_Silhouete_cos"] = sample_metrics

    return cluster_metrics


def apply_pca(df, pca_components, transpose=True, samples_col="samples", genes_col="genes"):
    """For a given DataFrame we apply Principal Component Analysis

    Args:
        df (DataFrame): The data to be processed
        samples_names ([list]): Neeed to add back to the returned df
        genes_names ([pd.Series]): Neeed to add back to the returned df
        pca_components ([int]): number of components
        transpose (bool, optional): This is needed so that we know how to add the genes/samples back to the df. Defaults to True.

    Returns:
        [DataFrame]: The PCA with samples and genes
    """
    # apply PCA
    samples_names = df.index.values
    cols = df.columns.values
    pca = PCA(n_components=pca_components)
    pca_bu = pca.fit_transform(df)
    print("Variation per principal component {} and the sum {:.02f}%".format(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_) * 100))

    # generate columns labels
    pca_col_labels = []
    for idx in range(pca_components):
        pca_col_labels.append("PC_" + str(idx + 1))  # start from 1

    pca_bu_df = pd.DataFrame(data=pca_bu, columns=pca_col_labels)
    # add dataframes labels accordingly
    dummy_df = df.copy(deep=True)
    if transpose:
        dummy_df.columns = cols
        dummy_df.insert(0, samples_col, samples_names)
        pca_bu_df.insert(0, samples_col, samples_names)
    else:
        dummy_df.columns = samples_names
        dummy_df.insert(0, genes_col, cols)
        pca_bu_df.insert(0, genes_col, cols)

    return pca_bu_df, pca


# plotting functions
def plot_clusters(df, plot_cols, x_label, y_label, exp_name, hover_data=None, marker_text=None, marker_text_position=None, pca_variance=None):
    """Function that plots the results from the clustering methods. It can received any number of columns values strings and it will display 3 on a row. This means that it will always display plots with 3 columns

    Args:
        df ([DataFrame]): At the moment only it supports 2d result DataFrames. This means can be either a t-sne or PCA
        plot_cols ([list]): A list of strings with the columns values to be ploted
        x_label([String]): The column of x which should be plotted, for t-sne it can be sne_2d_one. It has to be in df arg
        y_label ([String]): The column of y which should be plotted, for t-sne it can be sne_2d_two. It has to be in df arg
        colours_pool ([list]): The list of colours which are used for plotting
        exp_name ([String]): Name for experiment which is used for plots
        marker_text ([String], optional): The string which is a column name from the df. Defaults to None.
        marker_text_position ([String], optional): A string identified form plotly that's used to center accordingly the marker text. Defaults to None.
    """
    num_cols = 3
    num_rows = int(np.ceil(len(plot_cols) / num_cols))
    if hover_data == None:
        hover_data = plot_cols[:]
        hover_data.append("samples")

    fig = make_subplots(
        rows=num_rows, cols=num_cols, subplot_titles=plot_cols, horizontal_spacing=0.1, vertical_spacing=0.1, shared_xaxes=True, shared_yaxes=True
    )

    traces = []
    for plot_col in plot_cols:
        hover_data.append(plot_col)
        trace = trace_2d(
            df, plot_col, x_label, y_label, hover_cols=hover_data, dot_colour=plot_col, marker_text=marker_text, marker_text_position=marker_text_position
        )
        traces.append(trace)
        hover_data.remove(plot_col)

    # add the traces to the subplot
    subplot_titles = []
    idx_row, idx_col = 1, 1
    for trace in traces:
        subplot_titles.append(trace["name"])
        fig.add_trace(trace, row=idx_row, col=idx_col)
        # What we do here, we increment the column and row idxs. This means we increment the column at each iteration and reset it when it is divisible with the max number of columnbs. After that we increment the row idx
        if idx_col % num_cols == 0:
            idx_col = 0
            idx_row += 1
        idx_col += 1

    if pca_variance is not None:
        x_label = x_label + " ({:.02f}%)".format(pca_variance[0] * 100)
        y_label = y_label + " ({:.02f}%)".format(pca_variance[1] * 100)

    layout = go.Layout(title_text=exp_name)

    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="Black")), selector=dict(mode="markers"))
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(layout, height=1500)
    return fig


def trace_2d(df, title, x_axis, y_axis, hover_cols=None, dot_colour=None, marker_text=None, marker_text_position=None):
    """This creates a 2D trace that it can be later be used for subplots in plotly. Easy to customise, the only constriant is that the x and y axis has to be part of the df argument.

    Args:
        df ([DataFrame]): The data that contains the points to be plotted.
        colours_pool ([list]): The list of colours which are used for plotting
        title ([String]): The title of the plot
        x_axis ([String]): X label, it has to be part of the DataFrame
        y_axis ([String]): Y label, it has to be part of the DataFrame
        hover_cols ([list], optional): [description]. Defaults to (centroids and samples for backwards compatibility).
        dot_colour ([String], optional): The string which is the column name of the df and by that it's done the colouring. Defaults to centroids (for backwards compatibility).
        marker_text ([String], optional): The string which is a column name from the df. Defaults to None.
        marker_text_position ([String], optional): A string identified form plotly that's used to center accordingly the marker text. Defaults to None.

    Returns:
        [trace]: The trace object that was created
    """
    hover, colours, markers = [], [], {}
    mode_markers = "markers"
    text_list = []
    if hover_cols == None:
        hover_cols = ["centroids", "samples", "diagnosis", "labels_ter_avg", "labels_ter_sd"]
    if dot_colour == None:
        dot_colour = "centroids"
    if marker_text != None:
        mode_markers = "markers+text"
        text_list = df[marker_text].values
        if marker_text_position == None:
            marker_text_position = "bottom left"
    for _, row in df.iterrows():
        centroid = row[dot_colour]
        # create the hover data
        hover_string = ""
        for hover_col in hover_cols:
            hover_string += "<br>%s=%s" % (hover_col, str(row[hover_col]))
        hover_string += "<br>" + x_axis + "=%{x}" + "<br>" + y_axis + "=%{y}"
        hover.append(hover_string)

        colours.append(colours_pool[centroid])

    markers["color"] = colours
    markers["size"] = 6
    trace = dict(
        type="scatter",
        x=df[x_axis],
        y=df[y_axis],
        hovertemplate=hover,
        showlegend=True,
        name=title,
        marker=markers,
        mode=mode_markers,
        text=text_list,
        textposition=marker_text_position,
    )
    return trace


def plot_cluster_metrics(metrics, exp_name, hide_text=False):
    """Creates the the figures for a given DataFrame of metrics.

    Args:
        metrics (DataFrame): Holds the metrics values, columns names being represented by the algorithm for which the metrics
        algorithm_names ([list]): List of strings of the algorithms that are being run
        exp_name ([String]): Name of the experiment
        hide_text ([Bool]): If True the points text is hidden. This is usefull when there are lots of points. Defaults to False
    """
    fig = go.Figure()
    # define x axis
    metrics_names = ["Silhoute_euclidean", "Silhoute_manhattan", "Silhoute_cosine", "Calinski_habrasz", "Davies_bouldin"]
    mode, text, traces = "lines+markers+text", metrics["Cluster"].values, []
    random_x = np.linspace(0, 15, metrics.shape[0])
    if hide_text:
        mode = "lines+markers"
    for metrics_name in metrics_names:
        trace = go.Scatter(x=random_x, y=metrics[metrics_name], mode=mode, text=text, hoverinfo="all", textposition="top right")
        traces.append(trace)

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "Silhouette euclidean (higher the better)",
            "Silhouette manhattan (higher the better)",
            "Silhouette cosine (higher the better)",
            "Calinski-Harabrasz (higher the better)",
            "Davies-Bouldin (lower the better)",
        ],
        shared_yaxes=False,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    for idx, trace in enumerate(traces):
        fig.add_trace(trace, row=int(idx / 2) + 1, col=idx % 2 + 1)

    layout = go.Layout(title_text="Different clustering metrics " + exp_name)
    fig.update_layout(layout)
    fig.show()


def mark_best_3(ref, metrics, metric="Silhoute_cosine"):
    # Remove the experiments w/ CS_2 and CS_3
    ascending = False
    if metric == "Davies_bouldin":
        ascending = True
    best_sill = metrics[~metrics["Exp"].str.contains("2|3")].sort_values(by=[metric], ascending=ascending)[:3]

    offset = best_sill[metric].mean() / 50

    annotations = []
    i = 0
    for i_dx, row in best_sill.iterrows():
        i += 1
        for j in range(1, i + 1):
            annotations.append(
                dict(x=row["Cluster"], y=row[metric] + offset * i * j, text="x", showarrow=False, xref="x{}".format(ref), yref="y{}".format(ref))
            )
    return annotations


def display_metrics(metrics, exp_title="Metrics", show_individual=False, verbose=True):
    """
        Receives the Metrics DataFrame from comp_exp function above.

    Args:
        metrics ([DataFrame]): It needs to contain
        exp_name (str, optional): [description]. Defaults to "".
    """
    # other sillhouettes available (Silhoute_euclidean', 'Silhoute_manhattane')

    metrics["cluster_type"] = ["-".join(cluster.split("_")[:1]) for cluster in metrics["Cluster"]]

    title_1 = "Cosine Sillhouette (higher the better)"
    fig_1 = px.bar(metrics, y="Silhoute_cosine", x="Cluster", title="{}. {}".format(exp_title, title_1), color="cluster_type")
    fig_1.update_yaxes(title_text="Sillhouette Avg")
    fig_1.update_xaxes(title_text="Experiment")
    annotations_1 = mark_best_3(1, metrics, metric="Silhoute_cosine")

    title_2 = "Calinski_habrasz (higher the better)"
    fig_2 = px.bar(metrics, y="Calinski_habrasz", x="Cluster", title="{}. {}".format(exp_title, title_2), color="cluster_type")
    fig_2.update_yaxes(title_text="Calinski Habrasz Avg")
    fig_2.update_xaxes(title_text="Experiment")
    fig_2.update_layout(xaxis={"categoryorder": "array", "categoryarray": metrics["Cluster"]})
    annotations_2 = mark_best_3(2, metrics, metric="Calinski_habrasz")

    title_3 = "Davies_bouldin (lower the better)"
    fig_3 = px.bar(metrics, y="Davies_bouldin", x="Cluster", title="{}. {}".format(exp_title, title_3), color="cluster_type")
    fig_3.update_yaxes(title_text="Davies Bouldin Avg")
    fig_3.update_xaxes(title_text="Experiment")
    fig_3.update_layout(xaxis={"categoryorder": "array", "categoryarray": metrics["Cluster"]})
    annotations_3 = mark_best_3(3, metrics, metric="Davies_bouldin")

    # Extract the traces
    figs = [fig_1, fig_2, fig_3]
    all_traces = []
    for fig in figs:
        fig_traces = []
        for trace in range(len(fig["data"])):
            fig_traces.append(fig["data"][trace])
        all_traces.append(fig_traces)

    subplots = make_subplots(rows=3, cols=1, subplot_titles=[title_1, title_2, title_3], shared_xaxes=True, vertical_spacing=0.05)

    for idx, fig_traces in enumerate(all_traces):
        for traces in fig_traces:
            subplots.append_trace(traces, col=1, row=idx + 1)

    # the subplot as shown in the above image
    layout = go.Layout(title_text="{}. Combined metrics ".format(exp_title))
    subplots.update_layout(layout, annotations=annotations_1[:3] + annotations_1 + annotations_2 + annotations_3)

    visited = []
    for trace in subplots["data"]:
        trace["name"] = trace["name"].split(",")[0]
        if trace["name"] not in visited:
            trace["showlegend"] = True
            visited.append(trace["name"])
        else:
            trace["showlegend"] = False

    if show_individual:
        fig_1.show()
        fig_2.show()
        fig_3.show()

    return subplots


def plot_meta_scores(df, y_axis="IFNG", classification="Basal_type", size="infiltration_score"):
    # infg_df["2019CC_Ba/Sq"] = infg_df["2019CC_Ba/Sq"].astype(float)
    # infg_df["2019CC_Stroma-rich"] = infg_df["2019CC_Stroma-rich"].astype(float)

    category_order = {
        "Basal_type": ["IFNG-IFNG (High) - 42", "Basal-IFNG (Medium) - 38", "Basal-Basal (Low) - 40"],
        "infiltration_label": ["High Inf", "Medium Inf", "No Inf"],
    }

    title = "Classification by {}. IFNG vs infiltration score; Colouring by Estimate score".format(classification)
    fig1 = px.scatter(df, x="ESTIMATE_score", y=y_axis, color=classification, title=title, category_orders=category_order, size=size)
    fig1.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    # fig1.show()

    title = "Classification by {}. IFNG vs infiltration score; Colouring by Imune rich score".format(classification)
    fig = px.scatter(df, x="Immune_score", y=y_axis, color=classification, title=title, category_orders=category_order, size=size)

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    # fig.show()

    title = "Classification by {}. IFNG vs infiltration score; Colouring by Stroma rich score".format(classification)
    fig = px.scatter(df, x="Stromal_score", y=y_axis, color=classification, title=title, category_orders=category_order, size=size)

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    # fig.show()

    return fig1


# Helper functions
def order_labels_size(df_series):
    """This function ensures that the the cluster labels are always in the order of the cluster sizes. This means that the cluster label 0 will correspond for the largest cluster and for n-clusters the n-1 will be the cluster with the lowest members.

    Args:
        df_series (Pandas Series): The cluster labels to be mapped

    Returns:
        [Pandas Series]: New mappped pandas series
    """
    # ordered by the number of frequency
    cluster_count = [clust for clust, _ in Counter(df_series).most_common()]
    new_labels = list(range(len(df_series.unique())))
    dic = dict(zip(cluster_count, new_labels))
    # print("Remap of the old vs new labels",dic)
    return df_series.map(dic)


def elbow_method(df, min_k, max_k, label):
    """Apply elbow method on K-means for a given data (DataFrame)

    Args:
        df ([DataFrame]): Input data
        min_k ([Int]):
        max_k ([Int]):

    Returns:
        [plotly.Figure, list]: Return the figure to be plotted and the sum squared distances
    """
    sum_dist, K = [], range(min_k, max_k)
    for k in K:
        km = cluster.KMeans(n_clusters=k, n_init=10)
        km = km.fit(df)
        sum_dist.append(km.inertia_)

    data = go.Scatter(x=list(K), y=sum_dist, mode="lines+markers")
    layout = go.Layout(
        title=f"{label}. Elbow method based on K-means inertia (sum sqrd dist)",
        xaxis=go.layout.XAxis(title="Clusters"),
        yaxis=go.layout.YAxis(title="Sum of squared distances"),
        title_y=0.9
    )
    fig = go.Figure(data=data, layout=layout)


    return fig, sum_dist
