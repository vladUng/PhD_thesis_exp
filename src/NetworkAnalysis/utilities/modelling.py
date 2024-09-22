#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   modelling.py
@Time    :   2022/07/06 14:12:01
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Purpose of this file is to contain the core computational methods
"""

import os
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

# plotting
import plotly.graph_objects as go
from fcmeans import FCM
from lifelines import KaplanMeierFitter
from matplotlib import cm
from plotly.subplots import make_subplots
from sklearn import cluster, metrics, mixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from .helpers import save_fig

from lifelines.statistics import multivariate_logrank_test


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

##### Metadata #####


def add_metadata(df, df_meta, tissue_type="BU"):
    if tissue_type == "BU":
        bu_add_metadata(df, df_meta)
    elif tissue_type == "TCGA":
        tcga_add_metadata(df, df_meta)
    else:
        print("Not implemented type: {}".format(tissue_type))


def bu_add_metadata(df, df_meta):
    """Adds to a df metadata columns with information such as TER avg, std, tissue type etc.
        Note: It doesn't all the metadata, but some selected, which was needed.  \n

    Args:
        df ([DataFrame]): Has the rows as samples and the columns either PC or t-sne values. Also, it's the one to whom the metadata is appended it  \n
        df_meta ([DataFrame]): Df that has the samples metadat, like age, TER etc  \n

    Returns:
        [LabelEncoder]: This it's returned as it may be useful to classify later.  \n
        [list]: Column with all the metadata added  \n
    """
    diff_seq, tissue_types, diagnosis = [], [], []
    ter_avgs, ter_sds = [], []
    labels_ter_avg, labels_ter_sd, labels_date, labels_cystometric = [], [], [], []

    # remove the white characters be aware this will change the argument passed
    df_meta["sample"] = df_meta["sample"].str.strip()
    # Add TER avg and std, but first we have to impute some values
    df_meta["TER_avg"].fillna(df_meta["TER_avg"].median(), inplace=True)
    df_meta["TER_sd"].fillna(df_meta["TER_sd"].median(), inplace=True)
    # add labels
    label_encoder = LabelEncoder()
    df_meta["labels_date"] = label_encoder.fit_transform(df_meta["date"])
    df_meta["labels_cystometric_capacity"] = label_encoder.fit_transform(df_meta["cystometric_capacity"])

    # add the sequencing machine column, N/A if the tissue type doesn't fit
    for sample_name in df["samples"]:
        sample_id = sample_name.split("-")[0].strip()
        tissue_type = sample_name.split("-")[1].strip()
        # tissue types
        tissue_types.append(tissue_type)
        # diff_seq
        sample = df_meta[df_meta["sample"] == sample_id]
        if len(sample["diff_sequencer"].values) > 0:
            if tissue_type == "D":  # seperate for diff and P0 tissue
                diff_seq.append(sample["diff_sequencer"].values[0])
            elif tissue_type == "P0":
                diff_seq.append(sample["P0_sequencer"].values[0])
            else:
                diff_seq.append("N/A")
        else:
            diff_seq.append("N/A")

        # ter avg and standard deviation
        ter_avg = sample["TER_avg"].values[0]
        ter_sd = sample["TER_sd"].values[0]
        # labels for ter_avg
        if ter_avg < 100:
            labels_ter_avg.append("absent")
        elif ter_avg > 100 and ter_avg < 500:
            labels_ter_avg.append("weak")
        elif ter_avg > 500:
            labels_ter_avg.append("tight")

        # labels for ter_sd
        if ter_sd < 100:
            labels_ter_sd.append("absent")
        elif ter_sd > 100 and ter_sd < 500:
            labels_ter_sd.append("weak")
        elif ter_sd > 500:
            labels_ter_sd.append("tight")

        # add to the arrays
        ter_avgs.append(ter_avg)
        ter_sds.append(ter_sd)
        labels_date.append(sample["labels_date"].values[0])
        labels_cystometric.append(sample["labels_cystometric_capacity"].values[0])
        # clinical diagnosis
        diagnosis.append(sample["clin_diagnosis"].values[0])

    # add the data to the given df
    df["tissue_type"] = tissue_types
    df["diff_sequencer"] = diff_seq
    df["diagnosis"] = diagnosis
    df["ter_avg"] = ter_avgs
    df["labels_ter_avg"] = labels_ter_avg
    df["ter_sd"] = ter_sds
    df["labels_ter_sd"] = labels_ter_sd
    df["labels_date"] = labels_date
    df["labels_cystometric_capacity"] = labels_cystometric

    # we may need this later to unpack some of the encoded values
    # TODO: delete this
    cols = ["samples", "tissue_type", "labels_ter_avg", "ter_avg", "labels_cystometric_capacity", "labels_date", "labels_ter_sd", "ter_sd", "diff_sequencer"]
    return cols


def tcga_add_metadata(df, df_meta):
    # This requires that both df_meta and df have samples columns
    df_meta_cpy = df_meta.copy(deep=True)
    df_meta_cpy.set_index("sample", inplace=True)
    df.set_index("Sample", inplace=True)

    # first numeric data
    for col in ["cigarettes_per_day", "bmi", "weight", "height"]:
        df_meta[col] = df_meta[col].replace("--", np.nan).astype(float)
        df_meta[col].fillna(df_meta[col].median(), inplace=True)
        df[col] = df_meta_cpy[col]

    df["2019_consensus_classifier"] = df_meta_cpy["2019_consensus_classifier"]
    df["gender"] = df_meta_cpy["gender"]
    df["TCGA408_classifier"] = df_meta_cpy["TCGA408_classifier"]

    df.reset_index().rename(columns={"Sample": "samples"}, inplace=True)


##### Methods #####


def apply_pca(df, samples_names, genes_names, pca_components, transpose=True, samples_col="samples", genes_col="genes"):
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
        dummy_df.columns = genes_names
        dummy_df.insert(0, samples_col, samples_names)
        pca_bu_df.insert(0, samples_col, samples_names)
    else:
        dummy_df.columns = samples_names
        dummy_df.insert(0, genes_col, genes_names)
        pca_bu_df.insert(0, genes_col, genes_names)

    return pca_bu_df, pca


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

        average_linkage = cluster.AgglomerativeClustering(linkage="average", metric=params["metric"], n_clusters=params["n_clusters"])

        fcm = FCM(n_clusters=params["n_clusters"], m=params["fuzziness"])

        # average_linake_2 = cluster.AgglomerativeClustering(linkage="average", n_clusters=params["n_clusters"]);

        clustering_algorithms = (
            ("RawKMeans", kmeans),
            ("FuzzyCMeans", fcm),
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
    data: pd.DataFrame,
    selected_genes: list,
    raw_metadata_t: pd.DataFrame,
    rob_comp=None,
    n_clusters=None,
    default_base=None,
    selected_clusters=None,
    show_figures=True,
    show_consensus=True,
    custom_points=None,
    custom_points_2=None,
    sb_samples=None,
    pca_data=False,
    n_comp=2,
):
    """
    
    The data has to be a DataFrame of samples x genes

    Args:
        data (pd.DataFrame): _description_
        selected_genes (list): _description_
        raw_metadata_t (pd.DataFrame): _description_
        rob_comp (_type_, optional): _description_. Defaults to None.
        n_clusters (_type_, optional): _description_. Defaults to None.
        default_base (_type_, optional): _description_. Defaults to None.
        selected_clusters (_type_, optional): _description_. Defaults to None.
        show_figures (bool, optional): _description_. Defaults to True.
        show_consensus (bool, optional): _description_. Defaults to True.
        custom_points (_type_, optional): _description_. Defaults to None.
        custom_points_2 (_type_, optional): _description_. Defaults to None.
        sb_samples (_type_, optional): _description_. Defaults to None.
        pca_data (bool, optional): _description_. Defaults to False.
        n_comp (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """

    data = data.sort_index()
    # Process results
    samples = list(data.index.values)

    # Show output
    pca_bu_df, pca_plot = apply_pca(
        df=data, samples_names=samples, genes_names=selected_genes, pca_components=2, transpose=True, samples_col="Sample", genes_col="Genes"
    )

    pca_model = {}
    if pca_data:
        pca = PCA(n_components=n_comp)
        initial_data = data
        data = pd.DataFrame(pca.fit_transform(data), index=initial_data.index)
        pca_model["pca"] = pca
        pca_model["data"] = data
        print("Variation per principal component {} and the sum {:.02f}%".format(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_) * 100))
        print("PCA score ", pca.score(initial_data))

    datasets = []
    if n_clusters is None:
        for cluster_no in range(3, 14):
            datasets.append((data, {"name": "CS_{}".format(cluster_no), "n_clusters": cluster_no}))
    else:
        datasets = [(data, {"name": "CS_{}".format(n_clusters), "n_clusters": n_clusters})]

    if selected_clusters is None:
        selected_clusters = ["Ward", "Birch", "SpectralClustering", "RawKMeans", "GaussianMixture"]

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
            "metric": "cityblock",
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
        metrics = compute_cluster_metrics(cluster_models, data)
        if 0:
            plot_cluster_metrics(metrics, exp_name)

        metrics["Exp"] = exp_name
        metrics["Cluster"] = metrics["Cluster"] + "_" + exp_name
        all_metrics = pd.concat([all_metrics, metrics])

    # make the biggest cluster is 0
    # pca_bu_df = pca_bu_df.sort_values(by="Sample").reset_index(drop=True)
    pca_bu_df = pd.concat([pca_bu_df.set_index("Sample"), all_outputs], axis=1)
    all_outputs.reset_index().rename(columns={"index": "Samples"})

    for col in all_outputs.columns:
        pca_bu_df[col] = order_labels_size(pca_bu_df[col])
        all_outputs[col] = order_labels_size(pca_bu_df[col])

    all_outputs = pca_bu_df.copy(deep=True)

    dummy_meta = raw_metadata_t[raw_metadata_t["Samples"].isin(samples)].rename(columns={"Samples": "sample"}).sort_values(by="sample").reset_index(drop=True)
    if show_figures:
        pca_bu_df = pca_bu_df.reset_index().rename(columns={"index": "Sample"})
        add_metadata(pca_bu_df, dummy_meta, tissue_type="TCGA")
        plot_cols = list(pca_bu_df.columns[2:-7])

        if custom_points is not None:
            pca_bu_df["Clustering difference"] = 0
            pca_bu_df.loc[pca_bu_df.index.isin(custom_points), "Clustering difference"] = 1
            pca_bu_df["Clustering difference"] = order_labels_size(pca_bu_df["Clustering difference"])
            plot_cols.append("Clustering difference")

        if custom_points_2 is not None:
            pca_bu_df["Clustering difference 2"] = 0
            pca_bu_df.loc[pca_bu_df.index.isin(custom_points_2), "Clustering difference 2"] = 1
            pca_bu_df["Clustering difference 2"] = order_labels_size(pca_bu_df["Clustering difference 2"])
            plot_cols.append("Clustering difference 2")
            # pca_bu_df['metric_labels'] =  order_labels_size(raw_metadata_t["metric_labels"])
            # plot_cols.append("metric_labels")

        if show_consensus:
            plot_cols = ["TCGA408_classifier"] + plot_cols  # "consenus" add consensus
            pca_bu_df["consenus"] = pd.factorize(pca_bu_df["2019_consensus_classifier"])[0]
            pca_bu_df["TCGA408_classifier"] = pd.factorize(pca_bu_df["TCGA408_classifier"])[0]
            pca_bu_df["consenus"] = order_labels_size(pca_bu_df["consenus"])
            pca_bu_df["TCGA408_classifier"] = order_labels_size(pca_bu_df["TCGA408_classifier"])

        if rob_comp is not None:
            # transform labels to codes
            for col in rob_comp.columns:
                rob_comp[col] = order_labels_size(rob_comp[col])
                plot_cols.append(col)
            pca_bu_df = pd.concat([pca_bu_df, rob_comp], axis=1)

        if sb_samples is not None:
            plot_cols = ["ifnq_split", "rarg_split"] + plot_cols

            pca_bu_df["ifnq_split"] = order_labels_size(dummy_meta.set_index("sample")["ifnq_split"])
            pca_bu_df["rarg_split"] = order_labels_size(dummy_meta.set_index("sample")["rarg_split"])

        if len(plot_cols) / 9 > 1:
            no_plots = int(len(plot_cols) / 9)

            for no_plot in range(0, no_plots + 1):
                fig = plot_clusters(
                    pca_bu_df,
                    plot_cols[no_plot * 9 : (no_plot + 1) * 9],
                    x_label="PC_1",
                    y_label="PC_2",
                    exp_name="Various cluster methods I",
                    hover_data=list(pca_bu_df.columns[-7:]),
                    pca_variance=pca_plot.explained_variance_ratio_,
                )
                fig.show()
        else:
            fig = plot_clusters(
                pca_bu_df,
                plot_cols,
                x_label="PC_1",
                y_label="PC_2",
                exp_name="Various cluster methods",
                hover_data=list(pca_bu_df.columns[-7:]),
                pca_variance=pca_plot.explained_variance_ratio_,
            )
            fig.show()

    # add metadata
    dummy_meta.set_index("sample", inplace=True)

    all_outputs["TCGA408_classifier"] = dummy_meta["TCGA408_classifier"]
    all_outputs["consensus"] = dummy_meta["2019_consensus_classifier"]
    all_outputs = all_outputs.reset_index().rename(columns={"index": "Sample"})

    return all_outputs, models, all_metrics, pca_model


##### Metrics #####


def compute_cluster_metrics(cluster_models, data, id=""):
    """Calculates the ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’ metrics for a given set of cluster models

    Args:
        cluster_models ([list]): List of cluster models such as Kmeans, Birch, Ward etc.
        data ([DataFrame]): The data used to trained the cluster models
        id ([String]): The id to append to the cluster names. This is useful for a set of experiments. Defaults to ""

    Returns:
        [DataFrame]: DataFrame used for storing the cluster metrics
    """

    clusters_metrics, cluster_names = [], []
    for name, cluster_model in cluster_models:
        if name not in ["GaussianMixture", "FuzzyCMeans"]:
            labels = cluster_model.labels_
        else:
            labels = cluster_model.predict(data.values)

        # ‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’
        silhoute_metric = metrics.silhouette_score(data, labels, metric="euclidean")
        silhoute_metric_2 = metrics.silhouette_score(data, labels, metric="manhattan")
        silhoute_metric_3 = metrics.silhouette_score(data, labels, metric="cosine")
        calinski_harabasz_metric = metrics.calinski_harabasz_score(data, labels)
        davies_bouldin_metric = metrics.davies_bouldin_score(data, labels)

        clusters_metrics.append([silhoute_metric, silhoute_metric_2, silhoute_metric_3, calinski_harabasz_metric, davies_bouldin_metric])
        cluster_names.append(name + id)

    metrics_names = ["Silhoute_euclidean", "Silhoute_manhattan", "Silhoute_cosine", "Calinski_habrasz", "Davies_bouldin"]
    cluster_metrics = pd.DataFrame(np.around(clusters_metrics, decimals=5), columns=metrics_names)
    cluster_metrics.insert(0, "Cluster", cluster_names)

    return cluster_metrics


def silhouette_analysis(data_tpm, algorithm_name="KMeans", n_comp=2, show_labels=None):
    pca = PCA(n_components=n_comp)
    pca_bu = pca.fit_transform(data_tpm.values)
    X = pca_bu
    min_size, max_size = 3, 14

    print("Variation per principal component: {}".format(pca.explained_variance_ratio_))
    pc_1_var = round(pca.explained_variance_ratio_[0]*100, 2)
    pc_2_var = round(pca.explained_variance_ratio_[1]*100, 2)

    best_score = {
        "silhouette": -1.1,
        "algorithm": "None",
        "n_clusters": 0,
    }
    comb_sil = pd.DataFrame()
    figures = []  # List to store figures
    plt.ioff()
    for n_clusters in range(min_size, max_size):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1.0])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Select algorithm based on the input
        if algorithm_name == "Ward":
            clusterer = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        elif algorithm_name == "Avg":
            clusterer = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
        elif algorithm_name == "GMM":
            clusterer = mixture.GaussianMixture(n_components=n_clusters, covariance_type="diag")
            clusterer.fit(X)
            cluster_labels = clusterer.predict(X)
        elif algorithm_name == "FuzzyCMeans":
            clusterer = FCM(n_clusters=n_clusters, m=1.2)
            clusterer.fit(X)
            cluster_labels = clusterer.predict(X)
        else:
            clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=10, n_init=10)

        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels, metric="cosine")
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        silhouette_df = pd.DataFrame(data=sample_silhouette_values, columns=["Silhouette"])
        silhouette_df["labels"] = cluster_labels
        silhouette_df['K'] = n_clusters
        comb_sil = pd.concat([comb_sil, silhouette_df], axis=0)

        if best_score["silhouette"] < silhouette_avg:
            best_score["silhouette"] = silhouette_avg
            best_score["n_clusters"] = n_clusters
            best_score["algorithm"] = clusterer

        y_lower = 10
        fontsize = 24
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = match_color_pattern_plotly(i) if show_labels else cm.get_cmap("Dark2_r")(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

            label_text = return_label(i) if show_labels else str(i)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, label_text, fontsize=fontsize)

            y_lower = y_upper + 10

        ax1.set_title(f"K-Means K={n_clusters}", fontsize=fontsize)
        ax1.set_xlabel("Silhouette score", fontsize=fontsize)
        ax1.set_ylabel("Cluster label", fontsize=fontsize)

        ax1.tick_params(axis='both', which='major', labelsize=fontsize)
        # Hide tick values for y-axis
        ax1.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        

        # Red line for average silhouette score
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.axvline(x=0.5, color="blue", linestyle="--")

        
        colors = [match_color_pattern_plotly(i) for i in cluster_labels] if show_labels else cm.get_cmap("Dark2_r")(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(pca_bu[:, 0], pca_bu[:, 1], marker=".", lw=0.9, alpha=1.0, c=colors, edgecolor="k", s=400)
        ax2.set_title(f"PCA plot - KMeans K={n_clusters}",  fontsize=fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize)
        ax2.set_xlabel(f"PC_1 ({pc_1_var}%)",  fontsize=fontsize)
        ax2.set_ylabel(f"PC_2 ({pc_2_var}%)",  fontsize=fontsize)

        figures.append(fig)  # Add the figure to the list

    print("\nBest score:", best_score)
    return best_score, comb_sil, figures  # Return the list of figures


def negative_silhouette_samples(data, n_comp=6, n_clusters=5):
    # PCA data
    pca = PCA(n_components=n_comp)
    pca_bu = pca.fit_transform(data.values)
    X = pca_bu

    clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=10, n_init=10, max_iter=1000)
    cluster_labels = order_labels_size(pd.Series(clusterer.fit_predict(X)))

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    dummy = pd.DataFrame(sample_silhouette_values, columns=["Sillhouette"])
    dummy["Sample"] = data.index

    negative_samples = dummy[dummy["Sillhouette"] < 0]["Sample"]

    return negative_samples


##### Plotting #####


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
            " Davies-Bouldin (lower the better)",
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


def mark_best_3(ref, metrics, metric="Silhoute_cosine"):
    # Remove the experiments w/ CS_2 and CS_3
    ascending = False
    if metric == "Davies_bouldin":
        ascending = True

    best_sill = metrics[~metrics["Exp"].str.contains("2|3")].sort_values(by=[metric], ascending=ascending)[:3]

    offset = best_sill[metric].mean() / 50

    annotations = []
    i = 0
    for idx, row in best_sill.iterrows():
        i += 1
        for j in range(1, i + 1):
            annotations.append(
                dict(x=row["Cluster"], y=row[metric] + offset * i * j, text="*", showarrow=False, xref="x{}".format(ref), yref="y{}".format(ref))
            )
    return annotations


def display_metrics(metrics, exp_title="Metrics", show_individual=False, file_path=None):
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
    fig_1.update_layout(xaxis={"categoryorder": "array", "categoryarray": metrics["Cluster"]})
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

    subplots = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[title_1, title_2, title_3],
        shared_xaxes=False,
    )

    for idx, fig_traces in enumerate(all_traces):
        for traces in fig_traces:
            subplots.append_trace(traces, row=1, col=idx + 1)

    # the subplot as shown in the above image
    layout = go.Layout(title_text="{}. Combined metrics ".format(exp_title))
    subplots = subplots.update_layout(layout, annotations=annotations_1[:3] + annotations_1 + annotations_2 + annotations_3)

    if file_path:
        # Create folders if needed
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)

        # layout = go.Layout(
        #     # paper_bgcolor='rgba(0,0,0,0)',
        #     # plot_bgcolor='rgba(0,0,0,0)',
        #     title_text="{}. Combined metrics ".format(exp_title),
        #     xaxis_title="Cluster models",
        #     yaxis_title="Score",
        # )

        # subplots.update_layout(layout)
        # subplots.write_image("{}/{}_20-02.png".format(file_path, exp_title),  width=1920, height=800, scale=3)

        save_fig(name=f"cluster_models/{exp_title}", fig=subplots, base_path=file_path, width=1920, height=700)

    if show_individual:
        fig_1.show()
        fig_2.show()
        fig_3.show()

    return subplots


##### Utilities #####
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


def match_color_pattern_plotly(label):
    pallette = px.colors.qualitative.Plotly
    if label == 1:  # LumP
        return pallette[0]
    elif label == 0:  # LumInf
        return pallette[1]
    elif label == 4:  # Mixed
        return pallette[2]
    elif label == 3:  # Ba/Sq small
        return pallette[4]
    elif label == 2:  # Ba/Sq big
        return pallette[3]
    else:
        return pallette[label]


def return_label(cluster_no):
    if cluster_no == 1:  # LumP
        return "LumP"
    elif cluster_no == 0:  # LumInf
        return "LumInf"
    elif cluster_no == 4:  # Mixed
        return "Mixed"
    elif cluster_no == 3:  # Ba/Sq small
        return "Ba/Sq small"
    elif cluster_no == 2:  # Ba/Sq big
        return "Ba/Sq large"
    else:
        return str(cluster_no)


def add_labels(df, label_name="KMeans_5_labels", gc=38):
    df.loc[df["RawKMeans_CS_5"] == 2, label_name] = "Large Ba/Sq"
    df.loc[df["RawKMeans_CS_5"] == 3, label_name] = "Small Ba/Sq"
    df.loc[df["RawKMeans_CS_5"] == 0, label_name] = "Lum Inf/NS"
    df.loc[df["RawKMeans_CS_5"] == 1, label_name] = "LumP"
    df.loc[df["RawKMeans_CS_5"] == 4, label_name] = "NE"

    if gc == 42:
        df.loc[df["RawKMeans_CS_5"] == 0, label_name] = "LumP"
        df.loc[df["RawKMeans_CS_5"] == 1, label_name] = "Lum Inf/NS"

    return df, label_name


def encode_lund(row):
    if row["Lund2017.subtype"] == "Ba/Sq":
        return 0
    elif row["Lund2017.subtype"] == "Ba/Sq-Inf":
        return 1
    elif row["Lund2017.subtype"] == "UroB":
        return 2
    elif row["Lund2017.subtype"] == "Mes-like":
        return 3
    elif row["Lund2017.subtype"] == "GU-Inf":
        return 4
    elif row["Lund2017.subtype"] == "Sc/NE-like":
        return 5
    else:
        return -1


def encode_infiltrate(row):
    inflamatory_type = row["InflammatoryInfiltrateResponseTSPresent"].lower()
    if inflamatory_type == "absent" or "absent" in inflamatory_type:
        return 0
    elif "minimal" in inflamatory_type:
        return 1
    else:
        return 2


##### Survival #####


def survival_plot(df, df_meta, classifier, selected_labels=None, color_map=None):

    # Pre-processing
    dmy_meta = df_meta.loc[df_meta["Samples"].isin(df["Sample"])][["Samples", "days_to_last_follow_up", "days_to_death"]]
    df = pd.concat([df.set_index("Sample"), dmy_meta.set_index("Samples")], axis=1)

    labels = list(df[classifier].unique())
    if selected_labels:
        labels = selected_labels

    models, all_df, dmy_df = [], [], pd.DataFrame()
    for label in labels:
        kmf = KaplanMeierFitter()

        # process the data
        sel_df = df[df[classifier] == label][["days_to_last_follow_up", "days_to_death"]].replace("--", 0).astype(int)
        sel_df["last_contact"] = sel_df[["days_to_last_follow_up", "days_to_death"]].max(axis=1)
        sel_df["dead"] = np.where(sel_df["days_to_death"] > 0, True, False)

        kmf.fit(sel_df["last_contact"], event_observed=sel_df["dead"], label=[label])
        models.append(kmf)

        # prepare df for plotting and change the scale from days to month
        survival_df = kmf.survival_function_.copy(deep=True)

        disease = [survival_df.columns[0][0]] * survival_df.shape[0]
        survival_df.rename(columns={survival_df.columns[0][0]: "chance"}, inplace=True)
        survival_df.reset_index(inplace=True)
        chance = [value[0] for value in survival_df["chance"].astype(float).values]
        timeline = [value[0] / 30 for value in survival_df["timeline"].astype(int).values]

        survival_df = pd.DataFrame()
        survival_df["disease"] = disease
        survival_df["timeline"] = timeline
        survival_df["chance"] = chance
        all_df.append(survival_df)
        dmy_df = pd.concat([dmy_df, sel_df], axis=0)

    all_df = pd.concat(all_df[:]).reset_index(drop=True)
    all_df["disease"] = all_df["disease"].astype(str)

    # plot the survival
    fig = px.line(all_df, x="timeline", y="chance", color="disease", markers=True, line_shape="hv", color_discrete_map=color_map, color_discrete_sequence=px.colors.qualitative.G10)
    fig.update_yaxes(title_text="Survival rate")
    fig.update_xaxes(title_text="Time (months)", range=[-1, 60])

    # Significance
    dmy_df[classifier] = df[classifier]
    survival_sig(dmy_df, model=classifier)

    return fig


def survival_comp(df, df_meta, classifier_1, classifier_2, selected_labels_1=None, selected_labels_2=None, color_map=None):
    dmy_meta = df_meta.loc[df_meta["Samples"].isin(df["Sample"])][["Samples", "days_to_last_follow_up", "days_to_death"]]
    df = pd.concat([df.set_index("Sample"), dmy_meta.set_index("Samples")], axis=1)

    labels = list(df[classifier_1].unique())
    if selected_labels_1:
        labels = selected_labels_1

    if selected_labels_2:
        labels.extend(selected_labels_2)
    else:
        labels.extend(list(df[classifier_2].unique()))

    labels = list(set(labels))  # eliminate duplicates

    models, all_df, dmy_df = [], [], pd.DataFrame()
    for label in labels:
        kmf = KaplanMeierFitter()
        sel_df = pd.DataFrame()
        # process the data
        if df[df[classifier_1] == label].shape[0]:
            sel_df = df[df[classifier_1] == label][["days_to_last_follow_up", "days_to_death"]]

        if df[df[classifier_2] == label].shape[0]:
            sel_df = df[df[classifier_2] == label][["days_to_last_follow_up", "days_to_death"]]

        sel_df = sel_df.replace("--", 0).astype(int)
        # dmy_1 = df[df[classifier_2] == label][["days_to_last_follow_up", "days_to_death"]].replace("--", 0).astype(int)
        sel_df["last_contact"] = sel_df[["days_to_last_follow_up", "days_to_death"]].max(axis=1)
        sel_df["dead"] = np.where(sel_df["days_to_death"] > 0, True, False)

        kmf.fit(sel_df["last_contact"], event_observed=sel_df["dead"], label=[label])
        models.append(kmf)

        # prepare df for plotting and change the scale from days to month
        survival_df = kmf.survival_function_.copy(deep=True)

        # pre-processing the data
        disease = [survival_df.columns[0][0]] * survival_df.shape[0]
        survival_df.rename(columns={survival_df.columns[0][0]: "chance"}, inplace=True)
        survival_df.reset_index(inplace=True)
        chance = [value[0] for value in survival_df["chance"].astype(float).values]
        # divide by the days in a month
        timeline = [value[0] / 30 for value in survival_df["timeline"].astype(int).values]

        # store results in a dataFrame
        survival_df = pd.DataFrame()
        survival_df["disease"] = disease
        survival_df["timeline"] = timeline
        survival_df["chance"] = chance
        all_df.append(survival_df)
        sel_df['Labels'] = label
        dmy_df = pd.concat([dmy_df, sel_df], axis=0)

    all_df = pd.concat(all_df[:])
    all_df["disease"] = all_df["disease"].astype(str)

    # Ploting
    fig = px.line(all_df, x="timeline", y="chance", color="disease", markers=True, line_shape="hv", color_discrete_map=color_map)
    fig.update_yaxes(title_text="Survival rate")
    fig.update_xaxes(title_text="Time (months)", range=[-1, 60])

    # Significance
    survival_sig(dmy_df, model='Labels')

    return fig

def survival_sig(df: pd.DataFrame, model: str):
    df = df.reset_index().rename(columns={"index": "Sample"}).copy(deep=True)
    classifier = model

    dmy = df[["days_to_last_follow_up", "days_to_death"]].replace("--", 0).astype(int)
    dmy[classifier] = df[classifier]
    dmy["last_contact"] = dmy[["days_to_last_follow_up", "days_to_death"]].max(axis=1).div(30)

    labels = list(df[model].unique())
    dmy = dmy[dmy[classifier].isin(labels)]

    results = multivariate_logrank_test(dmy["last_contact"], dmy[classifier], dmy["days_to_death"])
    results.print_summary()
    print("p={0:.6f}".format(results.p_value))

#### Scores #####


def plot_meta_scores(df, y_axis="IFNG", classification="Basal_type", size="infiltration_score"):

    figs = []

    category_order = {
        "Basal_type": ["IFNG-IFNG (High) - 42", "Basal-IFNG (Medium) - 38", "Basal-Basal (Low) - 40"],
        "infiltration_label": ["High Inf", "Medium Inf", "No Inf"],
    }

    title = "Classification by {}. IFNG vs infiltration score; Colouring by Estimate score".format(classification)
    fig = px.scatter(df, x="ESTIMATE_score", y=y_axis, color=classification, title=title, category_orders=category_order, size=size)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    figs.append(fig)

    title = "Classification by {}. IFNG vs infiltration score; Colouring by Imune rich score".format(classification)
    fig = px.scatter(df, x="Immune_score", y=y_axis, color=classification, title=title, category_orders=category_order, size=size)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    figs.append(fig)

    title = "Classification by {}. IFNG vs infiltration score; Colouring by Stroma rich score".format(classification)
    fig = px.scatter(df, x="Stromal_score", y=y_axis, color=classification, title=title, category_orders=category_order, size=size)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    figs.append(fig)

    return figs


def compute_infg_score(infg_genes, tpms_df):
    total_genes = ["IFNG"] + list(infg_genes["gene"])
    ifng_score_df = np.log2(tpms_df[tpms_df["genes"].isin(total_genes)].rename(columns={"genes": "Sample"}).set_index("Sample").transpose() + 1)

    ifng_score_df["IFNG_score"] = np.sum(ifng_score_df / ifng_score_df.max(), axis=1)
    return ifng_score_df["IFNG_score"] / ifng_score_df["IFNG_score"].max()


#### Gene contribution #####


def get_closest_furthest(cluster_models, outputs, pca, label, labels=None, num=25, verbose=False):
    """
        Function that receives the cluster models and outputs the genes that contributed the most/least for a particular sub-cluster and the centroids

    Args:
        cluster_models (list): A list of clustering models used in the function comp_exp
        outputs (DataFrame): The results from applying the above models
        pca (dictionary): The reduced data ("data") and the PCA model ("pca")
        label (string): The subtype for each to get the genes/
        labels (list, optional): The list of labels. Defaults to None.
        num (int, optional): Number of genes. Defaults to 25.

    Returns:
        DataFrame, DataFrame, DataFrame: top genes, bottom genes and the centroids
    """

    # First Cluster Label - Cluster label in order of the cluster size - Bladder subtype
    #  0-0 - "Lum Inf/NS", 1-4 - "Small Ba/Sq", 2-3 - "Mixed", 3-1 - "Lum P", 4-2 -"Large Ba/Sq";

    _, kmeans = cluster_models[0][1][0]

    model_name = "RawKMeans_CS_5"

    if labels is None:
        labels = ["Lum Inf/NS", "Small Ba/Sq", "Mixed", "LumP", "Large Ba/Sq"]

    # In order to do the fit, we need get the initial dataset used we need to check if we get the same results as with our methods
    clusters_predict = order_labels_size(pd.Series(data=kmeans.predict(pca["data"])))
    # check allignment

    if verbose:
        if clusters_predict.equals(outputs[model_name]):
            print("✅ Data aligned")
        else:
            print("❌ Missmatch between output labels and predicted labels")

    # transform
    k_trans = pd.DataFrame(kmeans.transform(pca["data"]), columns=labels)
    top_samples = pd.concat([outputs[[model_name, "Sample"]], k_trans], axis=1).set_index("Sample")

    top = top_samples.sort_values(by=[label], ascending=True).iloc[:num]
    bottom = top_samples.sort_values(by=[label], ascending=False).iloc[:num]

    centroids = pd.DataFrame(kmeans.cluster_centers_.transpose(), columns=labels, index=["PC_1", "PC_2", "PC_3", "PC_4", "PC_5"])
    centroids = centroids.reindex(sorted(centroids.columns), axis=1)

    return top, bottom, centroids


def get_contribution(samples, working_tpm, selected_genes, most=True, num=100, verbose=False):
    # the same as data_tpm but with the gene
    data_tpm_w_genes = working_tpm[working_tpm["genes"].isin(selected_genes)]
    data_tpm_w_genes.set_index("genes", inplace=True)
    data_tpm_w_genes = data_tpm_w_genes.subtract(data_tpm_w_genes.std(axis=1) / data_tpm_w_genes.median(axis=1), axis=0)

    cmn_g, all_g = {}, {}
    for idx, sample in enumerate(samples):
        # 'not most' is what we want as we want to sort values descending for the genes which contributed the most;
        # otherwise for the genes that contributed the least OR the genes that have low TPM values in the selected sample
        genes = data_tpm_w_genes[sample].sort_values(ascending=not most).iloc[:num].index.values
        if idx:
            cmn_g = set(genes) & cmn_g
            all_g = set(genes) | all_g
        else:
            cmn_g, all_g = set(genes), set(genes)

    if verbose:
        print("There are {} samples and from the top {} genes: ".format(len(samples), num))
        print("{} genes are common across the samples".format(len(cmn_g)))
        print(" = {}".format(list(cmn_g)))
        print("{} genes are unique across the samples".format(len(all_g)))

    return cmn_g, all_g
