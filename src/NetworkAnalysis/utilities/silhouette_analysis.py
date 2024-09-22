#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   silhouette_analysis.py
@Time    :   2023/10/03 11:43:50
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Sillhouette analysis of any clusters
'''


from collections import Counter

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import cluster, mixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score


def silhoutte_analyssis(data, algorithm_name="KMeans", apply_pca=True, n_comp=2, min_k=2, max_k = 10, show_labels=None, show_figs=False):

    # Generating the sample data from make_blobs
    # This particular setting has one distinct cluster and 3 clusters placed close
    # together.

    pca_model = PCA(n_components=n_comp)
    pca = pca_model.fit_transform(data.values)
    if apply_pca:
        X = pca
    else:
        X = data.values
    # min_size, max_size = 3, 10

    print('Variation per principal component {}'.format(
        pca_model.explained_variance_ratio_))

    best_score = {"sillhouette": -1.1, "algorithm": "None", "n_clusters": 0}
    silhouette_stats = []
    k_range = range(min_k, max_k)
    for n_clusters in k_range:

        if algorithm_name == "Ward":
            clusterer = cluster.AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward")
            cluster_labels = clusterer.fit_predict(X)
        elif algorithm_name == "Avg":
            clusterer = cluster.AgglomerativeClustering(
                n_clusters=n_clusters, linkage="average")
            cluster_labels = clusterer.fit_predict(X)
        elif algorithm_name == "GMM":
            clusterer = mixture.GaussianMixture(
                n_components=n_clusters, covariance_type='diag')
            clusterer.fit(X)
            cluster_labels = clusterer.predict(X)
        else:
            clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=10, n_init=10)
            cluster_labels = order_labels_size(
                pd.Series(clusterer.fit_predict(X)))

        # elif algorithm_name is "FuzzyCMeans":
            # fuzzines = (1 + (max_size - n_clusters)/max_size)
            # the fuzziness arg is the one deciding the clusters. So, we make it proportional to the number of cluster
            # clusterer = FCM(n_clusters=n_clusters, m=1.2)
            # clusterer.fit(X)
            # cluster_labels = order_labels_size(pd.Series(clusterer.predict(X)))
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters

        # avg score
        # silhouette_avg = silhouette_score(X, cluster_labels, metric="cosine")
 
        # silhoutte per sample
        sill_samples = silhouette_samples(X, cluster_labels, metric="cosine")
        sill_df = pd.DataFrame(data=sill_samples, columns=["cosine"])
        sill_df["labels"] = cluster_labels 

        # keep the stats
        sill_avg, median, std, variance = sill_df["cosine"].mean(), sill_df["cosine"].median(), sill_df["cosine"].std(), sill_df["cosine"].var()
        silhouette_stats.append([sill_avg, median, std, variance])

        print("For n_clusters = {}. The average, std = {:.4f}, {:.4f} silhouette_score (cosine)".format(n_clusters, sill_avg, std))
   
        if best_score["sillhouette"] < sill_avg:
            best_score["sillhouette"] = sill_avg
            best_score["n_clusters"] = n_clusters
            best_score["algorithm"] = clusterer
            best_score["sillhouete_samples"] = sill_df

        if show_figs: 
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sill_samples[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                # choose the color
                if show_labels:
                    color = match_color_pattern_plotly(i)
                else:
                    color = cm.Dark2(float(i) / n_clusters)
                    
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)

                #display the label on the graphs
                if show_labels:
                    ax1.text(0.0, y_lower + 0.05 * size_cluster_i, return_label(i))
                else:
                    ax1.text(-0.05, y_lower + 0.05 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plots for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=sill_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            if show_labels:
                colors = [match_color_pattern_plotly(i) for i in cluster_labels]
            else:
                colors = cm.Dark2(cluster_labels.astype(float) / n_clusters)
                
            ax2.scatter(pca[:, 0], pca[:, 1], marker='.', lw=0, alpha=1.0,
                        c=colors, edgecolor='k', s=175)
            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for {} clustering on sample data "
                        "with n_clusters = {}".format(algorithm_name, n_clusters)),
                        fontsize=14, fontweight='bold')
            plt.show()

    stats_df = pd.DataFrame(silhouette_stats, columns=["Mean", "Median","Std", "Var"])
    stats_df["K"] = list(k_range)
    print("\n Best score K = {}".format(best_score["n_clusters"]))
    return best_score, stats_df 

################ Helper functions ################

def match_color_pattern_plotly(label):
    pallette = px.colors.qualitative.Plotly
    if label == 1: #LumP
        return pallette[0]
    elif label == 0: #LumInf
        return pallette[1]
    elif label == 4: #Mixed
        return pallette[2]
    elif label == 3: #Ba/Sq small
        return pallette[4]
    elif label == 2: #Ba/Sq big
        return pallette[3]
    else:
        return pallette[label]

def return_label(cluster_no): 
    if cluster_no == 1: #LumP
        return "LumP"
    elif cluster_no == 0: #LumInf
        return "LumInf"
    elif cluster_no == 4: #Mixed
        return "Mixed"
    elif cluster_no == 3: #Ba/Sq small
        return "Ba/Sq small"
    elif cluster_no == 2: #Ba/Sq big
        return "Ba/Sq large"
    else:
        return str(cluster_no)

def order_labels_size(df_series):
    """ This function ensures that the the cluster labels are always in the order of the cluster sizes. This means that the cluster label 0 will correspond for the largest cluster and for n-clusters the n-1 will be the cluster with the lowest members.

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
