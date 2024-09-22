#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   GraphHelper.py
@Time    :   2023/10/18 13:23:29
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   File that contains helper functions to analyse the different networks
"""

import numpy as np
import pandas as pd

# ploting libs=
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from .utilities import clustering as cs
from .utilities import pre_processing as pre
from .utilities import sankey_consensus_plot as sky
from .GraphToolExp import GraphToolExperiment as GtExp
from .utilities.helpers import save_fig
from .ExperimentSet import ExperimentSet
from .NetworkOutput import NetworkOutput


########## Clustering analysis ##########
def plot_individual_metric(metrics_df, pca=True, export=False, offset_db=4, base_path=None):
    metrics = ["Silhoute_cosine", "Calinski_habrasz", "Davies_bouldin"]

    for metric in metrics:
        if pca:
            title = "{}. PCA".format(metric)
        else:
            title = "{}. No PCA".format(metric)

        ascending = False
        if metric == "Davies_bouldin":
            ascending = True
            title += " (lower the better)"
        else:
            title += " (higher the better)"

        fig_1 = px.bar(metrics_df, y=metric, x="Cluster", title="{} {}".format("", title), color="cluster_type")
        fig_1.update_yaxes(title_text="{} Avg".format(metric))
        fig_1.update_xaxes(title_text="Experiment")

        if metric == "Davies_bouldin":
            fig_1.update(layout_yaxis_range=[0, offset_db])

        best_sill = metrics_df[~metrics_df["Exp"].str.contains("2|3")].sort_values(by=[metric], ascending=ascending)[:3]

        offset = best_sill[metric].mean() / 5

        i = 0
        for _, row in best_sill.iterrows():
            i += 1
            fig_1.add_annotation(x=row["Cluster"], y=row[metric] + offset, text="*" * i, showarrow=False)

        if export:
            # save_fig(
            #     fig_1,
            #     title="{}".format(title),
            #     filename="/cluster_models/{}_best3".format(title),
            #     path=base_path,
            #     x_label="Experiment",
            #     y_label="{} Avg".format(metric),
            #     width=1920,
            #     height=800,
            # )
            save_fig(name="{}".format(title), fig=fig_1, width=1920, height=800, base_path=base_path)

        fig_1.show()


def run_clusters(exp: NetworkOutput, label="", show_figs=False, individual_plots=False, ge_exp=None, norm=False):
    if ge_exp is None:
        data = exp.mevsMut
    else:
        data = ge_exp

    if norm:
        data = data / data.max()

    selected_clusters = ["Birch", "RawKMeans", "GaussianMixture", "Ward", "SpectralClustering", "Avg"] 

    # run experiments
    outputs, _, all_metrics, _ = cs.compare_exp(
        data, rob_comp=None, n_clusters=None, selected_clusters=selected_clusters, show_figures=False, show_consensus=True, pca_data=False
    )
    outputs.set_index("Sample", inplace=True)

    fig = {}
    if show_figs:
        # Plot the metrics
        fig = cs.display_metrics(all_metrics, f"Cluster metrics for {exp.type}", show_individual=False, verbose=True)

        if individual_plots:
            plot_individual_metric(all_metrics, pca=False, offset_db=4)

    new_cols = [col + "_" + label for col in outputs.columns[2:]]
    outputs.columns = ["PC_1", "PC_2"] + new_cols

    return outputs, fig, all_metrics


def sill_distrib(metrics, label="Standard", points=None, figures_path=None):
    comb_df = pd.DataFrame()
    for _, row in metrics.iterrows():
        comb_df = pd.concat([comb_df, pd.DataFrame(row["Samples_Silhouete_cos"][1], columns=[row["Cluster"]])], axis=1)
    comb_df = comb_df.melt(var_name="Exp", value_name="Sillhouette (cosine)")
    comb_df = pd.concat([comb_df, (comb_df["Exp"].str.split("_", expand=True)[[0, 2]]).rename(columns={0: "Model", 2: "Cluster Size"})], axis=1)

    fig = px.box(comb_df, x="Cluster Size", y="Sillhouette (cosine)", color="Model", points=points, title=label)
    fig = fig.update_traces(boxmean=True)
    if figures_path is not None:
        fig = fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                xanchor="center",
                y=1.05,
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=16, color="#003366"),
            ),
            font=dict(size=14),
            xaxis=dict(tickfont=dict(size=16)),
            yaxis=dict(tickfont=dict(size=16)),
            title="",
        )
        save_fig(name="{}_sill_spread".format(label), fig=fig, base_path=figures_path, width=1720, height=600)
    return fig, comb_df


def rank_cs_metrics(metrics: pd.DataFrame, label="std", figures_path=None, path=None):
    top_3 = pd.DataFrame()
    scores = ["Silhoute_cosine", "Calinski_habrasz", "Davies_bouldin"]
    for score in scores:
        ascending = False
        if score == "Davies_bouldin":
            ascending = True
        best_score = metrics.loc[~metrics["Exp"].str.contains("2|3")].sort_values(by=[score], ascending=ascending)[:3]
        r_score = score.split("_")[0][:3] + "_" + score.split("_")[1][:3]
        best_score["Score"] = r_score
        sel_df = best_score[["Cluster"]].transpose()
        sel_df.columns = ["{}_{}".format(r_score, idx) for idx in range(1, 4)]
        top_3 = pd.concat([top_3, sel_df], axis=1)

    abreviate_cs = {"Avg": "Avg", "RawKMeans": "KM", "Ward": "Wrd", "Birch": "Brch", "SpectralClustering": "Spec"}
    dmy_df = top_3.loc["Cluster"].str.split("_", expand=True)
    dmy_df[0].replace(abreviate_cs, inplace=True)
    abb_models = dmy_df[0] + "_" + dmy_df[2].astype(str)

    sel_df = pd.concat([top_3.transpose(), pd.Series(abb_models, name=label)], axis=1).drop(columns=["Cluster"])

    if path != None:
        filename = "{}.tsv".format(label)
        sel_df.reset_index(names="Metric_Rank").to_csv(figures_path + filename, sep="\t")

    return sel_df


def prcs_top3_metrics(comb_df: pd.DataFrame, label="top3", figures_path=None):
    # comb_df = pd.concat([top_std, top_rwd, top_pen], axis=1)
    # print(comb_df.columns)

    # melt the data
    tst_df = comb_df.melt(value_name="gen")
    tst_df = pd.concat([tst_df, tst_df["gen"].str.split("_", expand=True).rename(columns={0: "model", 1: "size"})], axis=1)
    tst_df["cat_gen"] = tst_df["gen"].astype("category").cat.codes
    tst_df["cat_model"] = tst_df["model"].astype("category").cat.codes
    tst_df["cat_size"] = tst_df["size"].astype("category").cat.codes

    figs = []
    for col in ["cat_gen", "cat_model", "cat_size"]:
        # update to numeric values
        coord_val = {row["gen"]: row[col] for _, row in tst_df.iterrows()}
        coord_val = comb_df.replace(coord_val)

        # compute the color
        # z_color = coord_val.replace(tst_df[col].value_counts().to_dict()).values
        z_color = []
        for net in tst_df["variable"].unique():
            sel = tst_df.loc[tst_df["variable"] == net]
            to_replace = sel[col].value_counts().to_dict()
            z_color.append(sel[col].replace(to_replace).values)

        z_color = np.array(z_color).T

        # the string value like the model, size etc
        orig_col = col.split("_")[-1]
        z_text = {row[col]: row[orig_col] for idx, row in tst_df.iterrows()}
        z_text = coord_val.replace(z_text)

        fig = px.imshow(coord_val, text_auto=True, aspect="auto", color_continuous_scale="sunset")
        fig = fig.update_traces(z=z_color, text=z_text, texttemplate="%{text}")

        fig = fig.update_layout(
                font=dict(size=18),
                xaxis=dict(tickfont=dict(size=18)),
                yaxis=dict(tickfont=dict(size=18)),
                title="",
        )

        if figures_path != None:
            save_fig(name="top3_cs_{}_{}".format(orig_col, label), fig=fig, base_path=figures_path, width=900, height=500, margin=0.02)

        figs.append(fig)

    if figures_path != None:
        comb_df.to_excel("{}/top3_cs_{}.xlsx".format(figures_path, label))

    return figs


def find_pcs(exp_df: pd.DataFrame):
    """

    The DataFrame needs to be in the form of n_samples x m_features

    Args:
        exp_df (pd.DataFrame): The data to which PCA is applied
    """

    for n_comp in range(2, exp_df.shape[1] - 1):
        pca_model = PCA(n_components=n_comp)
        pca_model.fit_transform(exp_df.values)

        if pca_model.explained_variance_ratio_.sum() >= 0.9:
            break

    print("Sum of 90% variance at PC: {}".format(n_comp))

    ## Number of components that there is a smaller change of 1%
    for n_comp in range(2, exp_df.shape[1] - 1):
        pca_model = PCA(n_components=n_comp)
        pca_model.fit_transform(exp_df.values)

        if np.where(np.diff(-pca_model.explained_variance_ratio_) <= 0.01)[0].size != 0:
            break

    print("Change < 1% at PC: {}".format(n_comp))


########## Analysis of gene representation ##########
def mev_modcon_genes(sel_exp: NetworkOutput, ref_tpms: pd.DataFrame, num_genes=100):
    sort_col = "ModCon_{}".format(sel_exp.type)
    sel_exp.mevsMut, meta = sel_exp.get_mevs(tpms=ref_tpms, modCon=sel_exp.modCons, sort_col=sort_col, num_genes=num_genes, verbose=True)

    data = []
    for key, val in meta.items():
        data.append((key, len(val["matched"]), len(val["not_matched"])))

    data_df = pd.DataFrame(data, columns=["Comm", "Matched", "Not matched"])
    data_df = data_df.melt(id_vars="Comm", value_vars=["Matched", "Not matched"], var_name="Type", value_name="Num genes")
    data_df["Comm"] = data_df["Comm"].astype(str)

    return data_df


def scatter_network_stats(df, metric="degree", height=700, width=1200):
    degree_cols = [col for col in df.columns if metric in col]

    # prepare dataframe for plotting
    # 1. Metl the dataframes
    dmy_df = pd.melt(df[degree_cols], ignore_index=False, value_name=metric, var_name="TF_metric")
    dmy_df["is_tf"] = df["is_tf"]
    # 2. extract the number of TF
    dmy_df["TF_num"] = [int(elem.split("_")[1].split("TF")[0]) for elem in dmy_df["TF_metric"]]
    dmy_df.sort_values(by=["TF_num", "is_tf"], ascending=True, inplace=True)
    dmy_df.reset_index(names="gene", inplace=True)

    fig = px.scatter(
        dmy_df,
        x="gene",
        y=metric,
        color="is_tf",
        # color_discrete_sequence=["red", "blue"],
        category_orders={"is_TF": ["1", "0"]},
        facet_col="TF_num",
        title="Healthy gc42, standard 5K. Metric: {}".format(metric),
        facet_col_wrap=4,
        height=height,
        width=width,
    )
    return fig, dmy_df


def get_network_metrics(exp_set: ExperimentSet, selected_exps=None):
    if selected_exps is None:
        selected_exps = exp_set.get_exp_labels()

    # Select experiments
    all_nodes_df = pd.DataFrame()
    for exp in exp_set.get_exps():
        if exp.type not in selected_exps:
            continue

        # proces the network stats graph
        g_stats = exp.compute_graph_stats()
        tf_type = exp.name.split("_")[-1]
        remap_cols = {col: f"{col}_{tf_type}" for col in g_stats.columns}
        g_stats.rename(columns=remap_cols, inplace=True)

        # proces the nodes_df
        dmy = exp.nodes_df.copy(deep=True)
        # dmy.rename(columns={"Modularity Class": "Leiden"}, inplace=True)
        used_cols = ["TF", "IVI"]
        dmy = dmy[used_cols]
        remap_cols = {col: f"{col}_{tf_type}" for col in used_cols}
        dmy.rename(columns=remap_cols, inplace=True)

        # combine them
        dmy_comb = pd.concat([g_stats, dmy], axis=1)
        # dmy_comb["tf_num"] = tf_type.split("TF")[0]

        all_nodes_df = pd.concat([all_nodes_df, dmy_comb], axis=1)

    # add marker for a gene if is TF
    all_nodes_df = pd.concat([all_nodes_df, exp.nodes_df["TF"].astype(str)], axis=1)
    # all_nodes_df["is_tf"] = exp.nodes_df["TF"].astype(str)
    all_nodes_df = all_nodes_df.rename(columns={"TF": "is_tf"})
    return all_nodes_df


def prep_net_metrics(std_nt: NetworkOutput, rwrd_nt: NetworkOutput, pen_nt: NetworkOutput):
    std_nt.export_to_gephi()
    rwrd_nt.export_to_gephi()
    pen_nt.export_to_gephi()

    std_sts = std_nt.compute_graph_stats()
    rwrd_sts = rwrd_nt.compute_graph_stats()
    pen_sts = pen_nt.compute_graph_stats()

    std_sts["IVI"] = std_nt.nodes_df["IVI"]
    rwrd_sts["IVI"] = rwrd_nt.nodes_df["IVI"]
    pen_sts["IVI"] = pen_nt.nodes_df["IVI"]

    std_sts["Type"] = "Standard"
    rwrd_sts["Type"] = "Reward"
    pen_sts["Type"] = "Penalised"

    combined_df = pd.concat([std_sts, rwrd_sts, pen_sts], axis=0).reset_index(names="gene")

    combined_df["TF"] = combined_df["gene"].apply(lambda gene: std_nt.nodes_df.loc[gene]["TF"])
    combined_df["TF"].fillna(0, inplace=True)
    combined_df["TF"] = combined_df["TF"].astype(str)

    return combined_df


def prep_single_net_metrics(network: NetworkOutput, label="Standard"):
    network.export_to_gephi()

    sts = network.compute_graph_stats()
    sts["IVI"] = network.nodes_df["IVI"]

    sts.reset_index(names="gene", inplace=True)
    sts["TF"] = sts["gene"].apply(lambda gene: network.nodes_df.loc[gene]["TF"])
    sts["TF"].fillna(0, inplace=True)
    sts["TF"] = sts["TF"].astype(str)
    sts["Type"] = label

    return sts


def plot_net_metrics(metrics_df, label, log_y=False, color="Type", figs_path=None, filename="NetMetrics"):
    metrics = ["degree", "pageRank", "closeness", "betwenees", "IVI"]

    figs, titles = [], []
    for metric in metrics:
        figs.append(
            px.box(metrics_df, y=metric, x="Type", color=color, points="all", log_y=log_y, hover_data=["gene", "Type", "TF", metric],
                            # category_orders={'Type': ['Tum Std', 'Non-Tum Std', 'Tum Rwd', 'Non-Tum Rwd']}
                    ) 
        )
        titles.append(metric)

    subplots_config = {
        "num_cols": 2,
        "shared_x": True,
        "shared_y": False,
        "h_spacing": None,
        "v_spacing": 0.05,
        "main_title": "Different network metrics {}".format(label),
        "height": 700,
        "width": None,
        "y_title": None,
        "x_title": None,
        "specs": [[{}, {}], [{}, {}], [{"colspan": 2}, None]],
    }

    fig = helper_multiplots(figs, titles, subplots_config)
    if log_y:
        fig = fig.update_yaxes(type="log")

    if figs_path:
        fig.update_layout(
            title="",
            xaxis=dict(
                tickfont=dict(size=20),
            ),
            yaxis=dict(
                tickfont=dict(size=20),
            ),
            font=dict(size=18),
            template='ggplot2'
            # paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(marker_size=8)
        save_fig(name=filename, fig=fig, base_path=figs_path, width=1400, height=1000, margin=0.02)

    return fig


########## Overview of Sankey/Leiden plots ##########
def plot_sankey_leiden(exps: ExperimentSet, sky_fig: dict, rename_cols: str, tf="6", label=""):
    _, all_leiden = exps.avg_leiden_scores()

    # filtering an pre-processing
    all_leiden = all_leiden.loc[all_leiden["TF"] == tf]
    all_leiden.loc[all_leiden["Modifier"] == "beta", "Modifier"] = "Penalised"

    figs, titles = [], []
    for metric in ["ModuleNum", "ModularityScore"]:
        fig = px.box(all_leiden, x="Modifier", y=metric, color="TF", points="all", category_orders={"Modifier": ['Standard', "Reward", "Penalised"]})
        figs.append(fig)

    figs.append(sky_fig)

    titles.append("Communities Number")
    titles.append("Modularity Score")
    titles.append("")  # None for Sankey

    subplots_config = {
        "num_cols": 2,
        "shared_x": True,
        "shared_y": False,
        "h_spacing": None,
        "v_spacing": 0.1,
        "main_title": "Modularity scores vs community size {}".format(label),
        "height": 700,
        "width": None,
        "y_title": None,
        "x_title": None,
        "specs": [[{}, {}], [{"colspan": 2, "type": "sankey"}, None]],
    }

    fig = helper_multiplots(figs, titles, subplots_config)

    start_x = 0.00
    step = 1 / (len(rename_cols.values()) - 1)
    for idx, col_name in enumerate(rename_cols.values()):
        x_pos = round(start_x + (idx * step), 2)
        fig.add_annotation(x=x_pos, y=0.47, xref="paper", yref="paper", text=col_name, showarrow=False, font=dict(size=20, color="black"))

    return fig


def prep_sankey_leiden(exps: ExperimentSet, vu_output: pd.DataFrame, tf=6, no_genes="4K", no_K=6, chosen_cs_model="RawKMeans"):
    comb_std, _, _ = run_clusters(exps.exps["standard_{}_{}TF".format(no_genes, tf)], label="std_tf{}".format(tf))
    comb_norm3, _, _ = run_clusters(exps.exps["norm3_{}_{}TF".format(no_genes, tf)], label="norm3_tf{}".format(tf), show_figs=False)
    comb_norm3.drop(columns=["PC_1", "PC_2"], inplace=True)
    comb_beta, _, _ = run_clusters(exps.exps["beta_{}_{}TF".format(no_genes, tf)], label="beta_tf{}".format(tf))
    comb_beta.drop(columns=["PC_1", "PC_2"], inplace=True)

    sel_exp = exps.exps["standard_4K_{}TF".format(tf)]
    comb_df = pd.concat([comb_std, comb_norm3, comb_beta, vu_output], axis=1).dropna()

    reorder_cols = [
        "TCGA408_classifier",
        "KMeans_labels_6",
        "{}_CS_{}_std_tf{}".format(chosen_cs_model, no_K, tf),
        "{}_CS_{}_norm3_tf{}".format(chosen_cs_model, no_K, tf),
        "{}_CS_{}_beta_tf{}".format(chosen_cs_model, no_K, tf),
        "2019_consensus_classifier",
    ]

    rename_cols = {
        reorder_cols[0]: "TCGA",
        reorder_cols[1]: "CA + IFNg",
        reorder_cols[2]: "Standard",
        reorder_cols[3]: "Reward",
        reorder_cols[4]: "Penalised",
        reorder_cols[5]: "Consensus",
    }

    _, sky_fig = sky.main(
        df=comb_df.rename(columns=rename_cols),
        reorder_cols=list(rename_cols.values()),
        title="Best for {}. Comp between {} ".format(sel_exp.type, ", ".join(reorder_cols)),
        retMeta=True,
    )

    return exps, sky_fig, rename_cols


def gene_stats(df: pd.DataFrame, tf_list: pd.DataFrame, mut_df: pd.DataFrame, mut_count=1):
    df["mut_count"] = mut_df["count"]
    df.fillna(0, inplace=True)

    # Compute stats
    stats = {}

    stats_idx = ["Mut", "Mut_TF"]  # "TF"]
    data = []
    for stat in stats_idx:
        df_key = f"{stat}_df"
        if stat == "Mut":
            stats[df_key] = df.loc[df["mut_count"] >= mut_count]
            # stats[f"{stat}_prct"] = stats[df_key].shape[0] / df.shape[0] * 100
        # elif stat == "TF":
        #     stats[df_key] = df.loc[df.index.isin(tf_list)]
        elif stat == "Mut_TF":
            included_tf = df[df.index.isin(tf_list)]
            stats[df_key] = included_tf.loc[included_tf["mut_count"] >= mut_count]
            # stats[f"{stat}_prct"] = stats[df_key].shape[0] / included_tf.shape[0] * 100

        stats[f"{stat}_prct"] = stats[df_key].shape[0] / df.shape[0] * 100
        stats[f"{stat}_prct"] = round(stats[f"{stat}_prct"], 2)
        stats[f"{stat}_num"] = stats[df_key].shape[0]
        data.append([stats["{}_num".format(stat)], stats["{}_prct".format(stat)]])

    stats_df = pd.DataFrame(
        data,
        columns=["Num", "Prct"],
        index=stats_idx,
    )

    stats_df["Burden"] = ">={}".format(mut_count)

    return stats, stats_df


def stats_mut_burden(exp_df: pd.DataFrame, tf_list: pd.DateOffset, mut_df: pd.DataFrame, type="a_type"):
    comb_df = pd.DataFrame()
    for mut_count in range(0, 11, 1):
        _, dmy_df = gene_stats(exp_df, tf_list, mut_df, mut_count=mut_count)

        comb_df = pd.concat([comb_df, dmy_df], axis=0)

    comb_df["Type"] = type

    return comb_df


########## Difference in gene selection ##########
def extract_gene_sel(exp: NetworkOutput, ref_ge: pd.DataFrame, num_genes=3000, verbose=True):
    """
    Find the differences in genes between a network and highest std/med variance

    Args:
        exp (PGCNAOutput): The experiments were to perform
        ref_ge (pd.DataFrame): Gene expression reference
        num_genes (int, optional): Number of genes selected for the highest std/median. Defaults to 3000.

    Returns:
        stats (dict): Useful stats
    """
    # get the Cluster analysis genes
    cs_genes = pre.select_genes(ref_ge.reset_index(), no_genes=num_genes, relative_selection=True)

    # generate MEVs for the given experiment
    modCon = exp.get_ModCon()
    if (type(exp) == GtExp) or (hasattr(exp, 'sbm_method')):  # check doesn't work for the first experiments with sbm
        sort_col = "ModCon_{}_gt".format(exp.type)
    else:
        sort_col = "ModCon_{}".format(exp.type)

    # sort_col = "ModCon_{}_gt".format(exp.type)

    exp.mevsMut, meta = exp.get_mevs(tpms=ref_ge, modCon=modCon, sort_col=sort_col, num_genes=100, verbose=True)

    net_sel_genes, net_all_genes = [], []
    for key, val in modCon.items():
        net_sel_genes.extend(meta[key]["modCon_genes"])
        net_all_genes.extend(val.index)

    stats = {
        "diff_all": set(cs_genes) - set(net_all_genes),
        "diff_sel": set(cs_genes) - set(net_sel_genes),
        "cmn_sel": set(cs_genes) & set(net_sel_genes),
        "cmn_all": set(cs_genes) & set(net_all_genes),
        "cs_genes": cs_genes,
        "net_sel_genes": net_sel_genes,
    }

    if verbose:
        print(f"CS vs Network (all). There are {len(stats['diff_all'])} different genes. ")
        print(f"CS vs Network (sel). There are {len(stats['diff_sel'])} different genes. ")
        print(f"CS vs Network (sel). Common genes: {len(stats['cmn_sel'])}")
        print(f"CS vs Network (all). Common genes: {len(stats['cmn_all'])}")
        print(f"Network selected: {len(stats['net_sel_genes'])}")
        print(f"Highest relative/std {len(stats['cs_genes'])}")

    return stats


def prep_net_vs_ca(ge_df: pd.DataFrame, mut_df: pd.DataFrame, gt_genes, cs_genes):
    """
    Function that prepares the given Gene Expression DataFrame to mark the genes included by both Network and Clustering Analysis method

    Args:
        ge_df (pd.DataFrame): All the GE dataframe
        mut_df (pd.DataFrame): The TCGA mut df for the mutation count
        gt_genes (_type_): The genes selected by the network approach
        cs_genes (_type_): The genes selected by the cluster analysis approach

    Returns:
        dmy_df (DataFrame): Procssed dataframe
    """
    dmy_df = ge_df[ge_df.index.isin(list(set(cs_genes + gt_genes)))].copy(deep=True)
    median_raw = dmy_df.median(axis=1)
    dmy_df = np.log2(dmy_df + 1)
    std, median = dmy_df.std(axis=1), dmy_df.median(axis=1)
    dmy_df["rel_var"] = std / median
    dmy_df["std"] = std
    dmy_df["median_log"] = median
    dmy_df["median_raw"] = median_raw

    dmy_df["type"] = "Both"
    dmy_df.loc[dmy_df.index.isin(list(set(cs_genes) - set(gt_genes))), "type"] = "CS"
    dmy_df.loc[dmy_df.index.isin(list(set(gt_genes) - set(cs_genes))), "type"] = "Network"

    dmy_df["mut_count"] = mut_df["count"]
    dmy_df.fillna(0, inplace=True)
    dmy_df["mut_count"] = dmy_df["mut_count"] + 1

    return dmy_df


def plot_net_vs_ca(prep_df: pd.DataFrame, log=False, annotations=False, ann_add=None, ann_rm=None, ann_chg=None):
    """
    Plotting the gene selection between Network and Clustering anlasysis

    Args:
        prep_df (pd.DataFrame): The DataFrame after prep func
        log (bool, optional): Axes in log scale. Defaults to False.
        annotations (bool, optional): Add annotations. Defaults to False.
        ann_add (list, optional): Annotate the given list of genes. Defaults to [].
        ann_rm (list, optional): Remove the annotations for the given list. This is used for declutering. Defaults to [].
        ann_chg (list, optional): Change the annotation offset for a given list of genes. Defaults to [].

    Returns:
        Plotly figure: Ploly Figure
    """
    fig = px.scatter(
        prep_df.reset_index(),
        y="mut_count",
        x="rel_var",
        color="type",
        title="Network vs Clustering Analysis",
        size="median_raw",
        size_max=60,
        height=700,
        hover_data=["genes", "mut_count", "rel_var"],
        log_y=log,
        log_x=log,
        # marginal_x="box",
        # marginal_y="box",
    )

    fig = fig.update_yaxes(title="Mut count")
    fig = fig.update_xaxes(title="Relative variance log2(TPM+1)")

    if annotations:
        marker_genes, num_genes = [], 20

        # prep_df = prep_df.loc[prep_df.index.values != "TP53"]
        marker_genes.extend(prep_df.sort_values(by=["rel_var"], ascending=False).index.values[:num_genes])
        marker_genes.extend(prep_df.sort_values(by=["mut_count"], ascending=False).index.values[:num_genes])

        if len(ann_add) > 0:
            marker_genes.extend(ann_add)
        if len(ann_rm) > 0:
            marker_genes = set(marker_genes) - set(ann_rm)

        counter = 0
        offset = 15
        for idx, row in prep_df.loc[prep_df.index.isin(marker_genes)][["rel_var", "mut_count"]].iterrows():
            yshift = offset
            if counter % 2 == 0:
                yshift = -offset

            if idx in ann_chg:
                yshift = offset
            elif idx == "":
                yshift = -offset

            if log:
                fig.add_annotation(
                    x=np.log10(row["rel_var"]), y=np.log10(row["mut_count"]), text=idx, showarrow=False, valign="bottom", xshift=yshift, yshift=yshift
                )
            else:
                fig.add_annotation(x=row["rel_var"], y=row["mut_count"], text=idx, showarrow=False, valign="bottom", xshift=yshift, yshift=yshift)

    return fig


##### Helper functions - These should be in a seperate file/class #####
def plot_leiden(exps: ExperimentSet, tf="6", label=""):
    # filtering an pre-processing
    _, all_leiden = exps.avg_leiden_scores()

    all_leiden = all_leiden.loc[all_leiden["TF"] == tf]
    all_leiden.loc[all_leiden["Modifier"] == "beta", "Modifier"] = "Penalty"

    figs, titles = [], []
    for metric in ["ModuleNum", "ModularityScore"]:
        fig = px.box(all_leiden, x="Modifier", y=metric, color="TF", points="all")
        figs.append(fig)

    titles.append("Communities Number")
    titles.append("Modularity Score")

    subplots_config = {
        "num_cols": 2,
        "shared_x": True,
        "shared_y": False,
        "h_spacing": None,
        "v_spacing": 0.05,
        "main_title": f"Modularity scores vs community size TF-{tf} {label}",
        "height": 500,
        "width": None,
        "y_title": None,
        "x_title": None,
        "specs": None,
    }

    fig = helper_multiplots(figs, titles, subplots_config)
    return fig


########## General plotting functions ##########


def plot_metrics_distrib(df, metric="degree", exp_label="Experiment", tf_range=None, x_range=None, path=None, filename="MetricsDistrib"):
    # metric = "degree"
    # df = all_nodes_df.copy(deep=True)
    if tf_range is None:
        tf_range = range(3, 15)

    metric_cols = [col for col in df.columns if metric in col]

    # prepare dataframe for plotting
    # 1. Melt the dataframes
    dmy_df = pd.melt(df[metric_cols], ignore_index=False, value_name=metric, var_name="TF_metric")
    dmy_df["is_tf"] = df["is_tf"]
    # 2. extract the number of TF and limit to 15
    dmy_df["TF_num"] = [int(elem.split("_")[1].split("TF")[0]) for elem in dmy_df["TF_metric"]]
    dmy_df.sort_values(by=["TF_num", "is_tf"], ascending=True, inplace=True)
    dmy_df.reset_index(names="gene", inplace=True)
    dmy_df = dmy_df[dmy_df["TF_num"].isin(tf_range)]

    figs, titles = [], []
    for num_tf in dmy_df["TF_num"].unique():
        is_tf = dmy_df.loc[(dmy_df["TF_num"] == num_tf) & (dmy_df["is_tf"] == "1")][metric]
        non_tf = dmy_df.loc[(dmy_df["TF_num"] == num_tf) & (dmy_df["is_tf"] == "0")][metric]
        hist_data = [is_tf, non_tf]
        group_labels = ["is_tf", "non_tf"]

        fig = ff.create_distplot(hist_data, group_labels, show_rug=False)
        # fig = fig.update_layout(xaxis=dict(tickmode="linear", tick0=3, dtick=5))
        if x_range is not None:
            fig = fig.update_layout(xaxis=dict(range=x_range))
        figs.append(fig)
        titles.append("plot_net_metricsTF_{}".format(num_tf))

    subplots_config = {
        "num_cols": 4,
        "shared_x": False,
        "shared_y": True,
        "h_spacing": 0.03,
        "v_spacing": 0.1,
        "main_title": "{} for {}. KDE distribution.".format(metric, exp_label),
        "height": 700,
        "width": None,
        "x_title": metric,
        "y_title": "Probability",
        "specs": None,
    }

    fig = helper_multiplots(figs, titles, subplots_config)

    # zoom in if specified
    if x_range is not None:
        fig = fig.update_xaxes(range=x_range)

    if path:
        save_fig(name=filename, fig=fig, base_path=path, width=1700, height=900)

    return fig


def plot_mut_rep(df: pd.DataFrame, title: str):
    fig = px.bar(
        df.reset_index(names="Gene_Type"),
        x="Burden",
        # y="Prct",
        y='Num',
        color="Type",
        barmode="group",
        facet_row="Gene_Type",
        title=title,
        text_auto=True,
        height=700,
    )

    fig = fig.update_layout(font=dict(size=16))
    return fig


def helper_multiplots(figs, subplot_tiles, config=None):
    if config is None:
        config = {
            "num_cols": 2,
            "shared_x": False,
            "shared_y": True,
            "h_spacing": 0.05,
            "v_spacing": 0.15,
            "main_title": "Subplots",
            "height": 700,
            "width": None,
            "y_title": "Y-axis",
            "x_title": "X-axis",
            "specs": None,
        }

    num_cols = config["num_cols"]
    num_rows = int(np.ceil(len(figs) / num_cols))

    subplot = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_tiles,
        shared_xaxes=config["shared_x"],
        shared_yaxes=config["shared_y"],
        horizontal_spacing=config["h_spacing"],
        vertical_spacing=config["v_spacing"],
        specs=config["specs"],
    )

    idx_row, idx_col = 1, 1
    for i, fig in enumerate(figs):
        for trace in range(len(fig["data"])):
            subplot.add_trace(fig["data"][trace], row=idx_row, col=idx_col)

        if idx_col % num_cols == 0:
            idx_col = 0
            idx_row += 1
        idx_col += 1

    layout = go.Layout(title_text=config["main_title"])

    subplot = subplot.update_layout(layout, height=config["height"], width=config["width"], showlegend=False, title_y=0.9)
    subplot = subplot.update_xaxes(title_text=config["x_title"])
    subplot = subplot.update_yaxes(title_text=config["y_title"])

    # Remove duplicate legend items
    # visited = []
    # for trace in subplot["data"]:
    #     if trace["name"] is None:
    #         continue

    #     trace["name"] = trace["name"].split(",")[0]
    #     if trace["name"] not in visited:
    #         trace["showlegend"] = True
    #         visited.append(trace["name"])
    #     else:
    #         trace["showlegend"] = False

    return subplot


def helper_multiplots_hist(figs, subplot_titles, df: pd.DataFrame, config=None, color_cols=None):

    colors = px.colors.qualitative.G10 + px.colors.qualitative.D3 + px.colors.qualitative.Alphabet

    if config is None:
        config = {
            "num_cols": 3,
            "shared_x": False,
            "shared_y": False,
            "h_spacing": 0.04,
            "v_spacing": 0.12,
            "main_title": "Metadata exploration",
            "height": 1000,
            "width": None,
            "y_title": None,
            "x_title": None,
            "specs": None,
        }

    num_cols = config["num_cols"]
    num_rows = int(np.ceil(len(figs) / num_cols))

    # Assuming `dmy_df` contains the column 'dendrogram_label'
    color_map = {}
    if color_cols != None:
        uniq_values = []
        for col in color_cols:
            uniq_values.extend(df[col].unique())
        uniq_values = list(set(uniq_values))

        # Create a dictionary to map each label to a color
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(uniq_values)}

    subplot = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=config["shared_x"],
        shared_yaxes=config["shared_y"],
        horizontal_spacing=config["h_spacing"],
        vertical_spacing=config["v_spacing"],
        specs=config["specs"],
    )

    idx_row, idx_col = 1, 1
    visited = set()
    for idx, fig in enumerate(figs):
        for trace in fig.data:
            # Modify the trace directly
            trace_color = color_map[trace.name.split(": ")[1]] if ":" in trace.name else color_map[trace.name]

            showlegend = False
            if trace.name not in visited:
                showlegend = True
                visited.add(trace.name)

            modified_trace = go.Histogram(
                x=trace.x,
                name=trace.name,
                legendgroup=f"group_{idx}",  # Assign a unique legend group
                marker=dict(color=trace_color),
                showlegend=showlegend,
                texttemplate="%{y}",  # Display the count of each bar
            )
            subplot.add_trace(modified_trace, row=idx_row, col=idx_col)

        if idx_col == num_cols:
            idx_col = 1
            idx_row += 1
        else:
            idx_col += 1

    subplot.update_layout(height=config["height"], title_text=config["main_title"], showlegend=True)
    return subplot


def update_legend(fig, pos_x=0.95):
    return fig.update_layout(
        height=1200,
        title_y=0.97,
        title_x=0.5,
        legend=dict(
            title="",
            orientation="h",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=pos_x,
            bgcolor="rgba(0,0,0,0)",
        ),
    )


########## Helpers##########


def remap_columns(output_df):
    """
    Remaps the column names of the given DataFrame based on a predefined mapping dictionary.

    Args:
        output_df (pd.DataFrame): The DataFrame whose column names need to be remapped.

    Returns:
        pd.DataFrame: The DataFrame with remapped column names.
    """
    abreviate_cs = {"Avg": "Avg", "RawKMeans": "KM", "Ward": "Wrd", "Birch": "Brch", "SpectralClustering": "Spec", "GaussianMixture": "GMM"}
    remap_cols = {}
    for col in output_df.columns:
        dmy = col.split("_")[0]
        if dmy in abreviate_cs:
            remap_cols[col] = col.replace(dmy, abreviate_cs[dmy])
        else:
            remap_cols[col] = col

    return output_df.rename(columns=remap_cols)


def color_scale(color_scale="Sunset", num_points=100):

    import re  # Regular expression library for extracting RGB values

    # Assuming colors is a list of strings like 'rgb(75, 41, 145)'
    colors = px.colors.sample_colorscale(color_scale, samplepoints=num_points, colortype="rgb")

    # Function to convert an RGB string 'rgb(x, y, z)' to Hex
    def rgb_string_to_hex(rgb_string):
        # Extract the numerical values using regular expression
        numbers = re.findall(r"\d+", rgb_string)
        # Convert the string numbers to integers
        rgb = [int(num) for num in numbers]
        # Convert to hexadecimal
        return "#{0:02x}{1:02x}{2:02x}".format(rgb[0], rgb[1], rgb[2])

    # Convert each RGB string to Hex
    hex_colors = [rgb_string_to_hex(color) for color in colors]

    return hex_colors


def add_stats_to(nodes_df: pd.DataFrame, tpm_df: pd.DataFrame):

    nodes_df["mean"] = tpm_df.loc[nodes_df.index].mean(axis=1).round(4)
    nodes_df["median"] = tpm_df.loc[nodes_df.index].median(axis=1).round(4)
    nodes_df["std"] = tpm_df.loc[nodes_df.index].std(axis=1).round(4)
    nodes_df["var"] = tpm_df.loc[nodes_df.index].var(axis=1).round(4)
    return nodes_df


def export_top_modCon_genes(nodes_df: pd.DataFrame, path: str, top_n=50, metric="median"):
    comm_genes = {}
    ascending = metric == "ModCon_Rank"
    for com in nodes_df["max_b"].unique():
        genes = nodes_df[nodes_df["max_b"] == com].sort_values(by=metric, ascending=ascending).index

        comm_genes[com] = list(genes)[:50]
        # print(f"markers_{com} = {genes}")
        # print(f"\n\n custom_traces.append({{'genes': markers_{com}, 'title': 'Com_{com}'}})")

    comm_genes = pd.DataFrame().from_dict(comm_genes, orient="index").T
    comm_genes.to_csv(f"{path}/Top_{top_n}_{metric}.tsv", sep="\t", index=False)

    return comm_genes

########## Check functions ##########


def check_ctrls(controls: dict):
    ref_tf_ctrls = {}
    for key, ctrl in controls.items():
        same_across_ctrl, ref_tf = True, {}
        for tf, exp in ctrl["exps"].items():
            if len(ref_tf) == 0:
                ret_tf = set(exp.tf_list)
            else:
                if len(set(exp.tf_list) & set(ret_tf)) > 0:
                    print("TF ", tf)
                    same_across_ctrl = False
                    break

        ref_tf_ctrls[key] = ret_tf
        if not same_across_ctrl:
            print("Different TF list across Ctrl {}".format(key))

    diff_across_ctrl = True
    for key_1, val_1 in ref_tf_ctrls.items():
        ref_tf = val_1
        for key_2, val_2 in ref_tf_ctrls.items():
            if key_1 == key_2:
                continue

            # if they have all the genes common; We expect that there will be some random genes shared across the contro
            if len(val_1 & val_2) == len(ret_tf):
                print("Ctrl {} & Ctrl {} are the same".format(key_1, key_2))
                diff_across_ctrl = False
                break

        if not diff_across_ctrl:
            break

    if diff_across_ctrl:
        print("Different TF list across Controls ")
