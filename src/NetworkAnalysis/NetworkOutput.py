#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   PGCNA_output.py
@Time    :   2023/02/21 15:54:50
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Don't be lazy fill this in!
"""

import os

import igraph as ig
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import zscore

from .utilities import clustering as cs
from .utilities import markers as mk


class NetworkOutput:
    name: str
    exp_meta: pd.DataFrame
    type: str
    graph: ig.Graph
    meta_path: str
    pgcna_path: str
    graph_type: str
    leiden_top3: pd.DataFrame
    meta_df: pd.DataFrame

    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame
    tpm_df: pd.DataFrame
    mut_tf: pd.DataFrame

    def __init__(self, graph, exp_meta, mut_df, exps_path, rel_path="", exp_type="PGCNA", **kwargs):
        self.exps_path = exps_path
        self.exp_meta = exp_meta

        # choosing the experiment
        if exp_type == "PGCNA":
            self.init_pgcna(graph, exp_meta, mut_df, rel_path)
        else:
            self.init_inet(graph, exp_meta, mut_df, base_path=kwargs["base_path"])
        self.exp_type = exp_type

    def init_pgcna(self, graph, exp_meta, mut_df, rel_path):
        self.name = exp_meta["expName"]
        exp_comp = self.name.split("_")
        genesKept = int(int(exp_meta["genesKept"]) / 1000)
        self.type = f"{exp_comp[0]}_{genesKept}K"
        self.graph = graph
        self.graph_type = "ig"
        exp_meta["edgesPG"] = int(exp_meta["edgesPG"])
        exp_meta["edgesTF"] = int(exp_meta["edgesTF"])

        # check if the exps diviate from the 'standard' values
        self.type = f"{self.type}_{exp_meta['edgesTF']}TF"
        if exp_meta["edgesPG"] != 3:
            self.type = f"{self.type}_{exp_meta['edgesPG']}EPG"

        if "resolution" in exp_meta.index:
            self.type = f"{self.type}_{exp_meta['resolution']}res"

        # temporary
        if rel_path == "":
            files_used = exp_meta["filesUsed"]
        else:
            files_used = os.path.abspath(os.path.join(rel_path, exp_meta["filesUsed"]))

        # setting the path
        self.meta_path = f"/CORR_MATRIX_GMF0/{files_used.split('/')[-1]}_RetainF{exp_meta['retainF']}_metaInf.txt"
        self.pgcna_path = f"{self.exps_path}/PGCNA/{self.name}/EPG{exp_meta['edgesPG']}/"

        ###### loading dataframes ######
        #  Leiden
        leidenalg_master = pd.read_csv(self.pgcna_path + "/LEIDENALG/moduleInfoBest.txt", sep="\t")
        self.leiden_top3 = leidenalg_master.iloc[:3].copy(deep=True)

        #  get all the tpms used in that run
        all_tpms = pd.read_csv(files_used, sep="\t", index_col="genes")
        #  meta df
        self.meta_df = pd.read_csv(self.pgcna_path + self.meta_path, sep="\t", header=None)

        # processing
        #  rename meta columns
        self.meta_df.columns = ["genes", "q2E", "varWithin", "varAcross"]
        #  only the best run of leiden
        best_mod = str(leidenalg_master.iloc[0]["Mod#"]) + ".csv"
        self.leiden_best = pd.read_csv(self.pgcna_path + "/LEIDENALG/BEST/ClustersTxt/" + best_mod)
        #  only the genes ones used in PGCNA
        self.tpm_df = all_tpms.loc[self.leiden_best["Id"]]
        #  mut_df
        dmy_df = pd.DataFrame(
            data=mut_df.loc[mut_df.index.isin(self.leiden_best["Id"])],
            index=self.leiden_best["Id"],
        )
        dmy_df.fillna(0, inplace=True)
        self.mut_df = dmy_df

    def init_inet(self, graph: ig.Graph, exp_meta: pd.Series, mut_df: pd.DataFrame, base_path: str):
        # Same as for PGCNA parsing
        self.name = exp_meta["expName"]
        genesKept = int(int(exp_meta["genesKept"]) / 1000)
        self.type = f"{exp_meta['modifierType']}_{genesKept}K"
        self.graph = graph
        self.graph_type = "ig"
        exp_meta["edgesPG"] = int(exp_meta["edgesPG"])
        exp_meta["edgesTF"] = int(exp_meta["edgesTF"])
        self.base_path = base_path

        self.type = f"{self.type}_{exp_meta['edgesTF']}TF"
        # TODO do we need this?
        if exp_meta["edgesPG"] != 3:
            self.type = f"{self.type}_{exp_meta['edgesPG']}EPG"

        if ("resolution" in exp_meta.index) and (not np.isnan(exp_meta["resolution"])):
            self.type = f"{self.type}_{exp_meta['resolution']}res"

        if "sbm_method" in exp_meta.index and not pd.isnull(exp_meta["sbm_method"]):
            self.type = f"{self.type}_{exp_meta['sbm_method']}"
            self.sbm_method = exp_meta["sbm_method"]
            self.graph_type = "gt"

        # This is *different* from the PGCNA parsing
        ### Defining the file paths
        tpm_path = os.path.abspath(os.path.join(base_path, exp_meta["input_ge_file"]))
        self.pgcna_path = f"{self.exps_path}/Networks/{self.name}/EPG{exp_meta['edgesPG']}/"
        self.meta_path = f"{self.pgcna_path}/gene_stats.tsv"
        leiden_best_path = f"{self.pgcna_path}/Leiden/Best/"

        ###### loading dataframes ######
        #  Leiden
        leidenalg_master = pd.read_csv(self.pgcna_path + "/Leiden/summary_leiden.tsv", sep="\t")
        self.leiden_top3 = leidenalg_master.iloc[:3].copy(deep=True)

        #  get all the tpms used in that run
        all_tpms = pd.read_csv(tpm_path, sep="\t", index_col="gene")

        #  meta df
        self.meta_df = pd.read_csv(self.meta_path, sep="\t")

        ####### processing
        #  rename meta columns; this is was needed for PGCNA parsing. Kept it for consistency
        self.meta_df.columns = ["genes", "q2E", "varWithin", "varAcross"]
        #  Select the best run
        best_mod = f"{leidenalg_master.iloc[0]['Mod#']}.tsv"
        self.leiden_best = pd.read_csv(f"{leiden_best_path}/leiden_best_{best_mod}", sep="\t", index_col="Gene")

        # align with the PGCNA parsers
        all_tpms.index.names = ["genes"]
        self.meta_df.index.names = ["genes"]

        self.leiden_best.index.names = ["Id"]
        self.leiden_best.reset_index(inplace=True)
        #### TPMs
        # only the genes ones used in PGCNA
        self.tpm_df = all_tpms.loc[self.leiden_best["Id"]]
        #### Mutations
        dmy_df = pd.DataFrame(
            data=mut_df.loc[mut_df.index.isin(self.leiden_best["Id"])],
            index=self.leiden_best["Id"],
        )
        dmy_df.fillna(0, inplace=True)
        self.mut_df = dmy_df

    #### PGCNA scores ####
    def compute_scores(self):
        """

        Gets both MEV and ModCon scores to which the graph stats are added.

        Args:
            exp (dict): information about the experiments

        Returns:
            df, df, list: _description_
        """
        # a dictionary with key - the idx of the community and the value the ModCon

        modCons = self.get_ModCon()
        mevs = self.get_mevs(self.tpm_df, modCons, num_genes=50)

        self.graph_stats = self.compute_graph_stats()
        success = True
        for key in modCons.keys():
            genes = modCons[key].index.values
            modCons[key] = pd.concat([modCons[key], self.mut_df["count"], self.graph_stats], axis=1).dropna()
            if modCons[key].shape[0] != len(genes):
                print(f"❌ Failed data merger for (Mod {key})!")
                success = False

        if not success:
            print(f"❌ There was an error in merging the data for {self.name} ")
            raise ValueError("Problem in merging the data")

        self.modCons = modCons
        self.mevs = mevs

    def get_ModCon(self):
        leiden_best = self.leiden_best
        mut_df = self.mut_df
        meta_df = self.meta_df

        # create a DataFrame with the components for the modCon equation
        modifier = self.type.split("_")[0]
        col = "conn_{}".format(modifier)

        # all_genes = []  # can be used for testing
        modCons = {}
        for mod_class in leiden_best["Modularity Class"].unique():
            genes = leiden_best.loc[leiden_best["Modularity Class"] == mod_class]["Id"].values

            conn_g = []
            for gene in genes:
                # 1st condition, get all the edges that the current gene is the Source and are inside the module
                # 2nd condition, get all the edges that current is the Target and are inside the module
                in_edges = self.edges_df[(self.edges_df["Source"] == gene) | (self.edges_df["Target"] == gene)]
                in_edges = in_edges[in_edges["Source"].isin(genes)]
                in_edges = in_edges[in_edges["Target"].isin(genes)]
                weights_sum = in_edges['Weight'].sum()
                if weights_sum == 0:
                    print(f"Weighted sum = 0 for Com {mod_class} !!")
                conn_g.append([gene, weights_sum])

            conn_df = pd.DataFrame(conn_g, columns=["gene", col]).set_index("gene")
            working_df = pd.concat([conn_df, meta_df.loc[meta_df["genes"].isin(genes)].set_index("genes"), mut_df[mut_df.index.isin(genes)]["count"]], axis=1)

            # 5. Workout the ModCon and save it
            working_df["ModCon_{}".format(self.type)] = (
                (working_df[col] ** 2) * working_df["q2E"] * working_df["varWithin"] * (100 - working_df["varAcross"]) / 100
            )

            # sort by the current exp type
            modCons[mod_class] = working_df.sort_values(by=["ModCon_{}".format(self.type), "varAcross"], ascending=[False, False])

        self.modCons = modCons
        return modCons

    def get_mevs(self, tpms, modCon, sort_col="ModCon", num_genes=25, verbose=False):
        mevs = pd.DataFrame(index=tpms.columns)
        tpms_log = np.log2(tpms + 1)
        info = {}  # used for verbose
        for key, value in modCon.items():
                
            data = value.sort_values(by=sort_col, ascending=False).iloc[:num_genes].copy(deep=True)
            genes = data.index.values
            df = tpms_log[tpms_log.index.isin(genes)].transpose()

            # 1. Per gene, standardize (z-score) the quantile normalized log2 TPM
            for gene in df.columns:
                tst = (df[gene] - df[gene].min()) / (df[gene].max() - df[gene].min())
                # tst = percentileofscore(tst, tst)
                df[gene] = zscore(tst)

            # 2.Sum up the z-scores to give the mev
            mevs[f"Com_{key}"] = df.sum(axis=1)

            # Useful information
            not_found, matched = [], []
            if len(genes) != df.shape[1] and verbose:
                not_found = list(set(genes) - set(df.columns.values))
                matched = list(set(genes) & set(df.columns.values))
                # print("Comm {}. Genes not found {}/{}({:.2f}%)".format(key, not_found, len(genes), not_found/len(genes)*100)
            else:
                matched = list(genes)

            if df.shape[1] == 0:
                print(f"No genes matched in Com {key}")
                continue

            if verbose:
                info[key] = {"modCon_genes": genes, "matched": matched, "not_matched": not_found, "mevs": df.sum(axis=1)}

        return mevs, info

    def get_iMevs(self, h_tpms: pd.DataFrame, tum_tpms: pd.DataFrame, modCon: dict, sort_col="ModCon", num_genes=25, verbose=False, **kwargs):

        # Integrated mevs
        i_mevs = pd.DataFrame(index=tum_tpms.columns) 

        # Mevs for the same dataset
        # t_mevs = pd.DataFrame(index=h_tpms.columns)
        t_log2 = np.log2(tum_tpms + 1)
        h_log2 = np.log2(h_tpms + 1)

        # Check if there are mutations
        mut_offset = 0
        if "mut_df" in kwargs.keys():
            mut_offset = 1
            mut_df = kwargs.pop("mut_df")

            if "offset" in kwargs.keys():
                mut_offset = kwargs.pop('offset')

            print(f"### Running with mut_ofset {mut_offset}")

        info = {}  # used for verbose
        for key, value in modCon.items():
            data = value.sort_values(by=sort_col, ascending=False).iloc[:num_genes].copy(deep=True)
            genes = data.index.values

            # find the genes in the dataset
            df_int = t_log2[t_log2.index.isin(genes)].transpose()

            # Neeed to compute the stats
            df_h = h_log2[h_log2.index.isin(genes)].transpose()

            # Only use the genes common
            cmn_genes = set(df_int.columns) & set(genes)
            diff_genes = set(genes) - set(df_int.columns)

            # 1. Per gene, standardize (z-score) the normalized log2 TPM
            for gene in cmn_genes:
                # Normalised and get the precentiles for both datasets
                h_norm = (df_h[gene] - df_h[gene].min()) / (df_h[gene].max() - df_h[gene].min())
                t_norm = (df_int[gene] - df_int[gene].min()) / (df_int[gene].max() - df_int[gene].min())
                # Stats
                h_mean, h_std = h_norm.mean(), h_norm.std()

                # z-scores
                ## Integrated Mevs
                df_int[gene] = (t_norm - h_mean) / h_std

            ## Adding the representation of the mutations
            if mut_offset != 0:
                sel_df = mut_df.loc[list(cmn_genes)]
                df_int = df_int + sel_df.T * mut_offset

            # 2.Sum up the z-scores to give the mev
            i_mevs[f"Com_{key}"] = df_int.sum(axis=1)

            # Useful information
            not_found, matched = [], []
            if len(genes) != df_int.shape[1] and verbose:
                not_found = list(set(genes) - set(tum_tpms.columns.values))
                matched = list(set(genes) & set(tum_tpms.columns.values))

            if df_h.shape[1] == 0:
                print(f"No genes matched in Com {key}")
                continue

            if verbose:
                info[int(key)] = {
                    "modCon_genes": genes,
                    "matched": matched,
                    "not_matched": not_found,
                    "diff_genes": diff_genes,
                    "cmn_genes": cmn_genes,
                    "h_norm": h_norm,
                    "t_norm": t_norm,
                }

        return i_mevs, info

    ### Connectivity ####
    def get_connectivity(self):
        leiden_best = self.leiden_best

        # all_genes = []  # can be used for testing
        all_conn_mod = dict()
        for mod_class in leiden_best["Modularity Class"].unique():
            genes = leiden_best.loc[leiden_best["Modularity Class"] == mod_class]["Id"].values

            conn_module = []
            for gene in genes:
                in_edges = self.edges_df[
                    ((self.edges_df["Source"] == gene) & (self.edges_df["Target"].isin(genes)))
                    | ((self.edges_df["Target"] == gene) & (self.edges_df["Source"].isin(genes)))
                ]["Target"]
                out_edges = self.edges_df[(self.edges_df["Source"] == gene) | (self.edges_df["Target"] == gene)]["Target"]
                conn_module.append([gene, in_edges.shape[0], out_edges.shape[0], list(in_edges.values), list(out_edges.values)])

            all_conn_mod[mod_class] = conn_module

            all_df = pd.DataFrame()

        # put them in a dataframe
        for key, val in all_conn_mod.items():
            new_df = pd.DataFrame(val, columns=["gene", "in_edges", "out_edges", "in_genes", "out_genes"])
            new_df["comm"] = key
            all_df = pd.concat([all_df, new_df], axis=0)

        self.genes_conn_edges = all_df

        return all_df

    ### Network stats ####
    def ge_comm(self):
        comm_tpms = {}
        median_vals, std_df = pd.DataFrame(index=self.tpm_df.columns), pd.DataFrame(index=self.tpm_df.columns)

        for mod_class in self.leiden_best["Modularity Class"].unique():
            genes = self.leiden_best.loc[self.leiden_best["Modularity Class"] == mod_class]["Id"].values
            comm_tpms[mod_class] = self.tpm_df.loc[self.tpm_df.index.isin(genes)]

            median_vals = pd.concat([median_vals, comm_tpms[mod_class].median().rename("Comm_{}".format(mod_class))], axis=1)
            std_df = pd.concat([std_df, comm_tpms[mod_class].std().rename("Comm_{}".format(mod_class))], axis=1)

        med_df = median_vals.melt(var_name="Comm", value_name="Median", ignore_index=False).reset_index(names=["Sample"])
        std_df = std_df.melt(var_name="Comm", value_name="Std", ignore_index=False).reset_index(names=["Sample"])
        combined_stats = pd.concat([std_df, med_df["Median"]], axis=1)
        return combined_stats

    #### Graph functions ####
    def compute_graph_stats(self):
        graph_stats = pd.DataFrame(index=self.graph.vs[:]["name"])
        graph_stats["betwenees"] = self.graph.betweenness(directed=False, weights="weight")
        graph_stats["closeness"] = self.graph.closeness(weights="weight")
        graph_stats["degree"] = self.graph.degree()
        graph_stats["strength"] = self.graph.strength()
        graph_stats["pageRank"] = self.graph.pagerank(directed=False, weights="weight")
        graph_stats["hubScore"] = self.graph.hub_score()

        return graph_stats

    def mergeSourceTarget(self, col_name):
        """
        Merge the 'Source' and 'Target' columns of the edges dataframe into a new 'Edge' column.
        Set the 'Edge' column as the index of the dataframe.
        Rename the 'Weight' column to the specified 'col_name'.

        Parameters:
            col_name (str): The name to be assigned to the renamed 'Weight' column.

        Returns:
            None
        """
        self.edges_df["Edge"] = self.edges_df["Source"] + "-" + self.edges_df["Target"]
        self.edges_df.set_index("Edge", inplace=True)
        self.edges_df.rename(columns={"Weight": col_name}, inplace=True)

    #### Export functions ####
    def export_to_gephi(self, save=False, sbm_df=None, state_idx=0, hsbm=False, **kwargs):
        # Build the edges and nodes DFs
        filename, path  = self.name, self.exps_path
        edge_list = []
        if self.graph_type == "ig":
            for edge in self.graph.es:
                source_vertex_id = edge.source
                target_vertex_id = edge.target
                source_vertex = self.graph.vs[source_vertex_id]
                target_vertex = self.graph.vs[target_vertex_id]
                weight = edge["weight"]
                edge_list.append((source_vertex["name"], target_vertex["name"], weight))

            nodes_df = pd.DataFrame([node["name"] for node in self.graph.vs], columns=["Id"])
        else:
            for edge in self.graph.edges():
                source_vertex = self.graph.vp.gene[edge.source()]
                target_vertex = self.graph.vp.gene[edge.target()]
                weight = self.graph.ep.weight[(edge.source(), edge.target())]
                edge_list.append((source_vertex, target_vertex, weight))

            nodes_df = pd.DataFrame(list(self.graph.vp.gene), columns=["Id"])

        edge_df = pd.DataFrame(edge_list, columns=["Source", "Target", "Weight"])

        edge_df["Type"] = "undirected"
        edge_df["fromAltName"] = edge_df["Source"]
        edge_df["toAltName"] = edge_df["Target"]

        # Build the Nodes DataFrame
        nodes_df["Label"] = nodes_df["Id"]  # Gephi requirement(?)
        nodes_df["Gene"] = nodes_df["Id"]  # Needed to for the filter script
        nodes_df.set_index("Id", inplace=True)

        nodes_df = pd.concat(
            [
                nodes_df,
                self.leiden_best.set_index("Id")["Modularity Class"],
                self.mut_df["count"],
            ],
            axis=1,
        )

        # TODO: No need for these now but need to update the lists
        nodes_df = mk.addTF(nodes_df, gene_col="Label")
        top_genes = 100
        # if there is a modCon add the ranking
        if (self.graph_type == 'ig' and hasattr(self, "modCons")):
            for modCon, value in self.modCons.items():
                dmy = value.sort_values(by=["ModCon_{}".format(self.type)], ascending=False).reset_index(names="Id").iloc[:top_genes]
                dmy["Rank"] = dmy.index + 1
                dmy.set_index("Id", inplace=True)
                nodes_df.loc[nodes_df["Modularity Class"] == modCon, "ModCon_Rank"] = dmy["Rank"]
                # offset needed to make it easy to use in gephi
                nodes_df["ModCon_Rank"] = nodes_df["ModCon_Rank"].fillna(top_genes + 10)
        elif self.graph_type == 'gt': 
            # Add the communities information
            if "com_df" in kwargs.keys():
                nodes_df = pd.concat([nodes_df, kwargs.pop("com_df")], axis=1)

            # Add the modCon Rank
            if hasattr(self, "gt_modCon"):
                for modCon, value in self.gt_modCon.items():
                    dmy = value.sort_values(by=["ModCon_{}_gt".format(self.type)], ascending=False).reset_index(names="Id").iloc[:top_genes]
                    dmy["Rank"] = dmy.index + 1
                    dmy.set_index("Id", inplace=True)
                    nodes_df.loc[nodes_df["max_b"] == modCon, "ModCon_Rank"] = dmy["Rank"]
                    # offset needed to make it easy to use in gephi
                    nodes_df["ModCon_Rank"] = nodes_df["ModCon_Rank"].fillna(top_genes + 10)

        ### check if there is an IVI file
        ivi_path = f"{path}/PGCNA/{self.name}/IVI.tsv"
        if self.exp_type == "iNet":
            ivi_path = f"{path}/Networks/{self.name}/IVI.tsv"
        if os.path.exists(ivi_path):
            ivi_df = pd.read_csv(ivi_path, sep="\t")
            ivi_df.index.names = ["Id"]
            if self.exp_type == "iNet":
                nodes_df = pd.concat([nodes_df, ivi_df.reset_index().set_index("gene")[["IVI"]]], axis=1)
            else: 
                nodes_df = pd.concat([nodes_df, ivi_df], axis=1)

        if sbm_df is not None:
            if hsbm:
                # nodes_df = pd.concat([nodes_df, sbm_df], axis=1)
                # the concatenation gives some error in gephi
                for col in sbm_df.columns:
                    nodes_df[col] = sbm_df[col]
            else:
                nodes_df[f"SBM_{state_idx}"] = sbm_df["max_b"]
                nodes_df[f"SMB_stable_{state_idx}_num"] = sbm_df["stable_gene_num"]

        tf_ctrl_path = "../../data/tf_ctrl.csv"
        if os.path.exists(tf_ctrl_path):
            genes = pd.read_csv(tf_ctrl_path).index
            nodes_df["ctrl_tf"] = 0
            nodes_df.loc[nodes_df.index.isin(genes), "ctrl_tf"] = 1

        if save:
            gephi_path = f"{path}/PGCNA/{filename}/"
            if self.exp_type == "iNet":
                gephi_path = f"{path}/Networks/{filename}/"

            nodes_df.index.names=['Id']
            nodes_df.to_csv(f"{gephi_path}/{filename}_nodes.tsv", sep="\t")
            edge_df.to_csv(
                f"{gephi_path}/{filename}_edges.tsv",
                sep="\t",
                index=False,
            )

        self.nodes_df = nodes_df
        self.edges_df = edge_df

    def export_morpheus_mevs(self, vu_output, name, tum=True, col='RawKMeans_CS_5', path=''):
        if hasattr(self, "mevsMut"):
            # Export if for morpheus
            mevs_name = f"mevs_{name}.tsv"

            # Morpheus doesn't like columns of strings
            dmy = None
            if tum:
                transform_cols = ["TCGA408_classifier", "consensus", "Lund2017.subtype"]
                vu_cols = []
                for col in transform_cols:
                    vu_output[col] = vu_output[col].astype("category")
                    vu_output[col + "_num"] = vu_output[col].cat.codes
                    vu_cols.append(col + "_num")

                vu_cols = vu_cols + ["RawKMeans_CS_5"]
                dmy = pd.concat(
                    [
                        vu_output[
                            [
                                "RawKMeans_CS_5",
                                "TCGA408_classifier_num",
                                "consensus_num",
                                "Lund2017.subtype_num",
                            ]
                        ],
                        self.mevsMut,
                    ],
                    axis=1,
                )

                # deal with missing samples between communities and VU_CS
                missing_vu = list(set(self.mevsMut.index) - set(vu_output.index))
                print(f"Missing #{len(missing_vu)} from VU_CS: {missing_vu}\n")
                dmy.loc[missing_vu, vu_cols] = -1

                missing_mevs = list(set(vu_output.index) - set(self.mevsMut.index))
                print(f"Missing #{len(missing_mevs)} from MEVs: {missing_mevs}\n")
                dmy.drop(index=missing_mevs, inplace=True)
            else:
                dmy = self.mevsMut


            save_path = f"{self.exps_path}stats/h_{mevs_name}.tsv"
            if path != '':
                save_path = f"{path}/{mevs_name}.tsv"

            dmy.transpose().to_csv(save_path, sep="\t", index=True)

            return dmy
        else:
            raise ValueError("No MEVS mutation")

    def prep_for_go(self, num_sel=None, to_save=False, save_path=None):
        top_genes = {}
        for mod_con in self.modCons:
            if num_sel:
                sel_genes = list(self.modCons[mod_con].sort_values(by="ModCon_{}".format(self.type), ascending=False).index.values[:num_sel])
            else:
                sel_genes = list(self.modCons[mod_con].sort_values(by="ModCon_{}".format(self.type), ascending=False).index.values[:])

            top_genes[mod_con] = [sel_genes, "\n".join(sel_genes)]
        dmy = pd.DataFrame.from_dict(top_genes, orient="index").reset_index(names="Community").rename(columns={0: "Genes", 1: "GO format"})

        if to_save and save_path:
            folder_path = "{}/GeneOntology/{}/".format(save_path, self.type)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if num_sel:
                path = "{}/{}".format(folder_path, "h_comm_genes_{}.tsv".format(num_sel))
            else:
                path = "{}/{}".format(folder_path, "h_comm_genes.tsv")

            dmy.to_csv(path, index=False, sep="\t")

        return dmy

    @staticmethod
    def plot_individual_metric(metrics_df, exp_name=None, pca=True, offset_db=4):
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

            if exp_name != None:
                title = "{} {}".format(title, exp_name)

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

            fig_1.show()

    def run_clusters(self, label="dmy", showFigs=False, mevsMut=None, clusters_config=None):
        if mevsMut is None:
            # the linter didn't like the nested conditions
            if hasattr(self, "mevsMut"):
                data = self.mevsMut
        else:
            data = mevsMut

        selected_clusters = ["Birch", "RawKMeans", "GaussianMixture", "Ward", "SpectralClustering", "Avg"]
        # selected_clusters = ["RawKMeans", "DBSCAN", "OPTICS", "SpectralClustering", "Avg"]

        # run experiments
        outputs, _, all_metrics, _ = cs.compare_exp(
            data,
            rob_comp=None,
            n_clusters=None,
            selected_clusters=selected_clusters,
            show_figures=False,
            show_consensus=True,
            pca_data=False,
            n_comp=5,
            default_base=clusters_config,
        )
        outputs.set_index("Sample", inplace=True)

        if showFigs:
            # Plot the metrics
            # fig = cs.display_metrics(all_metrics, "Cluster metrics for {}".format(exp.type), show_individual=False)
            # fig.show()
            all_metrics["cluster_type"] = ["-".join(cluster.split("_")[:1]) for cluster in all_metrics["Cluster"]]
            NetworkOutput.plot_individual_metric(all_metrics, pca=False, offset_db=4, exp_name=self.type)

        new_cols = [col + "_" + label for col in outputs.columns[2:]]
        outputs.columns = ["PC_1", "PC_2"] + new_cols

        return outputs


