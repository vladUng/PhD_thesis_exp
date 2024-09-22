#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   main.py
@Time    :   2024/01/03 08:19:02
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   This is the main file of the network pipeline developed to create a network from TPMs and run different community detection algorithms.

The scripts has the following capabilities:
1. Create a co-regulated network based on TPMs
2. Selective edge pruning can configured
3. Run different community detection algorithm
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import plotly.express as px
import math as math
import igraph as ig
import leidenalg as la
import random
from scipy.stats import rankdata
import pickle as pickle
import graph_tool.all as gt


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = execution_time % 60
        print(f"Function {func.__name__} took {minutes}:{seconds:.2f} s to execute")
        return result

    return wrapper


class iNET:
    # Graph construction parameters
    edges_pg: int
    edges_sel: int
    genes_kept: int
    retain_f = 0.5  # 50%

    # Weight modifier parameter
    modifier_type = "standard"

    # Comm detection type
    graph_type = "gt"  # gt or ig - for leiden
    sbm_method = "hsbm"
    sbm_config: dict

    # Leiden algorithm parameter
    mod_type = "mod_max"  # or CPM (Constat Pots Model)
    resolution_parameter = None  # set only when the CPM mode is selected

    save_to_gephi = False

    # general
    g: ig.Graph
    exp_name: str

    # other settings
    gene_sel_type = "rel_std"
    round_decimal = 3

    meta_exsts: bool

    # folders:
    in_f: str
    out_f: str

    # Intermediate folders paths
    inpute_ge_file: str
    sel_ge_file: str
    mut_file: str
    master_strats: str
    output_folder: str
    prcsd_data_folder: str

    def __init__(
        self, exp_name, ge_file:str, input_folder:str, output_folder:str, genes_kept=5000, edges_pg=3, edges_sel=6, modifier_type="standard", mod_type="mod_max", **kwargs
    ) -> None:

        self.exp_name = exp_name
        self.genes_kept = genes_kept
        self.edges_pg = edges_pg
        self.edges_sel = edges_sel
        self.modifier_type = modifier_type
        self.mod_type = mod_type

        if "graph_type" in kwargs.keys():
            self.graph_type = kwargs["graph_type"]

        if "sbm_method" in kwargs.keys():
            self.sbm_method = kwargs["sbm_method"]

        if "sbm_config" in kwargs.keys():
            self.sbm_config = kwargs["sbm_config"]
            self.sbm_method = kwargs["sbm_config"]["method"]
        else:
            self.sbm_config = {"n_iter": 1000, "mc_iter": 10, "deg_cor": True}

        if mod_type == "CPM":
            self.resolution_parameter = kwargs["res_param"]

        self.set_paths(exp_name, in_f=input_folder, ge_file=ge_file, out_f= output_folder)

    def compute_g_matrix(self):
        """
        Computes the graph matrix by loading data, preprocessing TPMs, computing the correlation matrix,
        applying modifiers, reducing edges, and exporting the result to Gephi.

        Returns:
            pruned_df (DataFrame): Pruned correlation matrix.
            corr_df (DataFrame): Correlation matrix.
            ge_df (DataFrame): Preprocessed TPMs.
            sel_ge (DataFrame): Selected gene expression data.
        """

        # still load the data
        ge_df, sel_ge, mut_df = self.load_data(ge_path=self.input_ge_file, sel_ge_path=self.sel_ge_file, mut_path=self.mut_file)

        #
        meta_corr_df, prcsd_meta_df = self.load_prcsd_meta()
        if meta_corr_df is None:

            # Pre-process TPMs
            ge_df = self.filter_data(ge_df, num_genes=self.genes_kept, type=self.gene_sel_type)

            # Compute corr matrix
            corr_df = self.corr_matrix(df=ge_df).round(self.round_decimal)

            # apply modifiers
            corr_df = self.weight_modifier(ge_df=ge_df, corr_df=corr_df, mut_df=mut_df, modifier_type=self.modifier_type)

            # pruned_df = self.reduce_edges(corr_df=corr_df, sel_ge=sel_ge, num_e=self.edges_pg, sel_num_e=self.edges_sel)
            self.save_prcsd_data(corr_df, prcsd_meta_df)
        else:
            corr_df = meta_corr_df

        pruned_df = self.reduce_edges_2(corr_df=corr_df.copy(deep=True), ge_df=ge_df, sel_ge=sel_ge)


        # pruned_df.to_csv(f'{self.master_stats}/pruned_df.tsv', sep="\t", index=True)
        # corr_df.to_csv(f'{self.master_stats}/modified_corr_df.tsv', sep="\t",  index=True)
        # ge_df.to_csv(f'{self.master_stats}/ge_df.tsv', sep="\t",  index=True)

        return pruned_df, corr_df, ge_df, sel_ge

    def set_paths(self, exp_name:str, in_f:str, ge_file:str, out_f:str):
        """
        Set the file paths for input and output files.

        Args:
            exp_name (str): The name of the experiment.
            in_f (str): The input file directory.
            out_f (str): The output file directory.
        """
        self.input_ge_file = f"{in_f}/{ge_file}"
        # self.input_ge_file = f"{in_f}/healthy_data_10k_gc42_v3.tsv"
        self.sel_ge_file = f"{in_f}/TF_names_v_1.01.txt"
        self.mut_file = f"{in_f}/mut_tcgna_pgcna_v1.csv"
        # self.output_root = f"{out_f}/Networks/{exp_name}/"
        self.master_stats = f"{out_f}/Stats/"
        self.output_folder = f"{out_f}/Networks/{exp_name}/EPG{self.edges_pg}/"
        self.prcsd_data_folder = f"{out_f}/Processed/"

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if not os.path.exists(self.master_stats):
            os.makedirs(self.master_stats)

        if not os.path.exists(self.prcsd_data_folder):
            os.makedirs(self.prcsd_data_folder)

    def save_experiment(self, sel_ge: list):
        """
        Save a snapshot of the experiment run.

        Parameters:
        - sel_ge (list): List of selected genes.

        Returns:
        None
        """

        save_dict = {
            "exp_name": self.exp_name,
            "graph_type": self.graph_type,
            "sbm_method": self.sbm_method,
            "modifier_type": self.modifier_type,
            "edges_pg": self.edges_pg,
            "edges_sel": self.edges_sel,
            "genes_kept": self.genes_kept,
            "retain_f": self.retain_f,
            "gene_sel_type": self.gene_sel_type,
            "resolution_parameter": self.resolution_parameter,
            "input_ge_file": self.input_ge_file,
            "input_mut_file": self.mut_file,
        }

        # this holds a history of the experiments run
        masterFile = "stats_master.tsv"
        masterDf = pd.DataFrame()
        masterFilePath = os.path.join(self.master_stats, masterFile)
        if os.path.exists(masterFilePath):
            masterDf = pd.read_csv(masterFilePath, sep="\t", engine="pyarrow")

        masterDf = pd.concat([masterDf, pd.DataFrame.from_dict(save_dict, orient="index").T])
        masterDf.to_csv(masterFilePath, sep="\t", index=False)

        save_dict["graph"] = self.g
        save_dict["tf_list"] = sel_ge
        filename = self.exp_name + ".pickle"
        objPath = os.path.join(self.master_stats, filename)
        with open(objPath, "wb") as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ############ Create graph ############
    @measure_execution_time
    def load_data(self, ge_path: str, sel_ge_path: str, mut_path: str):
        """
        Load data from a file.

        Args:
            ge_path (str): The path to the file.
            sel_ge_path (str): The path to selected genes in edge pruning
            mut_path (str): The path to TCGA mutations

        Returns:
            pandas.DataFrame: The loaded data as a DataFrame.

        Raises:
            FileExistsError: If the file does not exist.
        """
        # check if path exists
        if not os.path.exists(ge_path):
            raise FileExistsError(f"There is no file for TPMs at {ge_path}")

        if not os.path.exists(mut_path):
            raise FileExistsError(f"There is no file for TCGA mutations at {mut_path}")

        sel_ge = None
        if sel_ge_path is not None:
            if not os.path.exists(sel_ge_path):
                raise FileExistsError(f"There is no file for Selected gene at {sel_ge_path}")

            sel_ge = np.genfromtxt(fname=sel_ge_path, delimiter="\t", skip_header=1, dtype="str")

        df = pd.read_csv(ge_path, index_col="gene", sep="\t", engine="pyarrow")
        df_mut = pd.read_csv(mut_path, index_col="gene", engine="pyarrow")
        return df, sel_ge, df_mut

    @measure_execution_time
    def corr_matrix(self, df: pd.DataFrame, method="spearman"):
        """
        Calculate the correlation matrix for a given DataFrame.

        Parameters:
            df (pd.DataFrame): The DataFrame for which the correlation matrix needs to be calculated.
            method (str, optional): The method used to calculate the correlation. Defaults to "spearman".

        Returns:
            pd.DataFrame: The correlation matrix.
        """

        if method == "partial_corr":
            corr_df = df.T.pcorr()
        else:
            corr_df = df.T.corr(method=method)
        return corr_df

    @measure_execution_time
    def reduce_edges(self, corr_df: pd.DataFrame, sel_ge: list, num_e=3, sel_num_e=6):
        """
        Reduce the number of edges in a correlation DataFrame by selecting a subset of genes.

        NOTE This is an attempt to use matrix multiplication which almost works, but it doesn't deal with duplicates, which is common when the values are rounded to 3 decimal places.
        Also the problem with this method is that it splits the DataFrame in two, the ones for TF and the others. I think that affects the number of edges

        Args:
            corr_df (pd.DataFrame): The correlation DataFrame.
            sel_ge (list): The list of genes to be selected.
            num_e (int, optional): The number of edges to keep for non-selected genes. Defaults to 3.
            sel_num_e (int, optional): The number of edges to keep for selected genes. Defaults to 6.

        Returns:
            pd.DataFrame: The pruned correlation DataFrame.
        """

        # For Debugging
        def check_counts(df: pd.DataFrame):
            """
            Calculate the number of non-zero values for each column in the given DataFrame and visualize the distribution.

            Args:
                pruned_df (pd.DataFrame): The DataFrame containing the data to be analyzed.

            Returns:
                None
            """
            counts = []
            for col in df.columns:
                non_zero_vals = (df[col] > 0.0).sum()
                counts.append((col, non_zero_vals))

            test_df = pd.DataFrame(counts, columns=["gene", "counts"])
            test_df["type"] = "normal"
            test_df.loc[test_df["gene"].isin(sel_ge), "type"] = "Selected"

            fig = px.histogram(test_df, x="counts", color="type", barmode="overlay", title="Distribution of number of connections")
            fig.show()

        # helps with debugging, not necessary
        # corr_df = iNET.sort_alphabetically(corr_df)
        rank_df = corr_df.rank(ascending=False)

        found_sel = list(set(corr_df.index) & set(sel_ge))
        genes = list(set(corr_df.index) - set(sel_ge))

        sel_df = rank_df.loc[found_sel].copy(deep=True)
        rest_df = rank_df.loc[genes].copy(deep=True)

        # account for the self-correlation -
        #  -- we add 1.5 to account for the starting from 0 AND
        #  the 0.5 to deal with duplicates
        sel_df = sel_df.where(sel_df <= float(sel_num_e + 1.5), 0)
        rest_df = rest_df.where(rest_df <= float(num_e + 1.5), 0)

        pruned_df = pd.concat([sel_df, rest_df], axis=0)
        # pruned_df = iNET.sort_alphabetically(pruned_df)

        pruned_df = corr_df[pruned_df != 0].fillna(0)

        return pruned_df

    @measure_execution_time
    def weight_modifier(self, ge_df: pd.DataFrame, corr_df: pd.DataFrame, mut_df: pd.DataFrame, modifier_type: str):
        """
        Modifies the weights of a correlation DataFrame based on a given modifier type.

        Args:
            corr_df (pd.DataFrame): The correlation DataFrame.
            mut_df (pd.DataFrame): The mutation DataFrame.
            modifier_type (str): The type of modifier to be applied.

        Returns:
            pd.DataFrame: The modified correlation DataFrame.
        """

        def rescaled(series: pd.Series, new_max: int):
            """Rescales a given series to a new range.

            Args:
                series (pd.Series): The series to be rescaled.
                new_max (int): The maximum value of the new range.

            Returns:
                pd.Series: The rescaled series.
            """
            new_min = -new_max
            x_norm = (series - series.min()) / (series.max() - series.min())
            x_scaled = x_norm * (new_max - new_min) + new_min

            return x_scaled

        def sigmoid_func(x, x0, offset=1) -> pd.DataFrame:
            """
            Sigmoid function

            Args:
                x (int/float): The variable
                x0 (center): Where the sigmoid is centered
                offset (int, optional): The elongation on y-axis. Defaults to 1.

                By default the sigmoid function starts goes from 1 to 2. The offset elongates on the y axis

            Returns:
                _type_: _description_
            """
            return (1 + math.exp(-(x - x0))) ** -1 * offset + 1

        if modifier_type == "standard":
            return corr_df

        # the first 2 lines ensures that all the mut_counts are found for the used genes
        ge_df["mut_count"] = mut_df['count']
        ge_df["mut_count"] = ge_df["mut_count"].fillna(0)
        
        mut = ge_df["mut_count"]
        mut_log = np.log2(mut + 1)
        max_log = mut_log.max()

        modifier = None
        if modifier_type == "norm3":
            modifier = (max_log + mut_log) / max_log
        elif modifier_type == "beta":
            modifier = (max_log - mut_log) / max_log
        elif modifier_type == "sigmoid":
            # 12, -12 was found through experimentations
            x0, offset, new_max = -8, 1, 12
            modifier = rescaled(mut, new_max=new_max)
            modifier = modifier.apply(sigmoid_func, args=(x0, offset))
        else:
            raise ValueError(f"Invalid modifier type: {modifier_type}")

        modifier.name = modifier_type
        modifier.replace(0, 1, inplace=True)

        # some adjustment
        corr_df = corr_df * modifier

        ge_df.drop(columns=['mut_count'], axis=1, inplace=True)

        return corr_df

    @measure_execution_time
    def create_graph(self, df: pd.DataFrame) -> ig.Graph:
        """
        Create a graph based on the given DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the graph data.

        Returns:
        g (ig.Graph): The created graph.
        edge_list (list of tuples): Needed to create graph tool
        """

        # remove self-loop
        dmy = df.replace(1, 0).values
        g = ig.Graph.Adjacency((dmy > 0).tolist(), mode="max")
        g.vs["name"] = df.index

        edges_list = []
        for edge in g.es:
            source_id, target_id = edge.tuple
            # dealing with 0
            source_target = df.loc[g.vs[source_id]["name"], g.vs[target_id]["name"]]
            target_source = df.loc[g.vs[target_id]["name"], g.vs[source_id]["name"]]
            weight = max(target_source, source_target)
            edge["weight"] = weight
            edges_list.append((g.vs[source_id]["name"], g.vs[target_id]["name"], weight))

        self.g = g
        return g, edges_list

    def create_gt(self, edges):
        """
        Creates the Graph Tool Graph based on the list of edges generated with PGCNA
        """
        edges_df = pd.DataFrame(edges, columns=["Source", "Target", "Weight"])

        list_of_edges = list(edges_df[["Source", "Target", "Weight"]].itertuples(index=False, name=None))
        gt_g = gt.Graph(directed=False)
        vertex_prop = gt_g.add_edge_list(edge_list=list_of_edges, hashed=True, eprops=[("weight", "double")])
        gt_g.vp.gene = vertex_prop

        self.gt_g = gt_g

        return gt_g

    @measure_execution_time
    def create_graph_from_gephi(self, gephi_path):
        # The role of this function is to create a graph from the Gephi file and then compared with the create_graph function.
        print("Work in progress")

    @measure_execution_time
    def export_to_gephi(self, pruned_df: pd.DataFrame, ge_df: pd.DataFrame, sel_ge: np.array):
        """
        Export the data to Gephi format.

        Args:
            pruned_df (pd.DataFrame): The pruned DataFrame containing the gene data.
            ge_df (pd.DataFrame): The DataFrame containing the gene expression data.
            sel_ge (np.array): The array of selected genes.

        Returns:
            None
        """

        iNET.print_msg("Exporting to Gephi")

        # create the nodes file
        nodes_df = pd.DataFrame(index=ge_df.index)
        nodes_df["Label"] = nodes_df.index
        nodes_df["TF"] = 0
        nodes_df.loc[nodes_df.index.isin(sel_ge), "TF"] = 1
        iNET.save_df(nodes_df, path=f"{self.output_folder}/Gephi/", filename="nodes.tsv")
        iNET.print_msg("Exported Nodes df")

        history = set()
        edges = []
        # create the edge file
        for gene_src, row in pruned_df.iterrows():
            # Select non-zero and different than itself
            sel_edges = row[(row != 0.0) & (row != 1.0)]

            new_items = []
            for gene_trg, weight in sel_edges.items():
                link_1 = gene_src + gene_trg
                link_2 = gene_trg + gene_src
                if (link_1 in history) or (link_2 in history):
                    continue

                edges.append((gene_src, weight, gene_trg))
                new_items.extend([link_1, link_2])

                history = history | set(new_items)

        edges = pd.DataFrame(edges, columns=["Source", "Weight", "Target"])
        edges["Type"] = "undirected"
        # do we need this?
        edges["fromAltName"] = edges["Source"]
        edges["toAltName"] = edges["Target"]

        iNET.save_df(edges, path=f"{self.output_folder}/Gephi/", filename="edges.tsv")
        iNET.print_msg("Exported Nodes df")
        iNET.print_msg("Gephi export completed")

    @measure_execution_time
    def compute_stats(self, ge_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics for gene expression data from:

            -- quartile to expression (from percentiles)
            -- median (Q2) varWithin
            -- varAcross -  Quartile Coefficient of Dispersion (from percentiles)

        These values are used to later compute ModuleConnection and ModuleEvaluation.


        NOTE: This is an adaptation of _generateGeneMeta_ function from PGCNA code (Care et al.). It returns the same values
        Args:
            ge_df (pd.DataFrame): DataFrame containing gene expression data.

        Returns:
            pd.DataFrame: DataFrame containing computed statistics for each gene.
        """

        def listToPercentiles(x: np.array) -> np.array:
            data = rankdata(x, "average") / float(len(x))
            return data * 100

        def expToPercentiles(expToPerc: dict, expA: np.array) -> list:
            return [expToPerc[e] for e in expA]

        def dataSpread(x):
            """
            Returns the min, Q1 (25%), median (Q2), Q3 (75%), max, IQR, Quartile Coefficient of Dispersion and IQR/Median (CV like)
            """

            q1 = float(np.percentile(x, 25, method="lower"))
            q2 = np.percentile(x, 50)
            q3 = float(np.percentile(x, 75, method="higher"))

            if (q2 == 0) or ((q3 + q1) == 0):
                return min(x), q1, q2, q3, max(x), abs(q3 - q1), 0, 0
            else:
                return min(x), q1, q2, q3, max(x), abs(q3 - q1), abs((q3 - q1) / (q3 + q1)), abs((q3 - q1) / q2)

        gene_stats = {}
        # Flatten the values to have all the expression
        expressionVals = ge_df.values.flatten()

        # Convert expression to percentiles
        expPercentiles = listToPercentiles(expressionVals)

        #  Calculate mapping from expression to percentile
        expToPerc = {}
        for i, e in enumerate(expressionVals):
            expToPerc[e] = expPercentiles[i]

        for i, gene in enumerate(ge_df.index):
            #  Convert from expression values to percentiles, so can work out (mean absolute deviation) MAD of percentiles
            genePercentiles = expToPercentiles(expToPerc, ge_df.values[i])

            try:
                minE, q1E, q2E, q3E, maxE, iqrE, qcodE, iqrME = dataSpread(ge_df.values[i])
            except:
                print("Issue with :", gene, ge_df.values[i])
                sys.exit()

            medianPercentiles = np.median(genePercentiles)
            # quartile to expression, median (Q2) varWithin, varAcross -  Quartile Coefficient of Dispersion
            gene_stats[gene] = [q2E, qcodE, medianPercentiles]

        gene_stats = pd.DataFrame.from_dict(gene_stats, orient="index", columns=["q2E", "qcodE", "median"])

        iNET.save_df(gene_stats, f"{self.output_folder}/", filename="gene_stats.tsv")
        return gene_stats

    ############ Community detection algorithms ############
    @measure_execution_time
    def run_leiden(self, g: ig.Graph, mod_type="mod_max", runs=100, res_param=None):
        """
        Runs the Leiden algorithm on a given graph.

        Args:
            g (ig.Graph): The input graph.
            mod_type (str, optional): The modularity type. Defaults to "mod_max".
            num (int, optional): The number of runs. Defaults to 100.
            res_param (float, optional): The resolution parameter. Defaults to None.

        Returns:
            pd.DataFrame, {}: A DataFrame containing the results of the Leiden algorithm and a dictionary containing the top 10 partitions

        """
        iNET.print_msg(f"Start Leiden with mod_type = {mod_type}; runs = {runs}; res_param={res_param}")
        print_count = int(runs / 10)

        graph_weights = g.es["weight"]
        scores, partitions = [], {}
        for run in range(1, runs + 1):
            seed = random.randint(0, 1e9)  # Generate random seed as leidenalg doesn't appear to do this (even though it says it does)

            if mod_type == "mod_max":
                partition = la.find_partition(
                    g, la.ModularityVertexPartition, weights=graph_weights, n_iterations=-1, seed=seed
                )  # n_iterations=-1 to converge on best local result
            else:
                partition = la.find_partition(g, la.CPMVertexPartition, weights=graph_weights, n_iterations=-1, seed=seed, resolution_parameter=res_param)

            part_sizes = partition.sizes()
            partitions[run] = partition
            scores.append(
                (
                    partition.modularity,
                    sum(part_sizes) / len(part_sizes),
                    len(part_sizes),
                    part_sizes,
                )
            )

            if run % print_count == 0:
                print(f"### Run {run}. ModularityScore={scores[-1][0]}; AvgModSize={scores[-1][1]}; ModuleNum={scores[-1][2]}")

        # Process results
        scores_df = pd.DataFrame(scores, columns=["ModularityScore", "AvgModSize", "ModuleNum", "ModuleSizes"])
        scores_df = scores_df.sort_values(by="ModularityScore", ascending=False).reset_index(drop=False, names="Run").reset_index(names="Mod#")
        scores_df["Mod#"] = scores_df["Mod#"] + 1

        # save top 10
        best_leidens = {}
        for _, row in scores_df.iterrows():
            sel_part = partitions[row["Run"] + 1]
            names = sel_part.graph.vs["name"]
            members = sel_part.membership

            partition_df = pd.DataFrame(zip(names, members), columns=["Gene", "Modularity Class"])

            iNET.save_df(partition_df.set_index("Gene"), f"{self.output_folder}Leiden/Best/", filename=f"leiden_best_{row['Mod#']}.tsv")

            best_leidens[row["Mod#"]] = sel_part
            if row["Mod#"] == 10:
                break

        iNET.save_df(scores_df, f"{self.output_folder}Leiden/", filename="summary_leiden.tsv")
        iNET.print_msg(f"Finished Leiden")

        return scores_df, best_leidens

    @measure_execution_time
    def run_sbm(self, gt_g: gt.Graph, n_iter=10000, mc_iter=10, deg_cor=True, distrib="real-exponential", verbose=True):

        if not hasattr(self, "states"):
            self.states = []
    
        state = gt.BlockState(gt_g, recs=[gt_g.ep.weight], rec_types=[distrib], deg_corr=deg_cor)

        print(f"### Equilibrate MCMC for {self.exp_name}")
        gt.mcmc_equilibrate(state, wait=n_iter / 10, mcmc_args=dict(niter=mc_iter), verbose=verbose)

        bs = []  # collect some partitions
        h = np.zeros(gt_g.num_vertices() + 1)  # to find the probabilities for a certain size
        dls = []  # description length history

        def collect_partitions(s):
            bs.append(s.b.a.copy())
            B = s.get_nonempty_B()
            h[B] += 1
            dls.append(s.entropy())

        print(f"### Collecton partitions for {self.exp_name}")

        # Now we collect partitions for exactly 1000 sweeps, at intervals
        # of 10 sweeps:
        # n_iter = 10000
        eq_res = gt.mcmc_equilibrate(state, force_niter=n_iter, mcmc_args=dict(niter=mc_iter), callback=collect_partitions, verbose=verbose)

        print(f"### Marginal property partitions for {self.exp_name}")
        # Disambiguate partitions and obtain marginals
        pmode = gt.PartitionModeState(bs, converge=True, relabel=True)
        print(f"### get all marginal for {self.exp_name}")
        pv = pmode.get_marginal(gt_g)

        new_state = {
            "name": f"state_{len(self.states) + 1}",
            "partition_meta": {"h": h, "n_iter": n_iter, "bs": bs},
            "dls": dls,
            "eq_res": eq_res,
            "pmode": pmode,
            "state": state,
            "pv": pv,
        }

        return new_state

    def run_hsbm(self, gt_g: gt.Graph, n_iter=10000, mc_iter=10, deg_cor=False):
        state = gt.NestedBlockState(gt_g, recs=[gt_g.ep.weight], rec_types=["real-exponential"], state_args=dict(deg_corr=deg_cor))

        print(f"### 1. Equilibrate MCMC. n_iter {n_iter / 10}")
        # We will first equilibrate the Markov chain
        gt.mcmc_equilibrate(state, wait=n_iter / 10, mcmc_args=dict(niter=mc_iter), verbose=True)  # 1000 / 10 standard

        bs = []  # collect some partitions
        h = [np.zeros(gt_g.num_vertices() + 1) for s in state.get_levels()]
        dls = [np.zeros(gt_g.num_vertices() + 1) for s in state.get_levels()]

        # to find the probabilities for a certain size
        def collect_partitions(s):
            for l, sl in enumerate(s.get_levels()):
                B = sl.get_nonempty_B()
                h[l][B] += 1
            dls.append(s.entropy())
            bs.append(s.get_bs())

        print(f"### 2. Collecton partitions. n_iter {n_iter}")
        eq_res = gt.mcmc_equilibrate(state, force_niter=n_iter, mcmc_args=dict(niter=mc_iter), callback=collect_partitions, verbose=True)  # 10000 / 10

        print("### 3. Marginal property partitions")
        # Disambiguate partitions and obtain marginals
        # pmode = gt.PartitionModeState(bs, nested=True, converge=False)
        # print("### 4. Collecting the data")
        # pv = pmode.get_marginal(gt_g)

        # Get consensus estimate
        # bs = pmode.get_max_nested()
        # state = state.copy(bs=bs)

        h_state_obj = {
            "name": "hstate",
            "partition_meta": {"h": h, "n_iter": n_iter, "bs": bs},
            "state": state,
            # "pv": pv,
            "dls": dls,
            "eq_res": eq_res,
        }

        return h_state_obj

    def dissambigue_hsbm_part(self, h_state_obj: dict):

        print("### 3. Marginal property partitions")
        # Disambiguate partitions and obtain marginals
        pmode = gt.PartitionModeState(h_state_obj["partition_meta"]["bs"], nested=True, converge=False)
        print("### 4. Collecting the data")
        pv = pmode.get_marginal(self.gt_g)

        # Get consensus estimate
        bs = pmode.get_max_nested()
        state = h_state_obj["state"].copy(bs=bs)

        h_state_obj["partition_meta"]["bs"] = bs
        h_state_obj["pmode"] = pmode
        h_state_obj["state"] = state
        h_state_obj["pv"] = pv

        return h_state_obj

    ############ Utilities ############
    def sort_alphabetically(df):
        df.sort_index(inplace=True)
        df = df[df.columns.sort_values()]
        return df

    @measure_execution_time
    def filter_data(self, ge_df: pd.DataFrame, num_genes=5000, type="rel_std"):
        """
        Filter the gene expression data based on a specified criterion.

        The standard is rel_std, but it also supports PGCNA style which is sorting by the median

        Parameters:
        ge_df (pd.DataFrame): The gene expression data.
        num_genes (int): The number of top genes to select.
        type (str): The type of criterion to use for filtering.

        Returns:
        pd.DataFrame: The filtered gene expression data.
        """
        if type == "rel_std":
            # so it doesn't divide by 0
            std_med = ge_df.std(axis=1) / ge_df.median(axis=1) # this is how it was run the sbm non_tum v3 (chapter 3 thesis)
            # std_med = ge_df.std(axis=1) / ge_df.mean(axis=1)

            top_genes = std_med.sort_values(ascending=False).index.values[:num_genes]
        elif type == "pgcna":
            std = ge_df.std(axis=1)
            top_genes = std.sort_values(ascending=False).index.values[:num_genes]
        else:
            top_genes = ge_df.index

        self.retain_f = round(len(top_genes) / ge_df.shape[0], 2)
        return ge_df.loc[ge_df.index.isin(top_genes)]

    @staticmethod
    def save_df(df: pd.DataFrame, path: str, filename: str):
        full_path = os.path.join(path, filename)

        # create directories if needed
        if not os.path.exists(path):
            os.makedirs(path)

        df.to_csv(full_path, sep="\t", index=True)

    @staticmethod
    def print_msg(message: str):
        print(f"####### {message} #######")

    def save_obj(obj: any, path: str, filename: str):
        full_path = os.path.join(path, filename)

        # create directories if needed
        if not os.path.exists(path):
            os.makedirs(path)

        with open(full_path, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ############ Intermediary files ############
    def load_prcsd_meta(self):

        master_path = f"{self.prcsd_data_folder}/master.tsv"
        if os.path.exists(master_path):

            master_path = f"{self.prcsd_data_folder}/master.tsv"

            # dropping the empty index from the load
            prcsd_meta_df = pd.read_csv(master_path, sep="\t", engine="pyarrow").drop(columns=[""])

            sel_exp = prcsd_meta_df[prcsd_meta_df["num_genes"] == str(self.genes_kept)]
            if sel_exp.shape[0] == 0:
                return None, prcsd_meta_df

            corr_df = pd.read_csv(sel_exp["path"].values[0], sep="\t", engine="pyarrow", index_col="gene")
            return corr_df, prcsd_meta_df

        return None, pd.DataFrame()

    def save_prcsd_data(self, weighted_df: pd.DataFrame, prcsd_meta_df: pd.DataFrame):

        if prcsd_meta_df is None:
            prcsd_meta_df = pd.DataFrame()

        corr_filename = f"corr_{self.modifier_type}.tsv"
        path_to_modifier = f"{self.prcsd_data_folder}/genes_{self.genes_kept}/"

        exp_meta = {
            "name": self.exp_name,
            "num_genes": self.genes_kept,
            "modifier": self.modifier_type,
            "edges_pg": self.edges_pg,
            "edges_sel": self.edges_sel,
            "path": f"{path_to_modifier}/{corr_filename}",
        }
        dmy_meta = pd.DataFrame().from_dict(exp_meta, orient="index").T
        if prcsd_meta_df.shape[0] == 0:
            prcsd_meta_df = dmy_meta
        else:
            prcsd_meta_df = pd.concat([prcsd_meta_df, dmy_meta], axis=0)

        iNET.save_df(weighted_df, path=path_to_modifier, filename=corr_filename)
        iNET.save_df(prcsd_meta_df.reset_index(drop=True), path=self.prcsd_data_folder, filename="master.tsv")

    def prcsd_meta_exists(self) -> bool:
        """
        Check if the processed data metadata file exists.

        Returns:
            bool: True if the metadata file exists, False otherwise.
        """
        master_path = f"{self.prcsd_data_folder}/master.tsv"
        if os.path.exists(master_path):
            prcsd_meta_df = pd.DataFrame(master_path, sep="\t", engine="pyarrow")
            if prcsd_meta_df[prcsd_meta_df["name"] == self.exp_name].shape[0] != 0:
                print("Meta info exists!")
                return True
        else:
            return False

    ##### Testing
    @measure_execution_time
    @staticmethod
    def test_unique_edges(edges: pd.DataFrame):
        for gene in edges["Source"].unique():
            src_df = set(edges[edges["Source"] == gene]["Target"])
            trg_df = set(edges[edges["Target"] == gene]["Source"])

            if src_df & trg_df:
                print(f"Duplicated links for {gene}")
                print(f"Source {src_df}; Target {trg_df}")

    @staticmethod
    def make_zero(row, num_elems=3):
        # if row.name not in ["AHR", "AKNA", "ZXDB"]:
        #     return row
        dmy_df = row.sort_values(ascending=False)
        dmy_df[num_elems + 1 :] = 0
        return dmy_df

    @staticmethod
    def bottomToZeroWithDuplicates(npA, n_dict: dict):
        """
        Set everything below n to zero,
        but deal with duplicates
        """

        num_e = n_dict[npA.name]
        unique = np.unique(npA)

        uniqueGTzero = len(unique[unique > 0])
        if num_e > uniqueGTzero:
            #  Deal with edgePG extending into negative correlations
            num_e = uniqueGTzero

        topIunique = np.argpartition(unique, -num_e)[-num_e:]
        toKeep = []
        for val in unique[topIunique]:
            # this ensures that the duplictates are kept too
            toKeep.extend(np.where(npA == val)[0])

        #  Mask and reverse
        mask = np.ones(len(npA), bool)
        mask[toKeep] = 0
        npA[mask] = 0

        return npA

    # This yields different results
    @measure_execution_time
    def reduce_edges_2(self, corr_df: pd.DataFrame, ge_df: pd.DataFrame, sel_ge: list):

        # Compute the number of edges
        num_e_dict = pd.Series([self.edges_pg + 1] * ge_df.shape[0], index=ge_df.index, name="number_edges")
        num_e_dict.loc[num_e_dict.index.isin(sel_ge)] = self.edges_sel + 1
        num_e_dict = num_e_dict.to_dict()

        dmy_df = corr_df.apply(iNET.bottomToZeroWithDuplicates, args=(num_e_dict,), axis=1)

        return dmy_df

    @measure_execution_time
    def run(self):

        # Dev
        def find_node(g: gt.Graph, gene_name='AHR'):
            for v in self.g.vertices():
                if self.g.vp['gene'][v] ==gene_name:
                    return v
                
        pruned_df, _, ge_df, sel_ge = self.compute_g_matrix()

        # stats df
        _ = self.compute_stats(ge_df)
        g, edges_list = self.create_graph(pruned_df)

        # Dev
        if 0:
            graph_stats = pd.DataFrame(index=g.vs[:]["name"])
            graph_stats["degree"] = g.degree()
            print(f"\n\n############ AHR iGRAPH degree {graph_stats.loc['AHR']['degree']}")  

            g2, edge_list = self.create_graph(pruned_df.T)
            graph_stats = pd.DataFrame(index=g.vs[:]["name"])
            graph_stats["degree"] = g.degree()
            # delete the large files
            # del pruned_df, ge_df

            graph_stats = pd.DataFrame(index=g.vs[:]["name"])
            graph_stats["degree"] = g2.degree()
            print(f"\n\n############ AHR iGRAPH 2 degree {graph_stats.loc['AHR']['degree']}")  


        # Running Leiden for comparison
        iNET.print_msg(f"Running Leiden")
        _, _ = self.run_leiden(g=g, mod_type="mod_max", runs=100)

        if self.graph_type == "gt":
            self.g = self.create_gt(edges_list)

            # Dev
            if 0:
                ahr_degree = self.g.get_total_degrees([find_node(self.g, gene_name='AHR')])
                print(f"\n\n############ AHR gt degree {ahr_degree}")   

            if self.sbm_method == "hsbm":
                iNET.print_msg(f"Running hSBM with {self.sbm_method}")
                state = self.run_hsbm(self.g, n_iter=self.sbm_config["n_iter"], mc_iter=self.sbm_config["mc_iter"], deg_cor=self.sbm_config["deg_cor"])

                # Sometimes disambiguity of the partitions it is failing
                iNET.save_obj(state, path=self.output_folder, filename=f"gt_{self.sbm_method}_{self.exp_name}.pickle")

                state = self.dissambigue_hsbm_part(state)
                
                # Update the graph object with the latest, after mcmc sweep
                self.gt_g = state['state'].g

            else:
                iNET.print_msg(f"Running SBM with {self.sbm_method}")
                state = self.run_sbm(self.g, n_iter=self.sbm_config["n_iter"], mc_iter=self.sbm_config["mc_iter"], deg_cor=self.sbm_config["deg_cor"])

                # Update the graph object with the latest, after mcmc sweep
                self.gt_g = state['state'].g

            iNET.save_obj(state, path=self.output_folder, filename=f"gt_{self.sbm_method}_{self.exp_name}.pickle")

        self.save_experiment(sel_ge=sel_ge)

        if self.save_to_gephi:
            self.export_to_gephi(pruned_df=pruned_df, ge_df=ge_df, sel_ge=sel_ge)


if 0:

    # input_folder = "data/"
    # output_folder = 'results/test_perf/'

    input_folder = "../../data/"
    output_folder = "../../results/test_perf/"

    genes_kept = 5000
    edges_pg, edges_sel, modifier = 3, 6, "standard"
    name = f"{modifier}_iNet_{edges_sel}TF_stdSort"
    graph_type = "gt"

    sbm_config = {"method": "hsbm", "n_iter": 1000, "mc_iter": 10, "deg_cor": True}  # or sbm

    ge_file = 'healthy_data_10K_gc42_v3.tsv'

    test_inet = iNET(
        exp_name=name,
        ge_file=ge_file,
        input_folder=input_folder,
        output_folder=output_folder,
        edges_pg=edges_pg,
        edges_sel=edges_sel,
        modifier_type=modifier,
        genes_kept=genes_kept,
        graph_type=graph_type,
        sbm_config=sbm_config,
    )

    pruned_df, corr_df, ge_df, sel_ge = test_inet.compute_g_matrix()

    # Compute corr matrix
    corr_df = test_inet.corr_matrix(df=ge_df).round(test_inet.round_decimal)
    rank_df = corr_df.rank(ascending=False, method="average")
    pruned_df = test_inet.reduce_edges_2(corr_df=corr_df.copy(deep=True), ge_df=ge_df, sel_ge=sel_ge)

    g, edges_list = test_inet.create_graph(pruned_df)

    gt_g = test_inet.create_gt(edges_list)

    test_inet.run_hsbm(n_iter=10, mc_iter=2, deg_cor=True)

    # rank_df.loc["CEBPA"].sort_values(ascending=True)[:12]
    # corr_df.loc["CEBPA"].sort_values(ascending=False)[:10]

    # apply modifiers

    print("Ended")
    if 0:
        start_time = time.time()
        num_e_dict = pd.Series([test_inet.edges_pg + 1] * ge_df.shape[0], index=ge_df.index, name="number_edges")
        print(f"Running time {(time.time() - start_time):.4f}")

        start_time = time.time()
        num_e_dict.loc[num_e_dict.index.isin(sel_ge)] = test_inet.edges_sel + 1
        print(f"Running time {(time.time() - start_time):.4f}")

        start_time = time.time()
        num_e_dict = num_e_dict.to_dict()
        print(f"Running time {(time.time() - start_time):.4f}")
