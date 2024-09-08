#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   GraphToolExp.py
@Time    :   2023/07/11 09:06:37
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   This file contains the class needed to analyse the graph tool
"""


import os

import _pickle as cPickle
import graph_tool.all as gt
import numpy as np
import pandas as pd
import sys

# ploting libs
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

# custom library
sys.path.append('/Users/vlad/Documents/Code/York/iNet_v2/src/')
# from ..PGCNA_Stats. import NetworkOutput
from NetworkAnalysis.ExperimentSet import NetworkOutput


class GraphToolExperiment(NetworkOutput):

    com_df: pd.DataFrame
    gt_modCon: dict
    pos: any
    mevsMut: pd.DataFrame
    gt_g: gt.Graph
    sbm_method: str

    # TODO: Rename these to reflect that are a dictionary that contain information about BlockModelwh
    hstateObj: dict
    states: list

    def __init__(self, exp: NetworkOutput, rel_path=""):
        super().__init__(exp.graph, exp.exp_meta, exp.mut_df, exp.exps_path, rel_path)

    @classmethod
    def from_pgcna_exp(cls, exp: NetworkOutput, rel_path="../"):
        obj = cls.__new__(cls)
        super(GraphToolExperiment, obj).__init__(exp.graph, exp.exp_meta, exp.mut_df, exp.exps_path, rel_path=rel_path)
        obj._value = exp
        return obj

    @classmethod
    def from_pgcna_inet(cls, exp: NetworkOutput, rel_path="../"):
        obj = cls.__new__(cls)
        super(GraphToolExperiment, obj).__init__(exp.graph, exp.exp_meta, exp.mut_df, exp.exps_path, rel_path=rel_path, exp_type="iNet", base_path=exp.base_path)

        sbm_method = exp.type.split("_")[-1]
        gt_g_path = f"{exp.exps_path}/Networks/{exp.name}/EPG3/gt_{sbm_method}_{exp.name}.pickle"

        if os.path.isfile(gt_g_path):
            with open(gt_g_path, "rb") as handle:
                part_state = cPickle.load(handle)
                if sbm_method == 'sbm': 
                    exp.states = [part_state]
                    gt_g = part_state['state'].g
                else:
                    exp.hstateObj = part_state
                    gt_g = part_state['state'].g
        else:
            raise ValueError("Couldn't loat gt tool {}".format(gt_g_path))

        exp.gt_g = gt_g
        obj._value = exp
        return obj

    # get the values from the original PGCNA experiment
    def __getattr__(self, name):
        if "_value" in self.__dict__:
            return getattr(self._value, name)
        raise AttributeError("no attribute: " + name)

    def create_gt(self):
        """
        Creates the Graph Tool Graph based on the list of edges generated with PGCNA
        """
        list_of_edges = list(self.edges_df[["Source", "Target", "Weight"]].itertuples(index=False, name=None))
        gt_g = gt.Graph(directed=False)
        vertex_prop = gt_g.add_edge_list(edge_list=list_of_edges, hashed=True, eprops=[("weight", "double")])
        gt_g.vp.gene = vertex_prop

        self.gt_g = gt_g

    def gt_modCon_MEV(self, all_tpms: pd.DataFrame, num_genes=100, is_imev=False, com_df = None, **kwargs):

        if not hasattr(self, "gt_modCon"):
            if com_df is not None:
                self.get_ModCon(com_df=com_df)
            else:
                if self.sbm_method == "sbm":
                    self.add_vp()
                else:
                    self.hsbm_add_vp()

                self.get_ModCon()

        is_mut = ("mut_df" in kwargs.keys())
        if (not is_imev) and (not is_mut):
            sort_col = "ModCon_{}_gt".format(self.type)
            self.mevsMut, _ = self.get_mevs(tpms=all_tpms, modCon=self.gt_modCon, sort_col=sort_col, num_genes=num_genes, verbose=False)
        elif (is_imev) and (not is_mut):
            sort_col = "ModCon_{}_gt".format(self.type)
            self.mevsMut, _ = self.get_iMevs(
                h_tpms=self.tpm_df, tum_tpms=all_tpms, modCon=self.gt_modCon, sort_col=sort_col, num_genes=num_genes, verbose=False)
        else:
            sort_col = "ModCon_{}_gt".format(self.type)
            self.mevsMut, _ = self.get_iMevs(h_tpms = self.tpm_df, tum_tpms=all_tpms, modCon=self.gt_modCon, sort_col=sort_col, num_genes=num_genes, verbose=False, mut_df=kwargs.pop('mut_df'), offset=kwargs.pop('offset'))

    # Sampling from posterior distrib
    def sample_posterior(self, n_iter=10000, mc_iter=10, deg_cor=True, distrib="real-exponential", verbose=True):
        gt_g = self.gt_g
        state = gt.BlockState(gt_g, recs=[gt_g.ep.weight], rec_types=[distrib], deg_corr=deg_cor)

        print("### Equilibrate MCMC for {}".format(self.type))
        gt.mcmc_equilibrate(state, wait=n_iter / 10, mcmc_args=dict(niter=mc_iter), verbose=verbose)

        bs = []  # collect some partitions
        h = np.zeros(gt_g.num_vertices() + 1)  # to find the probabilities for a certain size
        dls = []  # description length history

        def collect_partitions(s):
            bs.append(s.b.a.copy())
            B = s.get_nonempty_B()
            h[B] += 1
            dls.append(s.entropy())

        print("### Collecton partitions for {}".format(self.type))

        # Now we collect partitions for exactly 1000 sweeps, at intervals
        # of 10 sweeps:
        # n_iter = 10000
        eq_res = gt.mcmc_equilibrate(state, force_niter=n_iter, mcmc_args=dict(niter=mc_iter), callback=collect_partitions, verbose=verbose)

        print("### Marginal property partitions for {}".format(self.type))
        # Disambiguate partitions and obtain marginals
        pmode = gt.PartitionModeState(bs, converge=True, relabel=True)
        print("### get all marginal for {}".format(self.type))
        pv = pmode.get_marginal(gt_g)

        # keep track
        if not hasattr(self, "states"):
            self.states = []

        new_state = {
            "name": "state_{}".format(len(self.states) + 1),
            "partition_meta": {"h": h, "n_iter": n_iter, "bs": bs},
            "pmode": pmode,
            "state": state,
            "pv": pv,
            "dls": dls,
            "eq_res": eq_res,
        }

        self.states.append(new_state)

    def add_vp(self, state_idx=0, mut_df=None):
        gt_g = self.gt_g

        # check if pv and pmode exists
        if not hasattr(self, "states"):
            self.sample_posterior()

        state = self.states[state_idx]

        vp_marg = gt_g.new_vertex_property("object", state["pv"])
        gt_g.vp["marginal"] = vp_marg

        vp_max = gt_g.new_vertex_property("string", state["pmode"].get_max(gt_g))
        gt_g.vp["max_b"] = vp_max

        if mut_df is not None:
            comb = {}
            for idx in gt_g.vertex_index:
                comb[idx] = gt_g.vp.gene[idx]

            dmy_df = pd.DataFrame.from_dict(comb, orient="index", columns=["gene"]).reset_index(names="gt_idx").set_index("gene")
            dmy_df["mut_count"] = mut_df["count"]
            dmy_df.fillna(0, inplace=True)
            dmy_df["mut_count"] = dmy_df["mut_count"].astype(int)
            dmy_df.sort_values(by="gt_idx", inplace=True)

            vp_mut = gt_g.new_vertex_property("int", list(dmy_df["mut_count"].values))
            gt_g.vp["mut_count"] = vp_mut

    # Layout functions
    def layout(self):
        pos = gt.sfdp_layout(self.gt_g, eweight=self.gt_g.ep.weight)
        vb, eb = gt.betweenness(self.gt_g)
        self.pos = pos

    # overriding the method from the NetworkOutput
    def get_ModCon(self, state=0, com_df = None):
        if self.sbm_method == 'sbm':
            if com_df is None:
                com_df = self.get_gt_df(state_idx=state)
        else:
            if com_df is None:
                com_df, _ = self.hsbm_get_gt_df()
            com_df["max_b"] = com_df["P_lvl_0"]

        gen_coms = com_df["max_b"].reset_index().rename(columns={"index": "Id"})
        mut_df = self.mut_df
        meta_df = self.meta_df

        # create a DataFrame with the components for the modCon equation
        modifier = self.type.split("_")[0]
        col = "conn_{}".format(modifier)

        # all_genes = []  # can be used for testing
        modCons = {}
        for mod_class in gen_coms["max_b"].unique():
            genes = gen_coms.loc[gen_coms["max_b"] == mod_class]["Id"].values

            conn_g = []
            for gene in genes:
                # 1st condition, get all the edges that the current gene is the Source and are inside the module
                # 2nd condition, get all the edges that current is the Target and are inside the module
                in_edges = self.edges_df[(self.edges_df["Source"] == gene) | (self.edges_df["Target"] == gene)]
                # Removed as a way to deal with communities where nodes have degree_in = 0
                # in_edges = in_edges[in_edges["Source"].isin(genes)]
                # in_edges = in_edges[in_edges["Target"].isin(genes)]
                weights_sum = in_edges["Weight"].sum()
                if weights_sum == 0:
                    print(f"Weighted sum = 0 for Com {mod_class} !!")
                conn_g.append([gene, weights_sum])

            conn_df = pd.DataFrame(conn_g, columns=["gene", col]).set_index("gene")
            working_df = pd.concat([conn_df, meta_df.loc[meta_df["genes"].isin(genes)].set_index("genes"), mut_df[mut_df.index.isin(genes)]["count"]], axis=1)

            # 5. Workout the ModCon and save it
            working_df["ModCon_{}_gt".format(self.type)] = (
                (working_df[col] ** 2) * working_df["q2E"] * working_df["varWithin"] * (100 - working_df["varAcross"]) / 100
            )

            # sort by the current exp type
            modCons[mod_class] = working_df.sort_values(by="ModCon_{}_gt".format(self.type), ascending=False)

        self.gt_modCon = modCons
        return modCons

    # Experiment processing
    def get_gt_df(self, state_idx=0, compute=True):
        """
        The graph requires the following VertexPropertyMaps:
        * Marginal - marginal prob of each node
        * Max_b - the max partition prob of each node

        Args:
            gt_g (Graph): Graph-Tool Graph
            pmode (PartitionModeState): _description_
        """

        # check if there is already calculated
        if not compute and hasattr(self, "com_df"):
            return self.com_df

        gt_g = self.gt_g

        # get the maximum number of clusters
        state = self.states[state_idx]
        max_part = state["pmode"].get_B()
        num_nodes = gt_g.num_vertices()

        part_nodes = np.zeros((num_nodes, max_part + 2))
        genes = []
        for node in gt_g.iter_vertices():
            genes.append(gt_g.vp.gene[node])
            frac = np.array(gt_g.vp["marginal"][node])
            part_nodes[node, : frac.shape[0]] = frac
            part_nodes[node, -2] = node
            part_nodes[node, -1] = int(gt_g.vp["max_b"][node]) # offset to start from 0

        col_names = ["b_{}".format(idx) for idx in range(0, max_part)] + ['node_idx', 'max_b']
        com_df = pd.DataFrame(part_nodes, columns=col_names, index=genes)

        self.com_df = com_df
        return com_df

    def check_graph(self):
        """

        TODO Work in progress. Re-construct the edges_df matrix but from gt_g.

        It just needs to check if the two are identical
        """
        if not self.gt_g:
            self.create_gt()

        gt_g = self.gt_g
        for e in enumerate(gt_g.iter_edges()):
            source, target = gt_g.vp.gene[e[0]], gt_g.vp.gene[e[1]]
            weight = gt_g.ep.weight[(e[0], e[1])]
            print("{} - {}: {}".format(source, target, weight))

    def get_stable_genes(self, state_idx=0, prob_th=0.75):
        self.add_vp(state_idx=state_idx)
        com_df = self.get_gt_df(state_idx=state_idx)

        max_iters = com_df.max().max()
        th = max_iters * prob_th

        stable_genes = com_df.loc[com_df[com_df > th].dropna(how="all").index]
        com_df["stable_gene"] = "wobble"
        com_df.loc[stable_genes.index, "stable_gene"] = "stable"

        return com_df

    ### hsbm specific methods
    def hsbm_get_gt_df(self):
        """
        The graph requires the following VertexPropertyMaps:
        * Marginal - marginal prob of each node
        * Max_b - the max partition prob of each node

        Args:
            gt_g (Graph): Graph-Tool Graph
            pmode (PartitionModeState): _description_
        """
        gt_g = self.gt_g

        # get the maximum number of clusters
        max_part = self.hstateObj["pmode"].get_B()
        num_nodes = gt_g.num_vertices()

        # 2 for max-b and node_idx
        part_nodes = np.zeros((num_nodes, max_part + 2))
        genes = []
        for node in gt_g.iter_vertices():
            genes.append(gt_g.vp.gene[node])
            frac = np.array(gt_g.vp["marginal"][node])
            part_nodes[node, : frac.shape[0]] = frac
            # Last two position with the node idx and the partition with highest membership
            part_nodes[node, -2] = node
            part_nodes[node, -1] = int(gt_g.vp["max_b"][node]) # offset to start from 0

        col_names = ["b_{}".format(idx) for idx in range(0, max_part)] + ["node_idx", "level_0"]
        com_df = pd.DataFrame(part_nodes, columns=col_names, index=genes)

        #### adding the gene membership from all the levels

        # Get the non-1 levels
        non_zero_lvls = []
        levels = self.hstateObj["state"].get_levels()
        for idx, s in enumerate(levels):
            if s.get_N() == 1:
                break
            non_zero_lvls.append(idx)

        # create DF with the non-1 membersips
        lvl_parts = self.get_level_partitions(com_df)

        # check if the dataframes are alligned before mergin
        if not lvl_parts["P_lvl_0"].equals(com_df["level_0"].astype(int)):
            raise ValueError("The partition membership don't match, between the two dataframes")

        # merge
        com_df = pd.concat([com_df.drop(columns=["level_0"]), lvl_parts], axis=1)

        return com_df, lvl_parts.columns

    def get_level_partitions(self, com_df):
        levels = self.hstateObj["state"].get_levels()
        # create the dictionary
        lvls_parts = {"gene": []}
        for idx, lvl in enumerate(levels):
            lvls_parts[idx] = []
            if lvl.get_N() == 1:
                break

        # add values
        for gene, row in com_df.iterrows():
            # first, will be from the first level
            lvls_parts["gene"].append(gene)
            last_lvl_idx = row["node_idx"]
            for i, lvl in enumerate(levels):
                part = lvl.get_blocks()[last_lvl_idx]
                lvls_parts[i].append(part)
                last_lvl_idx = part
                if lvl.get_N() == 1:
                    break

        dmy = pd.DataFrame.from_dict(data=lvls_parts)
        dmy.columns = ["gene"] + list(["P_lvl_{}".format(lvl) for lvl in range(0, len(lvls_parts.keys()) - 1)])

        return dmy.set_index("gene")

    def hsbm_sample_posterior(self, n_iter=10000, mc_iter=10, deg_cor=False):
        gt_g = self.gt_g
        # TODO: Is this the best function?
        state = gt.NestedBlockState(gt_g, recs=[gt_g.ep.weight], rec_types=["real-exponential"], state_args=dict(deg_corr=deg_cor))

        print("### 1. Equilibrate MCMC. n_iter {}".format(n_iter / 10))
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

        print("\n\n\n### 2. Collecton partitions. n_iter {}".format(n_iter))
        eq_res = gt.mcmc_equilibrate(state, force_niter=n_iter, mcmc_args=dict(niter=mc_iter), callback=collect_partitions, verbose=True)  # 10000 / 10

        print("### 3. Marginal property partitions")
        # Disambiguate partitions and obtain marginals
        pmode = gt.PartitionModeState(bs, nested=True, converge=False)
        print("### 4. Collecting the data")
        pv = pmode.get_marginal(gt_g)

        # Get consensus estimate
        bs = pmode.get_max_nested()
        state = state.copy(bs=bs)

        print("### 5. Saving")
        self.hstateObj = {
            "name": "hstate",
            "partition_meta": {"h": h, "n_iter": n_iter, "bs": bs},
            "pmode": pmode,
            "state": state,
            "pv": pv,
            "dls": dls,
            "eq_res": eq_res,
        }

    def draw_hstate(self, name, level=None, lvl_idx=0, layout="radial"):
        if not hasattr(self, "hstateObj"):
            raise ValueError("No hstate")

        if level is None:
            filename = "hbm_{}_{}_{}_marginals_2.pdf".format(self.type, layout, name)
            self.hstateObj["state"].draw(
                layout=layout,
                vertex_text=self.gt_g.vp.gene,
                vertex_shape="pie",
                vertex_pie_fractions=self.hstateObj["pv"],
                # edge_color=gt.prop_to_size(self.gt_g.ep.weight, power=1, log=True),
                # eorder=self.gt_g.ep.weight, edge_pen_width=gt.prop_to_size(self.gt_g.ep.weight, 1, 4, power=1, log=True),
                # order=self.gt_g.ep.weight,
                # edge_pen_width=gt.prop_to_size(self.gt_g.ep.weight, 2, 8, power=1, log=True),
                # edge_gradient=[],
                output=filename,
            )
        else:
            filename = "hbm_lvl{}_{}_{}_{}_marginals.pdf".format(lvl_idx, self.type, layout, name)
            level.draw(output=filename)

    def print_summary(self):
        if not hasattr(self, "hstateObj"):
            raise ValueError("No hstate")

        self.hstateObj["state"].print_summary()

    def hsbm_add_vp(self, mut_df=None):
        # check if pv and pmode exists
        if not hasattr(self, "hstateObj"):
            raise ValueError("No hstate")

        gt_g = self.gt_g

        comb = []
        for idx in gt_g.vertex_index:
            comb.append({idx: gt_g.vp.gene[idx]})

        stateObj = self.hstateObj
        vp_meta = gt_g.new_vertex_property("object", comb)
        gt_g.vp["meta"] = vp_meta

        vp_marg = gt_g.new_vertex_property("object", stateObj["pv"])
        gt_g.vp["marginal"] = vp_marg

        vp_max = gt_g.new_vertex_property("int", stateObj["pmode"].get_max(gt_g))
        gt_g.vp["max_b"] = vp_max

        if mut_df is not None:
            comb = {}
            for idx in gt_g.vertex_index:
                comb[idx] = gt_g.vp.gene[idx]

            dmy_df = pd.DataFrame.from_dict(comb, orient="index", columns=["gene"]).reset_index(names="gt_idx").set_index("gene")
            dmy_df["mut_count"] = mut_df["count"]
            dmy_df.fillna(0, inplace=True)
            dmy_df["mut_count"] = dmy_df["mut_count"].astype(int)
            dmy_df.sort_values(by="gt_idx", inplace=True)

            vp_mut = gt_g.new_vertex_property("int", list(dmy_df["mut_count"].values))
            gt_g.vp["mut_count"] = vp_mut

    def hsbm_plot_posterior(self):
        """
        Plots the posterior probability for each layer of hSBM until only a single block is found by the highest probability

        Args:
            stop_lvl (int, optional): The level at which to stop plotting. Defaults to -1 so it doesn't stop at all

        Returns:
            None
        """
        if not hasattr(self, "hstateObj"):
            raise ValueError("No hstate")

        h = self.hstateObj["partition_meta"]["h"]
        n_iter = self.hstateObj["partition_meta"]["n_iter"]
        levels: gt.NestedBlockState = self.hstateObj["state"].get_levels()

        figs, titles = [], []
        for l, lvl_state in enumerate(levels):
            cluster_idxs = np.nonzero(h[l])[0]

            cluster_contrib = h[l][cluster_idxs] / n_iter
            cluster_idxs = list(map(str, cluster_idxs))
            figs.append(px.bar(x=cluster_idxs, y=cluster_contrib))
            titles.append("Layer {}".format(l))
            print(f"Lvl {l}, non-empty {lvl_state.get_nonempty_B()}; B {lvl_state.get_B()}; N: {lvl_state.get_N()}")

            if lvl_state.get_nonempty_B() == 1:
                break

        if figs == []:
            print("No figures to plot")
            return 

        config = {
            "num_cols": 4,
            "shared_x": False,
            "shared_y": True,
            "h_spacing": 0.05,
            "v_spacing": 0.15,
            "main_title": "Posterior probability for each layer of hSBM",
            "height": 900,
            "width": None,
            "y_title": "Y-axis",
            "x_title": "X-axis",
            "specs": None,
        }

        return GraphToolExperiment.helper_multiplots(figs, titles, config)

    # TODO: Develop this to accept gene name
    def get_gene_membership_levels(self, gene):
        levels = self.hier_state.get_levels()
        for s in levels:
            print(s)
            if s.get_N() == 1:
                break

        r = levels[0].get_blocks()[46]  # group membership of node 46 in level 0
        print(r)
        r = levels[1].get_blocks()[r]  # group membership of node 46 in level 1
        print(r)
        r = levels[2].get_blocks()[r]  # group membership of node 46 in level 2
        print(r)

    # TODO: Work in progress
    def model_selection(self):
        g = gt.collection.ns["foodweb_baywet"]

        # This network contains an internal edge property map with name
        # "weight" that contains the energy flow between species. The values
        # are continuous in the range [0, infinity].

        state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(recs=[g.ep.weight], rec_types=["real-exponential"]))

        # improve solution with merge-split

        for i in range(100):
            ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)

        state.draw(
            edge_color=gt.prop_to_size(g.ep.weight, power=1, log=True),
            ecmap=(plt.cm.inferno, 0.6),
            eorder=g.ep.weight,
            edge_pen_width=gt.prop_to_size(g.ep.weight, 1, 4, power=1, log=True),
            edge_gradient=[],
            output="foodweb-wsbm.pdf",
        )

    # Plotting functionsget
    def plot_partition_prob(self, state_idx=0):
        if not hasattr(self, "states"):
            raise AttributeError("No attribute partitions_meta")

        part_meta = self.states[state_idx]["partition_meta"]
        cluster_idxs = np.nonzero(part_meta["h"])[0]
        cluster_contrib = part_meta["h"][cluster_idxs] / part_meta["n_iter"]
        cluster_idxs = list(map(str, cluster_idxs))
        fig = px.bar(x=cluster_idxs, y=cluster_contrib, title="Marginal prob of the number of groups")

        return fig

    def plot_membership_changes(self, idx, prob_th=90):
        com_df = self.get_gt_df(state_idx=idx)

        # wobble df defined by <75
        max_iters = com_df.max().max()
        th = max_iters * prob_th

        wobble_df = com_df[com_df < th].dropna().copy(deep=True)
        dmy_df = pd.concat(
            [pd.DataFrame(wobble_df["max_b"].value_counts()), pd.DataFrame(com_df["max_b"].value_counts()).rename(columns={"count": "orig_size"})], axis=1
        ).sort_index()

        fig = px.line(dmy_df.reset_index(), x="max_b", y=["count", "orig_size"], markers=True)
        fig.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1))

        return fig

    def comp_posterior_distribcomp_posterior_distrib(self):
        figs, titles = [], []
        for idx, _ in enumerate(self.states):
            figs.append(self.plot_partition_prob(state_idx=idx))
            titles.append("Run {}".format(idx + 1))

        subplots_config = {
            "num_cols": 2,
            "shared_x": False,
            "shared_y": True,
            "h_spacing": 0.05,
            "v_spacing": 0.15,
            "main_title": "SBM #communities for exp {} in {} runs.".format(self.type, len(titles)),
            "height": 700,
            "width": None,
            "y_title": "#Comunities",
            "x_title": "Probability",
            "specs": None,
        }

        return GraphToolExperiment.helper_multiplots(figs, titles, subplots_config)

    # NOTE this only works for SBM
    def plot_genes_network_metrics(self, state_idx=0, prob_th=0.75):
        # get the stable genes based on the prob_th
        com_df = self.get_stable_genes(state_idx=state_idx, prob_th=prob_th)

        # get the iGraph stats
        graph_stats = self.compute_graph_stats()

        # The other metrics
        metrics = ["TF", "count", "IVI", "degree", "hubScore", "strength", "betwenees"]
        figs = []
        for metric in metrics:
            if metric in graph_stats.columns:
                com_df[metric] = graph_stats[metric]
            else:
                if metric != "TF":
                    # Need to make TF as category, to show nicely on the histogram
                    com_df[metric] = self.nodes_df[metric]
                else:
                    com_df[metric] = self.nodes_df[metric].astype(str)

            figs.append(px.histogram(com_df, x=metric, color="stable_gene", nbins=None, histnorm="percent"))

        # prepare for plotting
        titles = metrics

        subplots_config = {
            "num_cols": 3,
            "shared_x": False,
            "shared_y": False,
            "h_spacing": 0.03,
            "v_spacing": 0.1,
            "main_title": "Stable genes against network metrics. Network - {}. State_idx - {}".format(self.type, state_idx),
            "height": 900,
            "width": None,
            "y_title": None,
            "x_title": None,
            "specs": None,
        }

        return figs, titles, subplots_config

    # Network stats
    def compute_graph_stats(self):
        graph_stats = pd.DataFrame(index=[g for g in self.graph.vp["gene"]])

        # Degree related metrics
        vp = self.graph.degree_property_map(deg="total", weight=self.graph.ep["weight"])
        graph_stats["degree_w"] = np.round(vp.a, 5)

        vp = self.graph.degree_property_map(deg="total")
        graph_stats["degree_t"] = vp.a

        vp_bet, _ = gt.betweenness(self.graph, weight=self.graph.ep["weight"])
        graph_stats["betweenness"] = [g for g in vp]

        # Other centrality stats
        vp = gt.closeness(self.graph, weight=self.graph.ep["weight"])
        graph_stats["closeness"] = np.round(vp.a, 5)

        vp = gt.katz(self.graph, weight=self.graph.ep["weight"])
        graph_stats["katz"] = np.round(vp.a, 5)

        _, vp, _ = gt.hits(self.graph, weight=self.graph.ep["weight"])
        graph_stats["hits"] = vp.a

        vp = gt.central_point_dominance(self.graph, betweenness=vp_bet)
        graph_stats['central_point_dominance'] = round(vp, 5)

        # More degree metrics
        vp = self.graph.degree_property_map(deg="in", weight=self.graph.ep["weight"])
        graph_stats["degree_i"] = vp.a

        vp = self.graph.degree_property_map(deg="out", weight=self.graph.ep["weight"])
        graph_stats["degree_o"] = vp.a

        vp = gt.pagerank(self.graph)
        graph_stats["pageRank"] = vp.a

        return graph_stats

    # Overriding the one from PGCNA ouput
    def ge_comm(self):
        comm_tpms = {}
        median_vals, std_df = pd.DataFrame(index=self.tpm_df.columns), pd.DataFrame(index=self.tpm_df.columns)

        for mod_class in self.nodes_df['max_b'].unique():
            genes = self.nodes_df.loc[self.nodes_df['max_b'] == mod_class].index.values
            comm_tpms[mod_class] = self.tpm_df.loc[self.tpm_df.index.isin(genes)]

            median_vals = pd.concat([median_vals, comm_tpms[mod_class].median().rename("Comm_{}".format(mod_class))], axis=1)
            std_df = pd.concat([std_df, comm_tpms[mod_class].std().rename("Comm_{}".format(mod_class))], axis=1)

        med_df = median_vals.melt(var_name="Comm", value_name="Median", ignore_index=False).reset_index(names=["Sample"])
        std_df = std_df.melt(var_name="Comm", value_name="Std", ignore_index=False).reset_index(names=["Sample"])
        combined_stats = pd.concat([std_df, med_df["Median"]], axis=1)
        return combined_stats

    ######## Adding custom properties ########
    def add_gt_prop_draw(self, gt_g: gt.Graph, com_df: pd.DataFrame, tf_list: []):

        if not hasattr(self, "nodes_df"):
            self.export_to_gephi(save=False, com_df=com_df)

        self.nodes_df["ModCon_Rank"] = self.nodes_df["ModCon_Rank"].astype(int)
        hex_colors = GraphToolExperiment.color_scale(color_scale="Sunset", num_points=50)

        max_char, prcsd_genes, display_prop, is_tf, colors_rank = (6, [], [], [], [])
        for idx in gt_g.vertices():
            gene, mut_count, com = gt_g.vp["gene"][idx], gt_g.vp["mut_count"][idx], gt_g.vp['max_b'][idx]

            rank = self.nodes_df.loc[gene]["ModCon_Rank"]
            if rank > 25:
                colors_rank.append("#656369")
            else:
                colors_rank.append(hex_colors[rank - 1])

            if len(gene) > max_char:
                if "ENSG" in gene:
                    gene = "EN_" + gene[-3:]
                else:
                    gene = gene[:max_char] + "."

            if gene in tf_list:
                is_tf.append(1)
            else:
                is_tf.append(0)

            prcsd_genes.append(gene)
            display_prop.append(f'{gt_g.vp["gene"][idx]}: Mut {mut_count}, Com: {com}')

        vp_prcsd_gene = gt_g.new_vertex_property("string", prcsd_genes)
        gt_g.vp["prcsd_gene"] = vp_prcsd_gene

        vp_display_prop = gt_g.new_vertex_property("string", display_prop)
        gt_g.vp["display_prop"] = vp_display_prop

        vp_colors_rank = gt_g.new_vertex_property("string", colors_rank)
        gt_g.vp["colors_rank"] = vp_colors_rank

        vp_tf = gt_g.new_vertex_property("int", is_tf)
        gt_g.vp["is_tf"] = vp_tf

    ######## Helper methods ########
    def save(self, exp_type="sbm", is_hsbm=False, path=None):
        save_gt_exp = {"name": self.name, "gt_exp": self}
        if path is None:
            path = self.exps_path

        # folder location
        if is_hsbm:
            folder_path = f"{path}/hsbm/{exp_type}/"
        else:
            folder_path = f"{path}/sbm/{exp_type}/"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = f"gt_{exp_type}_{self.name}.pickle"
        full_path = folder_path + filename
        with open(full_path, "wb") as handle:
            cPickle.dump(save_gt_exp, handle, protocol=cPickle.HIGHEST_PROTOCOL)
        return full_path

    def export_gephi_coord(self, graph: gt.Graph, draw_results: tuple, path: str, name: str):

        test_array = np.array(list(draw_results[0]))

        x_cord = test_array[:, 0] * 40
        y_cord = test_array[:, 1] * 40

        vp_x = graph.new_vertex_property("double", x_cord)
        graph.vp["x_cord"] = vp_x
        graph.vp["x"] = vp_x

        vp_y = graph.new_vertex_property("double", y_cord)
        graph.vp["y_cord"] = vp_y
        graph.vp["y"] = vp_y

        graph.save(f"{path}/{name}.graphml")

    def export_morpheus(self, settings: dict):

        cs_num_1 = settings['cs_num_1']
        cs_num_2 = settings['cs_num_2']
        sel_mut = settings['sel_mut']
        cs_exp = settings['cs_exp']
        vu_output = settings['vu_output']
        label_col_cs = settings['label_col_cs']

        file_path= f"{settings['figures_path']}/{settings['filename']}.tsv"

        # MEVS
        mevsMut = self.mevsMut

        cs_cols = [f"RawKMeans_CS_{cs_num_1}_{label_col_cs}", f"RawKMeans_CS_{cs_num_2}_{label_col_cs}"]
        vu_cols = [
            "race",
            "gender",
            "bmi",
            "KMeans_labels_6",
            "2019_consensus_classifier",
            "TCGA408_classifier",
        ]

        # Selecting only the present mutations
        dmy_mut = sel_mut[sel_mut.index.isin(self.tpm_df.index)]

        mevs_cols, mut_cols = list(self.mevsMut.columns), list(dmy_mut.T.columns)
        mevs_cs = pd.concat(
            [
                cs_exp[cs_cols],
                vu_output[vu_cols],
                dmy_mut.T,
                mevsMut,
            ],
            axis=1,
        ).dropna()

        for col in cs_cols:
            mevs_cs[col] = mevs_cs[col].astype(str)

        mevs_cs = mevs_cs[vu_cols + cs_cols + mut_cols + mevs_cols]
        mevs_cs.transpose().to_csv(f"{file_path}", sep="\t", index=True)

        return mevs_cs

    ### Looking at the nodes ###
    def find_node(self, gene: str):
        for v in self.gt_g.iter_vertices():
            if self.gt_g.vp["gene"][v] == gene:
                return v, self.gt_g.vp['max_b'][v]

    def filter_graph(self, gene: str, show_own_com=True):

        graph = self.gt_g
        # Clear the previous filter
        graph.set_edge_filter(None)
        graph.set_vertex_filter(None)

        # idx, non_zero_part = find_comm_membership(gt_g, gene=gene, n_runs=10000)
        gene_idx, com = self.find_node(gene=gene)
        print(f"{gene}. Idx = {gene_idx}. Comm = {com}")

        vp_bool = graph.new_vertex_property("bool")
        neighbors = []
        for idx in graph.iter_out_neighbors(gene_idx):
            vp_bool[idx] = True
            neighbors.append(idx)

        # Show the community where is the gene we seek
        if show_own_com:
            max_b = np.array(list(graph.vp["max_b"])) #
            # For hSBM and newer exp it works graph.vp["max_b"]
            #  but for SBM and the control experiments it doesn't
            #  so I had to explicitly transfer the VP of max_b in a numpy array
            for idx in np.nditer(np.where(max_b == com)):
                vp_bool[idx] = True

        vp_bool[gene_idx] = True
        graph.set_vertex_filter(vp_bool)

        return neighbors

    def process_neigbhors(self, v_idxs:list):

        nodes_df = self.nodes_df
        neigbhors_df = nodes_df[nodes_df["node_idx"].isin(v_idxs)][["max_b", "count", "TF"]]

        stats_neigbhor = []
        neighbors_genes = pd.DataFrame()

        for com in neigbhors_df["max_b"].unique():
            sel_com = neigbhors_df[neigbhors_df["max_b"] == com]
            com_size = nodes_df[nodes_df["max_b"] == com].shape[0]
            mut_burden = sel_com[sel_com["count"] > 0].shape[0]
            num_tf = sel_com[sel_com["TF"] == 1].shape[0]

            # com rep
            com_ratio = round(sel_com.shape[0] / com_size, 4)

            stats_neigbhor.append((com, com_ratio, mut_burden, num_tf, sel_com.shape[0], com_size))
            neighbors_genes = pd.concat([neighbors_genes, sel_com], axis=0)

        stats_neigbhor = pd.DataFrame(stats_neigbhor, columns=["com", "com_ratio", "mut_burden", "num_TF", "num_found", "com_size"])
        stats_neigbhor = stats_neigbhor.sort_values(by=["num_found"], ascending=False)
        stats_neigbhor["com"] = stats_neigbhor["com"].astype(str)

        return stats_neigbhor, neighbors_genes

    def plot_overview_gene(self, gene_name: str):

        neigbhors = self.filter_graph(gene=gene_name)
        self.gt_g.set_vertex_filter(None)
        stats_neigbhors = self.process_neigbhors(v_idxs=neigbhors)

        fig1 = px.bar(stats_neigbhors, x="com", y=["num_found", "com_size"], barmode="group", text_auto=True)
        fig2 = px.bar(stats_neigbhors, x="com", y="com_ratio", barmode="group", text_auto=True)
        fig3 = px.bar(stats_neigbhors, x="com", y=["mut_burden"], barmode="group", text_auto=True)
        fig4 = px.bar(stats_neigbhors, x="com", y="num_TF", barmode="group", text_auto=True)

        figs = [fig1, fig2, fig3, fig4]
        titles = [
            "Num Found and Community sizes",
            "Com ratio representation",
            "Mut burden in the neigbhors",
            "Number of TF in the neigbhors",
        ]

        subplots_config = {
            "num_cols": 2,
            "shared_x": False,
            "shared_y": False,
            "h_spacing": 0.05,
            "v_spacing": 0.12,
            "main_title": f"Neigbhours overview for {gene_name} ({self.type})",
            "height": 700,
            "width": None,
            "y_title":None,
            "x_title": "Community",
            "specs": None,
        }

        fig = self.helper_multiplots(figs, titles, subplots_config)

        return fig

    @staticmethod
    def load_sbm_exps(exp_set, name="standard_4K", exp_type="h42", tf_range=None, is_hsbm=False):
        exps = {}
        entropy = pd.DataFrame()

        if tf_range is None:
            tf_range = range(3, 10, 1)

        for tf in tf_range:
            key = str(tf)
            sel_exp = exp_set.exps["{}_{}TF".format(name, tf)]
            exps[key] = GraphToolExperiment.load(exp_name=sel_exp.name, exp_type=exp_type, path=sel_exp.exps_path, is_hsbm=is_hsbm)

            exps[key].add_vp()
            dmy_df = pd.DataFrame(exps[key].states[0]["dls"], columns=["Entropy"])
            dmy_df["TF"] = key
            entropy = pd.concat([entropy, dmy_df], axis=0)

        return exps, entropy

    @staticmethod
    def load(exp_name, exp_type="sbm", is_hsbm=False, path="", legacy_path=True):
        filename = "gt_{}_{}.pickle".format(exp_type, exp_name)

        if legacy_path:
            if is_hsbm:
                full_path = "{}/hsbm/{}/{}".format(path, exp_type, filename)
            else:
                full_path = "{}/sbm/{}/{}".format(path, exp_type, filename)
        else:
            full_path = os.path.abspath(path)

        # full_path = os.path.abspath(full_path)
        if os.path.isfile(full_path):
            with open(full_path, "rb") as handle:
                exp = cPickle.load(handle)
                # we need to update the experiment's path with the current one
                exp["gt_exp"].exps_path = path
                print("### Loaded {}".format(exp["name"]))
                return exp["gt_exp"]
        else:
            raise ValueError("SBM exp not found at {}".format(full_path))

    @staticmethod
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

        subplot = subplot.update_layout(layout, height=config["height"], width=config["width"], showlegend=False)
        subplot = subplot.update_xaxes(title_text=config["x_title"])
        subplot = subplot.update_yaxes(title_text=config["y_title"])

        # Remove duplicate legend items
        visited = []
        for trace in subplot["data"]:
            if trace["name"] is None:
                continue

            trace["name"] = trace["name"].split(",")[0]
            if trace["name"] not in visited:
                trace["showlegend"] = True
                visited.append(trace["name"])
            else:
                trace["showlegend"] = False

        return subplot

    @staticmethod
    def create_mpl_fig(com_df: pd.DataFrame, size=6000, dpi=300):
        """
        Create a matplotlib figure with an empty plot and legend based on the provided DataFrame.

        Parameters:
        - com_df (pd.DataFrame): The DataFrame containing the data for plotting. This needs to contain 'max_b' column
        - size (int): The size of the figure in pixels (default: 6000).
        - dpi (int): The DPI (dots per inch) of the figure (default: 300).

        Returns:
        - fig (matplotlib.figure.Figure): The created matplotlib figure.
        - ax (matplotlib.axes.Axes): The axes of the created figure.
        """

        import matplotlib.cm

        # matplotlib.use("cairo")
        # Combine colors from two color maps
        default_clrs = list(matplotlib.cm.tab20.colors) + list(matplotlib.cm.tab20b.colors)

        default_cm = matplotlib.cm.colors.ListedColormap(default_clrs)

        # Ensure unique indices are sorted or in the desired order
        # colors_idxs = list(com_df["max_b"].unique())
        # used_colors = default_clrs[colors_idxs]

        cnorm = lambda x: x % len(default_clrs)

        used_colors = {}
        color_idxs = com_df["max_b"].unique()
        for comm in color_idxs:
            used_colors[comm] = list(default_cm(cnorm(comm)))

        pixels_x, pixels_y = size, size
        size_x, size_y = pixels_x / dpi * 2, pixels_y / dpi * 2

        # Create an empty plot with specified DPI and figure size
        fig, ax = plt.subplots(figsize=(size_x, size_y), dpi=dpi)

        # To display an empty plot with a legend, we can use a trick by plotting empty data
        for comm, color in used_colors.items():
            ax.plot([], [], color=color, label=f"Com {comm}")

        # Add legend with horizontal orientation
        ax.legend(loc='center')

        # Turn off axis
        ax.axis("off")

        return fig, ax, used_colors

    @staticmethod
    def save_legend(g: gt.Graph, path: str, name: str, size=3000, dpi=400):
        com_colors = {}
        for node in g.get_vertices():
            max_b = g.vp["max_b"][node]
            color = g.vp["fill_color"][node]
            com_colors[max_b] = color

        com_colors = dict(sorted(com_colors.items()))

        pixels_x, pixels_y = size, size
        size_x, size_y = pixels_x / dpi * 2, pixels_y / dpi * 2

        # Create an empty plot with specified DPI and figure size
        fig, ax = plt.subplots(figsize=(size_x, size_y), dpi=dpi)

        # To display an empty plot with a legend, we can use a trick by plotting empty data
        for comm, color in com_colors.items():
            ax.plot([], [], color=color, label=f"Com {comm}")

        # Add legend with horizontal orientation
        ax.legend(loc="center")

        # Turn off axis
        ax.axis("off")

        plt.savefig(f"{path}/{name}_legend.pdf")
        return com_colors

    @staticmethod
    # NOTE: this is also in GraphHelper but can't use that class as it will run in circular dependency
    def color_scale(color_scale='Sunset', num_points=100):

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
