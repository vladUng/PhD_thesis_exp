import os
import pickle

import pandas as pd

from .utilities import sankey_consensus_plot as sky
from .NetworkOutput import NetworkOutput


class ExperimentSet:
    def __init__(self, label, base_path, exp_path, mut_df, sel_sets=None, rel_path="", exp_type="PGCNA"):
        self.name = label
        self.path = os.path.join(base_path, exp_path)

        # read the master file and the experiments
        if exp_type == "PGCNA":
            raw_exps, self.master = ExperimentSet.read_experiments(self.path, sel_sets=sel_sets)
        else:
            raw_exps, self.master = ExperimentSet.read_experiments(self.path, sel_sets=sel_sets, exp_type="iNet")

        # check if everything is ok
        if len(raw_exps) != self.master.shape[0]:
            raise ImportError("Different length between master df and number of experiments")

        # unwrap experiments
        exps = {}
        for idx, exp_meta in self.master.iterrows():
            # just an extra check to see if things are ok
            if raw_exps[idx]["expName"] != exp_meta["expName"]:
                print("❗️ Check this experiment, there might be a problem in loading the data: {}".format(exp_meta["expName"]))

            # we need the base path for iNET only exps
            exp = NetworkOutput(raw_exps[idx]["graph"], exp_meta, mut_df, self.path, rel_path=rel_path, exp_type=exp_type, base_path=base_path)

            if "tf_list" in list(raw_exps[idx].keys()):
                exp.tf_list = raw_exps[idx]["tf_list"]

            key_val = exp.type
            exps[key_val] = exp
        self.exps = exps
        print("##### Experiment labels: ", exps.keys())

    @staticmethod
    def read_experiments(base_path, sel_sets=None, path="Stats/", exp_type="PGCNA"):
        """
        Each PCGNA run is stored in a master.csv. This function reads that .csv and returns a dataframe

        Args:
            path (_type_): _description_

        Returns:
            _type_: _description_
        """
        stats_path = base_path + path
        df = pd.read_csv(stats_path + "stats_master.tsv", sep="\t")
        # rename columns if exps are from iNet
        if exp_type == "iNet":
            remap_cols = {
                "exp_name": "expName",
                'modifier_type': "modifierType",
                "edges_pg": "edgesPG",
                "edges_sel": "edgesTF",
                "resolution_parameter": "resolution",
                "retain_f": "retainF",
                "genes_kept": "genesKept",
            }

            df.rename(columns=remap_cols, inplace=True)

        prcs_df = pd.DataFrame()
        exps = []
        for _, row in df.iterrows():
            # TODO the below code can be and needs to be improved!
            if sel_sets:
                toSkip = True
                for sel_set in sel_sets:
                    if row["genesKept"] == int(sel_set[0]) * 1000:  # 0 - num_genes, 1 - K
                        toSkip = False
                        break
                if toSkip:
                    continue

            picklePath = os.path.join(stats_path, row["expName"] + ".pickle")
            if os.path.exists(picklePath):
                with open(picklePath, "rb") as handle:
                    raw_exp = pickle.load(handle)

                    if exp_type == "iNet":
                        new_exp = {}
                        new_exp["expName"] = raw_exp["exp_name"]
                        new_exp["edgesPG"] = raw_exp["edges_pg"]
                        new_exp["edgesTF"] = raw_exp["edges_sel"]
                        new_exp["resolution"] = raw_exp["retain_f"]
                        new_exp["retainF"] = raw_exp["retain_f"]
                        new_exp["filesUsed"] = raw_exp["input_ge_file"]
                        new_exp["graph"] = raw_exp["graph"]
                        new_exp["genesKept"] = raw_exp["genes_kept"]

                        exps.append(new_exp)
                    else:
                        exps.append(raw_exp)
            else:
                print("No file at {}".format(picklePath))

            prcs_df = pd.concat([prcs_df, row], axis=1)

        return exps, prcs_df.transpose().reset_index()

    def get_exp_labels(self):
        return list(self.exps.keys())

    def get_exps(self):
        return self.exps.values()

    def generate_modConsMut(self, modifiers=[]):
        # The problem with the parallysed code is that it doesn't take the reference to the next code
        #  processes = []

        # for exp in self.exps.values():
        #     process = mp.Process(target=exp.get_ModCon_Mut, args=(modifiers,))
        #     processes.append(process)
        #     process.start()

        # # wait for all processes to complete
        # for process in processes:
        #     process.join()
        # # report that all tasks are completed
        # print('Done', flush=True)
        for exp in self.exps.values():
            # exp.get_ModCon_Mut(modifiers)
            exp.get_ModCon()

    def generate_Mevs(self, no_genes=100):
        for exp in self.get_exps():
            if "beta2" in exp.name or "beta3" in exp.name:
                continue

            sort_col = "ModCon_{}".format(exp.type)
            exp.mevsMut, _ = exp.get_mevs(exp.tpm_df, exp.modCons, sort_col=sort_col, num_genes=100)

    def combine_edges(self):
        combined_edges = pd.DataFrame()
        for idx, exp in enumerate(self.get_exps()):
            col = "Weight_{}".format(exp.type)
            exp.mergeSourceTarget(col_name=col)

            if idx == 0:
                first_exp = self.get_exp_labels()[0]
                combined_edges = self.exps[first_exp].edges_df[["Source", "Target", col]].copy(deep=True)
            else:
                combined_edges = pd.concat([combined_edges, exp.edges_df[col]], axis=1)

        # populate all the source-target based on the index. This is needed as the experiments have different number of edges

        source, target = [], []
        for index in combined_edges.index:
            split = index.split("-")
            source.append(split[0])
            target.append(split[1])
        combined_edges["Source"] = source
        combined_edges["Target"] = target

        self.combined_edges = combined_edges
        return combined_edges

    def combine_nodes(self, mod_type="Leiden"):
        combined_nodes, comm_class = pd.DataFrame(), []

        if mod_type == 'Leiden':
            for idx, exp in enumerate(self.get_exps()):
                col = f"Leiden_{exp.type}"
                dmy_df = exp.nodes_df.rename(columns={"Modularity Class": col})
                dmy_df[col] = exp.type + "_" + dmy_df[col].astype(str)
                if idx == 0:
                    combined_nodes = dmy_df.copy(deep=True)
                else:
                    combined_nodes = pd.concat([combined_nodes, dmy_df[col].astype(str)], axis=1)

                comm_class.append(col)

                combined_nodes.fillna("NA", inplace=True)
        else:
            for idx, exp in enumerate(self.get_exps()):
                col = f"max_b_{exp.type}"
                dmy_df = exp.nodes_df.rename(columns={"max_b": col})
                dmy_df[col] = exp.type + "_" + dmy_df[col].astype(str)
                if idx == 0:
                    combined_nodes = dmy_df[["count", "ctrl_tf", col]].copy(deep=True)
                else:
                    combined_nodes = pd.concat([combined_nodes, dmy_df[col].astype(str)], axis=1)

                comm_class.append(col)

            # For pyarrow compatibility
            combined_nodes.fillna(pd.NA, inplace=True)

        self.combined_nodes = combined_nodes
        return combined_nodes, comm_class

    def combine_network_metric(self, stats):
        metrics = []
        for metric in stats:
            metric_df = pd.DataFrame()
            for key, val in self.exps.items():
                metric_df = pd.concat(
                    [
                        metric_df,
                        val.graph_stats[metric].rename("{}_{}".format(key, metric)),
                    ],
                    axis=1,
                )

            metrics.append(metric_df)

        self.networks_metrics = pd.concat(metrics, axis=1)
        return self.networks_metrics

    def export_to_gephi(self, label=None, save=True):
        if label:
            self.exps[label].export_to_gephi(save=save)
        else:
            for exp in self.exps.values():
                exp.export_to_gephi(save=save)

    def export_to_graphml(self):
        for exp in self.get_exps():
            filename = exp.name
            exp.graph.write_graphml("{}/PGCNA/{}/Graph_{}.graphml".format(exp.exps_path, filename, filename))

    def align_communities(self, source_exp, target_exp, mod_type):
        combined_nodes = self.combined_nodes
        # generate the mapping of experiments
        if mod_type == "Leiden":
            reoder_cols = ["Leiden_{}".format(source_exp), "Leiden_{}".format(target_exp)]
        else:
            reoder_cols = ["max_b_{}".format(source_exp), "max_b_{}".format(target_exp)]
            combined_nodes[reoder_cols] = combined_nodes[reoder_cols].fillna("NA")
        meta, _ = sky.main(
            combined_nodes,
            reorder_cols=reoder_cols,
            title="Community comparison between " + ", ".join(reoder_cols),
            retMeta=True,
        )

        # pre-process the data
        sourceTargetDf, labels = meta[0], meta[1]
        # below are the mapped labels of the groups used in Sankey (k) and the unique number given by the algorithm
        mapped_values = {k: v for k, v in enumerate(labels)}

        sourceTargetDf["SourceLabel"] = sourceTargetDf["sourceID"].map(mapped_values)
        sourceTargetDf["TargetLabel"] = sourceTargetDf["targetID"].map(mapped_values)

        # 3 Gives the number of paramaters that are taken in account for each experiment
        sourceTargetDf["sourceExpType"] = ["_".join(val.split("_")[:3]) for val in sourceTargetDf["SourceLabel"].values]
        sourceTargetDf["targetExpType"] = ["_".join(val.split("_")[:3]) for val in sourceTargetDf["TargetLabel"].values]

        # It is important to sort this by the number of changes ("count)" so that we get the remapping where the communities matched the most
        dmy = sourceTargetDf[sourceTargetDf["sourceExpType"] == source_exp].sort_values(by="count", ascending=False).copy(deep=True)
        remap_dict, checked_communities = {}, []
        for _, row in dmy.iterrows():
            if row["TargetLabel"] == "NA":
                continue

            source = int(row["SourceLabel"].split("_")[-1])
            target = int(row["TargetLabel"].split("_")[-1])

            if target in checked_communities:
                continue

            checked_communities.append(target)
            remap_dict[int(target)] = int(source)

        # Add the missing values. Otherwise the replace function won't work properly
        max_label = self.exps[target_exp].nodes_df["Modularity Class"].max()
        for label in range(1, max_label + 1):
            if label not in remap_dict.keys():
                remap_dict[label] = label

        node_df = self.exps[target_exp].nodes_df.copy(deep=True)

        col_name = "Leiden_r_{}".format(target_exp)
        node_df = node_df.replace({"Modularity Class": remap_dict}).rename(columns={"Modularity Class": col_name})
        tst_df = pd.concat([combined_nodes, node_df[col_name]], axis=1)
        tst_df[col_name] = tst_df[col_name].fillna("NA")
        tst_df[col_name] = "r_" + target_exp + "_" + tst_df[col_name].astype(str)

        return tst_df[col_name], sourceTargetDf

    def run_comb_exp(self, no_genes, egTF=50, comps=[], mod_type='Leiden'):
        # get the combinations of experiments
        if not len(comps):
            exp_labels = self.get_exp_labels()
            comps = [
                ("standard_{}K_{}TF".format(no_genes, egTF), label) for label in exp_labels if ("standard" not in label) and ("{}K".format(no_genes) in label)
            ]

        # combine the nodes
        if not hasattr(self, "combined_nodes"):
            _, _ = self.combine_nodes(mod_type=mod_type)

        # remap for all experimends
        remap_nodes_df = self.combined_nodes.copy(deep=True)
        meta_dict = {}
        for source_exp, target_exp in comps:
            remap_series, sourceTarget = self.align_communities(source_exp, target_exp, mod_type=mod_type)
            remap_nodes_df = pd.concat([remap_nodes_df, remap_series], axis=1)
            meta_dict["{}-{}".format(source_exp, target_exp)] = sourceTarget.rename(columns={"count": "changes_comm"})

        return remap_nodes_df, meta_dict

    def comb_leiden_scores(self):
        leiden_scores = pd.DataFrame()
        for exp in self.get_exps():
            exp.leiden_top3["Exp"] = exp.type
            exp.leiden_top3["TF"] = exp.type.split("_")[-1].split("TF")[0]
            exp.leiden_top3["Modifier"] = exp.type.split("_")[0]
            if "Leiden Rank" not in exp.leiden_top3.columns:
                exp.leiden_top3.reset_index(names="Leiden Rank", inplace=True)

            exp.leiden_top3["AvgModuleNum"] = exp.leiden_top3["ModuleNum"].mean()
            leiden_scores = pd.concat([leiden_scores, exp.leiden_top3], axis=0)

        # # rename beta3 to beta
        leiden_scores.loc[leiden_scores["Modifier"] == "beta2", "Modifier"] = "Penalty"
        leiden_scores.loc[leiden_scores["Modifier"] == "norm3", "Modifier"] = "Reward"
        leiden_scores.loc[leiden_scores["Modifier"] == "standard", "Modifier"] = "Standard"

        return leiden_scores

    def avg_leiden_scores(self):
        leiden_means, all_leiden = [], pd.DataFrame()
        for exp in self.get_exps():
            lg_df = pd.read_csv(exp.pgcna_path + "/LEIDENALG/moduleInfoBest.txt", sep="\t")

            # processing the data
            tf = exp.type.split("_")[-1].split("TF")[0]
            modifier = exp.type.split("_")[0]
            lg_df["Exp"] = exp.type
            lg_df["TF"] = tf
            lg_df["Modifier"] = modifier

            all_leiden = pd.concat([all_leiden, lg_df.drop(columns=["ModuleSizes"])], axis=0)
            leiden_means.append((exp.type, tf, modifier, lg_df["ModularityScore"].mean(), lg_df["Mod#"].mean()))

        stats_df = pd.DataFrame(leiden_means, columns=["Exp", "TF", "Modifier", "Avg_ModularityScore", "Avg_Mod#"])
        stats_df.loc[stats_df["Modifier"] == "beta2", "Modifier"] = "Penalty"
        stats_df.loc[stats_df["Modifier"] == "norm3", "Modifier"] = "Reward"
        stats_df.loc[stats_df["Modifier"] == "standard", "Modifier"] = "Standard"
        stats_df["Avg_Mod#"] = stats_df["Avg_Mod#"].round()

        all_leiden.loc[all_leiden["Modifier"] == "beta2", "Modifier"] = "Penalty"
        all_leiden.loc[all_leiden["Modifier"] == "norm3", "Modifier"] = "Reward"
        all_leiden.loc[all_leiden["Modifier"] == "standard", "Modifier"] = "Standard"
        return stats_df, all_leiden

    def comp_modConRank(self):
        for exp in self.get_exps():
            for modCon, value in exp.modCons.items():
                dmy = value.sort_values(by=["ModCon_{}".format(exp.type)], ascending=False).reset_index(names="Id").iloc[:100]
                dmy["Rank"] = dmy.index + 1
                dmy.set_index("Id", inplace=True)
                exp.nodes_df.loc[exp.nodes_df["Modularity Class"] == modCon, "ModCon_Rank"] = dmy["Rank"]
                exp.nodes_df["ModCon_Rank"] = exp.nodes_df["ModCon_Rank"].fillna(0)

    ##### Clustering methods #####
    def get_clusters(self, vu_output=None, tf=10, showFigs=False):
        comb_std = self.exps["standard_4K_{}TF".format(tf)].run_clusters(label="std_tf{}".format(tf), show_figs=showFigs)
        comb_norm3 = self.exps["norm3_4K_{}TF".format(tf)].run_clusters(label="norm3_tf{}".format(tf), show_figs=showFigs)
        comb_norm3.drop(columns=["PC_1", "PC_2"], inplace=True)
        comb_beta = self.exps["beta_4K_{}TF".format(tf)].run_clusters(label="beta_tf{}".format(tf), show_figs=showFigs)
        comb_beta.drop(columns=["PC_1", "PC_2"], inplace=True)

        dfs = [comb_std, comb_norm3, comb_beta]
        if vu_output is not None:
            dfs.append(vu_output)

        comb_all = pd.concat(dfs, axis=1).dropna()
        return comb_all