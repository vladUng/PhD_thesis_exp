import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr

from .utilities import sankey_consensus_plot as sky
from .utilities.helpers import save_fig
from .ExperimentSet import ExperimentSet
from .NetworkOutput import NetworkOutput


class NetworkComp:

    source_exp: NetworkOutput
    target_exp: NetworkOutput

    source: str
    target: str
    name: str

    comp_df: pd.DataFrame

    def __init__(self, exp_set: ExperimentSet, no_genes, source, target, mod_type = 'Leiden'):
        exp_labels = exp_set.get_exp_labels()

        # backwards compatibility
        if mod_type == "Leiden":
            if "_".join(source.split("_")[-3:]) not in exp_labels:
                raise ValueError("Source exp {} not in experiments {}".format(source, exp_set.name))

            if "_".join(target.split("_")[-3:]) not in exp_labels:
                raise ValueError("Target exp {} not in experiments {}".format(source, exp_set.name))

        # This may fail with non-sbm exps if so un-comment the next lines
        egTF = int(source.split("TF")[0].split("_")[-1])
        # if len(source.split("_")[-1]) > 3:
        #     egTF = int(source.split("_")[-1][:2])
        # else:
        #     egTF = int(source.split("_")[-1][:1])

        remap_nodes_df, _ = exp_set.run_comb_exp(no_genes=no_genes, egTF=egTF, comps=[(source, target)], mod_type=mod_type)
        remap_nodes_df.rename(columns={"count": "mut_count"}, inplace=True)

        # cols in nodes_df
        if "sbm" in mod_type: 
            source_col = f"max_b_{source}"
            target_col = f"max_b_{target}"
        else:
            source_col = f"Leiden_{source}"
            target_col = f"Leiden_{target}"

        # pre-process the data
        remap_df = remap_nodes_df.loc[:, ["mut_count", source_col, target_col]]
        remap_df["Source"] = [val.split("_")[-1] if "NA" not in val else np.nan for val in remap_df[source_col].values]
        remap_df["Target"] = [val.split("_")[-1] if "NA" not in val else np.nan for val in remap_df[target_col].values]

        # fill the NA with 0
        remap_df.loc[remap_df["mut_count"] == "NA", "mut_count"] = 0

        remap_df = remap_df[~remap_df["Source"].isnull()]

        self.name = "{} vs. {}".format(source, target)
        self.no_genes = no_genes
        self.comp_df = remap_df.copy(deep=True)
        # the above 2 may need renaming
        self.source = source_col
        self.target = target_col

        self.source_exp = exp_set.exps[source]
        self.target_exp = exp_set.exps[target]
        # NOTE: The above assignment work for SBM Experiments and it should work for Leiden as well, but un-comment the below code if it doesn't
        # self.source_exp = exp_set.exps[("_".join(source.split("_")[-3:]))]
        # self.target_exp = exp_set.exps[("_".join(target.split("_")[-3:]))]

    def compute_comm_mut_stats(self):
        dmy = {}

        if self.comm_stats == None:
            df = self.comp_df.copy(deep=True)
            for comm, orig_num in df["Source"].value_counts().items():
                stats = []
                diff = abs(orig_num - df[df["Target"] == comm].shape[0])

                sel_mut_orig = df[df["Source"] == comm]["mut_count"]
                sel_mut_changes = df[(df["Source"] == comm) & (df["Target"] != comm)]["mut_count"]

                # orig, changes, ratio
                stats.append(orig_num)
                stats.append(diff)
                stats.append(round(diff / orig_num, 4))

                # Mut stats
                stats.append(sel_mut_orig.mean())
                stats.append(sel_mut_changes.mean())

                stats.append(sel_mut_orig.median())
                stats.append(sel_mut_changes.median())

                dmy[comm] = stats

            comm_stats = (
                pd.DataFrame.from_dict(
                    data=dmy,
                    orient="index",
                    columns=[
                        "Orig_num",
                        "Changes_num",
                        "Ratio",
                        "Mut_orig_mean",
                        "Mut_changes_mean",
                        "Mut_orig_median",
                        "Mut_changes_median",
                    ],
                )
                .reset_index()
                .rename(columns={"index": "comm"})
            )
            self.comm_stats = comm_stats

    def com_mut_distrib(self, include_source=False, toSave=False, path=None, binarySource=False, annotations=None, ann_add=[], ann_rm=[], ann_chg=[]):
        if not include_source:
            df = self.comp_df[self.comp_df["Target"] != self.comp_df["Source"]].copy(deep=True)
        else:
            df = self.comp_df.copy(deep=True)

        title = "{}. Gene comm changes and mut counts".format(self.name)
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Safe

        # This enables to show on the histogram if the data came from other communities (any) or itself
        color_map = None
        if binarySource:
            df.loc[df["Target"] != df["Source"], "Source"] = 0
            # assign blue to incoming genes
            color_map = {0: colors[0]}
            colors.pop(0)

        order = sorted(df["Source"].astype(int).unique())
        fig = px.strip(
            df.reset_index(),
            x="Target",
            y="mut_count",
            color="Source",
            height=600,
            category_orders={"Target": order},
            title=title,
            color_discrete_sequence=colors,
            color_discrete_map=color_map,
            hover_data=["Id", "mut_count", "Source", "Target"],
        )
        fig.update_layout(
            legend=dict(
                orientation="v",
                title="Source",
                yanchor="middle",
                y=0.70,
                xanchor="center",
                x=0.96,
                bgcolor="rgba(0,0,0,0)",
            ),
        )

        if annotations:
            marker_genes, num_genes = [], 20
            marker_genes.extend(df.sort_values(by=["mut_count"], ascending=False).index.values[:num_genes])

            if len(ann_add) > 0:
                marker_genes.extend(ann_add)
            if len(ann_rm) > 0:
                marker_genes = set(marker_genes) - set(ann_rm)

            counter = 0
            offset = 10
            for idx, row in df.loc[df.index.isin(marker_genes)][["Target", "Source", "mut_count"]].iterrows():
                align_left = True
                if counter % 2 == 0:
                    shift = offset
                    align_left = False
                else:
                    shift = -offset

                if idx in ann_chg:
                    if align_left:
                        shift = -offset * 5.5
                    else:
                        shift = +offset * 5.5

                align = "left" if align_left else "right"
                x = str(int(row["Target"]) - 1)

                if row.name in ["CDKN2A", "FAT2"]:
                    align = "left"
                    shift = -offset * 7
                if row.name in ["COL7A1"]:
                    shift = offset

                fig.add_annotation(x=x, y=row["mut_count"], text=idx, showarrow=False, valign="top", align=align, xshift=shift, yshift=0)

        if toSave and path:
            fig_name = "box_{}_{}".format(self.source_exp.type, self.target_exp.type)
            save_fig(name=fig_name, fig=fig, base_path=path)
        else:
            return fig

    def membership_change(self, include_source=False, toSave=False, path=None):
        if not include_source:
            df = self.comp_df[self.comp_df["Target"] != self.comp_df["Source"]].copy(deep=True)
        else:
            df = self.comp_df.copy(deep=True)

        title = "{}. Gene comm changes".format(self.name)

        order = sorted(df["Source"].unique())

        fig = px.histogram(
            df,
            x="Target",
            color="Source",
            height=600,
            category_orders={"Target": order},
            title=title,
        )
        # fig.update_layout(
        #     legend=dict(
        #         orientation="v",
        #         title="Source",
        #         yanchor="middle",
        #         y=0.75,
        #         xanchor="center",
        #         x=0.96,
        #         bgcolor="rgba(0,0,0,0)",
        #     ),
        # )
        if toSave and path:
            fig_name = "memberShip_{}_{}".format(self.source_exp.type, self.target_exp.type)
            save_fig(name=fig_name, fig=fig, base_path=path)
        else:
            return fig

    def plt_median_ge(self, map_names: dict, path = None):

        comp_dict = self.comp_ge_comm()

        figs = []
        for key, df in comp_dict.items():
            df["Comm_num"] = df["Comm"].str.split("_", expand=True)[1]
            title = "{}. Median Gene Expression across communities.".format(map_names[key])
            fig_name = "P0_{}_med".format(key)
            fig = px.box(df, x="Comm_num", y="Median", color="Comm_num", title=title, points="all")
            fig.update_layout(showlegend=False, xaxis_title="Community", yaxis_title="Median Gene Expression")
            fig.update_traces(boxmean=True)
            figs.append(fig)

            if path != None:
                fig.update_layout(
                    title = "",
                    xaxis=dict(tickfont=dict(size=14)),
                    yaxis=dict(tickfont=dict(size=14), title="Median TPM"),
                    font=dict(size=16),
                )
                save_fig(name=fig_name, fig=fig, base_path=path, width=1200, height=400, margin=0.02)

        return figs

    def sankey_plot(self, df=None, drop_na=True, toSave=False, path=None, source_label=None, target_label=None):
        reoder_cols = [format(self.source), format(self.target)]
        if df is None:
            df = self.comp_df.copy(deep=True)

        # remove the genes added by 7K exp
        if drop_na:
            sel_idxs = df[reoder_cols][df[format(self.source)] != "NA"].dropna().index
            df = df.loc[sel_idxs]


        # This
        if (source_label != None) and (target_label != None):

            # Rename the rows values
            row_vals = self.source.replace("Leiden_", "")
            df[self.source] = df[self.source].str.replace(row_vals, source_label)

            row_vals = self.target.replace("Leiden_", "")
            df[self.target] = df[self.target].str.replace(row_vals, target_label)

            # Rename the columns
            df = df.rename(columns={self.target: target_label, self.source: source_label})
            reoder_cols = [source_label, target_label]

        if toSave and path:
            _, fig = sky.main(df, reorder_cols=reoder_cols, title="Community comparison between " + ", ".join(reoder_cols), retMeta=True)
            fig_name =f"sankey_{self.source_exp.type}_{self.target_exp.type}"
            save_fig(name=fig_name, fig=fig, base_path=path)
        else:

            _, fig = sky.main(df, reorder_cols=reoder_cols, title="Community comparison between " + ", ".join(reoder_cols), retMeta=True)
            return fig

    def find_mut_stats(self, th=1, col="Source"):
        dmy = []
        df = self.comp_df.copy(deep=True)
        for comm in df[col].unique():
            sel_comm = df[df[col] == comm]
            mut_genes = sel_comm[sel_comm["mut_count"] >= th].shape[0]
            if sel_comm.shape[0] != 0:
                percent = round(mut_genes / sel_comm.shape[0] * 100, 3)
            else:
                percent = 0

            dmy.append([comm, mut_genes, sel_comm.shape[0], percent])
        return pd.DataFrame(dmy, columns=[col, "Mut_genes", "All", "Percent"]).sort_values(by="Percent", ascending=False).set_index(col)

    def comb_mut_stats(self, start=0, end=60, direction="Source", log10=False):
        ths = list(range(start, end + 1, 1))

        all_dfs = pd.DataFrame()
        for th in ths:
            new_col = "{}".format(th)
            mut_target = self.find_mut_stats(th=th, col=direction).rename(columns={"Mut_genes": new_col})
            if th == start:
                all_dfs = mut_target[["All", new_col]].copy(deep=True)
            else:
                all_dfs = pd.concat([all_dfs, mut_target[new_col]], axis=1)

        dmy_df = all_dfs.melt(ignore_index=False, var_name="Mut_th", value_name="# genes").reset_index()
        if log10:
            dmy_df["# genes"] = np.log10(dmy_df["# genes"] + 1)
        return dmy_df, all_dfs

    def plot_mut_evo(self, df, direction="Source"):
        title = "{} comm. Mutations count in each community as we increase Mut_count threshold".format(direction)
        fig = px.line(
            df,
            x="Mut_th",
            y="# genes",
            color=direction,
            title=title,
            markers=True,
            color_discrete_sequence=px.colors.qualitative.G10)
        fig.update_layout(
            legend=dict(
                orientation="h",
                title=direction,
                yanchor="middle",
                y=0.95,
                xanchor="center",
                x=0.6,
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        return fig

    def comp_ge_comm(self):
        ret_dict = {}
        source_mod = "_".join(self.source.split("_")[-3:])
        target_mod = "_".join(self.target.split("_")[-3:])

        ret_dict[source_mod] = self.source_exp.ge_comm()
        ret_dict[target_mod] = self.target_exp.ge_comm()

        return ret_dict

    def diff_in_com(self):
        remap_cols = {}
        for src_com in self.comp_df["Source"].unique():
            trgt_com = self.comp_df[self.comp_df["Source"] == src_com]["Target"].values[0]
            remap_cols[src_com] = trgt_com

        src_cols = [int(val) for val in remap_cols.values() if val is not np.nan]
        src_cols.sort()
        diff = set(self.target_exp.modCons.keys()) - set(src_cols)
        print("{} has the following communities in addition to {}: {}".format(self.target_exp.type, self.source_exp.type, diff))

    def plot_network_stats(self, nodes_df_1: pd.DataFrame, nodes_df_2: pd.DataFrame, label_1: str, label_2: str, path = None):

        # Add the stats
        graph_stats_1: pd.DataFrame = self.source_exp.compute_graph_stats()
        graph_stats_2: pd.DataFrame = self.target_exp.compute_graph_stats()

        graph_stats_1["mut_count"] = nodes_df_1["count"]
        graph_stats_2["mut_count"] = nodes_df_2["count"]

        graph_stats_1["TF"] = nodes_df_1["TF"]
        graph_stats_2["TF"] = nodes_df_2["TF"]

        # graph_stats_1["IVI"] = nodes_df_1["IVI"]
        # graph_stats_2["IVI"] = nodes_df_2["IVI"]

        # Define the metrics
        metrics = [
            # "degree_o",
            "degree_t",
            # "degree_w",
            "betweenness",
            "closeness",
            "katz",
            "pageRank",
            # "IVI",
            'mut_count'
            # "TF",
        ]

        # Set the number of rows and columns for the subplot
        num_cols = 2
        num_rows = round(len(metrics) / num_cols)
        nbins = 150

        # Create a figure and subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))

        # Flatten the axes array
        axes = axes.flatten()

        # Iterate over the metrics and plot the histograms
        for i, metric in enumerate(metrics):
            # The first plot is clearer than the second - swap them as preferred
            axes[i].hist(graph_stats_1[metric], bins=nbins, alpha=0.5, label=label_1, log=True)

            axes[i].hist(graph_stats_2[metric], bins=nbins, alpha=0.5, label=label_2,  log=True)

            # Set tick parameters
            axes[i].tick_params(axis='both', which='major', labelsize=20)
            
            if metric in ['degree_t', 'betweenness', 'IVI']:
                axes[i].xaxis.set_major_locator(plt.MaxNLocator(15))

            # Set the title and labels for the subplot
            axes[i].set_xlabel(metric, fontsize=21)
            # axes[i].set_ylabel('count',fontsize=21)

            # Add a legend to the subplot
            axes[i].legend(prop={'size': 16})
    

        # Adjust the spacing between subplots
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)

        if path:
            # Save the figure
            path = f"{path}/net_metrics_{label_1}_{label_2}.png"
            fig.savefig(path, dpi=300)


        # Show the plot
        return fig, graph_stats_1, graph_stats_2
    

    @staticmethod
    def plot_corr_matrix_coms(coms, height=700, title="Corr matrix", hide_up=False):

        # corr, pvalue = spearmanr(tum.exps["standard_4K_10TF"].mevsMut)
        coms.index.names = ["Mut_th"]
        dmy = coms.rename(columns={"All_genes": 0}).T
        dmy.columns = ["Com_{}".format(int(val)) for val in sorted(list(map(int, dmy.columns)))]

        corr, pvalue = spearmanr(dmy)

        # replace all the upper triangle values with None. So that we don't display the entire heatmap
        if hide_up:
            N = corr.shape[0]
            corr = [[corr[i][j] if i > j - 1 else None for j in range(N)] for i in range(N)]
            pvalue = [[pvalue[i][j] if i > j - 1 else None for j in range(N)] for i in range(N)]

        cols = dmy.columns
        corr_df = pd.DataFrame(corr, index=cols, columns=cols)

        # corr_df = dmy.corr(method="spearman")
        fig = px.imshow(corr_df, text_auto=True, height=height, title=title, aspect="auto")  # contrast_rescaling="infer" )

        return fig
