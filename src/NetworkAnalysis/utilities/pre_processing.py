import numpy as np
import pandas as pd


def filter_data(df, th, at_least_good_cols=3, idx_cols=None):
    """Filter a given DataFrame by getting removing of the rows that have elements <= threshold.

    Args:
        df ([DataFrame]): The DataFrame from where we have to filter the data. This is has the genes as columns and samples as rows
        th ([float]): Threshold value of the unexpressed genes
        at_least_good_cols (int, optional): [The number of samples that fulfill the conditions]. Defaults to 3.

    Returns:
        [DataFrame]: DataFrame
    """
    if idx_cols is None:
        idx_cols = ["genes"]
        
    # eliminating the first column
    df_prcsd = df.drop(idx_cols, axis=1)
    # compute the selected genes
    selected_genes_idxs = df_prcsd[df_prcsd >= th].dropna(thresh=at_least_good_cols).index.values
    selected_genes = df_prcsd.iloc[selected_genes_idxs]
    # add the genes names back
    cols = [df.loc[selected_genes_idxs, idx_col] for idx_col in idx_cols]
    cols.append(selected_genes)
    selected_genes = pd.concat(cols, axis=1)

    # reset indexes
    selected_genes.reset_index(drop=True, inplace=True)
    return selected_genes


def select_genes(tpm_df, no_genes=3347, good_th=0.5, relative_selection=True, use_median=True):
    """
     It selects the most relative varied genes in the given DataFrame for the given number

    Args:
        tcga_tpm_df ([Dataframe]): The dataframe from where to select
        no_genes_selected (int, optional): [Genes to select]. Defaults to 3347.

    Returns:
        [type]: [description]
    """

    # remove all the genes w/ that have a lower expression value from `th` in `>10%` across the samples
    # dummy_df = pd.concat([pd.DataFrame(tcga_tpm_df["genes"]), pd.DataFrame(np.log2(tcga_tpm_df.iloc[:, 1:] + 1))], axis=1)
    # dummy_df = filter_data(dummy_df, th=np.log2(1.5), at_least_good_cols=dummy_df.shape[1] * 0.9, idx_cols=["genes"])

    dummy_df = np.log2(tpm_df.set_index("genes") + 1)
    at_least_good_cols = round(dummy_df.shape[1] * good_th)
    print(f"For th {good_th} ==> at least non-NAN values {at_least_good_cols}")

    # Do the filtering
    dummy_df = dummy_df[dummy_df >= np.log2(1.5)].dropna(thresh=at_least_good_cols).copy(deep=True)

    # acros samples
    print("####### Gene selection, num genes: {} #######".format(no_genes))
    if relative_selection:

        dmy = dummy_df.median(axis=1)
        if not use_median:
            dmy = dummy_df.mean(axis=1)

        print("The genes selected by the highest standard deviation/median ration.")
        dummy_df["ratio"] = dummy_df.std(axis=1) / dmy
    else:
        print("The genes selected by the highest standard deviation; approached used by Robertson et al.")
        dummy_df["ratio"] = dummy_df.std(axis=1)

    most_varied_genes = list(dummy_df.sort_values(by="ratio", ascending=False).iloc[:no_genes].index)
    return most_varied_genes


def remove_duplicates(df):
    """
       Removes the samples that are duplicates (i.e. resamples and are marked by having 01A#). We want to keep the ones with 01B
    Args:
        df (DataFrame): TCGA dataframe (columns - samples, rows - genes)

    Returns:
        list: The list of duplicates
    """
    # get the ones that are duplicated
    duplicates = ["-".join(col.split("-")[:-1]) for col in df if "01B" in col]
    # get with their pairs
    to_remove = []
    for col in df.columns:
        splits = "-".join(col.split("-")[:-1])
        if splits not in duplicates:
            continue

        if "01A" in col:
            to_remove.append(col)

    return to_remove


def create_map_cols(tcga_tpm_df):
    """
     Remove the -01B and -01A - this needs to be run only once

    Args:
        tcga_tpm_df ([DataFrame]): where to remove

    Returns:
        [Dict]: Dictionary of the old vs new col name
    """
    mapping_cols = {}
    mapping_cols["genes"] = "genes"
    for col in tcga_tpm_df.columns.values[1:]:
        mapping_cols[col] = "-".join(col.split("-")[:-1])
    return mapping_cols


def transp_df(df, samples_label="Samples"):
    """Transpose a given df and makes the original df's first row the header

    Args:
        df ([DataFrame]): Dataframe to transpose
        samples_label (str, optional): [description]. Defaults to "Samples".

    Returns:
        [DataFrame]: The transposed dataframe
    """
    df_t = df.transpose()
    df_t.columns = df.iloc[:, 0]
    df_t.drop(df_t.index[0], inplace=True)
    df_t.index.names = [samples_label]
    df_t.reset_index(inplace=True)
    df_t.rename_axis("Index", axis=1, inplace=True)
    return df_t


def prep_data(tcga_tpm_df, tcga_metadata_df, consensus_classifier, num_genes=3500, at_least_good=0.9, remap_cols=True):
    samples_to_remove = remove_duplicates(tcga_tpm_df)

    # drop from metadata, tpm
    tcga_tpm_df.drop(samples_to_remove, axis=1, inplace=True)

    tcga_metadata_df.drop(samples_to_remove, axis=1, inplace=True)

    ##### Apply pre-processing by Robertson et al. #####
    most_varied_genes_3k = select_genes(tcga_tpm_df, no_genes=num_genes, relative_selection=True, good_th=at_least_good)

    ##### The different genes #####
    selected_genes = set(most_varied_genes_3k)
    print(len(most_varied_genes_3k))

    ##### Sync DataFrames #####
    metadata_t = transp_df(tcga_metadata_df)
    samples_tpm = ["-".join(sample.split("-")[:-1]) for sample in metadata_t["Samples"].values]

    # remove the -01B and -01A - this needs to be run only once
    if remap_cols:
        mapping_cols = create_map_cols(tcga_tpm_df)
        working_tpm = tcga_tpm_df.copy(deep=True).rename(columns=mapping_cols)
    else:
        mapping_cols = list(tcga_tpm_df.columns)
        working_tpm = tcga_tpm_df.copy(deep=True)

    common_samples = set(samples_tpm) - set(["TCGA-BL-A0C8", "TCGA-BL-A13J", "TCGA-FJ-A3Z9", "TCGA-BL-A13I"])

    # don't modify the original Dataframes
    working_metadata = metadata_t.copy(deep=True)

    working_metadata["Samples"] = samples_tpm
    working_metadata = working_metadata[working_metadata["Samples"].isin(common_samples)]
    # working_tpm.rename(columns=mapping_cols, inplace=True)
    working_tpm = working_tpm[["genes"] + list(common_samples)]

    # select only the common samples
    raw_metadata_t = working_metadata[working_metadata["Samples"].isin(common_samples)]

    # prepare data
    data_tpm = working_tpm.loc[tcga_tpm_df["genes"].isin(selected_genes)].copy(deep=True)
    data_tpm.set_index("genes", inplace=True)

    #
    consensus_classifier = consensus_classifier[consensus_classifier["Sample"].isin(common_samples)].set_index("Sample")
    raw_metadata_t.set_index("Samples", inplace=True)
    # raw_metadata_t.loc["2019_consensus_classifier"] = consensus_classifier["consensusClass_remapTCGA"]
    # raw_metadata_t.loc[consensus_classifier.index, "2019_consensus_classifier"] = consensus_classifier["consensusClass_remapTCGA"]

    # raw_metadata_t = pd.concat([raw_metadata_t, consensus_classifier[["consensusClass_remapTCGA", "TCGA_Robertson2017"]]], axis=1).rename(columns={
    #     "consensusClass_remapTCGA": "2019_consensus_classifier",
    #      "TCGA_Robertson2017":"TCGA408_classifier"}
    #      )

    raw_metadata_t["TCGA408_classifier"] = consensus_classifier["TCGA_Robertson2017"]
    raw_metadata_t["TCGA_2017_AM_remap"] = consensus_classifier["consensusClass_remapTCGA"]
    raw_metadata_t.reset_index(inplace=True)

    return data_tpm, working_tpm, raw_metadata_t, selected_genes, list(common_samples)
