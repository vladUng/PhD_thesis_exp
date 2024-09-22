#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   process_TPMs.py
@Time    :   2022/11/02 11:24:45
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Script used to replace the . in the genes with _ so that I can use the GEPHI script language. This used the nodes or edges w/ the following conventions: vGENE_NAME thus having . in a nambe is incompatible
'''


import pandas as pd

filename = "TPMs_selected_genes_v2.tsv"
path = "../../data/"

tpm_df = pd.read_csv(path + filename, sep="\t")

print("Are there any genes with . in their names? ", tpm_df[tpm_df["genes"].str.contains("\.")].shape[0])

for row, item in tpm_df[tpm_df["genes"].str.contains("\.")].iterrows():
    tpm_df.loc[tpm_df.index == row, 'genes'] = item["genes"].replace(".", "_")


print("Are there any genes with . in their names? ", tpm_df[tpm_df["genes"].str.contains("\.")].shape[0])

tpm_df.to_csv(path + filename, sep ="\t", index=False)