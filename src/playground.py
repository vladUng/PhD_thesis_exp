#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   playground.py
@Time    :   2024/01/15 18:32:05
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Test, experiment the iNet package
'''

from iNet.main import iNET

input_folder = "data/"
output_folder = 'results/test/'

# For debugging
# input_folder = "../data/"
# output_folder = '../results/test_perf/'
genes_kept = 5000
edges_pg, edges_sel, modifier = 3, 6, "sigmoid"

sbm_method = "hsbm" # sbm or hsbm
label = 'ctrl_1'
name = f"t_{label}_{sbm_method}_{modifier}_iNet_{edges_sel}TF"
graph_type = "gt"

ge_file = 'healthy_data_10K_gc42_v3.tsv'
# ge_file = 'TPMs_selected_genes_v3_13k_gc42.tsv'

sbm_config = {
    'method': sbm_method,
    'n_iter': 1000,
    'mc_iter': 10,
    'deg_cor': True
}

test_inet = iNET(exp_name=name, ge_file=ge_file, input_folder=input_folder, output_folder=output_folder, edges_pg=edges_pg, edges_sel=edges_sel, modifier_type=modifier, genes_kept=genes_kept, graph_type=graph_type, sbm_config=sbm_config)

test_inet.run()

