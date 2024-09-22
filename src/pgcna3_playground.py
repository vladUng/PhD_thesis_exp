#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   pgcna3_playground.py
@Time    :   2024/09/22 15:54:03
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   Example script on how to run the pgcna3. Inspired from PGCNA work from Care et al. 2019; see https://github.com/medmaca/PGCNA
"""


import importlib
import pickle
import time as time

import plotly.express as px
from plotly.colors import hex_to_rgb
import multiprocess as mp


import pgcna3 as pg

DATA_PATH = "../data/"
RESULTS_PATH = "../results/integration_v3/tst_iNet/"


def create_colours_obj(path):
    colours_rgb = [hex_to_rgb(col) for col in px.colors.qualitative.Light24]

    colours_rgb.extend([hex_to_rgb(col) for col in px.colors.qualitative.Dark24])
    colours_rgb.extend([hex_to_rgb(col) for col in px.colors.qualitative.Plotly])

    colours_rgb.extend([hex_to_rgb(col) for col in px.colors.qualitative.G10])
    colours_rgb.extend([hex_to_rgb(col) for col in px.colors.qualitative.T10])
    colours_rgb.extend([hex_to_rgb(col) for col in px.colors.qualitative.Alphabet])

    pickle.dump(colours_rgb, open(path + "colours.pickle", "wb"), protocol=2)


importlib.reload(pg)


######## test for comparing with iNet_v2 ########
if 1:
    start_time = time.time()
    no_K_genes = 4
    dataset = "healthy"
    modifiers = ["standard"]  # "sigmoid_v1", "standard",   "beta", "norm3"]
    exps = []
    for edges in range(3, 6):
        for edgePTF in range(3, 6):
            for modifier in modifiers:
                expName = f"{modifier}_int_{dataset}_{no_K_genes}K_{edgePTF}TF_{edges}PG"
                args = pg.PGCNAArgs(f"{RESULTS_PATH}/", DATA_PATH, expName)

                args.keepBigF = False
                args.retainF = no_K_genes / 10
                args.edgePTF = edgePTF
                args.expName = expName
                args.modifier = modifier  # "norm2" #"standard" "norm"
                if modifier == "beta":
                    args.mutMetric = "beta"
                elif modifier == "norm3":
                    args.mutMetric = "norm3"
                elif modifier == "sigmoid_v1":
                    args.mutMetric = "sigmoid_v1"
                else:
                    args.mutMetric = "count_norm"

            args.setEdges(edges)
            pg.main(args)
            exps.append(args)

    pool = mp.Pool(mp.cpu_count())
    results = pool.map(pg.main, exps)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n\n######## PGCNA experiments took {execution_time:.4f} seconds to execute\n\n")
