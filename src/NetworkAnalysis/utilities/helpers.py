#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   utilities.py
@Time    :   2023/06/08 08:55:34
@Author  :   Vlad Ungureanu
@Version :   1.0
@Contact :   vlad.ungureanu@york.ac.uk
@Desc    :   This is a strip down version of utilities from Subtypes paper.
'''
import numpy as np
import pandas as pd
import plotly.express as px
from lifelines import KaplanMeierFitter


def save_fig(name, fig, width=None, height=None, scale=None, base_path=None, margin=None):
    """ Saves a figure to the base path or othwerise to th specified path

    Args:
        name ([type]): Name of the figure
        fig ([type]): Figure object
        width ([type], optional): Default 1280.
        height ([type], optional): Default 720.
        scale ([type], optional): Default 2.
        base_path ([type], optional): Default root folder.
    """

    default_width, default_height, default_scale = 1280, 720, 2
    default_base_path = "./data/figures"

    # override arguments if needed
    if width != None:
        default_width = width
    if height != None:
        default_height = height
    if scale != None:
        default_scale = scale
    if base_path != None:
        default_base_path = base_path

    if margin != None:
        margin_x = width * margin
        margin_y = height * margin

        fig.update_layout(
            margin=dict(l=margin_x, r=margin_x, t=margin_y+10, b=0)
        )

    fig.write_image(default_base_path + name + ".png",width=default_width, height=default_height, scale=default_scale)


def survival_plot(df, df_meta, classifier, selected_labels=None, color_map=None):

    dmy_meta = df_meta.loc[df_meta.index.isin(df.index.values)][["days_to_last_follow_up", "days_to_death"]]
    df = pd.concat([df, dmy_meta], axis=1)

    labels = list(df[classifier].unique())
    if selected_labels:
        labels = selected_labels

    models = []
    all_df = []
    for label in labels:
        kmf = KaplanMeierFitter()

        # process the data

        # older version for non-pyarrow format
        # dmy = df[df[classifier] == label][["days_to_last_follow_up", "days_to_death"]].replace("--", 0).astype(int)

        # to make it compatabile with pyarrow
        dmy = df[df[classifier] == label][["days_to_last_follow_up", "days_to_death"]].astype(str).replace("--", pd.NA).fillna(0).astype(float)

        dmy["last_contact"] = dmy[["days_to_last_follow_up", "days_to_death"]].max(axis=1)
        dmy["dead"] = np.where(dmy["days_to_death"] > 0, True, False)

        kmf.fit(dmy["last_contact"], event_observed=dmy["dead"], label=[label])
        models.append(kmf)

        # prepare df for plotting and change the scale from days to month
        dmy_df = kmf.survival_function_.copy(deep=True)

        disease = [dmy_df.columns[0][0]] * dmy_df.shape[0]
        dmy_df.rename(columns={dmy_df.columns[0][0]: "chance"}, inplace=True) 
        dmy_df.reset_index(inplace=True)
        chance = [value[0] for value in dmy_df["chance"].astype(float).values] 
        timeline = [value[0] /30 for value in dmy_df["timeline"].astype(int).values]

        dmy_df = pd.DataFrame()
        dmy_df["disease"] = disease
        dmy_df["timeline"] = timeline
        dmy_df["chance"] = chance
        all_df.append(dmy_df)

    test = pd.concat(all_df[:]).reset_index(drop=True)
    test["disease"] = test["disease"].astype(str)

    fig = px.line(test, x="timeline", y="chance", color="disease", markers=True, line_shape="hv", color_discrete_map = color_map) 
    fig.update_yaxes(title_text="Survival rate")
    fig.update_xaxes(title_text="Time (months)")
    fig.update_xaxes(range=[-1, 60])
    return fig

def survival_comp(df, df_meta, classifier_1, classifier_2, selected_labels_1 = None, selected_labels_2 = None,  color_map=None):

    # survival_comp(tum_h_tf10_cs.drop(columns=["days_to_last_follow_up", "days_to_death"]),  vu_output, classifier_1 = model, classifier_2="KMeans_labels_6", color_map=None)

    dmy_meta = df_meta.loc[df_meta.index.isin(df.index)][["days_to_last_follow_up", "days_to_death"]]
    df = pd.concat([df, dmy_meta], axis=1)
    labels = list(df[classifier_1].unique())
    if selected_labels_1:
        labels = selected_labels_1

    if selected_labels_2:
        labels.extend(selected_labels_2)
    else:
        labels.extend(list(df[classifier_2].unique()))

    labels = list(set(labels)) #eliminate duplicates

    models = []
    all_df = []
    for label in labels:
        kmf = KaplanMeierFitter()

        dmy = pd.DataFrame()
        # process the data
        if df[df[classifier_1] == label].shape[0]: 
            dmy = df[df[classifier_1] == label][["days_to_last_follow_up", "days_to_death"]] 

        if df[df[classifier_2] == label].shape[0]:
            dmy = df[df[classifier_2] == label][["days_to_last_follow_up", "days_to_death"]]

        # older version
        # dmy = dmy.replace("--", 0).astype(int)

        # to make it compatabile with pyarrow
        dmy = dmy.replace("--", pd.NA).fillna(0).astype(float)

        # dmy_1 = df[df[classifier_2] == label][["days_to_last_follow_up", "days_to_death"]].replace("--", 0).astype(int)

        dmy["last_contact"] = dmy[["days_to_last_follow_up", "days_to_death"]].max(axis=1)
        dmy["dead"] = np.where(dmy["days_to_death"] > 0, True, False)

        kmf.fit(dmy["last_contact"], event_observed=dmy["dead"], label=[label])
        models.append(kmf)

        # prepare df for plotting and change the scale from days to month
        dmy_df = kmf.survival_function_.copy(deep=True)

        disease = [dmy_df.columns[0][0]] * dmy_df.shape[0]
        dmy_df.rename(columns={dmy_df.columns[0][0]: "chance"}, inplace=True) 
        dmy_df.reset_index(inplace=True)
        chance = [value[0] for value in dmy_df["chance"].astype(float).values] 
        timeline = [value[0] /30 for value in dmy_df["timeline"].astype(int).values]

        dmy_df = pd.DataFrame()
        dmy_df["disease"] = disease
        dmy_df["timeline"] = timeline
        dmy_df["chance"] = chance
        all_df.append(dmy_df)

    test = pd.concat(all_df[:]).reset_index(drop=True)
    test["disease"] = test["disease"].astype(str)

    fig = px.line(test, x="timeline", y="chance", color="disease", markers=True, line_shape="hv", color_discrete_map = color_map) 
    # fig.update_traces(line_width=7)

    fig.update_yaxes(title_text="Survival rate")
    fig.update_xaxes(title_text="Time (months)")

    fig.update_xaxes(range=[-1, 60])
    return fig 
