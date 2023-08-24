#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sinem Sasmaz
"""
import pandas as pd
import numpy as np


def calculate_hardness(t100_data):
    
    grb_names = t100_data["grbname"].values
    redshifts = t100_data["z"].values
    best_models = t100_data["t100s_best_model"].values
    pl_fluence_25_50 = t100_data["t100s_pl_fluence_25_50_kev"].values
    pl_fluence_100_150 = t100_data["t100s_pl_fluence_100_150_kev"].values
    cpl_fluence_25_50 = t100_data["t100s_cpl_fluence_25_50_kev"].values
    cpl_fluence_100_150 = t100_data["t100s_cpl_fluence_100_150_kev"].values
    pl_hardness = t100_data["t100s_pl_hardness"].values
    cpl_hardness = t100_data["t100s_cpl_hardness"].values

    num_files = len(grb_names)

    hardness_ratio = np.zeros(num_files)

    for i in range(0, num_files):
        if best_models[i] == "PL":
            if not np.isnan(pl_hardness[i]):
                hardness_ratio[i] = pl_hardness[i]
        if best_models[i] == "CPL":
            if not np.isnan(cpl_hardness[i]):
                hardness_ratio[i] = cpl_hardness[i]
        if best_models[i] != "PL" or best_models[i] != "CPL":
            if not np.isnan(pl_hardness[i]):
                hardness_ratio[i] = pl_hardness[i]
            else:
                hardness_ratio[i] = cpl_hardness[i]
                # print(hardness_ratio[i])

    t100_data["hardness_ratio"] = hardness_ratio
    # print(np.isnan(hardness_ratio))
    return t100_data

    # nan values -- all values are empty
