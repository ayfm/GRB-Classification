#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ayf

creates t1s and t100s files
"""

import os
import numpy as np
import pandas as pd
import data as dt
from env import dir_catalogs

def update_cols(frame):
    cols = [col.lower().strip() for col in frame.columns]
    frame.columns = cols
    return frame

def strip_cols(frame, cols=None):
    if not cols:
        cols = frame.columns
    for col in cols:
        frame.loc[:, col] = frame.loc[:, col].str.strip()
    return frame

def set_col_type(frame, col, type):
    # first, remove N/A values
    frame.loc[:, col] = frame.loc[:, col].apply(lambda _x: type(_x) \
                                                if not _x == "N/A" \
                                                else np.nan)
    return frame

def read_file(file_name, skiprows):
    # create file path
    fpath = os.path.join(dir_catalogs, "swift_data", file_name)
    # read data file
    frame = pd.read_csv(fpath, skiprows=skiprows, 
                      delimiter="|", skipinitialspace=True)
    # update col names
    frame = update_cols(frame)
    # update nan values in str typed columns
    if "comment" in frame.columns:
        frame.loc[:, "comment"] = frame.loc[:, "comment"].replace(np.nan, "")
    # return frame
    return frame

def save_frame(frame, file_name):
    # create file path
    fpath = os.path.join(dir_catalogs, file_name)
    # save as csv file
    frame.to_csv("{}.csv".format(fpath), sep=",", index=False)
    # save to excel
    frame.to_excel("{}.xlsx".format(fpath), index=False)


def check_dups(frame):
    n_obs = frame.shape[0]
    n_unique = frame["grbname"].unique().shape[0]
    print("Frame obs={}  unique={}".format(n_obs, n_unique))
    # print duplicated rows
    if n_obs != n_unique:
        dups = frame[frame.duplicated(subset=["grbname"], keep=False)] 
        print(dups)
        print(dups.loc[208] == dups.loc[209])
        print(dups.loc[210] == dups.loc[211]) 
        return dups
    
    return None


#%% summary_general.txt

# read summary file
summary = read_file("summary_general.txt", skiprows=22)
# strip columns
summary = strip_cols(summary, ["grbname", "t90"])
# how many obs? how many unique GRB name ?
dups = check_dups(summary)

# convert t90 values to numeric values
summary = set_col_type(summary, "t90", np.float)
# drop duplicates
summary.drop_duplicates(inplace=True)
assert summary["grbname"].duplicated().sum() == False

# keep only grb name and t90
summary = summary[["grbname", "t90"]]

#%% GRBlist_redshift_BAT.txt

# read redshift file
redshift = read_file("GRBlist_redshift_BAT.txt", skiprows=18)
# strip columns
redshift = strip_cols(redshift)
# how many obs? how many unique GRB name ?
dups = check_dups(redshift)

# convert redshift values to numeric values
for idx in redshift.index:
    # get redshift value
    z = redshift.loc[idx, "z"]
    z_comment = ""    
    # check index
    
    # if question mark
    if idx in [3, 47, 48, 51, 64, 79, 81, 91, 93, 96, 108, 111, 122, 299, 330, 
               191, 335, 374, 407]:
        z_comment = str(z)
        z = np.float(z.replace("(?)", ""))
    
    # if lower limit
    elif idx in [13, 21, 22, 45, 60, 54]:
        z_comment = "lower limit: {}".format(z)
        z = np.float(z.replace("<", "").replace("~", ""))
    
    # range
    elif idx in [31]:
        z_comment = "Range: {}.  Averaged.".format(z)
        p1, p2 = z.split("-")
        p1 = np.float(p1.strip())
        p2 = np.float(p2.strip())
        z = (p1+p2)/2
    
    # conflict ?
    elif idx in [66]:
        z_comment = "Two values: {}.  Averaged.".format(z)
        z = z.replace("(?)", "")
        p1, p2 = z.split("or")
        p1 = np.float(p1.strip())
        p2 = np.float(p2.strip())
        z = (p1+p2)/2
    
    # conflict ?
    elif idx in [80]:
        z_comment = "Multiple values: {}. ".format(z)
        z_comment += "The most decent one (Ref3: Knust et al. (2017)) is selected."
        z = 1.0
    
    # conflict ?
    elif idx in [188]:
        z_comment = "Multiple values: {}. ".format(z)
        z_comment += "The most decent one (Selsing et al. 2017) is selected."
        z = 2.211
    
    elif idx in [356]:
        z_comment = "Multiple values: {}".format(z)
        z_comment += "Selected Ref: Ref 2: Berger et al. GCN Circ. 5952"
        z = 0.111
    
    elif idx in [403]:
        z_comment = "Multiple values: {}".format(z)
        z_comment += "Selected Ref: https://sites.astro.caltech.edu/grbhosts/redshifts.html"
        z = 0.56
    
    else: 
        # try to convert it to the numeric value
        try:
            z = np.float(z)
        except:
            print("{} - {}".format(idx, z))
   
    # set redshift value and comment
    redshift.loc[idx, "z"] = z
    redshift.loc[idx, "z_comment"] = z_comment
    
# set dtype
redshift.loc[:, "z"] = redshift.loc[:, "z"].astype(np.float)
# drop cols
redshift = redshift[["grbname", "z", "z_comment"]]

#%% t1s_best_model.txt

# read redshift file
t1_best = read_file("t1s_best_model.txt", skiprows=7)
# strip columns
t1_best = strip_cols(t1_best)
# how many obs? how many unique GRB name ?
dups = check_dups(t1_best)

# rename cols
t1_best["t1s_best_model"] = t1_best["model"]
# drop cols
t1_best = t1_best[["grbname", "t1s_best_model"]]

#%% t100s_best_model.txt

# read file
t100_best = read_file("t100s_best_model.txt", skiprows=7)
# strip columns
t100_best = strip_cols(t100_best)
# how many obs? how many unique GRB name ?
dups = check_dups(t100_best)

# rename cols
t100_best["t100s_best_model"] = t100_best["model"]
# drop cols
t100_best = t100_best[["grbname", "t100s_best_model"]]

#%% t1s_summary_pow_parameters.txt

t1s_pow = read_file("t1s_summary_pow_parameters.txt", skiprows=20)
# strip columns
t1s_pow = strip_cols(t1s_pow, ["grbname", "alpha", "norm"])
# how many obs? how many unique GRB name ?
dups = check_dups(t1s_pow)

# drop duplicates
t1s_pow.drop_duplicates(inplace=True)
assert t1s_pow["grbname"].duplicated().sum() == False

# convert alpha values to float
t1s_pow = set_col_type(t1s_pow, "alpha", np.float)
# convert norm values to float
t1s_pow = set_col_type(t1s_pow, "norm", np.float)

# rename cols
t1s_pow["t1s_pl_alpha"] = t1s_pow["alpha"]
t1s_pow["t1s_pl_norm"] = t1s_pow["norm"]
# drop cols
t1s_pow = t1s_pow[["grbname", "t1s_pl_alpha", "t1s_pl_norm"]]

#%% t1s_summary_cutpow_parameters.txt

t1s_cutpow = read_file("t1s_summary_cutpow_parameters.txt", skiprows=24)
# strip columns
t1s_cutpow = strip_cols(t1s_cutpow, ["grbname", "alpha", "norm", "epeak"])
# how many obs? how many unique GRB name ?
dups = check_dups(t1s_cutpow)

# drop duplicates
t1s_cutpow.drop_duplicates(inplace=True)
assert t1s_cutpow["grbname"].duplicated().sum() == False

# convert alpha values to float
t1s_cutpow = set_col_type(t1s_cutpow, "alpha", np.float)
# convert norm values to float
t1s_cutpow = set_col_type(t1s_cutpow, "norm", np.float)
# convert e-peak values to float
t1s_cutpow = set_col_type(t1s_cutpow, "epeak", np.float)

# rename cols
t1s_cutpow["t1s_cpl_alpha"] = t1s_cutpow["alpha"]
t1s_cutpow["t1s_cpl_norm"] = t1s_cutpow["norm"]
t1s_cutpow["t1s_cpl_epeak"] = t1s_cutpow["epeak"]
# drop cols
t1s_cutpow = t1s_cutpow[["grbname", "t1s_cpl_alpha", "t1s_cpl_norm", 
                         "t1s_cpl_epeak"]]


#%% t100s_summary_pow_energy_fluence.txt
t100_pow_ef = read_file("t100s_summary_pow_energy_fluence.txt", 
                        skiprows=13)
# strip columns
t100_pow_ef = strip_cols(t100_pow_ef, ["grbname", "25_50kev", "100_150kev"])
# how many obs? how many unique GRB name ?
dups = check_dups(t100_pow_ef)
# drop duplicates
t100_pow_ef.drop_duplicates(inplace=True)
assert t100_pow_ef["grbname"].duplicated().sum() == False

# convert fluence(25_50) values to float
t100_pow_ef = set_col_type(t100_pow_ef, "25_50kev", np.float)
# convert norm values to float
t100_pow_ef = set_col_type(t100_pow_ef, "100_150kev", np.float)

# rename cols
t100_pow_ef["t100s_pl_fluence_25_50_kev"] = t100_pow_ef["25_50kev"]
t100_pow_ef["t100s_pl_fluence_100_150_kev"] = t100_pow_ef["100_150kev"]
# drop cols
t100_pow_ef = t100_pow_ef[["grbname", "t100s_pl_fluence_25_50_kev", 
                         "t100s_pl_fluence_100_150_kev"]]

#%% t100s_summary_cutpow_energy_fluence.txt
t100_cutpow_ef = read_file("t100s_summary_cutpow_energy_fluence.txt", 
                        skiprows=13)
# strip columns
t100_cutpow_ef = strip_cols(t100_cutpow_ef, ["grbname", "25_50kev", "100_150kev"])
# how many obs? how many unique GRB name ?
dups = check_dups(t100_cutpow_ef)
# drop duplicates
t100_cutpow_ef.drop_duplicates(inplace=True)
assert t100_cutpow_ef["grbname"].duplicated().sum() == False

# convert fluence(25_50) values to float
t100_cutpow_ef = set_col_type(t100_cutpow_ef, "25_50kev", np.float)
# convert norm values to float
t100_cutpow_ef = set_col_type(t100_cutpow_ef, "100_150kev", np.float)

# rename cols
t100_cutpow_ef["t100s_cpl_fluence_25_50_kev"] = t100_cutpow_ef["25_50kev"]
t100_cutpow_ef["t100s_cpl_fluence_100_150_kev"] = t100_cutpow_ef["100_150kev"]
# drop cols
t100_cutpow_ef = t100_cutpow_ef[["grbname", "t100s_cpl_fluence_25_50_kev", 
                         "t100s_cpl_fluence_100_150_kev"]]


#%% STOP
# raise Exception("STOP")


#%% merge t1s 

# create dataframe
merged_t1 = summary[["grbname"]].copy(deep=True)

# merge redshift values
merged_t1 = pd.merge(merged_t1, redshift, how="left", on="grbname")
# merge t90 values
merged_t1 = pd.merge(merged_t1, summary, how="left", on="grbname")
# merge best model
merged_t1 = pd.merge(merged_t1, t1_best, how="left", on="grbname")
# merge pow index params
merged_t1 = pd.merge(merged_t1, t1s_pow, how="left", on="grbname")
# merge cutpow index params
merged_t1 = pd.merge(merged_t1, t1s_cutpow, how="left", on="grbname")

# # save to csv
# merged_t1.to_csv("t1.csv", sep=",", index=False)
# # save to excel
# merged_t1.to_excel("t1.xlsx", index=False)
save_frame(merged_t1, "swift_t1s")

#%% merge t2s 

# create dataframe
merged_t100 = summary[["grbname"]].copy(deep=True)

# merge redshift values
merged_t100 = pd.merge(merged_t100, redshift, how="left", on="grbname")
# merge t90 values
merged_t100 = pd.merge(merged_t100, summary, how="left", on="grbname")
# merge best model
merged_t100 = pd.merge(merged_t100, t100_best, how="left", on="grbname")
# merge pow index params
merged_t100 = pd.merge(merged_t100, t100_pow_ef, how="left", on="grbname")
# merge cutpow index params
merged_t100 = pd.merge(merged_t100, t100_cutpow_ef, how="left", on="grbname")

# compute hardness
merged_t100["t100s_pl_hardness"] = merged_t100["t100s_pl_fluence_100_150_kev"]/merged_t100["t100s_pl_fluence_25_50_kev"]
merged_t100["t100s_cpl_hardness"] = merged_t100["t100s_cpl_fluence_100_150_kev"]/merged_t100["t100s_cpl_fluence_25_50_kev"]

# # save to csv
# merged_t100.to_csv("t100s.csv", sep=",", index=False)
# # save to excel
# merged_t100.to_excel("t100s.xlsx", index=False)
save_frame(merged_t100, "swift_t100s")









