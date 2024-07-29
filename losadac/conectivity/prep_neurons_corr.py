import glob
import os
import numpy as np
from typing import Dict, List
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.structures.population_data import PopulationData
from ephysvibe.trials import align_trials, select_trials
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm
from ephysvibe.stats import smetrics

seed = 1997

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import re
from collections import defaultdict
import itertools
import pickle


def get_neu_align(path, params, sp_sample=False):

    neu = NeuronData.from_python_hdf5(path)
    for it in params:
        sp, mask = neu.align_on(
            select_block=it["select_block"],
            select_pos=it["inout"],
            event=it["event"],
            time_before=it["time_before"],
            error_type=0,
        )
        endt = it["time_before"] + it["end"]
        stt = it["time_before"] + it["st"]
        setattr(neu, it["sp"], np.array(sp[:, :endt], dtype=it["dtype_sp"]))
        setattr(neu, it["mask"], np.array(mask, dtype=it["dtype_mask"]))
        setattr(neu, "st_" + it["event"] + "_" + it["inout"], np.array(stt, dtype=int))
        setattr(
            neu,
            "time_before_" + it["event"] + "_" + it["inout"],
            np.array(it["time_before"], dtype=int),
        )

    if ~sp_sample:
        setattr(neu, "sp_samples", np.array([]))

    return neu


def get_fr(neu, inout, sample, st, end, win=100):
    res = {}
    mask = getattr(neu, "mask_" + inout)
    sp = getattr(neu, "sample_on_" + inout)
    fr = firing_rate.moving_average(data=sp, win=win, step=1)[:, st:end]
    sample_id = neu.sample_id[mask]
    fr_samples = select_trials.get_sp_by_sample(fr, sample_id, [sample])
    frs = fr_samples[str(sample)]
    if frs.shape[0] < 40:
        return None
    nzeros = np.sum(np.sum(frs[:, 200:], axis=0) == 0)
    fr_mask = np.logical_or(
        np.mean(frs[:, 200:650]) * 1000 < 1, np.mean(frs[:, 650:]) * 1000 < 1
    )

    if np.logical_or(nzeros > 100, fr_mask):
        return None
    res["id"] = neu.cluster_group + "_" + str(neu.cluster_number)
    res["area"] = neu.area
    res["fr"] = frs
    return res


def sort_neu_by_area(neu1, neu2):
    idx_sorted = np.argsort([neu1["area"], neu2["area"]])
    n1_sorted = [neu1, neu2][idx_sorted[0]]
    n2_sorted = [neu1, neu2][idx_sorted[1]]
    return n1_sorted, n2_sorted


def compute_correlation(popu_fr, n1, n2):

    neu1, neu2 = sort_neu_by_area(popu_fr[n1], popu_fr[n2])

    trial_dur = neu1["fr"].shape[1]
    # corr = np.corrcoef(neu1['fr'].T,neu2['fr'].T)
    corr, _ = stats.spearmanr(neu1["fr"], neu2["fr"])
    if np.all(np.isnan(corr)):
        # corr = np.array([corr],dtype=np.float16)
        # p_val = np.array([p_val],dtype=np.float16)
        return None
    else:
        # p_val=p_val[:trial_dur,trial_dur:].astype(np.float16)
        corr = corr.round(decimals=3)[:trial_dur, trial_dur:].astype(np.float16)

        slices = [slice(0, 200), slice(200, 650), slice(650, None)]
        mean_corr = np.empty((len(slices), len(slices)))
        for i in range(len(slices)):
            for j in range(len(slices)):
                mean_corr[i, j] = np.nanmean(corr[slices[i], slices[j]])

        corr = mean_corr  # reduce_matrix(corr,2,2,np.nanmean)
    # corr = np.corrcoef(np.array(t_neus1).T,np.array(t_neus2).T)
    areas = neu1["area"] + "_" + neu2["area"]
    return {
        "areas": areas,
        "y": neu1["id"],
        "x": neu2["id"],
        "corr": corr,
        "fr_delay_y": np.mean(neu1["fr"][:, 650:]) * 1000,
        "fr_delay_x": np.mean(neu2["fr"][:, 650:], dtype=np.float16) * 1000,
        "fr_sample_y": np.mean(neu1["fr"][:, 200:650], dtype=np.float16) * 1000,
        "fr_sample_x": np.mean(neu2["fr"][:, 200:650], dtype=np.float16) * 1000,
    }  # ,'p':p_val}


# Define parameters
# Create one population per session
filepath = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/*/neurons/*neu.h5"
outputpath = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/correlation/"
path_list = glob.glob(filepath)
# Group paths per session
date_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
# Dictionary to group paths by date and time
grouped_paths = defaultdict(list)
# Iterate over the paths
for path in path_list:
    # Extract the date and time
    match = re.search(date_pattern, path)
    if match:
        date_time = match.group()
        grouped_paths[date_time].append(path)


# Parameters to preprocesss the spikes
params = [
    {
        "inout": "in",
        "sp": "sample_on_in",
        "mask": "mask_in",
        "event": "sample_on",
        "time_before": 300,
        "st": 0,
        "end": 1550,
        "select_block": 1,
        "win": 100,
        "dtype_sp": np.int8,
        "dtype_mask": bool,
    }
]
# TODO: add parameters to align also to test on

win = 100
inout = "in"
sample = 0
st = 100
end = 1150

for date_time in grouped_paths.keys():
    print(date_time)
    datepaths = grouped_paths[date_time]
    population = Parallel(n_jobs=-1)(
        delayed(get_neu_align)(neu, params) for neu in tqdm(datepaths)
    )

    comment = str(params)
    popu = PopulationData(population, comment=comment)
    popu.to_python_hdf5(outputpath + "corr_" + date_time + ".h5")

    for sample in [0, 11, 15, 51, 55]:
        print(sample)
        popu = PopulationData.from_python_hdf5(outputpath + "corr_" + date_time + ".h5")

        popu_fr = popu.execute_function(
            get_fr,
            win=win,
            inout=inout,
            sample=sample,
            st=st,
            end=end,
            n_jobs=-1,
            ret_df=False,
        )

        fr_dicts_only = [item for item in popu_fr if isinstance(item, dict)]

        if len(fr_dicts_only) == 0:
            continue

        numbers = range(len(fr_dicts_only))
        pairs = list(
            itertools.combinations(numbers, 2)
        )  # Compute each pair in parallel
        len(pairs)

        res = Parallel(n_jobs=-1)(
            delayed(compute_correlation)(fr_dicts_only, n1, n2)
            for n1, n2 in tqdm(pairs)
        )
        res_dicts_only = [item for item in res if isinstance(item, dict)]

        if len(res_dicts_only) == 0:

            continue

        with open(
            outputpath + "corr_s" + str(sample) + "_" + date_time + ".pkl", "wb"
        ) as fp:
            pickle.dump(res_dicts_only, fp)
