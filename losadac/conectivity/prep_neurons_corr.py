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


# Define parameters
# Create one population per session
filepath = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/*/neurons/*neu.h5"
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
    },
    {
        "inout": "out",
        "sp": "sample_on_out",
        "mask": "mask_out",
        "event": "sample_on",
        "time_before": 300,
        "st": 0,
        "end": 1550,
        "select_block": 1,
        "win": 100,
        "dtype_sp": np.int8,
        "dtype_mask": bool,
    },
]
# TODO: add parameters to align also to test on

date_time = "2022-12-14_10-50-49"
datepaths = grouped_paths[date_time]
population = Parallel(n_jobs=-1)(
    delayed(get_neu_align)(neu, params) for neu in tqdm(datepaths)
)

comment = str(params)
popu = PopulationData(population, comment=comment)
popu.to_python_hdf5("res/test.h5")

popu = PopulationData.from_python_hdf5("corr_" + date_time + ".h5")

numbers = range(len(popu.population))
pairs = list(itertools.combinations(numbers, 2))  # Compute each pair in parallel


def get_fr(neu, inout, sample, st, end, win=100):
    res = {}
    mask = getattr(neu, "mask_" + inout)
    sp = getattr(neu, "sample_on_" + inout)
    fr = firing_rate.moving_average(data=sp, win=win, step=1)[:, st:end]
    sample_id = neu.sample_id[mask]
    fr_samples = select_trials.get_sp_by_sample(fr, sample_id, [sample])
    res["id"] = neu.cluster_group + "_" + str(neu.cluster_number)
    res["area"] = neu.area
    res["fr"] = fr_samples[str(sample)]
    return res


win = 100
inout = "in"
sample = 11
st = 100
end = 1150

popu_fr = popu.execute_function(
    get_fr, win=win, inout=inout, sample=sample, st=st, end=end, n_jobs=-1, ret_df=False
)


def compute_correlation(popu_fr, n1, n2):
    neu1 = popu_fr[n1]
    neu2 = popu_fr[n2]
    trial_dur = neu1["fr"].shape[1]
    corr, p_val = stats.spearmanr(neu1["fr"], neu2["fr"])
    p_val = p_val[:trial_dur, trial_dur:]
    # corr = np.corrcoef(np.array(t_neus1).T,np.array(t_neus2).T)
    corr = corr.round(decimals=3)[:trial_dur, trial_dur:].astype(np.float16)
    areas = neu1["area"] + "_" + neu2["area"]
    return {"areas": areas, "y": neu1["id"], "x": neu2["id"], "corr": corr, "p": p_val}


res = Parallel(n_jobs=-1)(
    delayed(compute_correlation)(popu_fr, n1, n2) for n1, n2 in tqdm(pairs)
)
with open("./corr_" + date_time + ".pkl", "wb") as fp:
    pickle.dump(res, fp)
