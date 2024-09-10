# from preproc_tools import get_fr_by_sample, to_python_hdf5
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import json
from pathlib import Path
import h5py
from sklearn.svm import SVC
from scipy.spatial.distance import pdist
import pickle
import pandas as pd
from datetime import datetime
from ephysvibe.structures.population_data import PopulationData
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.structures.results import Results
from ephysvibe.trials import select_trials
from ephysvibe.trials.spikes import firing_rate
from typing import Dict, List
import tools_decoding

seed = 1997

path = "./"
totatest = 20
args = {
    "preprocessing": {
        "to_decode": "orient",
        "min_ntr": 25,
        "start_sample": -200,
        "end_sample": 850,
        "start_test": -400,
        "end_test": 500,
        "step": 10,
        "time_before_son": "time_before_son_in",
        "time_before_t1on": "time_before_t1on_in",
        "sp_son": "sp_son_in",
        "sp_t1on": "sp_t1on_in",
        "mask_son": "mask_son_in",
        "no_match": True,
    },
    # decoder
    "decoder": {"niterations": 10, "ntr_train": 30, "ntr_test": 10, "svc_c": 0.8},
    # workspace
    "workspace": {"output": "", "path": ""},
}

popu = PopulationData.from_python_hdf5(
    "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population/lip/2024_08_28_12_23_36/population.h5"
)

list_data = popu.execute_function(
    tools_decoding.preproc_for_decoding,
    **args["preprocessing"],
    ret_df=False,
)
list_data = [idata for idata in list_data if idata is not None]

model = SVC(
    kernel="linear",
    C=args["decoder"]["svc_c"],
    decision_function_shape="ovr",
    gamma="auto",
    degree=1,
)
rng = np.random.default_rng(seed)
niterations = args["decoder"]["niterations"]
ntr_train = args["decoder"]["ntr_train"]
ntr_test = args["decoder"]["ntr_test"]
to_decode = args["preprocessing"]["to_decode"]

# Decode
trial_duration = int(
    (
        (args["preprocessing"]["end_sample"] - args["preprocessing"]["start_sample"])
        + (args["preprocessing"]["end_test"] - args["preprocessing"]["start_test"])
    )
    / args["preprocessing"]["step"]
)

n_iters = 2
lat_data = np.empty([n_iters, trial_duration, trial_duration], dtype=np.float16)
mean_data = np.empty([n_iters, trial_duration, trial_duration], dtype=np.float16)
list_n_cells = np.empty([n_iters], dtype=np.int8)
for i in np.arange(n_iters):
    seeds = rng.choice(np.arange(0, 3000), size=niterations, replace=False)
    results = Parallel(n_jobs=5)(
        delayed(tools_decoding.run_decoder)(
            model, list_data, trial_duration, ntr_train, ntr_test, to_decode, seeds[it]
        )
        for it in tqdm(range(niterations))
    )

    all_perf, weights = [], []
    for idata in results:
        all_perf.append(idata[0])
        weights.append(idata[1])
    all_perf = np.array(all_perf)
    weights = np.array(weights)
    # plot results
    n_cells = len(list_data)
    list_n_cells[i] = n_cells
    data = all_perf.transpose(0, 2, 1)
    lat_data[i] = np.sum(data > 10, axis=0)
    mean_data[i] = np.mean(data, axis=0) / totatest
    # select n-1 neurons for the next iter
    mean_w = np.mean(np.abs(weights), axis=(0, 1))
    idx_sorted_w = np.argsort(mean_w)
    if i == 0:
        list_mean_w = mean_w[idx_sorted_w]
    idx_w = idx_sorted_w[:-1]
    new_list_data = [list_data[icell] for icell in idx_w]
    list_data = new_list_data
# save results
res = Results(
    "min_amount_cells.py",
    "path",
    lat_data=lat_data,
    mean_data=mean_data,
    list_mean_w=list_mean_w,
)
res.to_python_hdf5(path + "/test.h5")
