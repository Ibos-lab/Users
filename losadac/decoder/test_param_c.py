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

for svc_c in [10, 1, 0.8, 0.6]:
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
            "no_match": False,
        },
        # decoder
        "decoder": {
            "niterations": 1000,
            "ntr_train": 30,
            "ntr_test": 10,
            "svc_c": svc_c,
        },
        # workspace
        "workspace": {"output": "", "path": ""},
    }

    popu = PopulationData.from_python_hdf5(
        "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population/lip/2024_08_28_12_23_36/population.h5"
    )

    list_data = popu.execute_function(
        tools_decoding.preproc_for_decoding,
        **args["preprocessing"],
        percentile=1,
        ret_df=False,
    )
    list_data = [idata for idata in list_data if idata is not None]

    model = SVC(
        kernel="linear",
        C=args["decoder"]["svc_c"],
        decision_function_shape="ovr",
    )
    rng = np.random.default_rng(seed)
    niterations = args["decoder"]["niterations"]
    ntr_train = args["decoder"]["ntr_train"]
    ntr_test = args["decoder"]["ntr_test"]
    to_decode = args["preprocessing"]["to_decode"]

    # Decode
    trial_duration = int(
        (
            (
                args["preprocessing"]["end_sample"]
                - args["preprocessing"]["start_sample"]
            )
            + (args["preprocessing"]["end_test"] - args["preprocessing"]["start_test"])
        )
        / args["preprocessing"]["step"]
    )

    # Decode with entire population
    seeds = rng.choice(np.arange(0, 3000), size=niterations, replace=False)
    results = Parallel(n_jobs=-1)(
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
    data = all_perf.transpose(0, 2, 1)
    # select n-1 neurons for the next iter
    mean_w = np.mean(np.abs(weights), axis=(0, 1))
    idx_sorted_w = np.argsort(mean_w)
    list_mean_w = mean_w[idx_sorted_w]

    lat_data = np.sum(data > 10, axis=0)
    mean_data = np.mean(data, axis=0) / totatest

    # save results
    res = Results(
        "test_param_c.py",
        "path",
        lat_data=lat_data,
        mean_data=mean_data,
        list_mean_w=list_mean_w,
        n_cells=n_cells,
    )
    res.to_python_hdf5(path + f"/test_orient_c{svc_c}.h5")
