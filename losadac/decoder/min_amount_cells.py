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


svc_c = 0.001

itinfo = {
    "lip": {"step": 10, "maxit": 350},  # 210
    "pfc": {"step": 30, "maxit": 1900},  # 1700
    "v4": {"step": 30, "maxit": 1800},  # 1500
}


for to_decode in ["sampleid"]:
    for area in ["pfc", "v4"]:
        path = f"percentile_with_nonzero/{area}/{svc_c}/{to_decode}"

        if not os.path.exists(path):
            os.makedirs(path)

        args = {
            "preprocessing": {
                "to_decode": to_decode,
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
            f"/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population/{area}/2024_08_28_12_23_36/population.h5"
        )

        list_data = popu.execute_function(
            tools_decoding.preproc_for_decoding,
            **args["preprocessing"],
            percentile=True,
            cerotr=True,
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
                + (
                    args["preprocessing"]["end_test"]
                    - args["preprocessing"]["start_test"]
                )
            )
            / args["preprocessing"]["step"]
        )

        list_it = np.arange(0, itinfo[area]["maxit"], itinfo[area]["step"])
        for i, _ in enumerate(list_it):
            seeds = rng.choice(np.arange(0, 3000), size=niterations, replace=False)
            results = Parallel(n_jobs=-1)(
                delayed(tools_decoding.run_decoder)(
                    model,
                    list_data,
                    trial_duration,
                    ntr_train,
                    ntr_test,
                    to_decode,
                    seeds[it],
                )
                for it in tqdm(range(niterations))
            )

            all_perf, weights = [], []
            for idata in results:
                all_perf.append(idata[0])
                weights.append(idata[1])

            weights = np.array(weights, dtype=np.float32)
            # plot results
            n_cells = len(list_data)

            # select n-1 neurons for the next iter
            mean_w = np.mean(np.abs(weights), axis=(0, 1))

            idx_sorted_w = np.argsort(mean_w)
            # idx_sorted_w = idx_sorted_w[mean_w[idx_sorted_w] != 0]

            idx_w = idx_sorted_w[: -itinfo[area]["step"]]
            new_list_data = [list_data[icell] for icell in idx_w]
            list_data = new_list_data
            # save results
            res = Results(
                "min_amount_cells.py",
                "path",
                perf=np.array(all_perf, dtype=np.float32),
                list_mean_w=mean_w[idx_sorted_w],
                n_cells=n_cells,
            )
            res.to_python_hdf5(path + f"/{n_cells}cells_c{svc_c}_test_{to_decode}.h5")
