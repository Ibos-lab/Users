# from preproc_tools import get_fr_by_sample, to_python_hdf5
import glob
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.svm import SVC
from ephysvibe.structures.population_data import PopulationData
from ephysvibe.structures.results import Results
from typing import Dict, List
from decoder import tools_decoding

seed = 1997


def compute_decoding(preprocessing: Dict, decoder: Dict, paths: Dict):
    # preprocessing
    popu = PopulationData.from_python_hdf5(paths["input"])

    list_data = popu.execute_function(
        tools_decoding.preproc_for_decoding,
        **preprocessing,
        ret_df=False,
    )
    del popu
    list_data = [idata for idata in list_data if idata is not None]

    # Decode
    trial_duration = int(
        (
            (preprocessing["end_sample"] - preprocessing["start_sample"])
            + (preprocessing["end_test"] - preprocessing["start_test"])
        )
        / preprocessing["step"]
    )

    model = SVC(
        kernel="linear",
        C=decoder["svc_c"],
        decision_function_shape="ovr",
        gamma="auto",
        degree=1,
    )
    rng = np.random.default_rng(seed)
    niterations = decoder["niterations"]
    ntr_train = decoder["ntr_train"]
    ntr_test = decoder["ntr_test"]
    to_decode = preprocessing["to_decode"]
    n_neurons = decoder["n_neurons"]
    print(f"Number of cells: {len(list_data)}")
    # check n_neurons < len(list_cells)
    if n_neurons is not None:
        if len(list_data) <= n_neurons:
            n_neurons = None
            print(f"{n_neurons}<={len(list_data)}")

    seeds = rng.choice(np.arange(0, 3000), size=niterations, replace=False)
    results = Parallel(n_jobs=-1)(
        delayed(tools_decoding.run_decoder)(
            model=model,
            list_cells=list_data,
            trial_duration=trial_duration,
            ntr_train=ntr_train,
            ntr_test=ntr_test,
            to_decode=to_decode,
            seed=seeds[it],
            n_neurons=n_neurons,
        )
        for it in tqdm(range(niterations))
    )
    all_perf, weights = [], []
    for idata in results:
        all_perf.append(idata[0])
        weights.append(idata[1])
    all_perf = np.array(all_perf)
    weights = np.array(weights)
    # plot res1ults
    n_cells = len(list_data)
    data = all_perf.transpose(0, 2, 1)
    # select n-1 neurons for the next iter
    mean_w = np.mean(np.abs(weights), axis=(0, 1))
    idx_sorted_w = np.argsort(mean_w)
    list_mean_w = mean_w[idx_sorted_w]
    res = Results(
        "decode.py",
        "path",
        perf=np.array(all_perf, dtype=np.int16),
        **preprocessing,
        **decoder,
    )
    return res
