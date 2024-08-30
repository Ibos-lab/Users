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

    seeds = rng.choice(np.arange(0, 3000), size=niterations, replace=False)
    all_perf = Parallel(n_jobs=-1)(
        delayed(tools_decoding.run_decoder)(
            model,
            list_data,
            trial_duration,
            ntr_train,
            ntr_test,
            to_decode,
            seeds[it],
            n_neurons,
        )
        for it in tqdm(range(niterations))
    )

    res = Results(
        "decode.py",
        "path",
        perf=np.array(all_perf),
        **preprocessing,
        **decoder,
    )
    return res
