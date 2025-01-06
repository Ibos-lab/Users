"""Execute main function of the module plot_trials."""

from typing import Dict
from . import _pipeline
import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import numpy as np
from ephysvibe.structures.population_data import PopulationData
import pandas as pd


def compute_ns_roc(paths: Dict, params: Dict, **kwargs):
    print("start compute neutral and space roc")

    if "hydra" in params and params["hydra"]:
        output_dir = os.getcwd()
    elif "output_dir" in params:
        output_dir = params["output_dir"]
    else:
        output_dir = "./"
    popu = PopulationData.from_python_hdf5(paths["input"])
    res = Parallel(n_jobs=-1)(
        delayed(_pipeline.get_space_neutral_roc)(
            neu,
            start_sample=params["start_sample"],
            end_sample=params["end_sample"],
            st_target=params["st_target"],
            end_target=params["end_target"],
            st_bl=params["st_bl"],
            end_bl=params["end_bl"],
            cerotr=params["cerotr"],
            percentile=params["percentile"],
        )
        for neu in tqdm(popu.population)
    )
    df = pd.DataFrame(res)
    print("saving")
    df.to_csv("neutral_space_idx.csv")
