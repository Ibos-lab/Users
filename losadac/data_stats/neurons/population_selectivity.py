import glob
import numpy as np
from typing import Dict, List
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.structures.population_data import PopulationData
from ephysvibe.trials import select_trials
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from ephysvibe.stats import smetrics
from sklearn import metrics

seed = 1997


def get_selectivity_info(neu: NeuronData):

    res = {}
    res["nid"] = neu.get_neuron_id()

    samples = [11, 15, 51, 55, 0]
    for inout in ["in", "out"]:
        mask = getattr(neu, "mask_" + inout)
        sp = getattr(neu, "sample_on_" + inout)
        sp = firing_rate.moving_average(data=sp, win=100, step=1)[:, 300:]
        sample_id = neu.sample_id[mask]
        sp_samples = select_trials.get_sp_by_sample(sp, sample_id, samples)
        o1 = np.concatenate((sp_samples["11"], sp_samples["15"]))
        o5 = np.concatenate((sp_samples["51"], sp_samples["55"]))
        c1 = np.concatenate((sp_samples["11"], sp_samples["51"]))
        c5 = np.concatenate((sp_samples["15"], sp_samples["55"]))
        sample = np.concatenate(
            (sp_samples["11"], sp_samples["15"], sp_samples["51"], sp_samples["55"])
        )
        n0 = sp_samples["0"]
        # Check selectivity and latency
        color_lat, color_score = smetrics.get_selectivity(c1, c5, win=75, scores=True)
        color_selec = (
            np.nan
            if np.isnan(color_lat)
            else "c1" if color_score[color_lat] > 0 else "c5"
        )
        orient_lat, orient_score = smetrics.get_selectivity(o1, o5, win=75, scores=True)
        orient_selec = (
            np.nan
            if np.isnan(orient_lat)
            else "o1" if orient_score[orient_lat] > 0 else "o5"
        )
        neutral_lat, neutral_score = smetrics.get_selectivity(
            sample, n0, win=75, scores=True
        )
        neutral_selec = (
            np.nan
            if np.isnan(neutral_lat)
            else "NN" if neutral_score[neutral_lat] > 0 else "N"
        )

        res["color_lat_" + inout] = color_lat
        res["color_selec_" + inout] = color_selec
        res["color_score_" + inout] = color_score
        res["orient_lat_" + inout] = orient_lat
        res["orient_selec_" + inout] = orient_selec
        res["orient_score_" + inout] = orient_score
        res["neutral_lat_" + inout] = neutral_lat
        res["neutral_selec_" + inout] = neutral_selec
        res["neutral_score_" + inout] = neutral_score

    return res


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

    if ~sp_sample:
        setattr(neu, "sp_samples", np.array([]))

    return neu


# Define parameters
areas = ["lip", "pfc", "v4"]
subject = "Riesling"
# paths
filepaths = {
    "lip": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/lip/neurons/",
    "pfc": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/pfc/neurons/",
    "v4": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/v4/neurons/",
}
savepath = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/"
popu_path = {
    "lip": "",
    "pfc": "",
    "v4": "",
}
for area in areas:
    print(area)
    path = filepaths[area]
    neu_path = path + "*neu.h5"
    path_list = glob.glob(neu_path)

    params = [
        {
            "inout": "in",
            "sp": "sample_on_in",
            "mask": "mask_in",
            "event": "sample_on",
            "time_before": 500,
            "st": 0,
            "end": 1000,
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
            "time_before": 500,
            "st": 0,
            "end": 1000,
            "select_block": 1,
            "win": 100,
            "dtype_sp": np.int8,
            "dtype_mask": bool,
        },
    ]
    population_list = Parallel(n_jobs=-1)(
        delayed(get_neu_align)(neu, params) for neu in tqdm(path_list)
    )
    comment = "{'inout':'in','sp':'sample_on_in','mask':'mask_in','event':'sample_on','time_before':500,'st':0,'end':1000,'select_block':1,'win':100,'dtype_sp':np.int8,'dtype_mask':bool},{'inout':'out','sp':'sample_on_out','mask':'mask_out','event':'sample_on','time_before':500,'st':0,'end':1000,'select_block':1,'win':100,'dtype_sp':np.int8,'dtype_mask':bool}"
    population = PopulationData(population_list, comment=comment)
    population.to_python_hdf5(savepath + "population_selectivity_" + area + ".h5")

    df_selectivity = population.execute_function(
        get_selectivity_info, n_jobs=-1, ret_df=True
    )

    df_selectivity.to_csv(
        savepath + "population_selectivity_" + area + ".csv", index=False
    )
