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
import os


def get_selectivity_info(neu: NeuronData):
    win = 100
    res = {}
    res["nid"] = neu.get_neuron_id()
    samples = [11, 15, 51, 55, 0]
    inout_n0_fr = []
    inout_nn_fr = []
    for inout in ["in", "out"]:
        mask = getattr(neu, "mask_" + inout)
        sp = getattr(neu, "sample_on_" + inout)
        st = getattr(neu, "st_sample_on_" + inout)
        fr = firing_rate.moving_average(data=sp, win=100, step=1)[:, :-win]  #
        sample_id = neu.sample_id[mask]
        fr_samples = select_trials.get_sp_by_sample(fr, sample_id, samples)
        o1 = np.concatenate((fr_samples["11"], fr_samples["15"]))
        o5 = np.concatenate((fr_samples["51"], fr_samples["55"]))
        c1 = np.concatenate((fr_samples["11"], fr_samples["51"]))
        c5 = np.concatenate((fr_samples["15"], fr_samples["55"]))
        sample = np.concatenate(
            (fr_samples["11"], fr_samples["15"], fr_samples["51"], fr_samples["55"])
        )
        inout_nn_fr.append(sample)
        n0 = fr_samples["0"]
        inout_n0_fr.append(n0)
        # Check selectivity and latency
        color_lat, color_score, color_p = smetrics.get_selectivity(
            c1, c5, win=75, scores=True
        )
        color_selec = (
            np.nan
            if np.isnan(color_lat)
            else "c1" if color_score[color_lat] > 0 else "c5"
        )
        orient_lat, orient_score, orient_p = smetrics.get_selectivity(
            o1, o5, win=75, scores=True
        )
        orient_selec = (
            np.nan
            if np.isnan(orient_lat)
            else "o1" if orient_score[orient_lat] > 0 else "o5"
        )
        neutral_lat, neutral_score, neutral_p = smetrics.get_selectivity(
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
        res["color_p_" + inout] = color_p
        res["orient_lat_" + inout] = orient_lat
        res["orient_selec_" + inout] = orient_selec
        res["orient_score_" + inout] = orient_score
        res["orient_p_" + inout] = orient_p
        res["neutral_lat_" + inout] = neutral_lat
        res["neutral_selec_" + inout] = neutral_selec
        res["neutral_score_" + inout] = neutral_score
        res["neutral_p_" + inout] = neutral_p
        res["mean_fr_" + inout] = np.nanmean(sp[:, st:] * 1000)

    nnpos_lat, nnpos_score, nnpos_p = smetrics.get_selectivity(
        inout_nn_fr[0], inout_nn_fr[1], win=75, scores=True
    )
    nnpos_selec = (
        np.nan
        if np.isnan(nnpos_lat)
        else "NNin" if nnpos_score[nnpos_lat] > 0 else "NNout"
    )

    neutralpos_lat, neutralpos_score, neutralpos_p = smetrics.get_selectivity(
        inout_n0_fr[0], inout_n0_fr[1], win=75, scores=True
    )
    neutralpos_selec = (
        np.nan
        if np.isnan(neutralpos_lat)
        else "Nin" if neutralpos_score[neutralpos_lat] > 0 else "Nout"
    )

    res["nnpos_lat"] = nnpos_lat
    res["nnpos_selec"] = nnpos_selec
    res["nnpos_score"] = nnpos_score
    res["nnpos_p"] = nnpos_p
    res["neutralpos_lat"] = neutralpos_lat
    res["neutralpos_selec"] = neutralpos_selec
    res["neutralpos_score"] = neutralpos_score
    res["neutralpos_p"] = neutralpos_p
    return res


def check_fr_loc(neu: NeuronData, rf_loc: pd.DataFrame):
    nid = neu.get_neuron_id()
    rfloc = rf_loc[rf_loc["nid"] == nid]["rf_loc"].values[0]
    if rfloc == "ipsi":
        pos_code = neu.pos_code
        mask1 = pos_code == 1
        mask_1 = pos_code == -1
        pos_code[mask1] = -1
        pos_code[mask_1] = 1
        setattr(neu, "pos_code", pos_code)
    return neu


def get_neu_align(path, params, sp_sample=False, rf_loc=None):

    neu = NeuronData.from_python_hdf5(path)

    if rf_loc is not None:
        neu = check_fr_loc(neu, rf_loc)

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
    "lip": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/population_selectivity_lip.h5",
    "pfc": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/population_selectivity_pfc.h5",
    "v4": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/population_selectivity_v4.h5",
}
rf_loc_path = {
    "lip": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/activation_index/rf_loc_df_lip.csv",
    "pfc": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/activation_index/rf_loc_df_pfc.csv",
    "v4": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/activation_index/rf_loc_df_v4.csv",
}

if not os.path.exists(savepath):
    os.makedirs(savepath)
for area in areas:
    print(area)
    if not os.path.isfile(popu_path[area]):
        path = filepaths[area]
        neu_path = path + "*neu.h5"
        path_list = glob.glob(neu_path)

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
        rf_loc_df = None
        if bool(rf_loc_path):
            rf_loc_df = pd.read_csv(rf_loc_path[area])
        population_list = Parallel(n_jobs=-1)(
            delayed(get_neu_align)(neu, params, rf_loc=rf_loc_df)
            for neu in tqdm(path_list)
        )
        comment = str(params)
        population = PopulationData(population_list, comment=comment)
        print("Saving population.h5")
        population.to_python_hdf5(savepath + "population_selectivity_" + area + ".h5")
        population = PopulationData.from_python_hdf5(
            savepath + "population_selectivity_" + area + ".h5"
        )
    else:
        print("Reading population data")
        population = PopulationData.from_python_hdf5(popu_path[area])
    print("Computing selectivity")
    df_selectivity = population.execute_function(
        get_selectivity_info, n_jobs=-1, ret_df=True
    )

    df_selectivity.to_pickle(savepath + "population_selectivity_" + area + ".pkl")
