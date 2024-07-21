import glob
import os
import numpy as np
from typing import Dict, List
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.structures.population_data import PopulationData
import pandas as pd
from scipy import stats

seed = 1997


# Define functions
def get_fr_info(neu: NeuronData, params: List[Dict]):
    res = {}
    res["nid"] = neu.get_neuron_id()
    for it in params:
        allsp, mask = neu.align_on(
            select_block=it["select_block"],
            select_pos=it["inout"],
            event=it["event"],
            time_before=it["time_before"],
            error_type=0,
        )
        # Select trials by sample type (N or NN)
        sid = neu.sample_id[mask]
        smask = (sid != 0) if it["stype"] == "NN" else (sid == 0)
        if np.sum(smask) < 10:  # If there's no enough trials
            continue
        sp = allsp[smask]
        # Fr [sp/sec] in epoch (from 0 to it['end])
        endt = it["time_before"] + it["end"]
        stt = it["time_before"] + it["st"]
        epoch_fr = np.nanmean(sp[:, stt:endt]) * 1000
        res["fr_" + it["epoch"] + it["stype"] + it["inout"]] = epoch_fr
        # Maximum fr and latency
        sp_avg = firing_rate.moving_average(
            np.mean(sp, axis=0)[: endt + it["win"]], win=it["win"], step=1
        )[stt:endt]
        lat = np.nanargmax(sp_avg)
        res["maxfrlat_" + it["epoch"] + it["stype"] + it["inout"]] = lat
        max_fr = sp_avg[lat]
        res["maxfr_" + it["epoch"] + it["stype"] + it["inout"]] = max_fr * 1000
        if lat < 100:
            lat = 100
        max_fr_200 = np.mean(sp_avg[lat - 100 : lat + 100])
        res["maxfr200_" + it["epoch"] + it["stype"] + it["inout"]] = max_fr_200 * 1000
        # Comparison with bl
        blallsp, mask = neu.align_on(
            select_block=it["select_block"],
            select_pos=it["inout"],
            event="sample_on",
            time_before=200,
            error_type=0,
        )
        blsp = blallsp[smask, :200]
        epmaxsp = sp[:, stt + (lat - 100) : stt + lat + 100]
        epfixsp = sp[:, stt + 50 : stt + 250]
        ## Compute ratios
        bl_fr = np.mean(blsp)
        if bl_fr == 0:
            bl_fr = 1
        epmax_fr = np.mean(epmaxsp)
        epfix_fr = np.mean(epfixsp)
        spfix_bl_ratio = epfix_fr / bl_fr
        spmax_bl_ratio = epmax_fr / bl_fr
        res["spfix_bl_ratio_" + it["epoch"] + it["stype"] + it["inout"]] = (
            spfix_bl_ratio
        )
        res["spmax_bl_ratio_" + it["epoch"] + it["stype"] + it["inout"]] = (
            spmax_bl_ratio
        )
        ## Compute p-value
        tr_fr_bl = np.mean(blsp, axis=1)
        tr_maxfr = np.mean(epmaxsp, axis=1)
        if np.all((tr_fr_bl - tr_maxfr) == 0):
            continue
        _, p_maxfr = stats.wilcoxon(tr_fr_bl, tr_maxfr)
        res["p_maxfr_" + it["epoch"] + it["stype"] + it["inout"]] = p_maxfr
    return res


# Define parameters
areas = ["pfc", "v4"]
subject = "Riesling"
# paths
filepaths = {
    "lip": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/lip/neurons/",
    "pfc": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/pfc/neurons/",
    "v4": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/v4/neurons/",
}

for area in areas:
    print(area)
    path = filepaths[area]
    neu_path = path + "*neu.h5"
    path_list = glob.glob(neu_path)

    attr_dtype = {
        "sp_samples": np.float16,
        "cluster_ch": np.float16,
        "cluster_number": np.float16,
        "block": np.float16,
        "trial_error": np.float16,
        "sample_id": np.float16,
        "test_stimuli": np.float16,
        "test_distractor": np.float16,
        "cluster_id": np.float16,
        "cluster_depth": np.float16,
        "code_samples": np.float16,
        "position": np.float16,
        "pos_code": np.float16,
        "code_numbers": np.float16,
    }
    popu = PopulationData.get_population(path_list[1500:], attr_dtype)

    params = []
    for iinout in ["in", "out"]:
        for istype in ["NN", "N"]:
            i_param = [
                {
                    "inout": iinout,
                    "stype": istype,
                    "epoch": "sample",
                    "event": "sample_on",
                    "time_before": 200,
                    "st": 0,
                    "end": 450,
                    "select_block": 1,
                    "win": 100,
                },
                {
                    "inout": iinout,
                    "stype": istype,
                    "epoch": "delay1",
                    "event": "sample_on",
                    "time_before": 200,
                    "st": 450,
                    "end": 850,
                    "select_block": 1,
                    "win": 100,
                },
                {
                    "inout": iinout,
                    "stype": istype,
                    "epoch": "delay2",
                    "event": "test_on_1",
                    "time_before": 500,
                    "st": -400,
                    "end": 0,
                    "select_block": 1,
                    "win": 100,
                },
            ]
            params.append(i_param)
    params = np.concatenate(params)

    df_fr = popu.execute_function(get_fr_info, params=params, n_jobs=-1, ret_df=True)
    df_fr.to_csv("population_fr_" + area + ".csv", index=False)
