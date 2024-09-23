import glob
import numpy as np
from typing import Dict, List
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.structures.population_data import PopulationData
from ephysvibe.structures.results import Results
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.trials import align_trials, select_trials
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from ephysvibe.stats import smetrics
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os


def select_trials_by_percentile(x: np.ndarray, mask: np.ndarray = None):
    ntr = x.shape[0]
    if mask is None:
        mask = np.full(ntr, True)

    mntr = x[mask].shape[0]

    if mntr < 2:
        return np.full(ntr, True)
    mean_trs = np.mean(x, axis=1)

    q25, q75 = np.percentile(mean_trs[mask], [25, 75])
    iqr = q75 - q25
    upper_limit = q75 + 1.5 * iqr
    lower_limit = q25 - 1.5 * iqr

    q1mask = mean_trs > lower_limit
    q2mask = mean_trs < upper_limit

    qmask = np.logical_and(q1mask, q2mask)
    return qmask


def check_trials(x, cerotr, percentile):
    masknocero = np.full(x.shape[0], True)
    maskper = np.full(x.shape[0], True)
    if cerotr:
        masknocero = np.sum(x, axis=1) != 0
    if percentile:
        maskper = select_trials_by_percentile(x, masknocero)
    mask = np.logical_and(masknocero, maskper)
    if np.sum(mask) < 10:
        mask = np.full(x.shape[0], True)
    return mask


def get_selectivity_info(
    neu: NeuronData,
    start_sample,
    end_sample,
    start_test,
    end_test,
    cerotr: bool,
    percentile: bool,
):
    res = {}
    res["nid"] = neu.get_neuron_id()
    samples = [11, 15, 51, 55, 0]
    inout_n0_fr = []
    inout_nn_fr = []
    for inout in ["in", "out"]:
        mask = getattr(neu, "mask_son_" + inout)
        sp_son = getattr(neu, "sp_son_" + inout)
        time_before_son = getattr(neu, "time_before_son_" + inout)

        idx_start_sample = time_before_son + start_sample
        idx_end_sample = time_before_son + end_sample

        fr_son = firing_rate.moving_average(data=sp_son, win=100, step=1)[
            :, idx_start_sample:idx_end_sample
        ]

        sp_t1on = getattr(neu, "sp_t1on_" + inout)
        time_before_t1on = getattr(neu, "time_before_t1on_" + inout)

        idx_start_test = time_before_t1on + start_test
        idx_end_test = time_before_t1on + end_test

        fr_ton = firing_rate.moving_average(data=sp_t1on, win=100, step=1)[
            :, idx_start_test:idx_end_test
        ]

        fr = np.concatenate([fr_son, fr_ton], axis=1)
        # check number of trials
        masktr = check_trials(fr, cerotr, percentile)
        fr = fr[masktr]
        sample_id = neu.sample_id[mask][masktr]
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meanfrson = np.nanmean(fr_son[:, np.abs(start_sample) :] * 1000)
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
        res["mean_fr_" + inout] = meanfrson

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


def compute_selectivity(preprocessing: Dict, paths: Dict):
    print("Reading population data")
    popu = PopulationData.from_python_hdf5(paths["input"])

    print("Computing selectivity")
    df_selectivity = popu.execute_function(
        get_selectivity_info, **preprocessing, n_jobs=-1, ret_df=True
    )
    return df_selectivity
