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
from ephysvibe.trials import select_trials
from ephysvibe.trials.spikes import firing_rate
from typing import Dict, List

seed = 1997


def check_number_of_trials(xdict, samples, min_ntr):

    for key in samples:
        if xdict[key].shape[0] < min_ntr:
            return False
    return True


pred_names = {
    "color": ["c1", "c5"],
    "orient": ["o1", "o5"],
    "sampleid": ["11", "15", "51", "55"],
    "neutral": ["n", "nn"],
}


def color_data(fr_samples: Dict, min_ntr: int):
    samples = ["11", "15", "51", "55"]
    enough_tr = check_number_of_trials(fr_samples, samples, min_ntr)
    if not enough_tr:
        return None
    c1 = np.concatenate([fr_samples["11"], fr_samples["51"]], axis=0)
    c5 = np.concatenate([fr_samples["15"], fr_samples["55"]], axis=0)
    color = {"c1": c1, "c5": c5}
    return color


def orient_data(fr_samples: Dict, min_ntr: int):
    samples = ["11", "15", "51", "55"]
    enough_tr = check_number_of_trials(fr_samples, samples, min_ntr)
    if not enough_tr:
        return None
    o1 = np.concatenate([fr_samples["11"], fr_samples["15"]], axis=0)
    o5 = np.concatenate([fr_samples["51"], fr_samples["55"]], axis=0)
    orient = {"o1": o1, "o5": o5}
    return orient


def sampleid_data(fr_samples: Dict, min_ntr: int):
    samples = ["11", "15", "51", "55"]
    enough_tr = check_number_of_trials(fr_samples, samples, min_ntr)
    if not enough_tr:
        return None
    return fr_samples


def neutral_data(fr_samples: Dict, min_ntr: int):
    samples = ["0", "11", "15", "51", "55"]
    enough_tr = check_number_of_trials(fr_samples, samples, min_ntr)
    if not enough_tr:
        return None
    n = fr_samples["0"]
    nn = np.concatenate(
        [fr_samples["11"], fr_samples["15"], fr_samples["51"], fr_samples["55"]], axis=0
    )
    neutral = {"n": n, "nn": nn}
    return neutral


def select_trials_by_percentile(
    x: np.ndarray, min_ntr: int = 2, mask: np.ndarray = None
):
    ntr = x.shape[0]
    if mask is None:
        mask = np.full(ntr, True)

    mntr = x[mask].shape[0]

    if mntr < min_ntr:
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


def check_trials(x, min_ntr, cerotr, percentile):
    masknocero = np.full(x.shape[0], True)
    maskper = np.full(x.shape[0], True)
    if cerotr:
        masknocero = np.sum(x, axis=1) != 0
    if percentile:
        maskper = select_trials_by_percentile(x, min_ntr, masknocero)
    mask = np.logical_and(masknocero, maskper)
    if np.sum(mask) < min_ntr:
        mask = np.full(x.shape[0], True)
    return mask


def preproc_for_decoding(
    neu: NeuronData,
    time_before_son: str,
    time_before_t1on: str,
    sp_son: str,
    sp_t1on: str,
    mask_son: str,
    to_decode: str,
    min_ntr: int,
    start_sample: int,
    end_sample: int,
    start_test: int,
    end_test: int,
    avgwin: int = 100,
    step: int = 10,
    zscore=True,
    no_match=False,
    return_id=False,
    percentile: list = False,
    cerotr: list = False,
):
    # Average fr across time
    idx_start_sample = int((getattr(neu, time_before_son) + start_sample) / step)
    idx_end_sample = int((getattr(neu, time_before_son) + end_sample) / step)
    idx_start_test = int((getattr(neu, time_before_t1on) + start_test) / step)
    idx_end_test = int((getattr(neu, time_before_t1on) + end_test) / step)
    sampleon = getattr(neu, sp_son)
    t1on = getattr(neu, sp_t1on)

    fr_son = firing_rate.moving_average(sampleon, win=avgwin, step=step)[
        :, idx_start_sample:idx_end_sample
    ]
    fr_t1on = firing_rate.moving_average(t1on, win=avgwin, step=step)[
        :, idx_start_test:idx_end_test
    ]

    fr = np.concatenate([fr_son, fr_t1on], axis=1)
    mask_son = getattr(neu, mask_son)
    sample_id = neu.sample_id[mask_son]
    if no_match:
        mask_no_match = np.where(
            neu.test_stimuli[mask_son, 0] == sample_id,
            False,
            True,
        )
        fr = fr[mask_no_match]
        sample_id = sample_id[mask_no_match]
    if len(fr) < 2:
        return None

    fr = np.array(fr, dtype=np.float32)
    fr_samples = select_trials.get_sp_by_sample(fr, sample_id)
    # check trials fr
    fr_new = []
    for isamp in fr_samples.keys():
        if ~np.all((np.isnan(fr_samples[isamp]))):
            masktr = check_trials(fr_samples[isamp], min_ntr, cerotr, percentile)
            fr_samples[isamp] = fr_samples[isamp][masktr]
            fr_new.append(fr_samples[isamp])
    fr_new = np.concatenate(fr_new, axis=0)
    if zscore:
        fr_std = np.std(fr_new, ddof=1, axis=0)
        fr_std = np.where(fr_std == 0, 1, fr_std)
        fr_mean = np.mean(fr_new, axis=0).reshape(1, -1)
        for isamp in fr_samples.keys():
            fr_samples[isamp] = (fr_samples[isamp] - fr_mean) / fr_std.reshape(1, -1)

    if to_decode == "color":
        data = color_data(fr_samples, min_ntr)
    elif to_decode == "orient":
        data = orient_data(fr_samples, min_ntr)
    elif to_decode == "sampleid":
        data = sampleid_data(fr_samples, min_ntr)
    elif to_decode == "neutral":
        data = neutral_data(fr_samples, min_ntr)
    else:
        raise ValueError(
            f"to_decode must be 'color' 'orient' 'sampleid' or 'neutral' but {to_decode} was given"
        )
    if return_id and data is not None:
        data = {neu.get_neuron_id(): data}
    return data


def pick_train_test_trials(idx_trials, train_ratio, rng):
    n_trials = len(idx_trials)
    tmp = rng.permutation(idx_trials)
    train = tmp[: int((n_trials * train_ratio))]
    test = tmp[int((n_trials * train_ratio)) :]
    return train, test


def run_decoder(
    model,
    list_cells,
    trial_duration,
    ntr_train,
    ntr_test,
    to_decode,
    seed,
    n_neurons=None,
):
    rng = np.random.default_rng(seed)
    total_n_cells = len(list_cells)
    idx_cell = np.arange(total_n_cells)
    if n_neurons is not None:
        idx_cell = rng.permutation(idx_cell)[:n_neurons]
        total_n_cells = n_neurons
    test_train_ratio = 1 - ntr_test / ntr_train
    topred = pred_names[to_decode]
    ntopred = len(topred)

    # Initialize arrays to store train and test data
    data_train = np.empty(
        [trial_duration, ntr_train * ntopred, total_n_cells], dtype=np.float32
    )
    data_test = np.empty(
        [trial_duration, ntr_test * ntopred, total_n_cells], dtype=np.float32
    )
    perf = np.empty([trial_duration, trial_duration], dtype=np.int16)
    weights = np.empty([trial_duration, total_n_cells], dtype=np.float16)
    y_train, y_test = [], []
    for i in range(ntopred):
        y_train.append(np.zeros(ntr_train) + i)
        y_test.append(np.zeros(ntr_test) + i)
    y_train, y_test = np.concatenate(y_train), np.concatenate(y_test)

    # Iterate through neurons to randomly pick trials
    for icell, idxc in enumerate(idx_cell):
        cell = list_cells[idxc]
        trials_train, trials_test = [], []
        for ipred in topred:
            trials = cell[ipred]
            idx_trials = np.arange(len(trials))
            train, test = pick_train_test_trials(idx_trials, test_train_ratio, rng)
            train = rng.choice(train, ntr_train, replace=True)
            test = rng.choice(test, ntr_test, replace=True)
            trials_train.append(trials[train])
            trials_test.append(trials[test])

        # build matrices of  [timestamp, trials, neurons] dimensions to feed to classifiers
        data_train[:, :, icell] = np.concatenate(trials_train, axis=0).T
        data_test[:, :, icell] = np.concatenate(trials_test, axis=0).T

    # train and test classifier
    for time_train in range(trial_duration):
        model.fit(data_train[time_train], y_train)
        for time_test in range(trial_duration):
            y_predict = model.predict(data_test[time_test])
            perf[time_train, time_test] = np.where(y_predict - y_test == 0)[0].shape[
                0
            ]  # / (ntr_test * ntopred)
            if time_train == time_test:
                if model.coef_.ndim > 1:
                    weights[time_train, :] = np.mean(model.coef_, axis=0).astype(
                        np.float16
                    )
                else:
                    weights[time_train, :] = model.coef_.astype(np.float16)
    return perf, weights


def compute_cross_decoding(
    model, list_neurons, trial_duration, ntr_train, ntr_test, to_decode, seed
):
    # TODO
    return
