from population_distance import preproc_tools
import glob
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
import h5py
from scipy.spatial.distance import pdist
import pandas as pd
from datetime import datetime
from ephysvibe.structures.population_data import PopulationData
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.structures.results import Results

seed = 1997


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


def get_neu_align_sample_test(path, params, sp_sample=False, rf_loc=None):

    neu = NeuronData.from_python_hdf5(path)
    if rf_loc is not None:
        neu = check_fr_loc(neu, rf_loc)

    sp_sample_on, mask_s = neu.align_on(
        select_block=params["select_block"],
        select_pos=params["inout"],
        event="sample_on",
        time_before=params["time_before_sample"],
        error_type=0,
    )

    sp_test_on, mask_t = neu.align_on(
        select_block=params["select_block"],
        select_pos=params["inout"],
        event="test_on_1",
        time_before=params["time_before_test"],
        error_type=0,
    )

    endt = params["time_before_sample"] + params["time_after_sample"]
    setattr(
        neu, "sp_sample", np.array(sp_sample_on[:, :endt], dtype=params["dtype_sp"])
    )
    setattr(neu, "mask_s", np.array(mask_s, dtype=params["dtype_mask"]))
    setattr(
        neu,
        "time_before_sample",
        np.array(params["time_before_sample"], dtype=int),
    )

    endt = params["time_before_test"] + params["time_after_test"]
    setattr(neu, "sp_test", np.array(sp_test_on[:, :endt], dtype=params["dtype_sp"]))
    setattr(neu, "mask_t", np.array(mask_t, dtype=params["dtype_mask"]))
    setattr(
        neu,
        "time_before_test",
        np.array(params["time_before_test"], dtype=int),
    )

    if ~sp_sample:
        setattr(neu, "sp_samples", np.array([]))
    return neu


def scrum_neutralsize_samepool(data, ntr, rng):

    nn = np.concatenate((data["11"], data["15"], data["51"], data["55"]), axis=0)
    size_nn = nn.shape[0]

    idx_tr = rng.choice(size_nn, size=ntr, replace=False)
    nn_trs = nn[idx_tr]

    idx_tr = rng.choice(data["0"].shape[0], size=ntr, replace=False)
    neutral_trs = data["0"][idx_tr]

    meanfr0 = np.mean(neutral_trs, axis=0)
    meanfr11 = np.mean(nn_trs, axis=0)
    meanfr15 = np.mean(nn_trs, axis=0)
    meanfr51 = np.mean(nn_trs, axis=0)
    meanfr55 = np.mean(nn_trs, axis=0)

    all_s = np.concatenate((neutral_trs, nn_trs), axis=0)

    idx_tr = rng.choice(len(all_s), size=ntr * 2, replace=False)
    g1 = np.mean(all_s[idx_tr[:ntr]], axis=0)
    g2 = np.mean(all_s[idx_tr[ntr:]], axis=0)

    return meanfr0, meanfr11, meanfr15, meanfr51, meanfr55, g1, g2


def get_distance(data, rng, min_trials, select_n_neu=100):

    g1mean, g2mean = [], []
    s0mean, s11mean, s15mean, s51mean, s55mean = [], [], [], [], []

    for idata in data:
        meanfr0, meanfr11, meanfr15, meanfr51, meanfr55, g1, g2 = (
            scrum_neutralsize_samepool(idata, min_trials, rng)
        )

        s0mean.append(meanfr0)
        s11mean.append(meanfr11)
        s15mean.append(meanfr15)
        s51mean.append(meanfr51)
        s55mean.append(meanfr55)
        g1mean.append(g1)
        g2mean.append(g2)

    neufr = {
        "0mean": s0mean,
        "11mean": s11mean,
        "15mean": s15mean,
        "51mean": s51mean,
        "55mean": s55mean,
        "g1mean": g1mean,
        "g2mean": g2mean,
    }

    fr_concat = np.concatenate(
        (
            neufr["0mean"],
            neufr["11mean"],
            neufr["15mean"],
            neufr["51mean"],
            neufr["55mean"],
        ),
        axis=1,
    )
    fr_group_concat = np.concatenate((neufr["g1mean"], neufr["g2mean"]), axis=1)

    n_neurons = fr_concat.shape[0]
    idx_neu = rng.choice(n_neurons, size=select_n_neu, replace=False)
    allsamp_fr = fr_concat[idx_neu]
    fr_group_concat = fr_group_concat[idx_neu]
    reshape_pc = allsamp_fr.reshape(select_n_neu, 5, -1)
    reshape_pc = np.concatenate(
        (reshape_pc[:, 0], np.mean(reshape_pc[:, 1:], axis=1)), axis=1
    ).reshape(select_n_neu, 2, -1)
    fr_groups = fr_group_concat.reshape(select_n_neu, 2, -1)
    dist_n_nn = []
    dist_fake_n_nn = []
    for i in range(reshape_pc.shape[-1]):

        dist_n_nn.append(pdist(np.array((reshape_pc[:, 0, i], reshape_pc[:, 1, i]))))
        dist_fake_n_nn.append(pdist(np.array((fr_groups[:, 0, i], fr_groups[:, 1, i]))))

    return {
        "dist_n_nn": np.array(dist_n_nn).reshape(-1),
        "dist_fake_n_nn": np.array(dist_fake_n_nn).reshape(-1),
        "n_neurons": n_neurons,
    }


def compute_distance(
    input,
    sp_son,
    sp_t1on,
    mask_son,
    start_sample,
    end_sample,
    start_test,
    end_test,
    time_before_son,
    time_before_t1on,
    avgwin,
    min_sp_sec,
    n_test,
    min_trials,
    nonmatch,
    norm,
    zscore,
    select_n_neu,
    nidpath,
    percentile,
    cerotr,
):

    # ------------------------------------------ Start preprocessing ----------------------------------------
    print("Compute distances")
    rng = np.random.default_rng(seed)
    res = {}

    popu = PopulationData.from_python_hdf5(input)
    include_nid = None
    if nidpath is not None:
        df_sel = pd.read_csv(nidpath)
        include_nid = df_sel["nid"].values
    all_fr_samples = popu.execute_function(
        preproc_tools.get_fr_by_sample,
        time_before_son=time_before_son,
        time_before_t1on=time_before_t1on,
        sp_son=sp_son,
        sp_t1on=sp_t1on,
        mask_son=mask_son,
        start_sample=start_sample,
        end_sample=end_sample,
        start_test=start_test,
        end_test=end_test,
        n_test=n_test,
        min_trials=min_trials,
        min_neu=False,
        nonmatch=nonmatch,
        avgwin=avgwin,
        n_sp_sec=min_sp_sec,
        norm=norm,
        zscore=zscore,
        include_nid=include_nid,
        n_jobs=-1,
        ret_df=False,
        cerotr=cerotr,
        percentile=percentile,
    )

    fr_dicts_only = [item for item in all_fr_samples if isinstance(item, dict)]

    print("start iterations")
    distance_data = []
    for _ in tqdm(range(1000)):
        dist = get_distance(
            fr_dicts_only,
            rng=rng,
            min_trials=min_trials,
            select_n_neu=select_n_neu,
        )
        distance_data.append(dist)

    all_dist_n_nn = []
    all_dist_fake_n_nn = []
    for asc in distance_data:
        all_dist_n_nn.append(asc["dist_n_nn"])
        all_dist_fake_n_nn.append(asc["dist_fake_n_nn"])
    all_dist_n_nn = np.array(all_dist_n_nn, dtype=np.float32)
    all_dist_fake_n_nn = np.array(all_dist_fake_n_nn, dtype=np.float32)
    res["dist_n_nn"] = all_dist_n_nn
    res["dist_fake_n_nn"] = all_dist_fake_n_nn
    res["n_neurons"] = asc["n_neurons"]

    res = Results("population_distance.py", input, distance=res)
    return res
