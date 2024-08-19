from preproc_tools import get_fr_by_sample, to_python_hdf5
import glob
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import json
from pathlib import Path
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
import pickle
import pandas as pd
from datetime import datetime
from ephysvibe.structures.population_data import PopulationData
from ephysvibe.structures.neuron_data import NeuronData

seed = 1997


def from_python_hdf5(load_path: Path):
    """Load data from a file in hdf5 format from Python."""
    with h5py.File(load_path, "r") as f:
        data = []
        for i_g in f.keys():
            group = f[i_g]
            dataset = {}
            for key, value in zip(group.keys(), group.values()):
                dataset[key] = np.array(value)
            data.append(dataset)
    f.close()
    return data


def get_neu_align_sample_test(path, params, sp_sample=False):

    neu = NeuronData.from_python_hdf5(path)

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


def compute_distance(data, rng, min_trials, select_n_neu=100):

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


def population_latency(params, kwargs):
    # --------------- Set variables ------------------
    areas = params["areas"]
    avgwin = params["avgwin"]
    min_sp_sec = params["min_sp_sec"]
    n_test = params["n_test"]
    min_trials = params["min_trials"]
    nonmatch = params["nonmatch"]
    norm = params["norm"]
    zscore = params["zscore"]
    select_n_neu = params["select_n_neu"]
    allspath = params["allspath"]
    nidpath = params["nidpath"]
    filepaths = params["filepaths"]
    outputpath = params["outputpath"]
    start_sample = params["start_sample"]
    end_sample = params["end_sample"]
    start_test = params["start_test"]
    end_test = params["end_test"]

    # --------------- Set variables ------------------
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ------------------------------------------ Start preprocessing ----------------------------------------
    if not bool(allspath) and kwargs is not None:
        outputpath = outputpath + date + "/"

        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        for area in areas:
            print(area)
            path = filepaths[area]
            neu_path = path + "*neu.h5"
            path_list = glob.glob(neu_path)

            listpopu = Parallel(n_jobs=-1)(
                delayed(get_neu_align_sample_test)(
                    path=path, params=kwargs, sp_sample=False
                )
                for path in tqdm(path_list)
            )
            comment = str(kwargs)
            spath = outputpath + area + "_preprocdata.h5"
            allspath[area] = spath
            print("saving")
            popu = PopulationData(listpopu, comment=comment)
            popu.to_python_hdf5(spath)

    print("Compute distances")
    rng = np.random.default_rng(seed)
    res = {"lip": {}, "v4": {}, "pfc": {}}

    for area in areas:
        print(area)
        path = allspath[area]
        popu = PopulationData.from_python_hdf5(path)
        include_nid = None
        if bool(nidpath):
            df_sel = pd.read_csv(nidpath[area])
            include_nid = df_sel["nid"].values
        all_fr_samples = popu.execute_function(
            get_fr_by_sample,
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
        )

        fr_dicts_only = [item for item in all_fr_samples if isinstance(item, dict)]

        print("start iterations")
        distance_data = []
        for _ in tqdm(range(1000)):
            dist = compute_distance(
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
        all_dist_n_nn = np.array(all_dist_n_nn)
        all_dist_fake_n_nn = np.array(all_dist_fake_n_nn)
        res[area]["dist_n_nn"] = all_dist_n_nn
        res[area]["dist_fake_n_nn"] = all_dist_fake_n_nn
        res[area]["n_neurons"] = asc["n_neurons"]
        print("saving")
        to_python_hdf5(
            dat=[res[area]], save_path=outputpath + area + "_population_dist.h5"
        )
    params["script"] = "population_latencies.py"
    with open(outputpath + "pipeline_parameters.json", "w") as f:
        json.dump(params, f, indent=4, sort_keys=True)
