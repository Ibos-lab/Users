# Pipeline for preprocessing data for pca analysis
from preproc_tools import get_neuron_sample_test_fr, to_python_hdf5
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

seed = 1997


def scrum_neutralsize_samepool(fr, ntr, rng):

    nn = np.concatenate((fr["11"], fr["15"], fr["51"], fr["55"]), axis=0)
    size_nn = nn.shape[0]

    idx_tr = rng.choice(size_nn, size=ntr, replace=False)
    nn_trs = nn[idx_tr]

    idx_tr = rng.choice(fr["0"].shape[0], size=ntr, replace=False)
    neutral_trs = fr["0"][idx_tr]

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


def compute_distance(data, rng, min_trials, select_n_neu=100):

    g1mean, g2mean = [], []
    s0mean, s11mean, s15mean, s51mean, s55mean = [], [], [], [], []

    for asc in data:
        fr = asc["fr"]
        if fr is not None:
            meanfr0, meanfr11, meanfr15, meanfr51, meanfr55, g1, g2 = (
                scrum_neutralsize_samepool(fr, min_trials, rng)
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


def population_latency(kwargs):
    # --------------- Set variables ------------------
    if kwargs is not None:

        areas = kwargs["areas"]
        avgwin = kwargs["avgwin"]
        min_sp_sec = kwargs["min_sp_sec"]
        n_test = kwargs["n_test"]
        min_trials = kwargs["min_trials"]
        code = kwargs["code"]
        nonmatch = kwargs["nonmatch"]
        norm = kwargs["norm"]
        zscore = kwargs["zscore"]
        select_n_neu = kwargs["select_n_neu"]
        time_before_sample = kwargs["time_before_sample"]
        start_sample = kwargs["start_sample"]
        end_sample = kwargs["end_sample"]
        time_before_test = kwargs["time_before_test"]
        start_test = kwargs["start_test"]
        end_test = kwargs["end_test"]
        allspath = kwargs["allspath"]
        nidpath = kwargs["nidpath"]
        filepaths = kwargs["filepaths"]
        outputpath = kwargs["outputpath"]

    # --------------- Set variables ------------------
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    outputpath = outputpath + date + "/"

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    # Compute idxs
    idx_start_sample = time_before_sample + start_sample
    idx_end_sample = time_before_sample + end_sample
    idx_start_test = time_before_test + start_test
    idx_end_test = time_before_test + end_test

    # ------------------------------------------ Start preprocessing ----------------------------------------
    if not bool(allspath):
        for area in areas:
            print(area)
            path = filepaths[area]
            neu_path = path + "*neu.h5"
            path_list = glob.glob(neu_path)
            include_nid = None
            if bool(nidpath):
                df_sel = pd.read_csv(nidpath[area])
                include_nid = df_sel["nid"].values

            data = Parallel(n_jobs=-1)(
                delayed(get_neuron_sample_test_fr)(
                    path=path,
                    time_before_sample=time_before_sample,
                    time_before_test=time_before_test,
                    idx_start_sample=idx_start_sample,
                    idx_end_sample=idx_end_sample,
                    idx_start_test=idx_start_test,
                    idx_end_test=idx_end_test,
                    n_test=n_test,
                    min_trials=min_trials,
                    min_neu=False,
                    nonmatch=nonmatch,
                    avgwin=avgwin,
                    n_sp_sec=min_sp_sec,
                    norm=norm,
                    zscore=zscore,
                    code=code,
                    include_nid=include_nid,
                )
                for path in tqdm(path_list)
            )

            spath = outputpath + area + "_preprocdata.pickle"
            allspath[area] = spath
            print("saving")
            with open(spath, "wb") as fp:
                pickle.dump(data, fp)

    print("Compute distances")
    rng = np.random.default_rng(seed)
    res = {"lip": {}, "v4": {}, "pfc": {}}

    for area in areas:
        print(area)
        path = allspath[area]
        with open(path, "br") as fp:
            data = pickle.load(fp)

        all_dist_n_nn = []
        all_dist_fake_n_nn = []
        print("start iterations")
        distance_data = Parallel(n_jobs=-1)(
            delayed(compute_distance)(
                data, rng=rng, min_trials=min_trials, select_n_neu=select_n_neu
            )
            for _ in tqdm(range(2))
        )
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
    kwargs["script"] = "population_latencies.py"
    with open(outputpath + "pipeline_parameters.json", "w") as f:
        json.dump(kwargs, f, indent=4, sort_keys=True)
