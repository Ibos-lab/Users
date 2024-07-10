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


seed = 1997


def scrum_eqsize(fr):

    meanfr0 = np.mean(fr["0"], axis=0)
    meanfr11 = np.mean(fr["11"], axis=0)
    meanfr15 = np.mean(fr["15"], axis=0)
    meanfr51 = np.mean(fr["51"], axis=0)
    meanfr55 = np.mean(fr["55"], axis=0)

    nn = np.concatenate((fr["11"], fr["15"], fr["51"], fr["55"]), axis=0)
    size_nn = nn.shape[0]

    all_s = np.concatenate((fr["0"], fr["11"], fr["15"], fr["51"], fr["55"]), axis=0)

    sizemax = np.sum(
        [
            len(fr["0"]),
            len(fr["11"]),
            len(fr["15"]),
            len(fr["51"]),
            len(fr["55"]),
        ]
    )
    idx_tr = rng.choice(len(all_s), size=sizemax, replace=False)
    g1 = np.mean(all_s[idx_tr[:size_nn]], axis=0)
    g2 = np.mean(all_s[idx_tr[size_nn:]], axis=0)

    return meanfr0, meanfr11, meanfr15, meanfr51, meanfr55, g1, g2


def scrum_neutralsize(fr, ntr=15):

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

    all_s = np.concatenate((fr["0"], fr["11"], fr["15"], fr["51"], fr["55"]), axis=0)

    idx_tr = rng.choice(len(all_s), size=ntr * 2, replace=False)
    g1 = np.mean(all_s[idx_tr[:ntr]], axis=0)
    g2 = np.mean(all_s[idx_tr[ntr:]], axis=0)

    return meanfr0, meanfr11, meanfr15, meanfr51, meanfr55, g1, g2


def scrum_neutralsize_newpool(fr, ntr=15):

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

    idx_tr = rng.choice(size_nn, size=ntr, replace=False)
    f_nn_trs = nn[idx_tr]
    idx_tr = rng.choice(fr["0"].shape[0], size=ntr, replace=False)
    f_neutral_trs = fr["0"][idx_tr]

    all_s = np.concatenate((f_neutral_trs, f_nn_trs), axis=0)

    idx_tr = rng.choice(len(all_s), size=ntr * 2, replace=False)
    g1 = np.mean(all_s[idx_tr[:ntr]], axis=0)
    g2 = np.mean(all_s[idx_tr[ntr:]], axis=0)

    return meanfr0, meanfr11, meanfr15, meanfr51, meanfr55, g1, g2


def scrum_neutralsize_samepool(fr, ntr):

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


def scrum_neutral_fixes_size(fr):

    ntr = fr["0"].shape[0]
    nn = np.concatenate((fr["11"], fr["15"], fr["51"], fr["55"]), axis=0)
    size_nn = nn.shape[0]

    idx_tr = rng.choice(size_nn, size=ntr, replace=False)
    nn_trs = nn[idx_tr]

    neutral_trs = fr["0"]

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


def scrum_neutral_fixes_size_difpool(fr):

    ntr = fr["0"].shape[0]
    nn = np.concatenate((fr["11"], fr["15"], fr["51"], fr["55"]), axis=0)
    size_nn = nn.shape[0]

    idx_tr = rng.choice(size_nn, size=ntr, replace=False)
    nn_trs = nn[idx_tr]

    neutral_trs = fr["0"]

    meanfr0 = np.mean(neutral_trs, axis=0)
    meanfr11 = np.mean(nn_trs, axis=0)
    meanfr15 = np.mean(nn_trs, axis=0)
    meanfr51 = np.mean(nn_trs, axis=0)
    meanfr55 = np.mean(nn_trs, axis=0)

    all_s = np.concatenate((fr["0"], fr["11"], fr["15"], fr["51"], fr["55"]), axis=0)

    idx_tr = rng.choice(len(all_s), size=ntr * 2, replace=False)
    g1 = np.mean(all_s[idx_tr[:ntr]], axis=0)
    g2 = np.mean(all_s[idx_tr[ntr:]], axis=0)

    return meanfr0, meanfr11, meanfr15, meanfr51, meanfr55, g1, g2


def scrum_eqsize_replacement(fr, ntr=30):

    nn = np.concatenate((fr["11"], fr["15"], fr["51"], fr["55"]), axis=0)
    size_nn = nn.shape[0]

    idx_tr = rng.choice(size_nn, size=ntr, replace=True)
    nn_trs = nn[idx_tr]

    idx_tr = rng.choice(fr["0"].shape[0], size=ntr, replace=True)
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


# Define parameters


areas = ["lip", "pfc", "v4"]
subject = "Riesling"
avgwin = 100
min_sp_sec = 1
n_test = 1
min_trials = 25
nonmatch = True  # if True: includes nonmatch trials
norm = False
zscore = True
# sample timing
time_before_sample = 500
start_sample = -200
end_sample = 450 + 400

# test timing
time_before_test = 500
start_test = -400
end_test = n_test * 450 + 200
# paths
filepaths = {
    "lip": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/lip/neurons/",
    "pfc": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/pfc/neurons/",
    "v4": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/v4/neurons/",
}
outputpath = (
    "./testv2_"
    + str(min_trials)
    + "tr_"
    + str(min_sp_sec)
    + "sp_Zscore_scrum_neutral_fixes_size/"
)
dataoutputpath = "./"

if not os.path.exists(outputpath):
    os.makedirs(outputpath)
# -------------------------------------------- End parameters ------------------------------------------

# Compute idxs
idx_start_sample = time_before_sample + start_sample
idx_end_sample = time_before_sample + end_sample
idx_start_test = time_before_test + start_test
idx_end_test = time_before_test + end_test
# total trial duration
trial_dur = end_sample - start_sample + end_test - start_test

allspath = {
    "lip": "/envau/work/invibe/USERS/LOSADA/Users/losadac/data_stats/neurons/preproc_data/lip_"
    + str(min_trials)
    + "tr_1sp_Zscore.pickle",
    "pfc": "/envau/work/invibe/USERS/LOSADA/Users/losadac/data_stats/neurons/preproc_data/pfc_"
    + str(min_trials)
    + "tr_1sp_Zscore.pickle",
    "v4": "/envau/work/invibe/USERS/LOSADA/Users/losadac/data_stats/neurons/preproc_data/v4_"
    + str(min_trials)
    + "tr_1sp_Zscore.pickle",
}
# ------------------------------------------ Start preprocessing ----------------------------------------
if not bool(allspath):
    for area in areas:
        path = filepaths[area]
        neu_path = path + "*neu.h5"
        path_list = glob.glob(neu_path)
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
            )
            for path in tqdm(path_list)
        )

        spath = (
            dataoutputpath
            + area
            + "_"
            + str(min_trials)
            + "tr_"
            + str(min_sp_sec)
            + "sp_Zscore.pickle"
        )
        allspath[area] = spath

        with open(spath, "wb") as fp:
            pickle.dump(data, fp)


rng = np.random.default_rng(seed)

res = {"lip": {}, "v4": {}, "pfc": {}}

pc_areas = {}
for area in areas:
    path = allspath[area]

    with open(path, "br") as fp:
        data = pickle.load(fp)

    all_distn = []
    all_distnn = []
    all_dist_n_nn = []
    n_comp = 100
    allidx_neu = []
    allreshape_pc = []
    alldist_fake_n_nn = []

    for _ in range(1000):

        s0, s11, s15, s51, s55 = [], [], [], [], []
        g1mean, g2mean = [], []
        gg1 = []
        gg2 = []
        s0mean, s11mean, s15mean, s51mean, s55mean = [], [], [], [], []

        for asc in data:
            fr = asc["fr"]
            if fr is not None:
                meanfr0, meanfr11, meanfr15, meanfr51, meanfr55, g1, g2 = (
                    scrum_neutral_fixes_size(fr)
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

        pc_areas[area] = {
            "n_neurons": fr_concat.shape[0],
            "allsamples_fr": fr_concat,
            "n_fr": neufr["0mean"],
            "fr_group_concat": fr_group_concat,
        }

        tot_nneu = pc_areas[area]["n_neurons"]

        idx_neu = rng.choice(tot_nneu, size=n_comp, replace=False)

        allidx_neu.append(idx_neu)
        allsamp_fr = pc_areas[area]["allsamples_fr"][idx_neu]
        fr_group_concat = pc_areas[area]["fr_group_concat"][idx_neu]

        reshape_pc = allsamp_fr.reshape(n_comp, 5, -1)
        reshape_pc = np.concatenate(
            (reshape_pc[:, 0], np.mean(reshape_pc[:, 1:], axis=1)), axis=1
        ).reshape(n_comp, 2, -1)

        fr_groups = fr_group_concat.reshape(n_comp, 2, -1)
        distn = []
        distnn = []
        dist_n_nn = []
        dist_fake_n_nn = []
        for i in range(reshape_pc.shape[-1]):

            reference = np.zeros(n_comp)
            distn.append(pdist(np.array((reshape_pc[:, 0, i], reference))))
            distnn.append(pdist(np.array((reshape_pc[:, 1, i], reference))))
            dist_n_nn.append(
                pdist(np.array((reshape_pc[:, 0, i], reshape_pc[:, 1, i])))
            )
            dist_fake_n_nn.append(
                pdist(np.array((fr_groups[:, 0, i], fr_groups[:, 1, i])))
            )

        all_distn.append(np.array(distn).reshape(-1))
        all_distnn.append(np.array(distnn).reshape(-1))
        all_dist_n_nn.append(np.array(dist_n_nn).reshape(-1))
        alldist_fake_n_nn.append(np.array(dist_fake_n_nn).reshape(-1))
    all_distn = np.array(all_distn)
    all_distnn = np.array(all_distnn)
    all_dist_n_nn = np.array(all_dist_n_nn)
    alldist_fake_n_nn = np.array(alldist_fake_n_nn)
    res[area]["all_distnn"] = all_distnn
    res[area]["all_distn"] = all_distn
    res[area]["allidx_neu"] = allidx_neu
    res[area]["all_dist_n_nn"] = all_dist_n_nn
    res[area]["alldist_fake_n_nn"] = alldist_fake_n_nn
    res[area]["n_neurons"] = fr_concat.shape[0]
    print(area)
    to_python_hdf5(dat=[res[area]], save_path=outputpath + area + "_population_dist.h5")
