from ephysvibe.structures.neuron_data import NeuronData
import numpy as np
from sklearn.decomposition import PCA, SparsePCA
import matplotlib.pyplot as plt
import os
import glob
from ephysvibe.trials import align_trials
from ephysvibe.trials.spikes import firing_rate
import platform
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py
from pathlib import Path
from typing import Dict, List
from ephysvibe.task import task_constants
import pca_tools

# import the animation and the HTML module to create and render the animation
from matplotlib import animation
from IPython.display import HTML
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy import stats
import seaborn as sns
from ephysvibe.stats import smetrics

seed = 2024


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


def to_python_hdf5(dat: List, save_path: Path):
    """Save data in hdf5 format."""
    # save the data
    with h5py.File(save_path, "w") as f:
        for i_d in range(len(dat)):
            group = f.create_group(str(i_d))

            for key, value in zip(dat[i_d].keys(), dat[i_d].values()):
                group.create_dataset(key, np.array(value).shape, data=value)
    f.close()


def z_score(X, with_std=False):
    # X: ndarray, shape (n_features, n_samples)
    # X=X/ np.max(X,axis=1).reshape(-1,1)
    ss = StandardScaler(with_mean=True, with_std=with_std)
    Xz = ss.fit_transform(X.T).T
    return Xz


def compute_pca(x, n_comp=50):
    model = PCA(n_components=n_comp)
    # C = model.components_
    # pc_s = C @ x
    pc_s = model.fit_transform(x.T).T
    return model, pc_s


# Load data
n_test = 1
# sample timing
time_before_sample = 500
start_sample = -200
end_sample = 450 + 400

# test timing
time_before_test = 500
start_test = -400
end_test = n_test * 450 + 200
# -------------------------------------------- End parameters ------------------------------------------

# Compute idxs
idx_start_sample = time_before_sample + start_sample
idx_end_sample = time_before_sample + end_sample
idx_start_test = time_before_test + start_test
idx_end_test = time_before_test + end_test
# total trial duration
trial_dur = end_sample - start_sample + end_test - start_test

# Define epochs
part1 = end_sample - start_sample
test1_st = part1 - start_test
test2_st = test1_st + 450
test3_st = test2_st + 450
test4_st = test3_st + 450
test5_st = test4_st + 450
idx_f = np.arange(0, 200, 2)
idx_s = np.arange(200, 200 + 450, 2)
idx_d1 = np.arange(200 + 450, part1, 2)
idx_d2 = np.arange(part1, test1_st, 2)
idx_t1 = np.arange(test1_st, test2_st, 2)
idx_t2 = np.arange(test2_st, test3_st, 2)
idx_t3 = np.arange(test3_st, test4_st, 2)
idx_t4 = np.arange(test4_st, test5_st, 2)
idx_aftert = np.arange(test2_st, trial_dur, 2)

if platform.system() == "Linux":
    basepath = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/pca/data/"
elif platform.system() == "Windows":
    basepath = "//envau_cifs.intlocal.univ-amu.fr/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/pca/data/"


n_comp = 100
path_list = [
    "/envau/work/invibe/USERS/LOSADA/Users/losadac/pca/test/lip/2024-06-06_12-38-40/lip_win100_test1_nonmatchTrue_min10tr_pca.h5",
    "/envau/work/invibe/USERS/LOSADA/Users/losadac/pca/test/pfc/2024-06-06_14-53-27/pfc_win100_test1_nonmatchTrue_min10tr_pca.h5",
    "/envau/work/invibe/USERS/LOSADA/Users/losadac/pca/test/v4/2024-06-06_15-00-01/v4_win100_test1_nonmatchTrue_min10tr_pca.h5",
]
pc_areas = {}
for path, area in zip(path_list, ["lip", "pfc", "v4"]):  #'lip',
    fr = from_python_hdf5(path)[0]
    fr_concat = np.concatenate(
        (fr["0mean"], fr["11mean"], fr["15mean"], fr["51mean"], fr["55mean"]), axis=1
    )

    print("%s %d" % (area, fr_concat.shape[0]))

    fr_concat = z_score(fr_concat, with_std=False)
    model, pc_s = compute_pca(fr_concat, n_comp=n_comp)
    pc_areas[area] = {
        "model": model,
        "pc": pc_s,
        "n_neurons": fr_concat.shape[0],
        "allsamples_fr": fr_concat,
        "n_fr": fr["0mean"],
        "nn_fr": fr["nnmean"],
    }


res = {"lip": {}, "v4": {}, "pfc": {}}
rng = np.random.default_rng(seed)
for area in ["lip", "v4", "pfc"]:
    all_distn = []
    all_distnn = []
    all_dist_n_nn = []
    n_comp = 100
    allidx_neu = []
    allreshape_pc = []
    tot_nneu = pc_areas[area]["n_neurons"]
    for _ in range(101):
        idx_neu = rng.choice(tot_nneu, size=n_comp, replace=False)
        allidx_neu.append(idx_neu)
        allsamp_fr = pc_areas[area]["allsamples_fr"][idx_neu]
        allsamp_mean = np.mean(allsamp_fr.reshape(n_comp, 5, -1), axis=1)

        n_fr = pc_areas[area]["n_fr"][idx_neu]
        nn_fr = pc_areas[area]["nn_fr"][idx_neu]
        # neurons_fr=np.concatenate((n_fr,nn_fr),axis=1)

        model, pc_s = compute_pca(allsamp_fr, n_comp=n_comp)
        reshape_pc = pc_s.reshape(n_comp, 5, -1)
        reshape_pc = np.concatenate(
            (reshape_pc[:, 0], np.mean(reshape_pc[:, 1:], axis=1)), axis=1
        ).reshape(n_comp, 2, -1)
        # pc_s =model.transform(np.concatenate((n_fr,nn_fr),axis=1).T).T
        # reshape_pc = pc_s.reshape(n_comp,2,-1)
        allsamp_pc = model.transform(allsamp_mean.T).T
        distn = []
        distnn = []
        dist_n_nn = []
        for i in range(reshape_pc.shape[-1]):
            # dist.append(signed_euclidean_distance(nnpc[:,i].T, npc[:,i].T))
            reference = np.zeros(n_comp)  # allsamp_pc[:,i] #
            distn.append(pdist(np.array((reshape_pc[:, 0, i], reference))))
            distnn.append(pdist(np.array((reshape_pc[:, 1, i], reference))))
            dist_n_nn.append(
                pdist(np.array((reshape_pc[:, 0, i], reshape_pc[:, 1, i])))
            )

        all_distn.append(np.array(distn).reshape(-1))
        all_distnn.append(np.array(distnn).reshape(-1))
        all_dist_n_nn.append(np.array(dist_n_nn).reshape(-1))
        allreshape_pc.append(reshape_pc)
    all_distn = np.array(all_distn)
    all_distnn = np.array(all_distnn)
    all_dist_n_nn = np.array(all_dist_n_nn)
    allreshape_pc = np.array(allreshape_pc)
    group1 = all_distnn
    group2 = all_distn
    roc_score, p = smetrics.compute_roc_auc(group1, group2)
    latency, _ = smetrics.find_latency(p_value=p, win=75, step=1, p_treshold=0.05)
    res[area]["roc_score"] = roc_score
    res[area]["latency"] = latency
    res[area]["p"] = p
    res[area]["all_distnn"] = all_distnn
    res[area]["all_distn"] = all_distn
    res[area]["allidx_neu"] = allidx_neu
    res[area]["all_dist_n_nn"] = all_dist_n_nn
    res[area]["reshape_pc"] = allreshape_pc
    print(area)
    print(latency - 200)
    to_python_hdf5(dat=[res[area]], save_path="./" + area + ".h5")
