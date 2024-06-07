# Pipeline for preprocessing data for pca analysis
from preproc_tools import get_neuron_sample_test_fr, to_python_hdf5
import glob
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
import json

# Define parameters
filepaths = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/lip/neurons/"
outputpath = "./test/"

area = "lip"
subject = "Riesling"
avgwin = 100
min_sp_sec = 5
n_test = 1
min_trials = 10
nonmatch = True  # if True: includes nonmatch trials

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


# ------------------------------------------ Start preprocessing ----------------------------------------
neu_path = filepaths + "*neu.h5"
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
        nonmatch=nonmatch,
        avgwin=avgwin,
        n_sp_sec=min_sp_sec,
    )
    for path in tqdm(path_list)
)

s0mean, s11mean, s15mean, s51mean, s55mean, nnmean = [], [], [], [], [], []
for asc in data:
    fr = asc["fr"]
    if fr is not None:
        s0mean.append(np.mean(fr["0"], axis=0))
        s11mean.append(np.mean(fr["11"], axis=0))
        s15mean.append(np.mean(fr["15"], axis=0))
        s51mean.append(np.mean(fr["51"], axis=0))
        s55mean.append(np.mean(fr["55"], axis=0))
        nn = np.concatenate((fr["11"], fr["15"], fr["51"], fr["55"]), axis=0)
        nnmean.append(np.mean(nn, axis=0))
neurons_fr = [
    {
        "0mean": s0mean,
        "11mean": s11mean,
        "15mean": s15mean,
        "51mean": s51mean,
        "55mean": s55mean,
        "nnmean": nnmean,
    }
]

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

save_path = outputpath + "/" + area + "/" + date + "/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

filename = (
    area
    + "_win"
    + str(avgwin)
    + "_test"
    + str(n_test)
    + "_nonmatch"
    + str(nonmatch)
    + "_min"
    + str(min_trials)
    + "tr_pca.h5"
)

save_path_fname = save_path + filename

to_python_hdf5(dat=neurons_fr, save_path=save_path_fname)

parameters = {
    "script": "preproc_pca_data.py",
    "date_time": date,
    "filepaths": filepaths,
    "outputpath": save_path_fname,
    "area": area,
    "subject": subject,
    "avgwin": avgwin,
    "min_sp_sec": min_sp_sec,
    "n_test": n_test,
    "min_trials": min_trials,
    "nonmatch": nonmatch,
    "time_before_sample": time_before_sample,
    "start_sample": start_sample,
    "end_sample": end_sample,
    "time_before_test ": time_before_test,
    "start_test": start_test,
    "end_test ": end_test,
    "idx_start_sample ": idx_start_sample,
    "idx_end_sample ": idx_end_sample,
    "idx_start_test ": idx_start_test,
    "idx_end_test": idx_end_test,
    "trial_dur ": trial_dur,
}


with open(save_path + "pipeline_parameters.json", "w") as f:
    json.dump(parameters, f, indent=4, sort_keys=True)
