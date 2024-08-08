import glob
import os
import numpy as np
from typing import Dict, List
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.structures.population_data import PopulationData
from ephysvibe.trials import align_trials, select_trials
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm
from ephysvibe.stats import smetrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import re
from collections import defaultdict
import itertools
import pickle

pathlocs = (
    "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/correlation/*.pkl"
)
filepath = (
    "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/correlation/mean_max/"
)
outputpath = (
    "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/correlation/mean_max/"
)
path_list = glob.glob(pathlocs)
# Group paths per session
date_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
# Dictionary to group paths by date and time
grouped_paths = defaultdict(list)
# Iterate over the paths
for path in path_list:
    # Extract the date and time
    match = re.search(date_pattern, path)
    if match:
        date_time = match.group()
        grouped_paths[date_time].append(path)
print(grouped_paths.keys())
slices = [slice(0, 200), slice(200, 650), slice(650, None)]
samp_groups = {}

for samp in [0, 11, 15, 51, 55]:
    print(samp)
    groups = {
        "liplip": [],
        "lippfc": [],
        "lipv4": [],
        "pfcpfc": [],
        "pfcv4": [],
        "v4v4": [],
    }
    for date_time in grouped_paths.keys():
        if not os.path.isfile(
            filepath + "corr_s" + str(samp) + "_" + date_time + ".pkl"
        ):
            continue
        with open(
            filepath + "corr_s" + str(samp) + "_" + date_time + ".pkl", "rb"
        ) as handle:
            corrlist = pickle.load(handle)

        for icorr in corrlist:

            if np.all(np.isnan(icorr["corr"])):
                continue
            areas = icorr["areas"].split("_")
            areas = "".join(areas)
            groups[areas].append(icorr["corr"])
    samp_groups[str(samp)] = groups
print("sort")
res_samples = {"0": {}, "11": {}, "15": {}, "51": {}, "55": {}}
for samp in [0, 11, 15, 51, 55]:
    # lipv4
    lipv4 = np.array(samp_groups[str(samp)]["lipv4"])
    if lipv4.shape[0] == 0:
        res_samples[str(samp)]["lipv4"] = {"slip_dv4": np.nan, "dlip_sv4": np.nan}
    else:
        slip_dv4 = lipv4[:, 1, 2]  # sample lip delay v4
        dlip_sv4 = lipv4[:, 2, 1]  # delay lip sample v4
        res_samples[str(samp)]["lipv4"] = {"slip_dv4": slip_dv4, "dlip_sv4": dlip_sv4}
    # lippfc
    lippfc = np.array(samp_groups[str(samp)]["lippfc"])
    if lippfc.shape[0] == 0:
        res_samples[str(samp)]["lippfc"] = {"slip_dpfc": np.nan, "dlip_spfc": np.nan}
    else:
        slip_dpfc = lippfc[:, 1, 2]  # sample lip delay pfc
        dlip_spfc = lippfc[:, 2, 1]  # delay lip sample pfc
        res_samples[str(samp)]["lippfc"] = {
            "slip_dpfc": slip_dpfc,
            "dlip_spfc": dlip_spfc,
        }
    # liplip
    liplip = np.array(samp_groups[str(samp)]["liplip"])
    if liplip.shape[0] == 0:
        res_samples[str(samp)]["liplip"] = {"slip_dlip": np.nan, "dlip_slip": np.nan}
    else:
        slip_dlip = liplip[:, 1, 2]  # sample lip delay lip
        dlip_slip = liplip[:, 2, 1]  # delay lip sample lip
        res_samples[str(samp)]["liplip"] = {
            "slip_dlip": slip_dlip,
            "dlip_slip": dlip_slip,
        }
    # pfcpfc
    pfcpfc = np.array(samp_groups[str(samp)]["pfcpfc"])
    if pfcpfc.shape[0] == 0:
        res_samples[str(samp)]["pfcpfc"] = {"spfc_dpfc": np.nan, "dpfc_spfc": np.nan}
    else:
        spfc_dpfc = pfcpfc[:, 1, 2]  # sample pfc delay pfc
        dpfc_spfc = pfcpfc[:, 2, 1]  # delay pfc sample lip
        res_samples[str(samp)]["pfcpfc"] = {
            "spfc_dpfc": spfc_dpfc,
            "dpfc_spfc": dpfc_spfc,
        }
    # pfcv4
    pfcv4 = np.array(samp_groups[str(samp)]["pfcv4"])
    if pfcv4.shape[0] == 0:
        res_samples[str(samp)]["pfcv4"] = {"spfc_dv4": np.nan, "dpfc_sv4": np.nan}
    else:
        spfc_dv4 = pfcv4[:, 1, 2]  # sample pfc delay v4
        dpfc_sv4 = pfcv4[:, 2, 1]  # delay pfc sample v4
        res_samples[str(samp)]["pfcv4"] = {"spfc_dv4": spfc_dv4, "dpfc_sv4": dpfc_sv4}
    # v4v4
    v4v4 = np.array(samp_groups[str(samp)]["v4v4"])
    if v4v4.shape[0] == 0:
        res_samples[str(samp)]["v4v4"] = {"sv4_dv4": np.nan, "dv4_sv4": np.nan}
    else:
        sv4_dv4 = v4v4[:, 1, 2]  # sample v4 delay v4
        dv4_sv4 = v4v4[:, 2, 1]  # delay v4 sample v4
        res_samples[str(samp)]["v4v4"] = {"sv4_dv4": sv4_dv4, "dv4_sv4": dv4_sv4}
print("plot")
f, ax = plt.subplots(5, 6, figsize=(20, 18), sharex=True, sharey=True)
for isamp, samp in enumerate(["0", "11", "15", "51", "55"]):
    sampdict = res_samples[samp]
    for ja, akey in enumerate(sampdict.keys()):
        adict = sampdict[akey]
        dkeys = list(adict.keys())
        if np.all(np.isnan(adict[dkeys[0]])):
            continue
        ax[isamp, ja].scatter(adict[dkeys[0]], adict[dkeys[1]])
        ax[isamp, ja].plot(
            [np.min(adict[dkeys[0]]), np.max(adict[dkeys[0]])],
            [np.min(adict[dkeys[0]]), np.max(adict[dkeys[0]])],
            "k",
        )
        ax[isamp, ja].set(xlabel=dkeys[0], ylabel=dkeys[1], title=samp + " " + akey)
f.savefig(outputpath + "sample_delay_corr4")
