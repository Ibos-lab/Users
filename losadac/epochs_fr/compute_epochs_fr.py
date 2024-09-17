import numpy as np
from typing import Dict
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.structures.population_data import PopulationData
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.trials import select_trials
import warnings


def compute_mean(data, st, end):
    if np.all(np.isnan(data)):
        return np.array([np.nan])
    return np.nanmean(data[:, st:end] * 1000, axis=1)


def get_fr_info(neu: NeuronData, start_sample, end_sample, start_test, end_test):
    res = {}
    res["nid"] = neu.get_neuron_id()
    samples = [11, 15, 51, 55, 0]
    for loc in ["in", "out"]:
        mask = getattr(neu, "mask_son_" + loc)
        # Sample on: get att and avg response
        sp_son = getattr(neu, "sp_son_" + loc)
        time_before_son = getattr(neu, "time_before_son_" + loc)
        idx_start_sample = time_before_son + start_sample
        idx_end_sample = time_before_son + end_sample
        fr_son = firing_rate.moving_average(data=sp_son, win=100, step=1)[
            :, idx_start_sample:idx_end_sample
        ]
        # Test 1 on: get att and avg response
        sp_t1on = getattr(neu, "sp_t1on_" + loc)
        time_before_t1on = getattr(neu, "time_before_t1on_" + loc)
        idx_start_test = time_before_t1on + start_test
        idx_end_test = time_before_t1on + end_test
        fr_ton = firing_rate.moving_average(data=sp_t1on, win=100, step=1)[
            :, idx_start_test:idx_end_test
        ]
        # Sample on: group trial by orientation and color
        sample_id = neu.sample_id[mask]
        son_oc = select_trials.get_sp_by_sample(fr_son, sample_id, samples)
        son_o1 = np.concatenate((son_oc["11"], son_oc["15"]))
        son_o5 = np.concatenate((son_oc["51"], son_oc["55"]))
        son_c1 = np.concatenate((son_oc["11"], son_oc["51"]))
        son_c5 = np.concatenate((son_oc["15"], son_oc["55"]))
        # Test 1 on: group trial by orientation and color
        t1on_oc = select_trials.get_sp_by_sample(fr_ton, sample_id, samples)
        t1on_o1 = np.concatenate((t1on_oc["11"], t1on_oc["15"]))
        t1on_o5 = np.concatenate((t1on_oc["51"], t1on_oc["55"]))
        t1on_c1 = np.concatenate((t1on_oc["11"], t1on_oc["51"]))
        t1on_c5 = np.concatenate((t1on_oc["15"], t1on_oc["55"]))
        # define idx
        sonst = np.abs(start_sample)
        soned = sonst + 450
        d1st = soned
        d1ed = d1st + 400
        d2ed = 400

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # mean fr by color/orientation
            res[f"son_c1_{loc}"] = compute_mean(son_c1, sonst, soned)
            res[f"son_c5_{loc}"] = compute_mean(son_c5, sonst, soned)
            res[f"son_o1_{loc}"] = compute_mean(son_o1, sonst, soned)
            res[f"son_o5_{loc}"] = compute_mean(son_c5, sonst, soned)
            res[f"d1_c1_{loc}"] = compute_mean(son_c1, d1st, d1ed)
            res[f"d1_c5_{loc}"] = compute_mean(son_c5, d1st, d1ed)
            res[f"d1_o1_{loc}"] = compute_mean(son_o1, d1st, d1ed)
            res[f"d1_o5_{loc}"] = compute_mean(son_o5, d1st, d1ed)
            res[f"d2_c1_{loc}"] = compute_mean(t1on_c1, 0, d2ed)
            res[f"d2_c5_{loc}"] = compute_mean(t1on_c5, 0, d2ed)
            res[f"d2_o1_{loc}"] = compute_mean(t1on_o1, 0, d2ed)
            res[f"d2_o5_{loc}"] = compute_mean(t1on_o5, 0, d2ed)
            # mean fr by sample
            res[f"son_o1c1_{loc}"] = compute_mean(son_oc["11"], sonst, soned)
            res[f"son_o1c5_{loc}"] = compute_mean(son_oc["15"], sonst, soned)
            res[f"son_o5c1_{loc}"] = compute_mean(son_oc["51"], sonst, soned)
            res[f"son_o5c5_{loc}"] = compute_mean(son_oc["55"], sonst, soned)
            res[f"son_n_{loc}"] = compute_mean(son_oc["0"], sonst, soned)
            res[f"d1_o1c1_{loc}"] = compute_mean(son_oc["11"], d1st, d1ed)
            res[f"d1_o1c5_{loc}"] = compute_mean(son_oc["15"], d1st, d1ed)
            res[f"d1_o5c1_{loc}"] = compute_mean(son_oc["51"], d1st, d1ed)
            res[f"d1_o5c5_{loc}"] = compute_mean(son_oc["55"], d1st, d1ed)
            res[f"d1_n_{loc}"] = compute_mean(son_oc["0"], d1st, d1ed)
            res[f"d2_o1c1_{loc}"] = compute_mean(t1on_oc["11"], 0, d2ed)
            res[f"d2_o1c5_{loc}"] = compute_mean(t1on_oc["15"], 0, d2ed)
            res[f"d2_o5c1_{loc}"] = compute_mean(t1on_oc["51"], 0, d2ed)
            res[f"d2_o5c5_{loc}"] = compute_mean(t1on_oc["55"], 0, d2ed)
            res[f"d2_n_{loc}"] = compute_mean(t1on_oc["0"], 0, d2ed)

    return res


def compute_epochs_fr(preprocessing: Dict, paths: Dict):
    print("Reading population data")
    popu = PopulationData.from_python_hdf5(paths["input"])

    print("Computing selectivity")
    df_fr = popu.execute_function(get_fr_info, **preprocessing, n_jobs=-1, ret_df=True)
    return df_fr
