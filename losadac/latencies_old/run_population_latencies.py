from population_latencies import population_latency
import numpy as np

# paths
filepaths = {
    "lip": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/lip/neurons/",
    "pfc": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/pfc/neurons/",
    "v4": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/v4/neurons/",
}
outputpath = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population_lat/data/neurons_no_selectivity_inout_out/"
selectivity_path = (
    "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/selectivity/"
)
nidpath = {
    "lip": selectivity_path + "lip_no_neutral_inout_selectivity.csv",
    "pfc": selectivity_path + "pfc_no_neutral_inout_selectivity.csv",
    "v4": selectivity_path + "v4_no_neutral_inout_selectivity.csv",
}

allspath = {
    # "lip": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population_lat/data/all_neurons_corrected_in/2024-08-21_16-50-55/lip_preprocdata.h5",
    # "pfc": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population_lat/data/all_neurons_corrected_in/2024-08-21_16-50-55/pfc_preprocdata.h5",
    # "v4": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population_lat/data/all_neurons_corrected_in/2024-08-21_16-50-55/v4_preprocdata.h5",
}

rf_loc_path = {
    "lip": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/activation_index/rf_loc_df_lip.csv",
    "pfc": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/activation_index/rf_loc_df_pfc.csv",
    "v4": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/activation_index/rf_loc_df_v4.csv",
}
# Define parameters
areas = ["lip", "pfc", "v4"]
avgwin = 100
min_sp_sec = 1
n_test = 1
min_trials = 25
inout = "out"  # in out
select_block = 1
nonmatch = True  # if True: includes nonmatch trials
norm = False
zscore = True
select_n_neu = 100
# sample timing
time_before_sample = 500
start_sample = -200
end_sample = 450 + 400
time_after_sample = end_sample + 200
# test timing
time_before_test = 500
start_test = -400
end_test = n_test * 450 + 200
time_after_test = end_test + 200
dtype_sp = np.int8
dtype_mask = bool

parameters = {}
parameters["areas"] = areas
parameters["avgwin"] = avgwin
parameters["min_sp_sec"] = min_sp_sec
parameters["n_test"] = n_test
parameters["min_trials"] = min_trials

parameters["nonmatch"] = nonmatch
parameters["norm"] = norm
parameters["zscore"] = zscore
parameters["select_n_neu"] = select_n_neu

parameters["start_sample"] = start_sample
parameters["end_sample"] = end_sample

parameters["start_test"] = start_test
parameters["end_test"] = end_test
parameters["allspath"] = allspath
parameters["nidpath"] = nidpath
parameters["filepaths"] = filepaths
parameters["rf_loc_path"] = rf_loc_path
parameters["outputpath"] = outputpath

preproc = {}
preproc["inout"] = inout
preproc["select_block"] = select_block
preproc["time_before_sample"] = time_before_sample
preproc["time_after_sample"] = time_after_sample
preproc["time_before_test"] = time_before_test
preproc["time_after_test"] = time_after_test
preproc["dtype_sp"] = dtype_sp
preproc["dtype_mask"] = dtype_mask

population_latency(parameters, preproc)
