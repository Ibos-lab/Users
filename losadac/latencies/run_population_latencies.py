from population_latencies import population_latency

# paths
filepaths = {
    "lip": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/lip/neurons/",
    "pfc": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/pfc/neurons/",
    "v4": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/v4/neurons/",
}
outputpath = "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/population_lat/data/all_neurons_in/"

nidpath = {
    # "lip": outputpath + "lip_no_neutral_inout_selectivity.csv",
    # "pfc": outputpath + "pfc_no_neutral_inout_selectivity.csv",
    # "v4": outputpath + "v4_no_neutral_inout_selectivity.csv",
}

allspath = {
    # "lip": "/envau/work/invibe/USERS/LOSADA/Users/losadac/data_stats/neurons/popudata_no_selectivity_neutralinout/lip_"
    # + str(min_trials)
    # + "tr_1sp_Zscore.pickle",
    # "pfc": "/envau/work/invibe/USERS/LOSADA/Users/losadac/data_stats/neurons/popudata_no_selectivity_neutralinout/pfc_"
    # + str(min_trials)
    # + "tr_1sp_Zscore.pickle",
    # "v4": "/envau/work/invibe/USERS/LOSADA/Users/losadac/data_stats/neurons/popudata_no_selectivity_neutralinout/v4_"
    # + str(min_trials)
    # + "tr_1sp_Zscore.pickle",
}

# Define parameters
areas = ["lip", "pfc", "v4"]
avgwin = 100
min_sp_sec = 1
n_test = 1
min_trials = 25
code = 1  # in out
nonmatch = True  # if True: includes nonmatch trials
norm = False
zscore = True
select_n_neu = 100
# sample timing
time_before_sample = 500
start_sample = -200
end_sample = 450 + 400

# test timing
time_before_test = 500
start_test = -400
end_test = n_test * 450 + 200

parameters = {}
parameters["areas"] = areas
parameters["avgwin"] = avgwin
parameters["min_sp_sec"] = min_sp_sec
parameters["n_test"] = n_test
parameters["min_trials"] = min_trials
parameters["code"] = code
parameters["nonmatch"] = nonmatch
parameters["norm"] = norm
parameters["zscore"] = zscore
parameters["select_n_neu"] = select_n_neu
parameters["time_before_sample"] = time_before_sample
parameters["start_sample"] = start_sample
parameters["end_sample"] = end_sample
parameters["time_before_test"] = time_before_test
parameters["start_test"] = start_test
parameters["end_test"] = end_test
parameters["allspath"] = allspath
parameters["nidpath"] = nidpath
parameters["filepaths"] = filepaths
parameters["outputpath"] = outputpath

population_latency(parameters)
