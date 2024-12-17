import receptive_field

processing = {}
paths = {
    "input_files": "//envau_cifs.intlocal.univ-amu.fr/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/pfc/neurons/",
    "input_rf_loc": "//envau_cifs.intlocal.univ-amu.fr/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/activation_index/rf_loc_df_pfc.csv",
}

receptive_field.run_rf(paths, processing)
