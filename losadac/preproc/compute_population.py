# pipeline start
# define functions to get populations processed in different useful ways
#
# * la idea es que reciban como input una lista de diccionarios. Cada
# diccionario contiene los parametros necesarios para alinear los spikes a un evento en particular
# * tambien puede recibir como input attributos que se desean eliminar del objeto neurona (se elimina la informacion pero no el atributo per se)

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.structures.population_data import PopulationData
import pandas as pd
import glob


def read_and_compute(path, params, rf_loc):
    neu = NeuronData.from_python_hdf5(path)
    neu = neu.get_neu_align(params=params, delete_att=["sp_samples"], rf_loc=rf_loc)
    return neu


def run_compute_population(paths, processing, **kwargs):
    params = []
    for idict in processing:
        params.append(processing[idict])
    neu_path = paths["input_files"] + "*neu.h5"
    path_list = glob.glob(neu_path)
    rf_loc = None
    if paths["input_rf_loc"] is not None:
        rf_loc = pd.read_csv(paths["input_rf_loc"])

    population = Parallel(n_jobs=-1)(
        delayed(read_and_compute)(path, params, rf_loc) for path in tqdm(path_list)
    )

    PopulationData(population).to_python_hdf5("population.h5")
