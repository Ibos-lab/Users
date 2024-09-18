import numpy as np
from typing import Dict
from ephysvibe.structures.neuron_data import NeuronData
from ephysvibe.structures.population_data import PopulationData
from ephysvibe.trials.spikes import firing_rate
from ephysvibe.trials import select_trials
import warnings








def plot_trials(paths: list):

        neu = NeuronData.from_python_hdf5(path)
        nid = neu.get_neuron_id()
       
            fig = neu.plot_sp_b1()
            s_path = os.path.normpath(path).split(os.sep)
            ss_path = s_path[-1][:-7]
            fig.savefig(
                plots_path + ss_path + ".png",
                format="png",
                bbox_inches="tight",
                transparent=False,
            )
            plt.close(fig)


