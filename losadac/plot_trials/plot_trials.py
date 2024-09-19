from ephysvibe.structures.neuron_data import NeuronData
from pathlib import Path
import matplotlib.pyplot as plt


def plot_trials(neupath: Path, format: str = "png"):

    neu = NeuronData.from_python_hdf5(neupath)
    nid = neu.get_neuron_id()
    fig = neu.plot_sp_b1()
    fig.savefig(
        f"{nid}.{format}",
        format=format,
        bbox_inches="tight",
        transparent=False,
    )
    plt.close(fig)
