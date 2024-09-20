"""Execute main function of the module plot_trials."""

from typing import Dict
from . import plot_trials
import glob
from joblib import Parallel, delayed
from tqdm import tqdm


def main(paths: Dict, params, **kwargs):
    print("start plot trials")
    path_list = glob.glob(paths["input"])

    Parallel(n_jobs=-1)(
        delayed(plot_trials.plot_trials)(
            neupath=path, format=params["format"], q1=params["q1"], q2=params["q2"]
        )
        for path in tqdm(path_list)
    )


if __name__ == "__main__":
    main()
