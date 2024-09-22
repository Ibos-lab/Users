from typing import Dict
import plot_trials
import glob
from joblib import Parallel, delayed
from tqdm import tqdm


# def main(paths: Dict, params, **kwargs):
#     print("start plot trials")
#     path_list = glob.glob(paths["input"])

#     Parallel(n_jobs=-1)(
#         delayed(plot_trials.plot_trials)(
#             neupath=path,
#             format=params["format"],
#             percentile=params["percentile"],
#             cerotr=params["cerotr"],
#         )
#         for path in tqdm(path_list)
#     )


"""Execute main function of the module plot_trials."""


def main(paths: Dict, params, **kwargs):
    print("start plot trials")
    path_list = glob.glob(paths["input"])

    for path in tqdm(path_list):
        plot_trials.plot_trials(
            neupath=path,
            format=params["format"],
            percentile=params["percentile"],
            cerotr=params["cerotr"],
        )


if __name__ == "__main__":
    paths = {
        "input": "/envau/work/invibe/USERS/IBOS/data/Riesling/TSCM/OpenEphys/new_structure/session_struct/lip/neurons/*neu.h5"
    }
    params = {"format": "png", "percentile": False, "cerotr": False}
    main(paths, params)
