"""Execute main function of the module population distance."""

from typing import Dict
from . import compute_population_distance


def main(preprocessing: Dict, paths: Dict, **kwargs):

    res = compute_population_distance.compute_distance(preprocessing, paths)
    print("saving")
    res.to_python_hdf5("population_distance.h5")


if __name__ == "__main__":
    main()
