"""Execute main function of the module selectivity."""

from typing import Dict
from . import compute_selectivity


def main(preprocessing: Dict, paths: Dict, **kwargs):

    df_selectivity = compute_selectivity.compute_selectivity(preprocessing, paths)
    print("saving")
    df_selectivity.to_pickle("population_selectivity.pkl")


if __name__ == "__main__":
    main()
