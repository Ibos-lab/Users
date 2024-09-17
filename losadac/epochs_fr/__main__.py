"""Execute main function of the module epochs_fr."""

from typing import Dict
from . import compute_epochs_fr


def main(preprocessing: Dict, paths: Dict, **kwargs):

    df_epochs_fr = compute_epochs_fr.compute_epochs_fr(preprocessing, paths)
    print("saving")
    df_epochs_fr.to_pickle("population_epochs_fr.pkl")


if __name__ == "__main__":
    main()
