"""Execute main function of the module epochs_fr."""

from typing import Dict
from . import plot_trials


def main(paths: Dict, **kwargs):

    plot_trials.plot_trials(paths)


if __name__ == "__main__":
    main()
