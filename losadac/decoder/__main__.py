"""Execute main function of the module decoder."""

from typing import Dict
from . import compute_decoding


def main(preprocessing: Dict, decoder: Dict, paths: Dict, **kwargs):

    res = compute_decoding.compute_decoding(preprocessing, decoder, paths)
    res.to_python_hdf5("performance_decoder.h5")


if __name__ == "__main__":
    main()
