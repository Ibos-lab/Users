"""Main script to run a pipeline."""

# Authors: Camila Losada, camilaalosada@gmail.com
# Date: 05/2024
from pathlib import Path
import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="/envau/work/invibe/USERS/LOSADA/Users/losadac/conf/",  # "C:/Users/camil/Documents/int/scripts/Users/losadac/conf/",  #
    config_name="config.yaml",
)
def main(cfg: DictConfig):
    # run pipeline
    params = {}
    # if "workspace" in cfg:
    #     params["workspace"] = cfg.workspace
    opt = hydra.utils.call(cfg.pipelines, **params)


if __name__ == "__main__":
    main()
