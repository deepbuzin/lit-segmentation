import sys
sys.path.append("..")

from pprint import pprint
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    trainer = hydra.utils.instantiate(cfg.trainer)

    # trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    valid_metrics = trainer.validate(model, datamodule=datamodule)
    pprint(valid_metrics)


if __name__ == "__main__":
    main()

