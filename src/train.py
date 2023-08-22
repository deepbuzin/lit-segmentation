import sys
from pathlib import Path
from pprint import pprint
import os

os.environ['WANDB_MODE'] = "disabled"
sys.path.append(Path(__file__).parent.parent.as_posix())

from params.model import model
from params.datamodule import datamodule
from params.trainer import trainer


def main() -> None:
    trainer.fit(model, datamodule=datamodule)
    valid_metrics = trainer.validate(model, datamodule=datamodule)
    pprint(valid_metrics)


if __name__ == "__main__":
    main()

