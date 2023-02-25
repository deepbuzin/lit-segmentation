import os
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from segmentator.model import FashionModel
from segmentator.pcs_dataset import PcsDataModule


def setup_training(data_root, batch_size=16, num_workers=4, arch="FPN", encoder_name="resnet34", max_epochs=10):
    datamodule = PcsDataModule(root=data_root, batch_size=batch_size, num_workers=num_workers)
    model = FashionModel(arch=arch, encoder_name=encoder_name, in_channels=3, out_classes=datamodule.num_classes)

    checkpoint_callback = ModelCheckpoint(monitor="train_loss", save_top_k=-1)
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min")
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    return {"model": model,
            "datamodule": datamodule,
            "trainer": trainer}


def train(model, datamodule, trainer):
    trainer.fit(model, datamodule=datamodule)
    valid_metrics = trainer.validate(datamodule=datamodule)
    pprint(valid_metrics)


if __name__ == "__main__":
    train_setting = dict(data_root="../pcs",
                         batch_size=16,
                         num_workers=os.cpu_count(),
                         arch="FPN",
                         encoder_name="resnet34",
                         max_epochs=10)
    train(**setup_training(**train_setting))

