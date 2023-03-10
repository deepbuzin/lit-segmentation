import sys
sys.path.append("..")

import os
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from segmentator.model import PcsModel
from segmentator.pcs_dataset import PcsDataModule


def setup_training(data_root, learning_rate=0.0001, batch_size=16, mixed_precision=False,
                   num_workers=4, arch="FPN", backbone="resnet34", max_epochs=10):
    datamodule = PcsDataModule(root=data_root, batch_size=batch_size, num_workers=num_workers)
    model = PcsModel(learning_rate=learning_rate, arch=arch, backbone=backbone, num_classes=datamodule.num_classes)

    # checkpoint_callback = ModelCheckpoint(monitor="train_loss", save_top_k=-1)
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=[early_stop_callback, lr_monitor],
        precision=16 if mixed_precision else 32,
        accumulate_grad_batches=4,
        # auto_scale_batch_size="binsearch",
        # auto_lr_find=True,
    )
    return {"model": model,
            "datamodule": datamodule,
            "trainer": trainer}


def train(model, datamodule, trainer):
    # trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    valid_metrics = trainer.validate(model, datamodule=datamodule)
    pprint(valid_metrics)


if __name__ == "__main__":
    train_setting = dict(data_root="../pcs",
                         learning_rate=1e-4,
                         batch_size=18,
                         mixed_precision=True,
                         num_workers=os.cpu_count(),
                         arch="fcn",
                         backbone="convnext_tiny",
                         max_epochs=100)
    train(**setup_training(**train_setting))

