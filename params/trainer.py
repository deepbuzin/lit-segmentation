from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from lib.callbacks.metrics import MetricsCallback
from lib.callbacks.wandb_logging import WandbLogCallback
from params.datamodule import project


wandb_logger = WandbLogger(project=project, log_model=True)


trainer = Trainer(
    logger=wandb_logger,
    log_every_n_steps=10,
    accelerator="gpu",
    precision="16-mixed",
    devices=1,
    max_epochs=30,
    callbacks=[
        ModelCheckpoint(save_last=True, monitor="val_loss", save_top_k=1, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        WandbLogCallback(num_samples=10),
        MetricsCallback(),
    ],
)
