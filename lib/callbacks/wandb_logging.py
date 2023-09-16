from typing import Optional, Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

import wandb
import numpy as np


class WandbLogCallback(Callback):
    def __init__(self, num_samples: int = 8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if not batch_idx == 0:
            return

        wandb_logger = pl_module.logger
        batch_size = batch["img"].shape[0]
        indices = np.random.choice(
            list(range(batch_size)), size=self.num_samples, replace=False
        )

        imgs = [img for img in batch["img"][indices]]
        gt_masks = batch["mask"][indices]
        pred_masks = outputs["pred_mask"][indices]

        columns = ["image", "gt mask", "pred mask", "step"]
        data = []
        for img, gt_m, pred_m in zip(imgs, gt_masks, pred_masks):
            data.append(
                [
                    wandb.Image(img),
                    wandb.Image(gt_m),
                    wandb.Image(pred_m),
                    trainer.global_step,
                ]
            )

        wandb_logger.log_table(
            key=f"val_samples/dataloader_idx_{dataloader_idx}",
            columns=columns,
            data=data,
        )
