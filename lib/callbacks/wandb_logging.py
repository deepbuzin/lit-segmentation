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

        imgs = [img for img in batch["img"][indices].cpu()]
        gt_masks = [gt_m for gt_m in batch["mask"][indices].cpu()]
        pred_masks = [pred_m for pred_m in outputs["pred_mask"][indices].cpu()]

        # columns = ["image", "gt mask", "pred mask", "step"]
        data = []

        for img, gt_m, pred_m in zip(imgs, gt_masks, pred_masks):
            data.append(

                    wandb.Image(
                        img.permute(1, 2, 0).numpy(),
                        masks={
                            "pred": {
                                "mask_data": pred_m.squeeze().numpy()
                            },
                            "gt": {
                                "mask_data": gt_m.squeeze().numpy()
                            }
                        }
                    ),

            )

        wandb_logger.log_image(
            key=f"val_samples/dataloader_idx_{dataloader_idx}",
            images=data,
        )
