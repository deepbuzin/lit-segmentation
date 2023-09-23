from typing import Optional, Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from torchmetrics.functional.classification import binary_jaccard_index


class MetricsCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        gt_masks = batch["mask"]
        pred_masks = outputs["pred_mask"].squeeze()

        metrics = {
            "iou": binary_jaccard_index(pred_masks, gt_masks),
        }
        pl_module.log_dict(metrics)
