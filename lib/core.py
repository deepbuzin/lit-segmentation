from functools import partial

import torch
import lightning.pytorch as pl
from timm.scheduler.scheduler import Scheduler


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model_instance: torch.nn.Module,
        loss_fn=None,
        optimizer_partial: partial = None,
        scheduler_partial: partial = None,
    ):
        super().__init__()
        self.model = model_instance
        self.loss_fn = loss_fn
        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial

    def forward(self, x):
        x = self.model(x)
        return x

    def common_step(self, batch, stage):
        image = batch["img"]
        assert image.ndim == 4
        assert (
            image.shape[2] % 32 == 0 and image.shape[3] % 32 == 0
        )  # Check that image dimensions are divisible by 32 to comply with network's downscaling factor

        mask = batch["mask"].float()
        assert mask.ndim == 3

        logits_mask = self.forward(image)["out"]
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        return {"loss": loss, "prob_mask": prob_mask, "pred_mask": pred_mask}

    def training_step(self, batch, batch_idx):
        out = self.common_step(batch, "train")
        self.log("train_loss", out["loss"])
        return out

    def validation_step(self, batch, batch_idx):
        out = self.common_step(batch, "val")
        self.log("val_loss", out["loss"])
        return out

    def configure_optimizers(self):
        assert self.optimizer_partial is not None
        assert self.scheduler_partial is not None

        optimizer = self.optimizer_partial(self.parameters())
        scheduler = self.scheduler_partial(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler: Scheduler, metric):
        scheduler.step(
            epoch=self.current_epoch
        )  # timm's scheduler need the epoch value
