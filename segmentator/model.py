from functools import partial

import torch
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
from timm.scheduler.scheduler import Scheduler


class SegmentationModel(pl.LightningModule):
    def __init__(self, model_instance: torch.nn.Module,
                 optimizer_partial: partial = None,
                 scheduler_partial: partial = None):
        super().__init__()
        self.model = model_instance
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        x = self.model(x)
        return x

    def common_step(self, batch, stage):
        image = batch["image"]
        assert image.ndim == 4  # Shape of the image should be (batch_size, num_channels, height, width)
        assert image.shape[2] % 32 == 0 and image.shape[3] % 32 == 0  # Check that image dimensions are divisible by 32 to comply with network's downscaling factor

        mask = batch["mask"].long()
        assert mask.ndim == 3  # Shape of the mask should be [batch_size, num_classes, height, width]

        logits_mask = self.forward(image)["out"]  # Predicted mask contains logits
        loss = self.loss_fn(logits_mask,
                            mask)  # Predicted mask contains logits, and loss_fn param `from_logits` is set to True

        prob_mask = logits_mask.sigmoid()  # convert mask values to probabilities
        pred_mask = (prob_mask > 0.5).float()  # apply thresholding

        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(pred_mask, dim=1).long(),
                                               mask.long(),
                                               mode="multiclass",
                                               num_classes=59)
        if stage == "train":
            self.training_step_outputs.append({"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
        elif stage == "val":
            self.validation_step_outputs.append({"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
        else:
            raise ValueError(f"Unknown stage: {stage}")
        return loss

    def common_epoch_end(self, outputs, stage):
        loss = outputs[-1]["loss"]
        tp = torch.cat([x["tp"] for x in outputs]).long()
        fp = torch.cat([x["fp"] for x in outputs]).long()
        fn = torch.cat([x["fn"] for x in outputs]).long()
        tn = torch.cat([x["tn"] for x in outputs]).long()

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")  # calculate IoU for each image and then compute mean over these scores
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")  # aggregate intersection and union over whole dataset and then compute IoU score

        metrics = {f"{stage}_per_image_iou": per_image_iou,
                   f"{stage}_dataset_iou": dataset_iou,
                   f"{stage}_loss": loss}
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, "train")

    def on_training_epoch_end(self):
        self.common_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, "val")

    def on_validation_epoch_end(self):
        self.common_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        if self.optimizer_partial is None or self.scheduler_partial is None:
            raise ValueError("Missing optimizer_partial or scheduler_partial")
        optimizer = self.optimizer_partial(self.parameters())
        scheduler = self.scheduler_partial(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler: Scheduler, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value

