import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from timm.optim import AdaBelief
from timm.scheduler import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler


class FashionModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, learning_rate=0.0001, **kwargs):
        super().__init__()
        self.model = smp.create_model(arch, encoder_name=encoder_name, encoder_weights="imagenet",
                                      in_channels=in_channels, classes=out_classes, **kwargs)
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.model(x)
        return x

    def common_step(self, batch, stage):
        image = batch["image"]
        assert image.ndim == 4  # Shape of the image should be (batch_size, num_channels, height, width)
        assert image.shape[2] % 32 == 0 and image.shape[
            3] % 32 == 0  # Check that image dimensions are divisible by 32 to comply with network's downscaling factor

        mask = batch["mask"].long()
        assert mask.ndim == 3  # Shape of the mask should be [batch_size, num_classes, height, width]

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask,
                            mask)  # Predicted mask contains logits, and loss_fn param `from_logits` is set to True

        prob_mask = logits_mask.sigmoid()  # convert mask values to probabilities
        pred_mask = (prob_mask > 0.5).float()  # apply thresholding

        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(pred_mask, dim=1).long(), mask.long(), mode="multiclass",
                                               num_classes=59)
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

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

    def training_epoch_end(self, outputs):
        return self.common_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.common_epoch_end(outputs, "val")

    def configure_optimizers(self):
        optimizer = AdaBelief(self.parameters(), lr=self.learning_rate, weight_decay=0.000001)
        scheduler = CosineLRScheduler(optimizer, t_initial=10, lr_min=0.0000003,
                                      cycle_decay=0.8, warmup_t=5, warmup_lr_init=0.00001)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler: Scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value

