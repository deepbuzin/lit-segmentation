from functools import partial

import timm
import torch
from lib.core import SegmentationModel
from lib.models.unet import UNet


num_classes = 1

backbone = timm.create_model("convnext_pico.d1_in1k",
                             pretrained=True,
                             features_only=True)

model_instance = UNet(encoder=backbone, num_classes=num_classes)


def binary_dice_loss(inputs, targets, smooth=1e-4):
    inputs = torch.nn.functional.sigmoid(inputs.squeeze())
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
    return 1 - dice


loss_fn = binary_dice_loss

optimizer_partial = partial(torch.optim.Adam, weight_decay=0.00001, lr=0.0001)

scheduler_partial = partial(timm.scheduler.cosine_lr.CosineLRScheduler,
                            t_initial=10,
                            lr_min=0.00001,
                            cycle_decay=0.5,
                            warmup_t=2,
                            warmup_lr_init=0.00001,
                            warmup_prefix=True,
                            cycle_limit=5,
                            cycle_mul=2)


model = SegmentationModel(model_instance=model_instance,
                          loss_fn=loss_fn,
                          optimizer_partial=optimizer_partial,
                          scheduler_partial=scheduler_partial)

ckpt_path = None
