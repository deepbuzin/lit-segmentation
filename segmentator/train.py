import os
from pprint import pprint

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from segmentator.model import FashionModel
from segmentator.pcs_dataset import PcsDataset


def get_datasets(root):
    train_transform = A.Compose(
        [
            A.SmallestMaxSize(256),
            A.RandomCrop(256, 256),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.SmallestMaxSize(256),
            A.CenterCrop(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    train_dataset = PcsDataset(root=root, mode='train', transform=train_transform)
    val_dataset = PcsDataset(root=root, mode='val', transform=val_transform)

    assert set(train_dataset.filenames).isdisjoint(set(val_dataset.filenames))
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    return train_dataset, val_dataset


def setup_training(data_root, batch_size=16, num_workers=4, arch="FPN", encoder_name="resnet34", max_epochs=10):
    train_dataset, val_dataset = get_datasets(data_root)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", save_top_k=-1)
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min")
    model = FashionModel(arch=arch, encoder_name=encoder_name, in_channels=3, out_classes=train_dataset.num_classes)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    return {"train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "model": model,
            "trainer": trainer}


def train(train_dataloader, val_dataloader, model, trainer):
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    valid_metrics = trainer.validate(model, dataloaders=val_dataloader, verbose=False)
    pprint(valid_metrics)


if __name__ == "__main__":
    train_setting = dict(data_root='../pcs',
                         batch_size=16,
                         num_workers=os.cpu_count(),
                         arch="FPN",
                         encoder_name="resnet34",
                         max_epochs=10)
    train(**setup_training(**train_setting))
