from pathlib import Path
from typing import List, Dict

import cv2
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from lib.data.model.wandb_art import DatasetCfg


class LeafDiseaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: Dict[str, List[DatasetCfg]],
        train_transforms=None,
        val_transforms=None,
        batch_size: int = 200,
        num_workers: int = 4,
        seed: int = 1337,
    ):
        super().__init__()

        self.cfg = cfg
        self.train_transform = train_transforms
        self.val_transform = val_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_dataset = None
        self.val_datasets = None

        self.num_classes = 2

    def fetch_datasets(self, mode):
        assert mode in {"train", "val"}

        datasets = []
        for ds in self.cfg[mode]:
            datasets.append(
                LeafDiseaseDataset(
                    root=ds.root,
                    transforms=self.train_transform
                    if mode == "train"
                    else self.val_transform,
                )
            )
        return datasets

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = ConcatDataset(self.fetch_datasets("train"))
            self.val_datasets = self.fetch_datasets(mode="val")

            print(f"Train size: {len(self.train_dataset)}")
            print(f"Val sizes: {[len(ds) for ds in self.val_datasets]}")

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.val_datasets = self.fetch_datasets(mode="val")

        if stage == "predict":
            self.val_datasets = self.fetch_datasets(mode="val")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                d,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for d in self.val_datasets
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                d,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for d in self.val_datasets
        ]

    def predict_dataloader(self):
        [
            DataLoader(
                d,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for d in self.val_datasets
        ]


class LeafDiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, transforms=None):
        self.root = root
        self.transforms = transforms

        self.img_path = self.root.joinpath("images")
        self.mask_path = self.root.joinpath("masks")

        self.file_paths: List[List[Path]] = self.load()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_paths[idx]

        img = cv2.imread(img_path.as_posix())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8)  # convert to binary mask
        mask = np.expand_dims(mask, axis=0)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        return {"img": img, "mask": mask}

    def load(self):
        file_paths = []
        imgs = self.img_path.glob("*")
        masks = list(self.mask_path.glob("*"))
        for img_p in imgs:
            mask_p = self.mask_path.joinpath(f"{img_p.stem}.png")
            assert mask_p in masks
            file_paths.append([img_p, mask_p])
        return file_paths
