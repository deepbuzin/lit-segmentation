import os
from glob import glob

import albumentations as A
import cv2
import pandas as pd
import lightning.pytorch as pl
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader


class PcsDataModule(pl.LightningDataModule):
    num_classes = 59

    def __init__(self, root: str = "../pcs", batch_size: int = 16, random_state: int = 1337, num_workers: int = 4):
        super().__init__()
        self.root = root

        self.train_transform = A.Compose(
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

        self.val_transform = A.Compose(
            [
                A.SmallestMaxSize(256),
                A.CenterCrop(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None

        self.random_state = random_state
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self):
        img_fnames = sorted(["/".join(f.split("/")[-3:]) for f in glob(f"{self.root}/png_images/IMAGES/*.png")])
        mask_fnames = [f"png_masks/MASKS/seg_{f[-8:-4]}.png" for f in img_fnames]
        for f in mask_fnames:
            if not os.path.exists(os.path.join(self.root, f)):
                print(f"Missing a mask for {f}")
        fnames = pd.DataFrame(data={"img": img_fnames, "mask": mask_fnames})
        train, val = train_test_split(fnames, test_size=0.1, random_state=self.random_state)

        train.to_csv(f"{self.root}/train.csv", header=None, index=None)
        val.to_csv(f"{self.root}/val.csv", header=None, index=None)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = PcsDataset(root=self.root, mode='train', transform=self.train_transform)
            self.val_dataset = PcsDataset(root=self.root, mode='val', transform=self.val_transform)

            assert set(self.train_dataset.filenames).isdisjoint(set(self.val_dataset.filenames))
            print(f"Train size: {len(self.train_dataset)}")
            print(f"Val size: {len(self.val_dataset)}")

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.val_dataset = PcsDataset(root=self.root, mode='val', transform=self.val_transform)

        if stage == "predict":
            self.val_dataset = PcsDataset(root=self.root, mode='val', transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class PcsDataset(torch.utils.data.Dataset):
    num_classes = 59

    def __init__(self, root, mode="train", transform=None):
        assert mode in {"train", "val"}
        self.root = root
        self.mode = mode
        self.transform = transform
        self.filenames = self._read_split()  # read train/val split

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_filename, mask_filename = self.filenames[idx]
        image = cv2.imread(os.path.join(self.root, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.root, mask_filename), cv2.IMREAD_UNCHANGED)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return dict(image=image, mask=mask)

    def _read_split(self):
        split_filename = "val.csv" if self.mode == "val" else "train.csv"
        split_path = os.path.join(self.root, split_filename)
        with open(split_path) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [tuple(x.split(",")) for x in split_data]
        return filenames
