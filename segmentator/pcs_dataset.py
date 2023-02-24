import cv2
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os
from glob import glob


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


def _prepare_splits():
    img_fnames = sorted([f[21:] for f in glob("pcs/png_images/IMAGES/*.png")])
    mask_fnames = [f"png_masks/MASKS/seg_{f[-8:-4]}.png" for f in img_fnames]
    for f in mask_fnames:
        if not os.path.exists(os.path.join("pcs", f)):
            print(f"Missing a mask for {f}")
    fnames = pd.DataFrame(data={"img": img_fnames, "mask": mask_fnames})
    train, val = train_test_split(fnames, test_size=0.1, random_state=1337)

    train.to_csv("pcs/train.csv", header=None, index=None)
    val.to_csv("pcs/val.csv", header=None, index=None)


if __name__ == "__main__":
    _prepare_splits()
