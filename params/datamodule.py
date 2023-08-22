import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from lib.data.leafs import LeafDiseaseDataModule

art_owner_entity = "andreybuzin"
project = "leafs"

TRAIN_DATASETS = []

VAL_DATASETS = []

train_run = wandb.init(project=project)

image_size = 224

train_transforms = A.Compose(
    [
        A.SmallestMaxSize(image_size + 20),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


val_transforms = A.Compose(
    [
        A.SmallestMaxSize(image_size),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

datamodule = LeafDiseaseDataModule()
