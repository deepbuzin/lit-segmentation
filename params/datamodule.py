import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from lib.data.model.wandb_art import DatasetCfg
from lib.data.leafs import LeafDiseaseDataModule

entity = "andreybuzin"
project = "leafs"

TRAIN_DATASETS = [DatasetCfg(art_id=f"{entity}/plants_arts/leaf_disease_train:v0")]

VAL_DATASETS = [DatasetCfg(art_id=f"{entity}/plants_arts/leaf_disease_train:v0")]

train_run = wandb.init(entity=entity, project=project, job_type="train")
for ds in TRAIN_DATASETS + VAL_DATASETS:
    ds.register_art(train_run)

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

datamodule = LeafDiseaseDataModule(
    cfg={"train": TRAIN_DATASETS, "val": VAL_DATASETS},
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    batch_size=200,
    num_workers=4,
)
