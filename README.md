# Segmentation with PyTorch Lightning

Here I will explore some approaches currently popular withing the segmentation domain.
The goal here is to get familiar with the Lightning framework, and gain better knowledge
of segmentation and deep learning computer vision in general.

## Data

The dataset used in this notebook is
[People Clothing Segmentation](https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation)
taken from Kaggle. It's quite small and clean, which makes it manageable
for the pet project scope.

## Tools

- [PyTorch Lightning](https://www.pytorchlightning.ai/) is used to build the training pipeline;
- [timm](https://github.com/huggingface/pytorch-image-models) is used to load pretrained feature extractors, optimizers and schedulers;
- [Albumentations](https://albumentations.ai/) is used to handle image preprocessing and augmentation;
- [Hydra](https://github.com/facebookresearch/hydra) is used to manage the configuration.





