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

The goal is to make each available architecture compatible with [timm](https://github.com/huggingface/pytorch-image-models) feature extractors.
The [Albumentations](https://albumentations.ai/) library is used to handle image preprocessing and augmentation.
General training pipeline is built using [PyTorch Lightning](https://www.pytorchlightning.ai/).





