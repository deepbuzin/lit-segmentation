# Segmentation with PyTorch Lightning

Here I will explore some approaches currently popular withing the segmentation domain.
The goal here is to get familiar with the Lightning framework, and gain better knowledge
of segmentation and deep learning computer vision in general.

This project is meant as a bootstrap for research in semantic segmentation, and currently
has zero warnings in place to prevent faulty configuration.

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


## How to

### Train a model

1. Configure the training pipeline: refer to the `configs` directory for reference;
2. Run `python3 train.py -cn <target config>`.

### Add a new architecture

1. Drop the code in `models` directory;
2. Put the factory function as a `model_instance._target_` in the pipeline configuration;
3. Bob's your uncle.

### Train on custom data

1. Create a datamodule with reference to `pcs_dataset.py`;
2. Add corresponding configuration in `configs/datamodules` directory;
3. Override the default datamodule in the pipeline configuration.