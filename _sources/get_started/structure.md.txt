# Library structure

## Introduction

Eztorch as a library has a specific organization for how and where the files are created. Names of folders and files should be informative of their content.

## General structure

The General tree structure of Eztorch is the following:

```bash
eztorch
├── docs
├── eztorch
├── run
└── tests
```

### Doc

In `docs/`, you will find source documentation files about Eztorch such as in this file. We aim to make Eztorch accessible to anyone.

### Eztorch

The folder `eztorch/` contains the actual library that we will detail below.

### Run

The folder `run/` contains various scripts to make training or evaluation. These scripts thanks a Hydra configuration should perform a task agnostically to specific classes or architectures and be able to generalize to any relevant Hydra configuration.

### Tests

The folder `tests/` contains various tests to ensure that components in Eztorch properly work.

As a general rule of thumb, the best would be to test individual components every time they are added, which is not the case for now.


## Library structure

The tree structure of the library is the following:

```bash
eztorch/eztorch
├── callbacks
├── configs
├── datamodules
├── datasets
├── evaluation
├── losses
├── models
├── optimizers
├── schedulers
├── transforms
├── utils
└── visualization
```

### Callbacks

The `callbacks/` folder contains [callbacks](https://pytorch-lightning.readthedocs.io/en/stable/api_references.html#callbacks) to feed the [trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) based on Pytorch-Lightning.

### Datamodules

The `datamodules/` folder contains [datamodules](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html?highlight=datamodule) for various image and video datasets.

### Datasets

The `datasets/` folder contains datasets tools for various image and video datasets.

Among these tools, there are:
- Datasets for videos, for example `datasets/hmdb51.py`.
- Wrapper around datasets, for example `datasets/dict_dataset.py`.
- Other tools such as clip samplers for video clip generation in `datasets/clip_samplers/`.

### Evaluation

The `evaluation/` folder contains the tools for evaluation networks.
For now, there are:
- Linear classifier evaluation for SSL protocol.
- Testing time augmentations functions to average predictions of several augmentations.

### Losses

The `losses/` folder contains individual component losses.

### Models

The `models/` folder contains models and tools to configure the various models supported by Eztorch for images and videos.

It follows the following structure:

```bash
eztorch/eztorch/models
├── heads
├── modules
├── siamese
└── trunks
```

#### Heads

The `heads/` folder contains various heads for your models.

Heads are the part of your models that should specialize to a task:
- Linear heads for classification task.
- MLP heads for self-supervised learning projections and predictions.
- ...

#### Modules

The `modules/` folder contains layers or modules for your models that accomplish a specific action.

It could be gathering tensors, or splitting batch normalization layers in parts, ...

#### Siamese

The `siamese/` folder contains siamese models for self-supervised learning.

Currently implemented methods are:
- MoCo (v1, v2, v3)
- ReSSL
- SCE
- SimCLR

#### Trunks

The `trunks/` folder contains the trunks, backbone, or encoders, used in Eztorch for Image and Video modalities.

Timm and Pytorchvideo models are also available.

### Optimizers

The `optimizers/` folder contains the custom optimizers (such as LARS) but also factories to easily instantiate them.

### Schedulers

The `schedulers/` folder contains the custom schedulers (such as Cosine Annealing) but also factories to easily instantiate them.

### Transforms

The `transforms/` folder contains all custom transforms defined by Eztorch for images and videos.

Transforms from Torchvision or Kornia are also available. The latter permits GPU transforms.

### Utils

The `utils/` folder contains all utils functions to help Eztorch functionalities and allow various training tricks.

### Visualization

The `visualization/` contains tools to visualize data.
