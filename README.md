[![Doc](https://img.shields.io/badge/doc-latest-blue.svg)](https://juliendenize.github.io/eztorch/index.html)
[![License](https://img.shields.io/badge/license-CeCILL--C-green.svg)](LICENSE)

<!-- start intro -->

# Eztorch

## Introduction

Eztorch is a library to make training, validation, and testing in Pytorch *easy* to perform image and video self-supervised representation learning and evaluate those representations on downstream tasks.

It was first developed to factorize code during [Julien Denize](https://juliendenize.github.io/)'s PhD thesis which was on *Self-supervised representation learning and applications to image and video analysis*.
It led to several academic contributions:
<!-- end intro -->

- [Similarity Contrastive Estimation for Self-Supervised Soft Contrastive Learning (WACV 2023)](./docs/source/contributions/sce_wacv.md)
- [Similarity Contrastive Estimation for Image and Video Soft Contrastive Self-Supervised Learning (MVAP 2023)](./docs/source/contributions/sce_mvap.md)
- [COMEDIAN: Self-Supervised Learning and Knowledge Distillation for Action Spotting using Transformers (WACV Workshops 2024)](./docs/source/contributions/comedian.md)

<!-- start readme+ -->

To ease the use of the code, [documentation](https://juliendenize.github.io/eztorch/index.html) has been built.


## How to Install

To install this repository you need to install a recent version of [**Pytorch (>= 2.)**](https://pytorch.org/get-started/locally/) and all Eztorch dependencies.

You can just launch the following command:
```bash
cd eztorch
conda create -y -n eztorch
conda activate eztorch
conda install -y pip
conda install -y -c conda-forge libjpeg-turbo
pip install -e .
pip uninstall -y pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

The argument `-e` makes a dev installation that allows you to make changes in the repository without needing to install the package again. It is optional.

If you want a lighter installation that only installs the main dependencies you need the requirement file by `requirements_lite.txt` and then launch the pip install.

## How to use

1. Read tutorials on Pytorch-Lightning and Hydra to be sure to understand those libraries.

2. Take a look at Eztorch [documentation](https://juliendenize.github.io/eztorch/index.html).

3. Use configs in ``eztorch/configs/run/`` or make your own

4. Pass your config to running scripts in ``run/`` folder.

Eztorch is a library, therefore you can import its components from anywhere as long as your Python environment has Eztorch installed.

```python
from eztorch.models.siamese import SCEModel

model = SCEModel(...)
```

## Dependencies

Eztorch relies on various libraries to handle different parts of the pipeline:
>**Why do something worse than people who know best?**

Its main dependencies are:

- [Pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/stable/) for easy setup of:
   - Preparing data through the datamodules
   - Models through the Lightning modules
   - Training, validating, and testing on various device types (CPU, GPU, TPU) with or without distributed training through the trainer
- [Hydra](https://hydra.cc/) to make configuration of your various experiments:
   - Write configurations in Python or Yaml
   - Enjoy hierarchical configuration
   - Let Hydra instantiate
   - Speak the same language in Bash or Python to configure your jobs
- [Torchaug](https://torchaug.readthedocs.io/en/stable/) for efficient GPU and batched data augmentations as a replacement to Torchvision when relevant.

For specific dependencies, we can cite:

- [Timm](https://github.com/rwightman/pytorch-image-models) to instantiate image models
- [Pytorchvideo](https://pytorchvideo.readthedocs.io/en/stable/) for video pipeline:
   - Clip samplers to select one or multiple clips per video
   - Datasets with decoders to read videos
   - Specific transforms for videos
   - Models for videos

## How to contribute

To contribute follow this process:

0. Make an issue if you find it necessary to discuss the changes with maintainers.

1. Checkout to a new branch.

2. Make your modifications.

3. Document your changes.

4. Ask for merging to main.

5. Follow the merging process with maintainers.

## Issue

If you found an error, have trouble making this work or have any questions, please open an [issue](https://github.com/juliendenize/eztorch/issues) to describe your problem.

## License

This project is under the CeCILL license 2.1.

<!-- end readme+ -->
