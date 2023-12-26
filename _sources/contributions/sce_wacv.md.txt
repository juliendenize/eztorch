# SCE (WACV 2023)

## Introduction

This repository contains the official [Pytorch](https://pytorch.org/) implementation of [Similarity Contrastive Estimation for Self-Supervised Soft Contrastive Learning](https://openaccess.thecvf.com/content/WACV2023/papers/Denize_Similarity_Contrastive_Estimation_for_Self-Supervised_Soft_Contrastive_Learning_WACV_2023_paper.pdf) (SCE) that has been published in the **IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2023**.

## Data preparation

Data preparation details are available [here](../get_started/prepare_data.md).

## Main results

The following results are the main ones reported in our paper:

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/similarity-contrastive-estimation-for-self/self-supervised-image-classification-on)](https://paperswithcode.com/sota/self-supervised-image-classification-on?p=similarity-contrastive-estimation-for-self)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<td align="center"><b>pretrain epochs</b></th>
<td align="center"><b>pretrain crops</b></th>
<td align="center"><b>ImageNet linear accuracy</b></th>
<td align="center"><b>ckpt</b></th>
<!-- TABLE BODY -->
<tr>
<td align="center">100</td>
<td align="center">2x224</td>
<td align="center">72.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1KJcd-EXfduoYJEcoVSAcPxt1Cux19cdU/view?usp=drive_link">Download</a></td>
</tr>
<tr>
<td align="center">200</td>
<td align="center">2x224</td>
<td align="center">72.7</td>
<td align="center"><a href="https://drive.google.com/file/d/1tA3ACMJmOuxQ35MVgPV2-ddnWB5JO-Tm/view?usp=drive_link">Download</a></td>
</tr>
<tr>
<td align="center">300</td>
<td align="center">2x224</td>
<td align="center">73.3</td>
<td align="center"><a href="https://drive.google.com/file/d/1IBU2JQqY7LNEJeMBBxMkVM-B8t-OJGz4/view?usp=drive_link">Download</a></td>
</tr>
<tr>
<td align="center">1000</td>
<td align="center">2x224</td>
<td align="center">74.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1hl3ic69u7VsTn04ZkU8R9ej1Z_IaF-gO/view?usp=drive_link">Download</a></td>
</tr>
<tr>
<td align="center">200</td>
<td align="center">2x224 + 192 + 160 + 128 + 96</td>
<td align="center">75.4</td>
<td align="center"><a href="https://drive.google.com/file/d/1sryGk08gJgq9dHeZT3fNs8_KmW069C0E/view?usp=drive_link">Download</a></td>
</tr>
</tbody></table>

You can find below the command lines to launch the configs to retrieve those results and above the checkpoint links.

## SCE Pretraining

We launched our experiments on a computational cluster configured via SLURM using for the two crops configuration 8 A100-80G GPUs and for the multi-crop configuration 16 A100-80G GPUs.

We provide below the commands using the [srun](https://slurm.schedmd.com/srun.html) command from SLURM that was inside a SLURM script. Pytorch-Lightning directly detects SLURM is used and configures accordingly the distributed training. We strongly suggest you refer to [Pytorch-Lightning's documentation](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu.html) to correctly set up a command line without srun if you do not have access to a slurm cluster.

### Pretraining for 100 epochs
```bash
output_dir=...
dataset_dir=...

config_path="../eztorch/configs/run/pretrain/sce/resnet50"
config_name="resnet50_imagenet"
seed=42

cd eztorch/run

srun --kill-on-bad-exit=1 python pretrain.py \
    -cp $config_path -cn $config_name\
    dir.data=$dataset_dir \
    dir.root=$output_dir \
    dir.exp='pretrain' \
    seed.seed=$seed \
    datamodule.train.loader.num_workers=8 \
    datamodule.val.loader.num_workers=8 \
    trainer.devices=8 \
    trainer.max_epochs=100 \
    model.optimizer.initial_lr=0.6
```
### Pretraining for 200 epochs
```bash
srun --kill-on-bad-exit=1 python pretrain.py \
    -cp $config_path -cn $config_name\
    dir.data=$dataset_dir \
    dir.root=$output_dir \
    dir.exp='pretrain' \
    seed.seed=$seed \
    datamodule.train.loader.num_workers=8 \
    datamodule.val.loader.num_workers=8 \
    trainer.devices=8
```

### Pretraining for 300 epochs
```bash
srun --kill-on-bad-exit=1 python pretrain.py \
    -cp $config_path -cn $config_name\
    dir.data=$dataset_dir \
    dir.root=$output_dir \
    dir.exp='pretrain' \
    seed.seed=$seed \
    datamodule.train.loader.num_workers=8 \
    datamodule.val.loader.num_workers=8 \
    trainer.devices=8 \
    trainer.max_epochs=300
```


### Pretraining for 1000 epochs
```bash
srun --kill-on-bad-exit=1 python pretrain.py \
    -cp $config_path -cn $config_name\
    dir.data=$dataset_dir \
    dir.root=$output_dir \
    dir.exp='pretrain' \
    seed.seed=$seed \
    datamodule.train.loader.num_workers=8 \
    datamodule.val.loader.num_workers=8 \
    trainer.devices=8 \
    trainer.max_epochs=1000 \
    model.optimizer.params.weight_decay=1.5e-6
```

### Pretraining for 200 epochs with multi-crop
```bash
config_name="resnet50_imagenet_five_crops"

srun --kill-on-bad-exit=1 python pretrain.py \
    -cp $config_path -cn $config_name \
    dir.data=$dataset_dir \
    dir.root=$output_dir \
    dir.exp='pretrain' \
    seed.seed=$seed \
    datamodule.train.loader.num_workers=8 \
    datamodule.val.loader.num_workers=8 \
    trainer.devices=8 \
    trainer.num_nodes=2
```
## Linear classification

Same as for pretraining we launched our experiment on a SLURM cluster.

```bash
eval_config_path="../eztorch/configs/run/evaluation/linear_classifier/sce/resnet50"
eval_config_name="resnet50_imagenet"
pretrain_checkpoint=...

srun --kill-on-bad-exit=1 python linear_classifier_evaluation.py \
    -cp $eval_config_path -cn $eval_config_name \
    dir.data=$dataset_dir \
    dir.root=$output_dir \
    dir.exp='linear_classifier_evaluation' \
    model.pretrained_trunk_path=$pretrain_checkpoint \
    seed.seed=$seed \
    datamodule.train.loader.num_workers=8 \
    datamodule.val.loader.num_workers=8 \
    trainer.devices=8
```

We consider by default you use checkpoints you pretrained yourselves.

If this is not the case and you downloaded the checkpoints we provide, do not forget to change the `model.trunk_pattern` config that searches the trunk pattern in the state dict:
```bash

srun --kill-on-bad-exit=1 python linear_classifier_evaluation.py
     ...
     model.trunk_pattern="" \
     ...
```

## Transfer Learning

For Transfer Learning evaluation we used code provided by several authors that we would like to thank for sharing their work. Below we redirect you to their Github for every transfer we have done.

All evaluations are based on the multi-crop checkpoint.
### Linear classifier on other datasets
To evaluate the transferability of our pretrained checkpoints on various datasets by training a linear classifier, we used the [ssl-transfer](https://github.com/linusericsson/ssl-transfer) repository.

<table>
    <tr>
        <th>Method</th>
        <th>Food101</th>
        <th>CIFAR10</th>
        <th>CIFAR100</th>
        <th>SUN397</th>
        <th>Cars</th>
        <th>Aircraft</th>
        <th>VOC2007</th>
        <th>DTD</th>
        <th>Pets</th>
        <th>Caltech101</th>
        <th>Flowers</th>
        <th>Avg.</th>
    </tr>
    <tr>
        <td>SimCLR</td>
        <td>72.8</td>
        <td>90.5</td>
        <td>74.4</td>
        <td>60.6</td>
        <td>49.3</td>
        <td>49.8</td>
        <td>81.4</td>
        <td>75.7</td>
        <td>84.6</td>
        <td>89.3</td>
        <td>92.6</td>
        <td>74.6</td>
    </tr>
    <tr>
        <td>BYOL</td>
        <td>75.3</td>
        <td>91.3</td>
        <td>78.4</td>
        <td>62.2</td>
        <td>67.8</td>
        <td>60.6</td>
        <td>82.5</td>
        <td>75.5</td>
        <td>90.4</td>
        <td>94.2</td>
        <td>96.1</td>
        <td>79.5</td>
    </tr>
    <tr>
        <td>NNCLR</td>
        <td>76.7</td>
        <td>93.7</td>
        <td>79.0</td>
        <td>62.5</td>
        <td>67.1</td>
        <td>64.1</td>
        <td>83.0</td>
        <td>75.5</td>
        <td>91.8</td>
        <td>91.3</td>
        <td>95.1</td>
        <td>80</td>
    </tr>
    <tr>
        <td><b>SCE</td>
        <td><b>77.7</b></td>
        <td><b>94.8</b></td>
        <td><b>80.4</b></td>
        <td><b>65.3</b></td>
        <td><b>65.7</b></td>
        <td><b>59.6</b></td>
        <td><b>84.0</b></td>
        <td><b>77.1</b></td>
        <td><b>90.9</b></td>
        <td><b>92.7</b></td>
        <td><b>96.1</b></td>
        <td><b>80.4</b></td>
    </tr>
    <tr>
        <td>Supervised</td>
        <td>72.3</td>
        <td>93.6</td>
        <td>78.3</td>
        <td>61.9</td>
        <td>66.7</td>
        <td>61.0</td>
        <td>82.8</td>
        <td>74.9</td>
        <td>91.5</td>
        <td>94.5</td>
        <td>94.7</td>
        <td>79.3</td>
    </tr>
</table>

### SVM classifier on PASCAL VOC 2007/2012
To evaluate the transferability of our pretrained checkpoints on PASCAL VOC by training an SVM classifier, we used the [PCL](https://github.com/salesforce/PCL) repository.

<table>
    <tr>
        <th>Method</th>
        <th>K = 16</th>
        <th>K = 32</th>
        <th>K = 64</th>
        <th>full</th>
    </tr>
    <tr>
        <td>MoCov2</td>
        <td>76.14</td>
        <td>79.16</td>
        <td>81.52</td>
        <td>84.60</td>
    </tr>
    <tr>
        <td>PCLv2</td>
        <td>78.34</td>
        <td>80.72</td>
        <td>82.67</td>
        <td>85.43</td>
    </tr>
    <tr>
        <td>ReSSL</td>
        <td>79.17</td>
        <td>81.96</td>
        <td>83.81</td>
        <td>86.31</td>
    </tr>
    <tr>
        <td>SwAV</td>
        <td>78.38</td>
        <td>81.86</td>
        <td>84.40</td>
        <td>87.47</td>
    </tr>
    <tr>
        <td>WCL</td>
        <td>80.24</td>
        <td>82.97</td>
        <td>85.01</td>
        <td>87.75</td>
    </tr>
    <tr>
        <td><b>SCE</b></td>
        <td><b>79.47</b></td>
        <td><b>83.05</b></td>
        <td><b>85.47</b></td>
        <td><b>88.24</b></td>
    </tr>
</table>

### Object detection and Mask Segmentation on COCO
To evaluate the transferability of our pretrained checkpoints on COCO by training a Mask R-CNN for object detection and mask segmentation, we used the [triplet](https://github.com/wanggrun/triplet) repository.

<table>
    <tr>
        <th>Method</th>
        <th>AP Box</th>
        <th>AP Mask</th>
    </tr>
    <tr>
        <td>MoCo</td>
        <td>40.9</td>
        <td>35.5</td>
    </tr>
    <tr>
        <td>MoCov2</td>
        <td>40.9</td>
        <td>35.5</td>
    </tr>
    <tr>
        <td>SimCLR</td>
        <td>39.6</td>
        <td>34.6</td>
    </tr>
    <tr>
        <td>BYOL</td>
        <td>40.3</td>
        <td>35.1</td>
    </tr>
    <tr>
        <td><b>SCE</b></td>
        <td><b>41.6</b></td>
        <td><b>36.0</b></td>
    </tr>
    <tr>
        <td>Truncated-Triplet</td>
        <td>41.7</td>
        <td>36.2</td>
    </tr>
    <tr>
        <td>Supervised</td>
        <td>40.0</td>
        <td>34.7</td>
    </tr>
</table>

## Issue

If you found an error, have trouble making this work or have any questions, please open an [issue](https://github.com/juliendenize/eztorch/issues) to describe your problem.

## Acknowledgment
This work was made possible by the use of the
Factory-AI supercomputer, financially supported by the Ile-de-France Regional Council.

## Citation

If you found our work useful, please consider citing us:

```
@InProceedings{Denize_2023_WACV,
    author    = {Denize, Julien and Rabarisoa, Jaonary and Orcesi, Astrid and H\'erault, Romain and Canu, St\'ephane},
    title     = {Similarity Contrastive Estimation for Self-Supervised Soft Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {2706-2716}
}
```
