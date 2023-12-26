# SCE Image & Video (MVAP 2023)

## Introduction

This repository contains the official [Pytorch](https://pytorch.org/) implementation of [Similarity Contrastive Estimation for Image and Video Soft Contrastive Self-Supervised Learning](https://link.springer.com/article/10.1007/s00138-023-01444-9) (SCE) that has been published in the journal **Machine Vision and Applications (2023)**.

## Data preparation

Data preparation details are available [here](../get_started/prepare_data.md).

## SCE for images

Doc is available [here](./sce_wacv.md).

## SCE for videos

We launched our experiments on computational clusters configured via SLURM using up to 16 A100-80G GPUs depending on the experiments.

We provide below the commands using the [srun](https://slurm.schedmd.com/srun.html) command from SLURM that was inside a SLURM script. Pytorch-Lightning directly detects SLURM is used and configures accordingly the distributed training. We strongly suggest you refer to [Pytorch-Lightning's](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu.html)[ documentation](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu.html) to correctly set up a command line without srun if you do not have access to a slurm cluster.


We launched our experiments on a computational cluster configured via SLURM.

### Main results

Results obtained on Kinetics 400. We provide the encoder checkpoints.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/similarity-contrastive-estimation-for-image/self-supervised-video-retrieval-on-hmdb51)](https://paperswithcode.com/sota/self-supervised-video-retrieval-on-hmdb51?p=similarity-contrastive-estimation-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/similarity-contrastive-estimation-for-image/self-supervised-video-retrieval-on-ucf101)](https://paperswithcode.com/sota/self-supervised-video-retrieval-on-ucf101?p=similarity-contrastive-estimation-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/similarity-contrastive-estimation-for-image/self-supervised-action-recognition-on-hmdb51)](https://paperswithcode.com/sota/self-supervised-action-recognition-on-hmdb51?p=similarity-contrastive-estimation-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/similarity-contrastive-estimation-for-image/self-supervised-action-recognition-on-ucf101)](https://paperswithcode.com/sota/self-supervised-action-recognition-on-ucf101?p=similarity-contrastive-estimation-for-image)

<table>
<thead>
  <tr>
    <th rowspan="2" align="center">Frames</th>
    <th rowspan="2" align="center">K400<br></th>
    <th colspan="2" style="text-align:center">UCF101<br></th>
    <th colspan="2" style="text-align:center">HMDB51<br></th>
    <th rowspan="2" style="text-align:center">ckpt</th>
  </tr>
  <tr>
    <th align="center">Acc 1</th>
    <th align="center">Retrieval 1</th>
    <th align="center">Acc 1<br></th>
    <th align="center">Retrieval 1</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">8</td>
    <td align="center">67.6</td>
    <td align="center">94.1</td>
    <td align="center">81.5</td>
    <td align="center">70.5</td>
    <td align="center">43.0</td>
    <td align="center"><a href="https://drive.google.com/file/d/1KpjJ9UF10fYATRnV-QNzEU2H3QIOZuuK/view?usp=drive_link">Download</a></td>
  </tr>
  <tr>
    <td align="center">16</td>
    <td align="center"><b>69.6</b></td>
    <td align="center"><b>95.3</b></td>
    <td align="center"><b>83.9</b></td>
    <td align="center"><b>74.7</b></td>
    <td align="center"><b>45.9</b></td>
    <td align="center"><a href="https://drive.google.com/file/d/1Bg-h6W6MA6Sq8gg8jPQqj5NWUyEaDZlh/view?usp=drive_link">Download</a></td>
  </tr>
</tbody>
</table>

### Pretraining

Define the output directory, experiment and datasets directory as well as the seed for all experiments.

```bash
output_dir=...
exp_dir=...
dataset_dir=...
seed=42
cd eztorch/run
```

#### R3D18 Kinetics200

Can be launched on 2 A100-80G GPUs.

```bash
config_path="../eztorch/configs/run/pretrain/sce/resnet3d18"
config_name="resnet3d18_kinetics200"

srun --kill-on-bad-exit=1 python pretrain.py \
     -cp $config_path -cn $config_name \
     dir.data=$dataset_dir \
     dir.root=$output_dir \
     dir.exp="pretrain" \
     seed.seed=$seed \
     datamodule.train.loader.num_workers=4 \
     datamodule.val.loader.num_workers=4 \
     trainer.devices=2
```

#### R3D50 Kinetics400 8 frames

Can be launched on 4 A100-80G GPUs.

```bash
config_path="../eztorch/configs/run/pretrain/sce/resnet3d50"
config_name="resnet3d50_kinetics400"

srun --kill-on-bad-exit=1 python pretrain.py \
     -cp $config_path -cn $config_name \
     dir.data=$dataset_dir \
     dir.root=$output_dir \
     dir.exp="pretrain" \
     seed.seed=$seed \
     datamodule.train.loader.num_workers=4 \
     datamodule.val.loader.num_workers=4 \
     trainer.devices=8
```

#### R3D50 Kinetics400 16 frames

Can be launched on 8 A100-80G GPUs.

```bash
config_path="../eztorch/configs/run/pretrain/sce/resnet3d50"
config_name="resnet3d50_kinetics400"

srun --kill-on-bad-exit=1 python pretrain.py \
     -cp $config_path -cn $config_name \
     dir.data=$dataset_dir \
     dir.root=$output_dir \
     dir.exp="pretrain" \
     seed.seed=$seed \
     datamodule.train.transform.transform.transforms.1.num_samples=16 \
     datamodule.train.loader.num_workers=4 \
     datamodule.val.loader.num_workers=4 \
     trainer.devices=8 \
     trainer.num_nodes=2
```

### Downstream tasks

For downstream tasks, we consider by default you use checkpoints you pretrained yourselves.

If this is not the case and you downloaded the checkpoints we provided, do not forget to change the `model.trunk_pattern` config that searches the trunk pattern in the state dict:
```bash

srun --kill-on-bad-exit=1 python downstream_script.py
     ...
     model.trunk_pattern="" \
     ...
```
#### Linear evaluation

##### R3D18 Kinetics200

```bash
eval_config_path="../eztorch/configs/run/evaluation/linear_classifier/sce/resnet3d18"
eval_config_name="resnet3d18_kinetics200_frame"
pretrain_checkpoint=...

srun --kill-on-bad-exit=1 python linear_classifier_evaluation.py \
     -cp $eval_config_path -cn $eval_config_name \
     dir.data=$dataset_dir \
     dir.root=$output_dir \
     dir.exp="linear_classifier_evaluation" \
     model.pretrained_trunk_path=$pretrain_checkpoint \
     datamodule.train.loader.num_workers=4 \
     datamodule.val.loader.num_workers=4 \
     seed.seed=$seed \
     trainer.devices=-1
```

##### R3D50 Kinetics400 8 frames

```bash
eval_config_path="../eztorch/configs/run/evaluation/linear_classifier/sce/resnet3d50"
eval_config_name="resnet3d50_kinetics400"
pretrain_checkpoint=...

srun --kill-on-bad-exit=1 python linear_classifier_evaluation.py \
    -cp $eval_config_path -cn $eval_config_name \
    dir.data=$dataset_dir \
    dir.root=$output_dir \
    dir.exp="linear_classifier_evaluation" \
    model.pretrained_trunk_path=$pretrain_checkpoint \
    datamodule.train.loader.num_workers=4 \
    datamodule.val.loader.num_workers=4 \
    seed.seed=$seed \
    trainer.devices=-1
```

##### R3D50 Kinetics400 16 frames

```bash
eval_config_path="../eztorch/configs/run/evaluation/linear_classifier/sce/resnet3d18"
eval_config_name="resnet3d50_kinetics400"
pretrain_checkpoint=...

srun --kill-on-bad-exit=1 python linear_classifier_evaluation.py \
     -cp $eval_config_path -cn $eval_config_name \
     dir.data=$dataset_dir \
     dir.root=$output_dir \
     dir.exp="linear_classifier_evaluation" \
     model.pretrained_trunk_path=$pretrain_checkpoint \
     datamodule.train.transform.transform.transforms.0.num_samples=16 \
     datamodule.train.loader.num_workers=5 \
     datamodule.val.loader.num_workers=5 \
     seed.seed=$seed \
     trainer.devices=-1
```

##### Testing

Validation can be quite long, in the code we evaluate only every 5 epochs. Two steps can speed things up:
1. Speed training:
     - removes validation and only saves the last checkpoint
     - performs a validation with only one crop instead of 30
2. Perform testing afterward.

To perform this, change the config for validation and launch a test after training (example for **Kinetics400 R3D50 16 frames**):

```bash
eval_config_path="../eztorch/configs/run/evaluation/linear_classifier/sce/resnet3d18"
eval_config_name="resnet3d50_kinetics400"
pretrain_checkpoint=...

srun --kill-on-bad-exit=1 python test.py.py \
     -cp $eval_config_path -cn $eval_config_name \
     dir.data=$dataset_dir \
     dir.root=$output_dir \
     dir.exp="linear_classifier_evaluation" \
     model.pretrained_trunk_path=$pretrain_checkpoint \
     model.optimizer.batch_size=512 \
     datamodule.train=null \
     datamodule.val=null \
     datamodule.test.loader.num_workers=3 \
     datamodule.test.global_batch_size=2 \
     datamodule.test.transform.transform.transforms.0.num_samples=16 \
     seed.seed=$seed \
     trainer=gpu \
     trainer.devices=1 \
     test.ckpt_by_callback_mode=best
```

#### Fine-tuning

We give here the configurations for fine-tuning a ResNet3d50 with 16 frames, but configs for other networks are available.

##### HMDB51

```bash
config_path="../eztorch/configs/run/finetuning/resnet3d50"
config_name="resnet3d50_hmdb51_frame"
pretrain_checkpoint=...
split=1

srun --kill-on-bad-exit=1 python supervised.py \
     -cp $config_path -cn $config_name \
     dir.data=$dataset_dir \
     dir.root=$output_dir \
     dir.exp="finetuning_hmdb51_split${split}" \
     model.pretrained_trunk_path=$pretrain_checkpoint \
     datamodule.split_id=$split \
     datamodule.train.loader.num_workers=4 \
     datamodule.val.loader.num_workers=4 \
     datamodule.decoder_args.frame_filter.num_samples=16 \
     seed.seed=$seed \
     trainer.devices=-1 \
     test=null \
```
##### UCF101

```bash
config_path="../eztorch/configs/run/finetuning/resnet3d50"
config_name="resnet3d50_ucf101_frame"
pretrain_checkpoint=...
split=1

srun --kill-on-bad-exit=1 python supervised.py \
     -cp $config_path -cn $config_name \
     dir.data=$dataset_dir \
     dir.root=$output_dir \
     dir.exp="finetuning_ucf101_split${split}" \
     model.pretrained_trunk_path=$pretrain_checkpoint \
     datamodule.split_id=$split \
     datamodule.train.loader.num_workers=4 \
     datamodule.val.loader.num_workers=4 \
     datamodule.decoder_args.frame_filter.num_samples=16 \
     seed.seed=$seed \
     trainer.devices=-1 \
     test=null \
```

##### Testing

Validation can be quite long, in the code we evaluate only every 5 epochs. Two steps can speed things up:
1. Speed training:
     - removes validation and only saves the last checkpoint
     - performs a validation with only one crop instead of 30
2. Perform testing afterward.

To perform this, change the config for validation and launch a test after training (example for **UCF101**):

```bash
config_path="../eztorch/configs/run/finetuning/resnet3d50"
config_name="resnet3d50_ucf101_frame"
pretrain_checkpoint=...
split=1

srun --kill-on-bad-exit=1 python test.py \
     -cp $config_path -cn $config_name \
     dir.data=$dataset_dir \
     dir.root=$output_dir \
     dir.exp="finetuning_hmdb51_split${split}" \
     model.pretrained_trunk_path=$pretrain_checkpoint \
     datamodule.train=null \
     datamodule.val=null \
     model.optimizer.batch_size=64 \
     datamodule.test.global_batch_size=2 \
     datamodule.test.loader.num_workers=4 \
     datamodule.decoder_args.frame_filter.num_samples=16 \
     trainer=gpu \
     seed.seed=$seed \
     trainer.devices=1 \
     test.ckpt_by_callback_mode=best
```

#### Retrieval

We give here the configurations for video retrieval using a ResNet3d50 with 16 frames, but configs for other networks are available.

It has two steps:
1. Features extraction
2. Retrieval

##### HMDB51

```bash
extract_config_path="../eztorch/configs/run/evaluation/feature_extractor/resnet3d50"
extract_config_name="resnet3d50_hmdb51_frame"
retrieval_config_path="../eztorch/configs/run/evaluation/retrieval_from_bank"
retrieval_config_name="default"

split=1
pretrain_checkpoint=...

# Extraction
srun --kill-on-bad-exit=1 python extract_features.py \
     -cp $extract_config_path -cn $extract_config_name \
     dir.data=$dataset_dir \
     dir.root=$output_dir \
     dir.exp="features_extraction_split${split}" \
     model.pretrained_trunk_path=$pretrain_checkpoint \
     datamodule.decoder_args.frame_filter.num_samples=16 \
     datamodule.train.loader.num_workers=3 \
     datamodule.val.loader.num_workers=3 \
     datamodule.train.global_batch_size=2 \
     datamodule.val.global_batch_size=2 \
     seed.seed=$seed \
     trainer.num_nodes=$SLURM_NNODES \
     datamodule.split_id=$split \
     trainer.max_epochs=1

# Retrieval
query_features="${output_dir}/features_extraction_split${split}/val_features.pth"
bank_features="${output_dir}/features_extraction_split${split}/train_features.pth"
query_labels="${output_dir}/features_extraction_split${split}/val_labels.pth"
bank_labels="${output_dir}/features_extraction_split${split}/train_labels.pth"

srun --kill-on-bad-exit=1 python retrieval_from_bank.py \
     -cp $retrieval_config_path -cn $retrieval_config_name \
     dir.root=$output_dir \
     dir.exp="retrieval_split${split}" \
     query.features_path=$query_features \
     query.labels_path=$query_labels \
     bank.features_path=$bank_features \
     bank.labels_path=$bank_labels
```

##### UCF101

```bash
extract_config_path="../eztorch/configs/run/evaluation/feature_extractor/resnet3d50"
extract_config_name="resnet3d50_ucf101_frame"
retrieval_config_path="../eztorch/configs/run/evaluation/retrieval_from_bank"
retrieval_config_name="default"

split=1
pretrain_checkpoint=...

# Extraction
srun --kill-on-bad-exit=1 python extract_features.py \
     -cp $extract_config_path -cn $extract_config_name \
     dir.data=$dataset_dir \
     dir.root=$output_dir \
     dir.exp="features_extraction_split${split}" \
     model.pretrained_trunk_path=$pretrain_checkpoint \
     datamodule.decoder_args.frame_filter.num_samples=16 \
     datamodule.train.loader.num_workers=3 \
     datamodule.val.loader.num_workers=3 \
     datamodule.train.global_batch_size=2 \
     datamodule.val.global_batch_size=2 \
     seed.seed=$seed \
     trainer.num_nodes=$SLURM_NNODES \
     datamodule.split_id=$split \
     trainer.max_epochs=1

# Retrieval
query_features="${output_dir}/features_extraction_split${split}/val_features.pth"
bank_features="${output_dir}/features_extraction_split${split}/train_features.pth"
query_labels="${output_dir}/features_extraction_split${split}/val_labels.pth"
bank_labels="${output_dir}/features_extraction_split${split}/train_labels.pth"
srun --kill-on-bad-exit=1 python retrieval_from_bank.py \
     -cp $retrieval_config_path -cn $retrieval_config_name \
     dir.root=$output_dir \
     dir.exp="retrieval_split${split}" \
     query.features_path=$query_features \
     query.labels_path=$query_labels \
     bank.features_path=$bank_features \
     bank.labels_path=$bank_labels
```

#### Action Localization on AVA and Recognition on SSV2

Generalization to Action Localization on AVA and Action Recognition on SSV2 was performed thanks to the [SlowFast](https://github.com/facebookresearch/SlowFast) repository. This repository supports the use of pytorchvideo models which we used as backbones.


## Issue

If you found an error, have trouble making this work or have any questions, please open an [issue](https://github.com/juliendenize/eztorch/issues) to describe your problem.


## Acknowledgment

This publication was made possible by the use of the Factory-AI supercomputer, financially supported by the Ile-de-France Regional Council and the HPC resources of IDRIS under the allocation 2022-AD011013575 made by GENCI.

## Citation

If you found our work useful, please consider citing us:

```
﻿@article{Denize_2023_MVAP,
  author={Denize, Julien and Rabarisoa, Jaonary and Orcesi, Astrid and H{\'e}rault, Romain},
  title={Similarity contrastive estimation for image and video soft contrastive self-supervised learning},
  journal={Machine Vision and Applications},
  year={2023},
  volume={34},
  number={6},
  doi={10.1007/s00138-023-01444-9},
}
```
