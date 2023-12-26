# COMEDIAN for Action Spotting (WACV Workshops 2024)

## Introduction

This repository contains the  official [Pytorch](https://pytorch.org/) implementation of [COMEDIAN: Self-Supervised Learning and Knowledge Distillation for Action Spotting using Transformers](https://openaccess.thecvf.com/content/WACV2024W/Pretrain/papers/Denize_COMEDIAN_Self-Supervised_Learning_and_Knowledge_Distillation_for_Action_Spotting_Using_WACVW_2024_paper.pdf) that has been published in the **IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops 2024**.

COMEDIAN is composed of three steps:
1. Pretraining via a self-supervised loss of the spatial transformer.
2. Pretraining via a knowledge distillation loss of the spatial and temporal transformers.
3. Fine-tuning to the action spotting task.

In next sections, we provide the code for all these steps for ViViT Tiny. All experiments can be launched on 2 A100-80G GPUs.

## Data preparation

Data preparation details are available [here](../get_started/prepare_data.md).

## Main results

Results obtained on test set from several architectures. We provide the associated checkpoints.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/comedian-self-supervised-learning-and/action-spotting-on-soccernet-v2)](https://paperswithcode.com/sota/action-spotting-on-soccernet-v2?p=comedian-self-supervised-learning-and)
<table>
<thead>
  <tr>
    <th style="text-align:center">Model</th>
    <th style="text-align:center">t-AmAP</th>
    <th style="text-align:center">Ckpts 🤗</th>
    <th style="text-align:center">Ckpts Gdrive</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">ViViT Tiny</td>
    <td align="center">70.7</td>
    <td align="center"><a href="https://huggingface.co/juliendenize/COMEDIAN-ViViT-tiny/tree/main">files</a></td>	
    <td align="center"><a href="https://drive.google.com/file/d/1iTTlVXXFLp9QzxlccfT2i44BMvuOyYgq/view?usp=drive_link">seed42</a> <a href="https://drive.google.com/file/d/1zfryhsRtJchJNfPRiA-u-r5CYc-j1_ub/view?usp=drive_link">seed203</a> <a href="https://drive.google.com/file/d/1qpNlU_-J42l0_53YN0xRCzfR6aN8BxWd/view?usp=drive_link">seed666</a></td>
  </tr>
  <tr>
    <td align="center">ViSwin Tiny</td>
    <td align="center">71.6</td>
    <td align="center"><a href="https://huggingface.co/juliendenize/COMEDIAN-ViSwin-tiny/tree/main">files</a></td>	
    <td align="center"><a href="https://drive.google.com/file/d/1zDVUKq8nRd5hVZIm49Ity-8GnLTa7DOh/view?usp=drive_link">seed42</a> <a href="https://drive.google.com/file/d/1QD52pB60d9u82urs6ZSVwRIpNTliv9pR/view?usp=drive_link">seed203</a> <a href="https://drive.google.com/file/d/11BGiR-yeJUJmY6FfobaRwa-3t_Ps4CdN/view?usp=drive_link">seed666</a></td>
  </tr>
  <tr>
    <td align="center">ViVit Tiny ensemble</td>
    <td align="center">72.0</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td align="center">ViSwin Tiny ensemble</td>
    <td align="center">73.1</td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
</tbody>
</table>

## Pretrain the spatial transformer

```bash
output_dir=...
dataset_dir=... # Path to the JSON file.
frame_dir=... # Path to the decoded videos.
seed=42

config_path="../eztorch/configs/run/pretrain/sce/vit"
config_name="vit_tiny_spatio_temporal_soccernet.yaml"

cd eztorch/run/

srun --kill-on-bad-exit=1 python pretrain.py \
    -cp $config_path -cn $config_name \
    dir.data=$dataset_dir \
    dir.root=$output_dir \
    dir.exp="pretrain_spatial" \
    seed.seed=$seed \
    datamodule.train.dataset.datadir=$train_dir \
    datamodule.train.dataset.video_path_prefix=$frame_dir \
    datamodule.train.loader.num_workers=4  \
    trainer.num_nodes=$SLURM_NNODES \
    seed.seed=$seed \
    model.coeff=1.
```
## Pretrain the spatial and temporal transformers

### Feature extraction

Example of Finetuning SCE ResNet3D50.

```bash
config_path="../eztorch/configs/run/evaluation/feature_extractor/resnet3d50"
config_name="resnet3d50_soccernet"

output_dir=...
dataset_dir=... # Path to the JSON file.
frame_dir=... # Path to the decoded videos.
pretrain_checkpoint=...


srun --kill-on-bad-exit=1 python extract_features.py \
    -cp $config_path -cn $config_name \
	trainer.max_epochs=1 \
	seed.seed=$seed \
	dir.data=$dataset_dir \
	dir.root=$output_dir \
	dir.exp="features_extraction/" \
	model.pretrained_trunk_path=$pretrain_checkpoint \
	model.filename="sce_finetuned_resnet3d50_4fps_4seconds_window" \
	datamodule.test.loader.num_workers=4 \
	datamodule.test.dataset.datadir=$dataset_dir \
    datamodule.test.dataset.video_path_prefix=$frame_dir \
	datamodule.test.global_batch_size=64
```

### PCA on the features

```bash
dataset=... # Path to the json
features_path=sce_finetuned_resnet3d50_4fps_4seconds_window # Path to the features
filename=... # Name of the features
pca_dim=512
save_path=... # Where to save PCA features

python datasets/pca_soccernet.py \
    --dataset-json $dataset \
    --video-zip-prefix "" \
    --features-path $features_path \
    --filename $filename \
    --dim $pca_dim \
    --save-path $save_path \
    --fps 2 \
    --task "action"
```

### Pretraining

```bash
config_path="../eztorch/configs/run/pretrain/sce_distill_tokens/vivit"
config_name="vivit_tiny_soccernet.yaml"
dataset_dir=... # Path to the JSON.
frame_dir=... # Path to the decoded videos.
feature_dir=... # Path to the features.
feature_filename=... # Name of the features.
seed=42

srun --kill-on-bad-exit=1 python pretrain.py \
    -cp $config_path -cn $config_name \
    dir.data=$dataset_dir \
    dir.root=$output_dir \
    dir.exp="pretrain_spatio_temporal/" \
    seed.seed=$seed \
    datamodule.train.dataset.datadir=$dataset_dir \
    datamodule.train.dataset.video_path_prefix=$frame_dir \
    datamodule.train.dataset.feature_args.dir=$feature_dir \
    datamodule.train.dataset.feature_args.filename=$feature_filename \
    datamodule.train.loader.num_workers=4 \
    model.trunk.transformer.weights_from=spatial \
    model.trunk.transformer.pretrain_pth="$output_dir/pretrain_spatial/pretrain_checkpoints/epoch"'\=99.ckpt' \
    model.optimizer.scheduler.params.warmup_start_lr=$warmup_start_lr \
    model.trunk.transformer.temporal_mask_ratio=0.25 \
    model.trunk.transformer.temporal_mask_token=True \
    model.trunk.transformer.temporal_mask_tube=2 \
    model.trunk.transformer.temporal_depth=6
```
## Fine-tuning

### Initialize classifier
```bash
config_path="../eztorch/configs/run/finetuning/vivit"
config_name="vivit_tiny_soccernet_uniform"

soccernet_labels_dir=... # Directory of ground truth labels.
labels_cache_dir_train=... # Where train model labels are cached
labels_cache_dir_val=... # Where val model labels are cached

train_dir=... # Path to the train JSON.
val_dir=... # Path to the val JSON.
frame_dir=... # Path to the decoded videos.

srun --kill-on-bad-exit=1 python supervised.py \
    -cp $config_path \
    -cn $config_name \
    dir.data="" \
    dir.root=$output_dir \
    dir.exp="pretrain_classifier" \
    seed.seed=$seed \
    datamodule.train.dataset.datadir=$train_dir \
    datamodule.train.dataset.video_path_prefix=$frame_dir \
    datamodule.train.dataset.label_args.cache_dir=$labels_cache_dir_train \
    datamodule.train.loader.num_workers=4 \
    datamodule.val.dataset.datadir=$val_dir \
    datamodule.val.dataset.video_path_prefix=$frame_dir  \
    datamodule.val.dataset.label_args.cache_dir=$labels_cache_dir_val \
    datamodule.val.loader.num_workers=4 \
    model.evaluation_args.SoccerNet_path=$soccernet_labels_dir \
    model.freeze_trunk=True \
    model.pretrained_trunk_path="$output_dir/pretrain_spatio_temporal/pretrain_checkpoints/epoch"'\=99.ckpt' \
    trainer.check_val_every_n_epoch=15 \
    trainer.max_epochs=30 \
    model.trunk.transformer.temporal_mask_ratio=0.25 \
    model.trunk.transformer.temporal_mask_token=True \
    model.trunk.transformer.temporal_mask_tube=2 \
    model.trunk.transformer.temporal_depth=6 \
    model.NMS_args.nms_type=soft \
    model.NMS_args.window=20 \
    model.NMS_args.threshold=0.001
```

### Global fine-tuning
```bash
srun --kill-on-bad-exit=1 python supervised.py \
    -cp $config_path \
    -cn $config_name \
    dir.data="" \
    dir.root=$output_dir \
    dir.exp="finetune_classifier_backbone" \
    seed.seed=$seed \
    datamodule.train.dataset.datadir=$train_dir \
    datamodule.train.dataset.video_path_prefix=$frame_dir \
    datamodule.train.dataset.label_args.cache_dir=$labels_cache_dir_train \
    datamodule.train.loader.num_workers=4 \
    datamodule.val.dataset.datadir=$val_dir \
    datamodule.val.dataset.video_path_prefix=$frame_dir  \
    datamodule.val.dataset.label_args.cache_dir=$labels_cache_dir_val \
    datamodule.val.loader.num_workers=4 \
    model.evaluation_args.SoccerNet_path=$soccernet_labels_dir \
    model.freeze_trunk=False \
    model.pretrained_path="$output_dir/pretrain_classifier/checkpoints/last.ckpt" \
    trainer.num_nodes=$SLURM_NNODES \
    trainer.check_val_every_n_epoch=5 \
    callbacks.model_checkpoint.every_n_epochs=5 \
    model.trunk.transformer.temporal_mask_ratio=0.25 \
    model.trunk.transformer.temporal_mask_token=True \
    model.trunk.transformer.temporal_mask_tube=2 \
    model.trunk.transformer.temporal_depth=6 \
    model.NMS_args.nms_type=soft \
    model.NMS_args.window=20 \
    model.NMS_args.threshold=0.001
```

### Testing

#### Inference

To make inference on data based on a checkpoint.

Example on the test split.

```bash
output_dir=...
test_dir=...
frame_dir=...
labels_cache_dir_test=... # Where test model labels are cached
soccernet_labels_dir=... # Directory of ground truth labels.
checkpoint_path=...

srun --kill-on-bad-exit=1 python test.py -cp $config_path -cn $config_name \
    dir.data=$test_dir \
    dir.root=$output_dir \
    dir.exp="test/" \
    seed.seed=$seed \
    datamodule.train=null \
    datamodule.val=null \
    datamodule.test.dataset.task=action \
    datamodule.test.dataset.datadir=$test_dir \
    datamodule.test.dataset.video_path_prefix=$frame_dir \
    datamodule.test.dataset.label_args.cache_dir=$labels_cache_dir_test \
    datamodule.test.dataset.label_args.radius_label=0.5 \
    datamodule.test.loader.num_workers=4 \
    datamodule.test.global_batch_size=64 \
    model.optimizer.batch_size=2 \
    model.evaluation_args.SoccerNet_path=$soccernet_labels_dir \
    model.evaluation_args.split="test" \
    model.trunk.transformer.temporal_depth=6 \
    model.save_test_preds_path="test_preds/" \
    model.prediction_args.remove_inference_prediction_seconds=12 \
    model.prediction_args.merge_predictions_type="max" \
    model.NMS_args.nms_type=soft \
    model.NMS_args.window=20 \
    model.NMS_args.threshold=0.001 \
    model.train_transform=null \
    model.val_transform=null \
    model.pretrained_path=$checkpoint_path \
    ++test.ckpt_path=null
```

#### Process predictions

The finetuning stores at each validation the raw predictions (before NMS) as well as the predicted ones that allow for trying different NMS parameters.

Example to use new NMS parameters from raw predictions on the validation split.

```bash
soccernet_labels_dir=... # Ground truths
test_dir=... # Path to the JSON

nms_type=...
nms_threshold=...
nms_window=...

raw_predictions_path=... # Path of the raw predictions.
process_path=... # Path to store new predictions

srun --kill-on-bad-exit=1 python evaluation_action_spotting.py \
    --soccernet-path=$soccernet_labels_dir \
    --predictions-path=$process_path \
    --preprocess-predictions-path=$raw_predictions_path \
    --dataset-path=$test_dir \
    --process-predictions \
    --nms-threshold=$nms_threshold \
    --nms-window=$nms_window \
    --nms-type=$nms_type \
    --fps=2 \
    --step-timestamp=0.5 \
    --split="valid" \
    --task "action"
```

#### Merge predictions

To improve results, you might want to merge several raw predictions. We provide a helper to do that.

```bash
dataset_path=... # Path to the JSON
saved_features_raw_path=... ... ... # Paths to the saved raw features
output_folder=... # Path to the output folder for merged features
kind_merge="average" # How to merge the features, default = "average"

srun --kill-on-bad-exit=1 python merge_soccernet_predictions.py \
    --predictions-path $saved_features_raw_path \
    --dataset-path=$dataset_path \
    --output-folder=$output_folder \
    --fps=2 \
    --kind-merge=$kind_merge \
    --task "action"
```

#### Evaluate predictions

You can evaluate the predictions from a folder or zip without processing.

```bash
soccernet_labels_dir=... # Ground truths
predictions_path=... # Path of predictions

srun --kill-on-bad-exit=1 python evaluation_action_spotting.py \
    --soccernet-path=$soccernet_labels_dir \
    --predictions-path=$predictions_path \
    --split="valid" \
    --task "action"
```

## Issue

If you found an error, have trouble making this work or have any questions, please open an [issue](https://github.com/juliendenize/eztorch/issues) to describe your problem.


## Acknowledgment
This publication was made possible by the use of the Factory-AI supercomputer, financially supported by the Ile-de-France Regional Council and the HPC resources of IDRIS under the allocation 2023-AD011014382 made by GENCI.


## Citation

If you found our work useful, please consider citing us:

```
@InProceedings{Denize_2024_WACVW,
    author    = {Denize, Julien and Liashuha, Mykola and Rabarisoa, Jaonary and Orcesi, Astrid and H\'erault, Romain},
    title     = {COMEDIAN: Self-Supervised Learning and Knowledge Distillation for Action Spotting Using Transformers},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2024},
    pages     = {530-540}
}
```
