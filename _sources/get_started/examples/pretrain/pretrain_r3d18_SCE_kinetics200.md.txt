# SCE R3D-18 on Kinetics200

In this guide, we provide steps to pretrain a R3D-18 using SCE on Kinetics200.

The first section will focus on defining the configuration and the second one to launch the training.

## Define the configuration

We first need to define the configuration to train SCE. The configuration is available [here](configs/pretrain_r3d18_SCE_kinetics200.yaml), and is detailed below.

### Define the Datamodule

The datamodule is the `Kinetics200DataModule`.

```yaml
datamodule:
  _target_: eztorch.datamodules.Kinetics200DataModule
  _recursive_: false
  datadir: ${..dir.data}
  video_path_prefix: ${.datadir}
```

For a video datamodule, it needs to be specified the decoder used such as Pyav, frames etc., along with its parameters. For this example, we use the frames decoder.

```yaml
datamodule:
  decoder: frame
  decoder_args:
    fps: 30
    frame_filter:
      subsample_type: uniform
      num_samples: 8
    time_difference_prob: 0.2
    num_threads_io: 4
    num_threads_decode: 4
    decode_float: true
```

For each set used during the fit of the model, usually training and validation, there needs to be passed information about the clip sampler, the transform applied to each clip and the configuration for the dataloaders.

```yaml
datamodule:
  train:
    dataset:
      datadir: ${...datadir}/train.csv
      video_path_prefix: ${...datadir}/train
    transform:
      _target_: eztorch.transforms.OnlyInputListTransform
      _recursive_: true
      transform:
        _target_: eztorch.transforms.video.RandomResizedCrop
        target_height: 224
        target_width: 224
        scale:
        - 0.2
        - 0.766
        aspect_ratio:
        - 0.75
        - 1.3333
        interpolation: bilinear
    clip_sampler:
      _target_: eztorch.datasets.clip_samplers.RandomMultiClipSampler
      num_clips: 2
      clip_duration: 2.56
      speeds:
      - 1
      jitter_factor: 0
    loader:
      drop_last: true
      num_workers: 5
      pin_memory: true
    global_batch_size: 512
```

### Define the model

We will use the `SCEModel`. SCE as a siamese self-supervised learning method defines several networks. It is composed of an online branch updated by backpropagation and a momentum target branch updated by the exponential moving average of the online branch.

The online branch consists of an encoder, or trunk, a projector and a predictor. The target branch has the same architecture as the online one without the predictor.

We first need to tell Hydra which model to instantiate:
```yaml
model:
  _target_: eztorch.models.siamese.SCEModel
  _recursive_: false
```

Each neural network architecture of SCE must also be defined:
- A trunk to learn representations such as ResNet3D18
```yaml
model:
    trunk:
    _target_: eztorch.models.trunks.create_video_head_model
    _recursive_: false
    model:
      _target_: eztorch.models.trunks.create_resnet3d_basic
      head: null
      model_depth: 18
    head:
      _target_: eztorch.models.heads.create_video_resnet_head
      activation: null
      dropout_rate: 0.0
      in_features: 512
      num_classes: 0
      output_size: [1, 1 ,1]
      output_with_global_average: true
      pool: null
      pool_kernel_size: [8, 7, 7]
```
- A projector, which is a rather small MLP network, to project data in a lower dimensional space invariant to data augmentation:
```yaml
model:
  projector:
    _target_: eztorch.models.heads.MLPHead
    activation_inplace: true
    activation_layer: relu
    affine: true
    bias: false
    dropout: 0.0
    dropout_inplace: true
    hidden_dims:
    - 1024
    - 1024
    input_dim: 512
    norm_layer: bn_1D
    num_layers: 3
    last_bias: false
    last_norm: true
    last_affine: false
    output_dim: 256
```
- A predictor, smaller than the projector, to predict the output projection of the target encoder
```yaml
model:
  predictor:
    _target_: eztorch.models.heads.MLPHead
    activation_inplace: true
    activation_layer: relu
    affine: true
    bias: false
    dropout: 0.0
    dropout_inplace: true
    hidden_dims:
    - 1024
    input_dim: 256
    norm_layer: bn_1D
    num_layers: 2
    last_bias: false
    last_norm: false
    last_affine: false
    output_dim: 256
```

Now we can provide the configuration to correctly configure the SCE model:
```yaml
model:
  coeff: 0.5
  final_scheduler_coeff: 0.0
  initial_momentum: 0.99
  mutual_pass: false
  normalize_outputs: true
  num_devices: -1
  num_global_crops: 2
  num_local_crops: 0
  num_splits: 0
  num_splits_per_combination: 2
  queue:
    size: 32768
    feature_dim: 256
  scheduler_coeff: null
  scheduler_momentum: cosine
  simulate_n_devices: 8
  shuffle_bn: false
  start_warmup_coeff: 1.0
  sym: true
  temp: 0.1
  temp_m: 0.05
  use_keys: false
  warmup_epoch_coeff: 0
  warmup_epoch_temp_m: 0
  warmup_scheduler_coeff: linear
  warmup_scheduler_temp_m: cosine
```

To optimize the parameters, we also provide the configuration for the optimizer and its scheduler:

```yaml
model:
    optimizer:
    _target_: eztorch.optimizers.optimizer_factory
    _recursive_: false
    exclude_wd_norm: false
    exclude_wd_bias: false
    name: lars
    params:
      momentum: 0.9
      trust_coefficient: 0.001
      weight_decay: 1.0e-06
    batch_size: 512
    initial_lr: 2.4
    layer_decay_lr: null
    scaler: linear
    scheduler:
      _target_: eztorch.schedulers.scheduler_factory
      _recursive_: false
      name: linear_warmup_cosine_annealing_lr
      params:
        max_epochs: 200
        warmup_epochs: 35
        warmup_start_lr: 0.0
        eta_min: 0.0
      interval: step
```

SCEModel supports GPU transform to speed up data augmentations for training and/or validation, and we specify the configuration of the contrastive transforms:
```yaml
model:
  train_transform:
    _target_: eztorch.transforms.ApplyTransformsOnList
    _recursive_: true
    transforms:
    - _target_: torchaug.batch_transforms.BatchVideoWrapper
      same_on_frames: true
      video_format: CTHW
      inplace: true
      transforms:
      - _target_: eztorch.transforms.Div255Input
        inplace: true
      - _target_: torchaug.batch_transforms.BatchRandomColorJitter
        brightness: 0.8
        contrast: 0.8
        hue: 0.2
        p: 0.8
        saturation: 0.4
        inplace: true
      - _target_: torchaug.batch_transforms.BatchRandomGrayscale
        p: 0.2
        inplace: true
      - _target_: torchaug.batch_transforms.BatchRandomGaussianBlur
        kernel_size: 23
        sigma:
        - 0.1
        - 2.0
        p: 1.0
        inplace: true
      - _target_: torchaug.batch_transforms.BatchRandomHorizontalFlip
        p: 0.5
        inplace: true
      - _target_: torchaug.transforms.Normalize
        mean:
        - 0.45
        - 0.45
        - 0.45
        std:
        - 0.225
        - 0.225
        - 0.225
        inplace: true
    - _target_: torchaug.batch_transforms.BatchVideoWrapper
      same_on_frames: true
      video_format: CTHW
      inplace: true
      transforms:
      - _target_: eztorch.transforms.Div255Input
        inplace: true
      - _target_: torchaug.batch_transforms.BatchRandomColorJitter
        brightness: 0.8
        contrast: 0.8
        hue: 0.2
        p: 0.8
        saturation: 0.4
        inplace: true
      - _target_: torchaug.batch_transforms.BatchRandomGrayscale
        p: 0.2
        inplace: true
      - _target_: torchaug.batch_transforms.BatchRandomGaussianBlur
        kernel_size: 23
        sigma:
        - 0.1
        - 2.0
        p: 0.1
        inplace: true
      - _target_: torchaug.batch_transforms.BatchRandomSolarize
        p: 0.2
        threshold: 0.5
        inplace: true
      - _target_: torchaug.batch_transforms.BatchRandomHorizontalFlip
        p: 0.5
      - _target_: torchaug.transforms.Normalize
        mean:
        - 0.45
        - 0.45
        - 0.45
        std:
        - 0.225
        - 0.225
        - 0.225
        inplace: true
```

### Configure the trainer

To run our experiment, we need to define a [trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api) from Pytorch-Lightning.

It allows us to specify the number and type of devices used, configure average mixed precision, and whether to use synchronized batch normalization or not, ...:

```yaml
trainer:
  _target_: lightning.pytorch.trainer.Trainer
  accelerator: gpu
  benchmark: true
  devices: -1
  max_epochs: 200
  num_nodes: 1
  precision: 16
  strategy:
    _target_: lightning.pytorch.strategies.DDPStrategy
    find_unused_parameters: false
    static_graph: false
  sync_batchnorm: true
```

Also, you should define the callbacks fired by the trainer such as the checkpointing for the model:
```yaml
callbacks:
  model_checkpoint:
    _target_: eztorch.callbacks.ModelCheckpoint
    dirpath: pretrain_checkpoints
    filename: '{epoch}'
    save_last: false
    save_top_k: -1
    mode: min
    every_n_epochs: 100
```

### Job configuration

Hydra allows you to configure its behavior to define a run directory to store your result, also used by Eztorch to change your `pwd`. You can also specify Python packages to retrieve configuration to inherit from or to include in your current config:

```yaml
hydra:
  searchpath:
    - pkg://eztorch.configs
  run:
    dir: ${...dir.run}
```

You can define the various directories for your experiment:
- the root of your experiments
- the current experiment
- the data

```yaml
dir:
  data: ???
  root: /output/
  exp: pretrain
  run: ${.root}/${.exp}
```

Finally, Pytorch-Lightning provides a nice tool to define seeds on all packages:
```yaml
seed:
  _target_: lightning.fabric.utilities.seed.seed_everything
  seed: 42
  workers: true
```

## Launch the pretraining

To launch the pretraining of SCE and use our current configuration, you have to call the right Python script with the location of the configuration.

Eztorch defines a pretrain `script` that provides you the script to launch pretraining **depending** on your hydra configuration.

The script to launch the experiments using SLURM is the following:

```bash
output_dir=... # The folder at the root of your experiment
dataset_dir=... # The folder containing the data

cd sce/run

config_path="../doc/examples/configs/"
config_name="pretrain_r3d18_SCE_kinetics200"
seed=42

srun --kill-on-bad-exit=1 python pretrain.py\
    -cp $config_path -cn $config_name\
    dir.data=$dataset_dir dir.root=$output_dir \
    dir.exp='pretrain' seed.seed=$seed \
    datamodule.train.loader.num_workers=3 \
    datamodule.val.loader.num_workers=3 \
    trainer.gpus=-1
```

Pytorch-lightning automatically detects we are using SLURM and through the srun command make the multi-GPU distributed training work.

As you can see, we provided the **relative path** to the configuration as well as its name to configure hydra with argparse-like arguments.

Configuration for our experiment is accessible the same way as in our Python code.
