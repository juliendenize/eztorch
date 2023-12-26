# Data preparation

## Image datasets

For image datasets, we strongly relied on Torchvision and adopted the same structure, therefore you can follow [their guidelines](https://pytorch.org/vision/main/datasets.html).

The dataset used are:
- [ImageNet](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html#torchvision.datasets.ImageNet)
- [Cifar10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10)
- [Cifar100](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html#torchvision.datasets.CIFAR100)
- [STL10](https://pytorch.org/vision/main/generated/torchvision.datasets.STL10.html#torchvision.datasets.STL10)

For ImageNet100 and [Tiny-ImageNet](https://www.kaggle.com/c/tiny-imagenet), we follow the [dataset folder](https://pytorch.org/vision/main/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder) structure.

In Eztorch, ImageNet100 is directly handled from the ImageNet folder.

## Video datasets
Generally, we follow from [Pytorchvideo](https://pytorchvideo.readthedocs.io/en/latest/data_preparation.html) with some adjustments and helpers.
### Kinetics400

1. [Download and extract](https://github.com/cvdfoundation/kinetics-dataset) the dataset.

2. Resize the shorter edge size of the videos to 256.
Prepare a CSV file that contains the path and the labels.

    ```bash
    cd eztorch

    input_folder="./kinetics_downscaled/train/"
    output_folder="./kinetics_downscaled/"
    output_filename="train.json"

    python run/datasets/create_video_files.py \
        --input-folder $input_folder \
        --output-folder $output_folder \
        --output-filename $output_filename
    ```

4. [Optional] Extract the frames to speed data loading.
    ```bash
    cd eztorch

    input_folder="./kinetics400_downscaled/train/"
    output_folder="./kinetics400_downscaled_extracted/train/"

    python run/datasets/process_video.py \
        --input-folder $input_folder \
        --output-folder $output_folder \
        --frames \
        --video-quality 1 \
        --fps 30
    ```

5. [Optional] Prepare a CSV file that contains the path, the labels and video duration.

    ```bash
    cd eztorch

    input_folder="./kinetics400_downscaled_extracted/train/"
    output_folder="./kinetics400_downscaled_extracted/"
    output_filename="train.json"

    python run/datasets/create_frames_video_files.py \
        --input-folder $input_folder \
        --output-folder $output_folder \
        --output-filename $output_filename
    ```

In our code, we do not use the frames extracted as they take more than 1To in storage. However, it could significantly reduce the decoding time, so it might suit you better. To reduce the storage cost, you can increase the parameter ``--video-quality`` that is used by ffmpeg for JPEG compression.

Keep in mind that our configuration files do not use the frame decoder but PYAV. Frame decoder is supported and you can take inspiration from Kinetics200 configs. It is also possible to consider other kind of decoders that we do not support such as DECORD and improve over what Eztorch propose.


### Kinetics200

1. Do steps 1 and 2 of the preprocessing of Kinetcs400 and create a folder containing symbolic links for videos in Kinetics200. Then perform step 3 of the preprocessing of Kinetics400.

2. [Optional] Extract the frames to speed data loading.
    ```bash
    cd eztorch

    input_folder="./kinetics200_downscaled/train/"
    output_folder="./kinetics200_downscaled_extracted/train/"

    python run/datasets/process_video.py \
        --input-folder $input_folder \
        --output-folder $output_folder \
        --frames \
        --video-quality 1 \
        --fps 30
    ```

3. [Optional] Prepare a CSV file that contains the path, the labels and video duration.
Although steps 2 and 3 are optional, we performed those and our configured scripts for Kinetics200 use a frame decoder. It is possible to update them to use a video decoder in case you do not want to extract the frames.


### HMDB51

1. [Download](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads) and extract the dataset.

2. [Optional] Extract the frames to speed data loading.
    ```bash
    cd eztorch

    input_folder="./hmdb51/"
    output_folder="./hmdb51_extracted/"

    python run/datasets/process_video.py \
        --input-folder $input_folder \
        --output-folder $output_folder \
        --frames \
        --video-quality 1 \
        --fps 30
    ```

3. [Optional] Prepare a CSV file that contains the path, the labels and video duration.

    ```bash
    cd eztorch

    input_folder="./hmdb51/"
    output_folder="./hmdb51_extracted/"

    python run/datasets/create_frames_video_files.py \
        --input-folder $input_folder \
        --output-folder $output_folder \
        --dataset hmdb51
    ```

Although steps 2 and 3 are optional, we performed those and our configured scripts for HMDB51 use a frame decoder. It is possible to update them to use a video decoder in case you do not want to extract the frames.

### UCF101

1. [Download](https://www.crcv.ucf.edu/data/UCF101.php) and extract the dataset.

2. [Optional] Extract the frames to speed data loading.
    ```bash
    cd eztorch

    input_folder="./ucf101/"
    output_folder="./ucf101_extracted/"

    python run/datasets/process_video.py \
        --input-folder $input_folder \
        --output-folder $output_folder \
        --frames \
        --video-quality 1 \
        --fps 25
    ```

3. [Optional] Prepare a CSV file that contains the path, the labels and videos duration.

    ```bash
    cd eztorch

    input_folder="./ucf101/"
    output_folder="./ucf101_extracted/"

    python run/datasets/create_frames_video_files.py \
        --input-folder $input_folder \
        --output-folder $output_folder \
        --dataset ucf101
    ```

Although steps 2 and 3 are optional, we performed those and our configured scripts for HMDB51 use a frame decoder. It is possible to update them to use a video decoder in case you do not want to extract the frames.

## SoccerNet

We followed the guide from the [challenge 2023 of SoccerNet](https://www.soccer-net.org/data) with some adjustments.

In our code, we use ``"val"`` split instead of ``"valid"`` split from SoccerNet to keep consistency with other datasets.

### Action Spotting

1. [Download the dataset](https://www.soccer-net.org/data#h.ov9k48lcih5g) and optionally the Baidu features. We downloaded the 224p low resolution.

2. Extract frames.

    ```bash
    cd eztorch

    fps=2
    input_folder="./SoccerNet_AS/"
    output_folder="./soccernet_as_extracted_${fps}fps/"
    split=train

    python run/datasets/extract_soccernet.py \
        --input-folder $input_folder \
        --output-folder $output_folder \
        --fps $fps \
        --split $split
    ```

The extraction also creates an annotation file for the split.

3. Precompute labels

    ```bash
    radius_label=0.5
    dataset_json=... # Path to the JSON.
    frame_dir=... # Path to the decoded videos.
    fps=2
    cache_dir=...

    python run/datasets/precompute_soccernet_labels.py \
        --radius-label $radius_label \
        --data-path $dataset_json \
        --path-prefix $frame_dir \
        --fps $fps \
        --cache-dir $cache_dir
    ```

4. [Optional] Merge annotation files.

    ```bash
    split_files=... ... ...
    output_folder=...
    output_filename=...

    python run/datasets/merge_soccernet_annotation_files.py \
        --split-files $split_files \
        --output-folder $output_folder \
        --output_filename $output_filename
    ```

5. [Optional] Merge labels' folders.

    ```bash
    labels_folders=... ... ...
    output_folder=...

    python run/datasets/merge_soccernet_annotation_files.py \
        --split-folders $labels_folders \
        --output-folder $output_folder \
    ```


### Ball Action Spotting


1. [Download the dataset](https://www.soccer-net.org/data#h.ykgf675j127d).


2. Extract frames.

    ```bash
    cd eztorch

    fps=25
    input_folder="./SoccerNet_ball/"
    output_folder="./soccernet_ball_extracted_${fps}fps/"
    split=train

    python run/datasets/extract_soccernet.py \
        --input-folder $input_folder \
        --output-folder $output_folder \
        --fps $fps \
        --split $split
    ```

## Issue
If you have trouble making this work or have any questions, open an [issue](https://github.com/juliendenize/eztorch/issues) to describe your problem.
