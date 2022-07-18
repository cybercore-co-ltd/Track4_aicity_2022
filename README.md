# Code release for AI City Challenge 2022 - Track 4 - Team 9

:boom::boom::boom: Paper: [Improving Domain Generalization by Learning without Forgetting: Application in Retail Checkout](https://arxiv.org/abs/2207.05422)

:boom::boom::boom: Our solution wins the 1st place Track 4. https://www.aicitychallenge.org/2022-challenge-winners/
## Table of contents

1. [Operating Systems and Hardware Specs](#operating-systems-and-hardware-specs)

- Introduce the OS and hardware we use.

3. [Dataset preparation and environment setup](#dataset-preparation-and-environment-setup)

- Give instructions to prepare datasets for training detector/classifier and set up environments.

4. [Train model](#train-model)

- Provide scripts to train detector and classifier models.

5. [Run detector and ensemble detection results](#run-detector-and-ensemble-detection-results)

- Provide inference script for detector and give instructions on how to run ensemble.

6. [Create submission file](#create-submission-file)

- Provide script to create submission file to submit to test server.

---

## Operating Systems and Hardware Specs

- Ubuntu 18.04
- Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
- 4 GPUs Quadro RTX 8000 48GB
- Driver: Version: 495.29.05, CUDA: 11.5
- RAM: 256GB

---

## Dataset preparation and environment setup

- If you use our trained models, you can download [here](http://118.69.233.170:60001/open/AICity/track4/ckpts.zip) and ignore the `dataset preparation` step. Unzip the .zip file in the code directory. You should see the following structure:
```bash
    work_dirs
    ├── lwof_frcnn_swin_shapenet_gan
    │   └── epoch_1.pth
    ├── repvgg_a0_4xb64_10e_aicity22t4
    │   └── epoch_10.pth
    ├── res2net50_4xb64_10e_aicity22t4
    │   └── epoch_10.pth
    └── swins_4xb256_10e_aicity22t4
        └── epoch_10.pth
```

- We split the dataset `Auto-retail-syndata-release` (provided by the challenge) into train and val sets. Please download the annotation files via the [link](http://118.69.233.170:60001/open/AICity/track4/anns.zip). Then, unzip it into `./data/Auto-retail-syndata-release/anns` as the following:
```bash
./data/Auto-retail-syndata-release/anns
├── train.txt
└── val.txt
```


### Dataset preparation

- We created datasets for training from random backgrounds and GANs pre-trained model.

- Please run the following command to create datasets:

```
cd tools/data_process
bash download.sh
cd  ../..
```

Dataset will be created in `data` folder with following structure:

```
./data
├── annotations
├── biggan_imagenet
├── CelebA_128x128_N2M2S64
├── LSUN_256x256_N2M2S128
├── random_bg
├── Auto-retail-syndata-release
└── ShapeNetRendering
```

<details>
  <summary>Folder description</summary>

    - "annotation" contains coco format json file for 'random_bg' images, 'random_bg' contain random backgrounds which were created to train our model.
    - "CelebA_128x128_N2M2S64" and "LSUN_256x256_N2M2S128" are folders containing random backgrounds which are created by GAN model at [COCO-GAN](https://hubert0527.github.io/COCO-GAN/).
    - "biggan_imagenet" contains random backgrounds which are created by [biggan](https://github.com/huggingface/pytorch-pretrained-BigGAN).
    - ShapeNetRendering contained ShapeNet images
    - "Auto-retail-syndata-release" contains AI city challenge dataset, please download and extract data by following organizers guide.

</details>

### Environment setup

<details>
  <summary>Create environment and install dependencies</summary>

    # create conda env
    conda create -n aicity22_track4 python=3.7 -y
    conda activate aicity22_track4
    # Install others
    conda install pytorch=1.10.0 torchvision cudatoolkit=10.2 -c pytorch -y
    conda install cython -y
    pip install click==7.1.2 openmim future tensorboard sklearn timm==0.4.12 tqdm torch_optimizer shapely scikit-learn scikit-image albumentations lap cython_bbox numba
    pip install git+https://github.com/thuyngch/cvut
    mim install mmcv-full==1.4.6 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html -y
    mim install mmcls==0.21.0 -y
    mim install mmsegmentation==0.22.0 -y
    mim install mmdet==2.22.0 -y
    python setup.py develop
    pip install Pillow==6.1 # Ignore the error

</details>

---

## Train model

- If you use our trained models, you can download [here](http://118.69.233.170:60001/open/AICity/track4/ckpts.zip) and ignore the `Train model` step. Unzip the .zip file and you should see the folder structure like in [Dataset preparation](#dataset-preparation) section.

### Train detection model

- The detection model must be trained using 4 GPUs to be able to pre-produce the result.
- Use the following script to train:
  <details>
    <summary>Train detector</summary>

      #!/usr/bin/env bash
      set -e

      ## make sure GPUs are not hang
      export NCCL_P2P_DISABLE=1

      #set which GPUs will be used to train
      export CUDA_VISIBLE_DEVICES=0,1,2,3

      export CUDA_LAUNCH_BLOCKING=1
      export MKL_NUM_THREADS=1
      export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

      #path to your model config file
      CFG="configs/ccdet/track4/lwof_frcnn_swin_shapenet_gan.py"

      #Set number of GPUs will be used to train
      GPUS_NUM=4

      #Command to train our detection model
      mim train mmdet $CFG --gpus $GPUS_NUM --launcher pytorch --no-validate

  </details>

Note: The detector's checkpoint will be saved at './work_dirs/lwof_frcnn_swin_shapenet_gan' folder

### Train classification model

- Following scripts will train 3 classification models namely RepVGG-A0, SwinS, and Res2Net50.
- Must use 4 GPUs to reproduce the results.
- The checkpoints will be saved under "./work_dirs"

<details>
  <summary>Train classifier</summary>

    #!/usr/bin/env bash
    set -e

    export GPUS=4
    export NCCL_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    CFG="configs/cccls/repvgg_a0_4xb64_10e_aicity22t4.py"
    WORKDIR="./work_dirs/repvgg_a0_4xb64_10e_aicity22t4"
    mim train mmcls $CFG --work-dir $WORKDIR --launcher pytorch --gpus $GPUS \
        --seed 0 --deterministic

    CFG="configs/cccls/swins_4xb256_10e_aicity22t4.py"
    WORKDIR="./work_dirs/swins_4xb256_10e_aicity22t4"
    mim train mmcls $CFG --work-dir $WORKDIR --launcher pytorch --gpus $GPUS \
        --seed 0 --deterministic

    CFG="configs/cccls/res2net50_4xb64_10e_aicity22t4.py"
    WORKDIR="./work_dirs/res2net50_4xb64_10e_aicity22t4"
    mim train mmcls $CFG --work-dir $WORKDIR --launcher pytorch --gpus $GPUS \
        --seed 0 --deterministic

</details>

---

## Run detector and ensemble detection results

### Run detector

- Following script will run 5 models located in `tools/tta_run/models`.

- `<path/to/videos>` is the path to test videos which are used to submit to the test server. For example, all videos are located under "./data/testB_videos", then `<path/to/videos>` will be `"./data/testB_videos"`.
- Run:
  ```bash
  mkdir data/full_video_pkl
  bash ./tools/tta_run/run.sh <path/to/videos> "./data/full_video_pkl"
  rm data/full_video_pkl/*.py
  ```
- The detection result will be saved under `"./data/full_video_pkl"` with the following structure:

```
  ./data/full_video_pkl
  ├── detectors_htc_r101_20e_coco
  │   ├── testA_1.pkl # This is output detection result for video name TestA_1.mp4
  ├── lwof_frcnn_swin_shapenet_gan
  │   ├── testA_1.pkl
  ...
```

### Ensemble detection results

- In the following scripts, `<path/to/videos>` is the path containing videos, it must be similar to `<path/to/videos>` in [Run detector](#run-detector).
- The ensemble results of videos will be saved at: `./cache/det_ensemble`

```
rm data/full_video_pkl/*.py
python tools/ensemble/0_0_organize_pkl_results.py
./tools/ensemble/0_1_test_time_ensemble_results_in_folder.sh <path/to/videos>
```

---

## Create submission file

- Following script will run classifier, tracker, and create file for submission.
- `<path/to/videos>` is same as `<path/to/videos>` in [Run detector](#run-detector).
- Videos for visualization are saved under: `./cache/cls_ensemble`
- Submission file is saved at: `./cache/submission.txt`, use this file to submit to test server.

```
./tools/create_submission_file.sh <path/to/videos>
```

---

## ABOUT US

We are Cybercore Co., Ltd. We proudly provide our award winning academic-level, state-of-the-art custom-made technologies to our customers in the field of image processing, image recognition and artificial intelligence (AI)

<p align="center">
  <img src="imgs/logo_cybercore.png">
</p>

<p align="center">
    ■Head Office <br />
    Malios 10th floor,
    2-9-1 Morioka-Eki Nishi-Dori,
    Morioka, Iwate, Japan 020-0045
    Tel : +81-19-681-8776 <br /><br />
    ■Tokyo Office <br />
    Toranomon-Hills Business Tower 15F,
    Minato-ku, Tokyo,  Japan 105-6490<br /><br />
    ■Vietnam (Affiliated company) <br />
    12th TNR Tower, 180-192 Nguyen Cong Tru St., District 1, HCMC
    Tel : (+84)028 710 77 710
</p>
