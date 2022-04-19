#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

if [[ ! -e ./cache ]]; then
    mkdir ./cache
fi

if [ -z "$1" ]; then
    echo "No videos path specified"
    exit 0
fi

if [ -z "$2" ]; then
    echo "No output path specified"
    exit 0
fi

model_name="lwof_frcnn_swin_shapenet_gan"
CFG="configs/ccdet/track4/lwof_frcnn_swin_shapenet_gan.py"
CKPT="work_dirs/lwof_frcnn_swin_shapenet_gan/epoch_1.pth"

TEMP_OUT_CFG="$2/${model_name}_cfg.py"
python tools/create_tta_cfg_det.py --cfg $CFG --out $TEMP_OUT_CFG

# DET_CACHE="cache/full_video_pkl/${model_name}"
OUT_DIR="$2/${model_name}"

VIDEO_DIR=$1
for VIDEO in $VIDEO_DIR/*mp4; do
    python tools/infer_det_video.py --cfg $TEMP_OUT_CFG --ckpt $CKPT \
        --det-thr 0.15 --video $VIDEO --out-dir $OUT_DIR \
        --save-pred --coco-filter
done