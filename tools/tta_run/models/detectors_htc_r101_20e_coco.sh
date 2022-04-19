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

model_name="detectors_htc_r101_20e_coco"
CFG="configs/ccdet/coco/detectors_htc_r101_20e_coco.py"
CKPT="https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r101_20e_coco/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth"

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