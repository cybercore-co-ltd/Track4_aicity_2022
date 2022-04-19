#!/usr/bin/env bash
set -e
AVAI_GPUS=(0 1)

DET_CACHE="cache/det_ensemble"
TRACK_CACHE="cache/track_ensemble"

CLS_CFG="configs/cccls/repvgg_a0_4xb64_10e_aicity22t4.py configs/cccls/swins_4xb256_10e_aicity22t4.py configs/cccls/res2net50_4xb64_10e_aicity22t4.py"
CLS_CKPT="work_dirs/repvgg_a0_4xb64_10e_aicity22t4/epoch_10.pth work_dirs/swins_4xb256_10e_aicity22t4/epoch_10.pth work_dirs/res2net50_4xb64_10e_aicity22t4/epoch_10.pth"
WEIGHT="0.2 0.5 0.3"
CLS_CACHE="cache/cls_ensemble"

DET_THR=0.4
CLS_THR=0.3
MATCH_THR=0.9
TEMP=1
ALPHA=1
BETA=1
GAMMA=1
MIN_AGE=10

SUBMISSION_FILE="cache/submission.txt"
VIDEO_DIR=$1

# run tracking
for VIDEO in $VIDEO_DIR/*mp4; do
    PKL="$DET_CACHE/$(basename $VIDEO .mp4).pkl"
    python tools/infer_track_video_from_det.py \
        --pkl $PKL --video $VIDEO \
        --det-thr $DET_THR --match-thr $MATCH_THR \
        --out-dir $TRACK_CACHE --save-pred & 
done
wait

# run classification
IDX=0
for VIDEO in $VIDEO_DIR/*mp4; do
    PKL="$TRACK_CACHE/$(basename $VIDEO .mp4).pkl"
    python tools/infer_cls_video_from_det_ensemble.py \
        --cfg $CLS_CFG --ckpt $CLS_CKPT --weight $WEIGHT \
        --video $VIDEO --pkl $PKL \
        --det-thr $DET_THR --cls-thr $CLS_THR \
        --out-dir $CLS_CACHE --save-pred --fp16
    IDX=$((IDX+1))
done

# create submission file
python tools/testA/create_submission_file.py \
    --video-dir $VIDEO_DIR --pkl-dir $CLS_CACHE --out $SUBMISSION_FILE \
    --temp $TEMP --alpha $ALPHA --beta $BETA --gamma $GAMMA --min-age $MIN_AGE
