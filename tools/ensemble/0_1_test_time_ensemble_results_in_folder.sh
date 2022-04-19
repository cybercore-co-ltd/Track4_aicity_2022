#!/usr/bin/env bash
set -e

# input
ALL_VIDEO_DIR=$1
val_log_tree="./tools/ensemble/val_log_tree_after_ensemble.pkl"
# ALL_VIDEO_DIR="./data/TestA"

PKLS="cache/process_results"

# output 
SAVE_ENSEMBLE_RESULTS="cache/det_ensemble"

for VIDEO_RESULT in $PKLS/*; do
    VIDEO_NAME=$(basename "$VIDEO_RESULT")
    TEMP_PKL="./cache/temp"

    # preprocess
    python tools/ensemble/1_preprocess_pkl.py \
        --save_pkl_dir $TEMP_PKL --input_pkl_dir $VIDEO_RESULT

    # ensemble using log tree
    CFG='configs/_base_/datasets/agnostic_track4_albu_random_bg_detection.py'
    save_test_time_ensemble_pkl="${SAVE_ENSEMBLE_RESULTS}/${VIDEO_NAME}.pkl"
    save_val_processed_pkl_dir=$TEMP_PKL
    python tools/ensemble/test_time_greedy_auto_ensemble_wbf.py \
        $CFG $save_val_processed_pkl_dir \
        $save_test_time_ensemble_pkl $val_log_tree \
        --video_dir "${ALL_VIDEO_DIR}/${VIDEO_NAME}.mp4" \
        --video_numclasses 1
done

# combine time, results
python tools/ensemble/_combine_timestamp_results.py \
    --ensemble_pkls_dir $SAVE_ENSEMBLE_RESULTS \
    --original_pkls_dir "data/full_video_pkl/lwof_frcnn_swin_shapenet_gan"