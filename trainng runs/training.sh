#!/bin/bash
set -euo pipefail

# Canonical VDS training entrypoint:
# 1. run the forward MPM rollout
# 2. render the rollout frames into videos
# 3. use SDS to update the simulation parameters

TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH:-../pretrained_models/output/tracking/a1_s1_460_200}"
MODEL_PATH="${MODEL_PATH:-../pretrained_models/model/a1_s1}"
DATASET_DIR="${DATASET_DIR:-../data}"
OUTPUT_DIR="${OUTPUT_DIR:-../output}"
WAN_CKPT_DIR="${WAN_CKPT_DIR:-../wan_5b_model}"
SDS_CFG="${SDS_CFG:-../bridge_sds/configs/sds_test.yaml}"
SAVE_NAME="${SAVE_NAME:-sds_mpm_rollout}"
ITERATIONS="${ITERATIONS:-2000}"
OPTIM="${OPTIM:-spsa}"
TAPED_SUBSTEPS_PER_FRAME="${TAPED_SUBSTEPS_PER_FRAME:-4}"
CONDITION_CAMERA_IDX="${CONDITION_CAMERA_IDX:-0}"
ACTOR="${ACTOR:-1}"
SEQUENCE="${SEQUENCE:-1}"
TRAIN_FRAME_START="${TRAIN_FRAME_START:-460}"
TRAIN_FRAME_COUNT="${TRAIN_FRAME_COUNT:-32}"
VERTS_START_IDX="${VERTS_START_IDX:-460}"

python ../train_sds_physics.py \
    --trained_model_path "${TRAINED_MODEL_PATH}" \
    --model_path "${MODEL_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --actor "${ACTOR}" \
    --sequence "${SEQUENCE}" \
    --train_frame_start_num "${TRAIN_FRAME_START}" "${TRAIN_FRAME_COUNT}" \
    --verts_start_idx "${VERTS_START_IDX}" \
    --wan_ckpt_dir "${WAN_CKPT_DIR}" \
    --sds_cfg "${SDS_CFG}" \
    --iterations "${ITERATIONS}" \
    --condition_camera_idx "${CONDITION_CAMERA_IDX}" \
    --taped_substeps_per_frame "${TAPED_SUBSTEPS_PER_FRAME}" \
    --optim "${OPTIM}" \
    --save_name "${SAVE_NAME}" \
    --random_init_params
