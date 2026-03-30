#!/bin/bash
set -euo pipefail

# Shared training launcher for VDS experiments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAINED_MODEL_PATH="${TRAINED_MODEL_PATH:-${REPO_DIR}/pretrained_models/output/tracking/a1_s1_460_200}"
MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/pretrained_models/model/a1_s1}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/output}"
WAN_CKPT_DIR="${WAN_CKPT_DIR:-${REPO_DIR}/wan_5b_model}"
SDS_CFG="${SDS_CFG:-${REPO_DIR}/bridge_sds/configs/sds_test.yaml}"
SAVE_NAME="${SAVE_NAME:-sds_mpm_rollout}"
ITERATIONS="${ITERATIONS:-2000}"
OPTIM="${OPTIM:-warp_autodiff}"
NUM_CAMS="${NUM_CAMS:-1}"
TIMESTEP_BIAS="${TIMESTEP_BIAS:-transition_core}"
TAPED_SUBSTEPS_PER_FRAME="${TAPED_SUBSTEPS_PER_FRAME:-4}"
CONDITION_CAMERA_IDX="${CONDITION_CAMERA_IDX:-0}"
USE_MASK="${USE_MASK:-0}"
USE_ATTENTION_SOFT_MASK="${USE_ATTENTION_SOFT_MASK:-0}"
ACTOR="${ACTOR:-1}"
SEQUENCE="${SEQUENCE:-1}"
TRAIN_FRAME_START="${TRAIN_FRAME_START:-460}"
TRAIN_FRAME_COUNT="${TRAIN_FRAME_COUNT:-121}"
VERTS_START_IDX="${VERTS_START_IDX:-460}"

ARGS=(
    --trained_model_path "${TRAINED_MODEL_PATH}"
    --model_path "${MODEL_PATH}"
    --dataset_dir "${DATASET_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --actor "${ACTOR}"
    --sequence "${SEQUENCE}"
    --train_frame_start_num "${TRAIN_FRAME_START}" "${TRAIN_FRAME_COUNT}"
    --verts_start_idx "${VERTS_START_IDX}"
    --wan_ckpt_dir "${WAN_CKPT_DIR}"
    --sds_cfg "${SDS_CFG}"
    --iterations "${ITERATIONS}"
    --num_cams "${NUM_CAMS}"
    --timestep_bias "${TIMESTEP_BIAS}"
    --condition_camera_idx "${CONDITION_CAMERA_IDX}"
    --taped_substeps_per_frame "${TAPED_SUBSTEPS_PER_FRAME}"
    --optim "${OPTIM}"
    --save_name "${SAVE_NAME}"
    --random_init_params
)

if [[ "${USE_MASK}" == "1" ]]; then
    ARGS+=(--use_mask)
fi

if [[ "${USE_ATTENTION_SOFT_MASK}" == "1" ]]; then
    ARGS+=(--use_attention_soft_mask)
fi

python "${REPO_DIR}/train_sds_physics.py" "${ARGS[@]}"
