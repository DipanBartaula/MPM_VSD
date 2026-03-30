#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SDS_CFG="${SDS_CFG:-${SCRIPT_DIR}/configs/attention_soft_mask_video.yaml}"
SAVE_NAME="${SAVE_NAME:-attention_soft_mask_video}"
NUM_CAMS="${NUM_CAMS:-1}"
TIMESTEP_BIAS="${TIMESTEP_BIAS:-transition_core}"
USE_ATTENTION_SOFT_MASK="${USE_ATTENTION_SOFT_MASK:-1}"

"${SCRIPT_DIR}/training.sh"
