#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SDS_CFG="${SDS_CFG:-${SCRIPT_DIR}/configs/multiview_batch_video.yaml}"
SAVE_NAME="${SAVE_NAME:-multiview_batched_video}"
NUM_CAMS="${NUM_CAMS:-4}"
TIMESTEP_BIAS="${TIMESTEP_BIAS:-transition_core}"

"${SCRIPT_DIR}/training.sh"
