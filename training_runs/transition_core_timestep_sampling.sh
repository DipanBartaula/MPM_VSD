#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SDS_CFG="${SDS_CFG:-${SCRIPT_DIR}/configs/single_view_video.yaml}"
SAVE_NAME="${SAVE_NAME:-transition_core_timestep_sampling}"
NUM_CAMS="${NUM_CAMS:-1}"
TIMESTEP_BIAS="${TIMESTEP_BIAS:-transition_core}"

"${SCRIPT_DIR}/training.sh"
