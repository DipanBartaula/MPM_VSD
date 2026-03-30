#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SDS_CFG="${SDS_CFG:-${SCRIPT_DIR}/configs/last_substep_tape.yaml}"
SAVE_NAME="${SAVE_NAME:-last_substep_tape}"
OPTIM="${OPTIM:-warp_autodiff}"
NUM_CAMS="${NUM_CAMS:-1}"
TIMESTEP_BIAS="${TIMESTEP_BIAS:-transition_core}"
TAPED_SUBSTEPS_PER_FRAME="${TAPED_SUBSTEPS_PER_FRAME:-1}"

"${SCRIPT_DIR}/training.sh"
