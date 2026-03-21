#!/usr/bin/env bash
# =============================================================================
# slurm_single_gsplat.sh
# SLURM job script: extract one 4D-DRESS frame + train a 3DGS model from it.
#
# Usage (direct sbatch):
#   sbatch slurm_single_gsplat.sh \
#       --export=TARGET=s170_t1_f021,SUBJECT=170,TAKE=1,FRAME=21
#
# Or called automatically by submit_all_gsplats.sh.
#
# Environment variables read at runtime:
#   TARGET   : human-readable name  (e.g. s170_t1_f021)
#   SUBJECT  : integer subject ID   (e.g. 170)
#   TAKE     : integer take ID      (e.g. 1)
#   FRAME    : integer frame index  (e.g. 21)
#   ITERS    : training iterations  (default: 7000)
#   ENV_NAME : conda env name       (default: gsplat_env)
#
# Output layout:
#   $OUTPUT_BASE/extracted/$TARGET/   ← multi-view images + cameras.pkl
#   $OUTPUT_BASE/models/$TARGET/      ← trained .ply model
# =============================================================================

#SBATCH --job-name=gsplat
#SBATCH --account=a168
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/gsplat_logs/gsplat_%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/dbartaula/gsplat_logs/gsplat_%x_%j.err

set -euo pipefail

# ── Job parameters (can be overridden via --export) ──────────────────────────
TARGET="${TARGET:-s170_t1_f021}"
SUBJECT="${SUBJECT:-170}"
TAKE="${TAKE:-1}"
FRAME="${FRAME:-21}"
ITERS="${ITERS:-7000}"
ENV_NAME="${ENV_NAME:-gsplat_train}"

# ── Paths ─────────────────────────────────────────────────────────────────────
MPMAVATAR_DIR="/iopsstor/scratch/cscs/dbartaula/MPMAvatar"
DATASET_ROOT="/iopsstor/scratch/cscs/dbartaula/4D-DRESS"
CONDA_BASE="/users/dbartaula/miniforge3"

OUTPUT_BASE="/iopsstor/scratch/cscs/dbartaula/gsplat_4ddress"
EXTRACTED_DIR="${OUTPUT_BASE}/extracted/${TARGET}"
MODEL_DIR="${OUTPUT_BASE}/models/${TARGET}"

mkdir -p /iopsstor/scratch/cscs/dbartaula/gsplat_logs

echo "================================================================"
echo " GSplat job: ${TARGET}"
echo "  Subject: ${SUBJECT}  Take: ${TAKE}  Frame: ${FRAME}"
echo "  Iters:   ${ITERS}"
echo "  Output:  ${OUTPUT_BASE}"
echo "  Node:    $(hostname)"
echo "  GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "================================================================"

# ── Activate conda env ───────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[env] Python: $(which python)"
echo "[env] torch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"

# ── Step 1: Extract frame ────────────────────────────────────────────────────
echo ""
echo "[1/2] Extracting frame ${FRAME} for s${SUBJECT} Take${TAKE} ..."

python "${MPMAVATAR_DIR}/bridge_sds/gsplat_4ddress/extract_frame.py" \
    --root    "${DATASET_ROOT}" \
    --subject "${SUBJECT}" \
    --take    "${TAKE}" \
    --frame   "${FRAME}" \
    --output_dir "${EXTRACTED_DIR}"

echo "[1/2] Extraction done → ${EXTRACTED_DIR}"

# ── Step 2: Train GSplat ──────────────────────────────────────────────────────
echo ""
echo "[2/2] Training GSplat (${ITERS} iters) ..."

cd "${MPMAVATAR_DIR}"     # ensures relative imports work

python bridge_sds/gsplat_4ddress/train_single_gsplat.py \
    --data_dir   "${EXTRACTED_DIR}" \
    --output_dir "${MODEL_DIR}" \
    --iterations "${ITERS}" \
    --sh_degree  3 \
    --device     cuda

echo "[2/2] Training done → ${MODEL_DIR}"

echo ""
echo "================================================================"
echo " DONE: ${TARGET}"
echo "  Model: ${MODEL_DIR}/point_cloud_final.ply"
echo "================================================================"
