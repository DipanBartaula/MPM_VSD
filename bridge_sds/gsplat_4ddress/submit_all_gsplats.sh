#!/usr/bin/env bash
# =============================================================================
# submit_all_gsplats.sh
# Submit all 12 gsplat jobs to SLURM on Bristen.
#
# Prerequisites:
#   1. conda activate gsplat_env  (or env already set)
#   2. Run from MPMAvatar/ root
#
# Usage:
#   cd MPMAvatar
#   bash bridge_sds/gsplat_4ddress/submit_all_gsplats.sh [--dry_run] [--iters N]
#
# Options:
#   --dry_run    Print sbatch commands without submitting
#   --iters N    Override training iterations (default: 7000)
#   --account X  CSCS account (default: reads SLURM_ACCOUNT env or prompts)
#   --skip_done  Skip targets whose model already exists
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MPMAVATAR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/slurm_single_gsplat.sh"

# ── Parse args ────────────────────────────────────────────────────────────────
DRY_RUN=0
ITERS=7000
ACCOUNT="${SLURM_ACCOUNT:-}"
SKIP_DONE=0
ENV_NAME="${GSPLAT_ENV:-gsplat_env}"
SCRATCH="${SCRATCH:-/iopsstor/scratch/cscs/${USER}}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry_run)   DRY_RUN=1; shift ;;
        --iters)     ITERS="$2"; shift 2 ;;
        --account)   ACCOUNT="$2"; shift 2 ;;
        --skip_done) SKIP_DONE=1; shift ;;
        --env)       ENV_NAME="$2"; shift 2 ;;
        --scratch)   SCRATCH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Resolve account ───────────────────────────────────────────────────────────
if [[ -z "${ACCOUNT}" ]]; then
    # Try to read from the slurm script header as fallback
    ACCOUNT=$(grep '#SBATCH --account' "${JOB_SCRIPT}" | awk '{print $3}' || echo "")
    if [[ "${ACCOUNT}" == "YOUR_ACCOUNT" || -z "${ACCOUNT}" ]]; then
        echo "[ERROR] CSCS account not set."
        echo "  Set it with: --account <account_name>"
        echo "  or: export SLURM_ACCOUNT=<account_name>"
        echo "  (Check yours with: sacctmgr show user ${USER} withassoc)"
        exit 1
    fi
fi

# ── All 12 targets ────────────────────────────────────────────────────────────
# Format: "NAME SUBJECT TAKE FRAME"
TARGETS=(
    "s170_t1_f021 170 1 21"
    "s170_t1_f060 170 1 60"
    "s170_t1_f100 170 1 100"
    "s185_t1_f021 185 1 21"
    "s185_t1_f060 185 1 60"
    "s185_t1_f100 185 1 100"
    "s190_t1_f021 190 1 21"
    "s190_t1_f060 190 1 60"
    "s190_t1_f100 190 1 100"
    "s191_t1_f021 191 1 21"
    "s191_t1_f060 191 1 60"
    "s191_t1_f100 191 1 100"
)

echo "================================================================"
echo " Submitting ${#TARGETS[@]} gsplat jobs to Bristen"
echo "  Account  : ${ACCOUNT}"
echo "  Iters    : ${ITERS}"
echo "  Env      : ${ENV_NAME}"
echo "  Scratch  : ${SCRATCH}"
echo "  Dry run  : ${DRY_RUN}"
echo "================================================================"
echo ""

mkdir -p "${MPMAVATAR_DIR}/logs"

SUBMITTED=0
SKIPPED=0

for entry in "${TARGETS[@]}"; do
    read -r TARGET SUBJECT TAKE FRAME <<< "${entry}"

    MODEL_PLY="${SCRATCH}/gsplat_4ddress/models/${TARGET}/point_cloud_final.ply"

    # Skip if already done
    if [[ "${SKIP_DONE}" -eq 1 && -f "${MODEL_PLY}" ]]; then
        echo "  [skip] ${TARGET} — model already exists"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    CMD=(
        sbatch
        "--account=${ACCOUNT}"
        "--job-name=gsplat_${TARGET}"
        "--export=ALL,TARGET=${TARGET},SUBJECT=${SUBJECT},TAKE=${TAKE},FRAME=${FRAME},ITERS=${ITERS},ENV_NAME=${ENV_NAME},SCRATCH=${SCRATCH}"
        "${JOB_SCRIPT}"
    )

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  [dry] ${CMD[*]}"
    else
        JOB_ID=$("${CMD[@]}" | awk '{print $NF}')
        echo "  [submitted] ${TARGET} → job ${JOB_ID}"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

echo ""
echo "================================================================"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo " Dry run complete. Remove --dry_run to submit."
else
    echo " Submitted: ${SUBMITTED}  Skipped: ${SKIPPED}"
    echo ""
    echo " Monitor with:"
    echo "   squeue -u ${USER}"
    echo " Check logs:"
    echo "   tail -f ${MPMAVATAR_DIR}/logs/gsplat_*.out"
    echo " Models saved to:"
    echo "   ${SCRATCH}/gsplat_4ddress/models/"
fi
echo "================================================================"
