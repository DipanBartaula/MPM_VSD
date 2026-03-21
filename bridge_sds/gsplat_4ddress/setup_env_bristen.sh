#!/usr/bin/env bash
# =============================================================================
# setup_env_bristen.sh
# Minimal conda environment for running gsplat (3DGS) from multi-view images.
# Targets CSCS Bristen (GH200 = aarch64 CPU + H100 GPU, CUDA 12.x).
#
# Run ONCE on a login node (or interactive node) before submitting jobs:
#   bash bridge_sds/gsplat_4ddress/setup_env_bristen.sh
#
# What this installs:
#   - PyTorch 2.3 + CUDA 12.1
#   - diff_gauss  (diff-gaussian-rasterization CUDA extension)
#   - simple-knn  (spatial k-NN CUDA extension)
#   - plyfile, pillow, pyyaml, tqdm, imageio, opencv-headless
#
# After install activate with:
#   conda activate gsplat_env
# =============================================================================

set -euo pipefail

ENV_NAME="${GSPLAT_ENV:-gsplat_env}"
MPMAVATAR_DIR="$(cd "$(dirname "$0")/../.." && pwd)"   # bridge_sds/../.. = MPMAvatar/

echo "============================================================"
echo " Setting up gsplat environment: ${ENV_NAME}"
echo " MPMAvatar root: ${MPMAVATAR_DIR}"
echo "============================================================"

# ── 0. Conda availability ────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda not found. On Bristen, load it with:"
    echo "  module load conda"
    echo "  # or: module load miniconda3"
    exit 1
fi

# ── 1. Create env if it doesn't exist ───────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[skip] Conda env '${ENV_NAME}' already exists."
else
    echo "[step 1] Creating conda env: ${ENV_NAME} (python=3.10)"
    conda create -y -n "${ENV_NAME}" python=3.10
fi

# ── 2. Activate ──────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "[step 2] Activated: $(which python)"

# ── 3. PyTorch (CUDA 12.1, works on H100/GH200) ─────────────────────────────
echo "[step 3] Installing PyTorch 2.3 + CUDA 12.1"
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 4. Core Python deps ──────────────────────────────────────────────────────
echo "[step 4] Installing core Python packages"
pip install \
    numpy==1.25.0 \
    Pillow \
    plyfile==1.0.3 \
    pyyaml \
    tqdm \
    imageio \
    "imageio[ffmpeg]" \
    scipy \
    matplotlib \
    opencv-python-headless

# ── 5. diff-gaussian-rasterization (CUDA extension) ─────────────────────────
# Build for H100 (sm_90) — GH200 uses the same H100 GPU die.
echo "[step 5] Building diff-gaussian-rasterization for sm_90 (H100/GH200)"
BUILD_DIR="/tmp/diff_gauss_build_$$"
mkdir -p "${BUILD_DIR}"
git clone --depth 1 \
    https://github.com/slothfulxtx/diff-gaussian-rasterization.git \
    "${BUILD_DIR}/diff-gaussian-rasterization"

TORCH_CUDA_ARCH_LIST="9.0" \
FORCE_CUDA=1 \
    pip install "${BUILD_DIR}/diff-gaussian-rasterization"

rm -rf "${BUILD_DIR}"
echo "[step 5] diff_gauss installed."

# ── 6. simple-knn (CUDA extension) ───────────────────────────────────────────
echo "[step 6] Building simple-knn for sm_90"
BUILD_DIR2="/tmp/simple_knn_build_$$"
mkdir -p "${BUILD_DIR2}"
git clone --depth 1 \
    https://gitlab.inria.fr/bkerbl/simple-knn.git \
    "${BUILD_DIR2}/simple-knn"

TORCH_CUDA_ARCH_LIST="9.0" \
FORCE_CUDA=1 \
    pip install "${BUILD_DIR2}/simple-knn"

rm -rf "${BUILD_DIR2}"
echo "[step 6] simple_knn installed."

# ── 7. Quick sanity check ────────────────────────────────────────────────────
echo ""
echo "[step 7] Sanity check"
python - <<'PYEOF'
import torch, diff_gauss, simple_knn, plyfile, PIL, cv2, yaml
print(f"  torch       : {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"  diff_gauss  : OK")
print(f"  simple_knn  : OK")
print(f"  plyfile     : OK")
print("  All OK!")
PYEOF

echo ""
echo "============================================================"
echo " Environment '${ENV_NAME}' ready."
echo " Activate with:  conda activate ${ENV_NAME}"
echo " Then run jobs:  bash bridge_sds/gsplat_4ddress/submit_all_gsplats.sh"
echo "============================================================"
