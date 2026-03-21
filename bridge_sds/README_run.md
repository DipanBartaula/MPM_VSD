# bridge_sds — SDS-guided Physics Parameter Optimisation

Bridge between **MPMAvatar** (Gaussian cloth simulation) and **Dreamcloth Phase3 / Wan I2V**
(video diffusion prior) for score-distillation–guided optimisation of cloth material parameters
φ = {D (density), E (Young's modulus), H (height scale)}.

---

## Repository layout

```
MPMAvatar/
└── bridge_sds/
    ├── __init__.py
    ├── configs/
    │   └── sds_phi.yaml            ← default hyper-parameters
    ├── runner_mpmavatar.py         ← given φ → rendered RGB clip
    ├── scorer_phase3.py            ← given clip → SDS scalar loss (Wan I2V)
    ├── physical_regularizers.py    ← L_pen + L_stretch + L_temporal_smooth
    ├── optimize_phi.py             ← SPSA optimisation loop  (CLI entry point)
    ├── utils_video_io.py           ← mp4 I/O, montage, checkpoint helpers
    └── README_run.md               ← this file
```

---

## Prerequisites

> **Current status:** `scorer_phase3.py` intentionally fails if Wan/Phase3 model
> loading is not implemented. Until `_load_model()` is completed on
> Clariden/Bristen, run with `--no_sds` (regularizers-only mode).

### 1. MPMAvatar appearance model (mandatory)

Run the standard MPMAvatar appearance training first:

```bash
bash MPMAvatar/scripts/appearance/actorshq_a1.sh
```

This produces the Gaussian model at `./output/tracking/a1_s1_460_200/`.

### 2. MPMAvatar preprocessing outputs (mandatory)

The following files must exist before calling the bridge:

| File | Description |
|------|-------------|
| `data/a1_s1/split_idx.npz` | cloth / body vertex split indices |
| `data/a1_s1/a1s1_uv.obj`   | cloth UV reference mesh (rest pose) |
| `data/a1_s1/smplx_fitted/XXXXXX/smplx_icp.obj` | SMPLX body mesh per frame |
| (optional) `data/a1_s1/joint_v_idx.npy` | cloth-body attachment vertex indices |

If `joint_v_idx.npy` is absent, a proximity-based heuristic is used automatically.

### 3. Python environment

```bash
pip install -r MPMAvatar/requirements.txt
pip install imageio[ffmpeg] pyyaml   # extra bridge dependencies
```

### 4. Wan I2V checkpoint (required for SDS; optional otherwise)

Place the Wan 2.1/2.2 I2V checkpoint on Clariden/Bristen and pass its path
via `--phase3_ckpt`.  If you want to run with **physical regularisers only**
(no SDS), use `--no_sds`.

---

## Quickstart — physical regularisers only (no SDS, no checkpoint needed)

```bash
cd MPMAvatar

python bridge_sds/optimize_phi.py \
    --save_name a1_s1_phys_only \
    --trained_model_path ./output/tracking/a1_s1_460_200 \
    --model_path          ./model \
    --dataset_dir         ./data \
    --dataset_type        actorshq \
    --actor 1 --sequence 1 \
    --train_frame_start_num 460 25 \
    --verts_start_idx 460 \
    --uv_path         ./data/a1_s1/a1s1_uv.obj \
    --split_idx_path  ./data/a1_s1/split_idx.npz \
    --test_camera_index 6 126 \
    --no_sds \
    --optim spsa \
    --iters 200
```

Expected output structure:
```
output/bridge_sds/
├── renders/a1_s1_phys_only/iter_00000.mp4
│                            iter_00100.mp4 ...
├── phis/a1_s1_phys_only/   phi_iter_00000.npz ...
│                            best_phi.json
└── logs/a1_s1_phys_only/   metrics.jsonl
```

---

## Full run — with SDS (Wan I2V on Clariden)

```bash
python bridge_sds/optimize_phi.py \
    --save_name a1_s1_sds \
    --trained_model_path ./output/tracking/a1_s1_460_200 \
    --model_path          ./model \
    --dataset_dir         ./data \
    --dataset_type        actorshq \
    --actor 1 --sequence 1 \
    --train_frame_start_num 460 25 \
    --verts_start_idx 460 \
    --uv_path          ./data/a1_s1/a1s1_uv.obj \
    --split_idx_path   ./data/a1_s1/split_idx.npz \
    --test_camera_index 6 126 \
    --phase3_ckpt      /path/to/wan_i2v.ckpt \
    --optim spsa \
    --iters 2000 \
    --lambda_sds 1.0 \
    --lambda_penetration 0.1 \
    --lambda_stretch 0.05 \
    --lambda_temporal_smooth 0.02
```

---

## Sanity check (10 iterations SPSA)

Run this to verify the pipeline end-to-end before submitting a full job:

```bash
python bridge_sds/optimize_phi.py \
    --save_name sanity \
    --trained_model_path ./output/tracking/a1_s1_460_200 \
    --dataset_dir ./data --dataset_type actorshq \
    --actor 1 --sequence 1 \
    --train_frame_start_num 460 25 \
    --uv_path ./data/a1_s1/a1s1_uv.obj \
    --split_idx_path ./data/a1_s1/split_idx.npz \
    --test_camera_index 6 \
    --no_sds --iters 10 --video_interval 5 --save_interval 5
```

What to verify:
- Loss values are finite (not NaN or Inf)
- φ values change between iterations (optimizer is active)
- `output/bridge_sds/renders/sanity/iter_00000.mp4` is a valid video
- `output/bridge_sds/phis/sanity/best_phi.json` contains reasonable values

---

## Resuming from checkpoint

```bash
python bridge_sds/optimize_phi.py \
    --save_name a1_s1_sds \
    ... (same flags as original run) ... \
    --resume ./output/bridge_sds/phis/a1_s1_sds/phi_iter_00500.npz
```

---

## Multi-view montage mode

Tile K camera views into a single wide frame before scoring.
Helps the SDS prior see 3D consistency across viewpoints.

```bash
    --test_camera_index 6 126 54 180 \
    --montage
```

---

## Key flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--phase3_ckpt` | None | Wan I2V checkpoint path |
| `--no_sds` | — | Disable SDS; physical regs only |
| `--optim` | `spsa` | `spsa` or `backprop` (falls back to spsa) |
| `--iters` | 2000 | Total optimisation iterations |
| `--clip_frame_num` | train_frame_num | Frames per clip per iteration |
| `--clip_jitter` | 5 | Random start-frame jitter range |
| `--montage` | False | Multi-view tiling |
| `--lambda_sds` | 1.0 | SDS loss weight |
| `--lambda_penetration` | 0.1 | Body penetration loss weight |
| `--lambda_stretch` | 0.05 | Cloth stretch loss weight |
| `--lambda_temporal_smooth` | 0.02 | Jitter penalty weight |
| `--video_interval` | 100 | Save debug mp4 every N iters |
| `--save_interval` | 50 | Save φ checkpoint every N iters |

---

## Adding Phase3 / Wan I2V (Clariden integration TODO)

Open `bridge_sds/scorer_phase3.py` and complete the `_load_model()` method:

```python
# In Phase3Scorer._load_model():
sys.path.insert(0, str(DREAMCLOTH_ROOT / "phase3"))

# Option A — Dreamcloth Phase3 lightweight model:
from video_diffusion_model import VideoDiffusionModel
self._model = VideoDiffusionModel(...)
ckpt = torch.load(self._ckpt_path, map_location="cpu")
self._model.load_state_dict(ckpt["model_state_dict"])
self._model.eval().to(self.device)
for p in self._model.parameters():
    p.requires_grad_(False)
self._model_ready = True

# Option B — Wan 2.1/2.2 I2V (already on Clariden):
from wan.modules.model import WanModel
from wan.configs import WAN_CONFIGS
cfg = WAN_CONFIGS["wan2.1-i2v-14B"]
self._model = WanModel.from_pretrained(self._ckpt_path, cfg)
self._model.eval().to(self.device)
self._model_ready = True
```

Then complete `_compute_sds()` with the real denoiser forward pass.

---

## VRAM requirements

| Component | Estimated VRAM |
|-----------|---------------|
| Gaussian renderer (512×512) | ~4 GB |
| MPM sim (grid 200³) | ~2 GB |
| Wan I2V 14B (FP16) | ~28 GB |
| **Total (single GPU)** | **~34 GB** |

With 200–300 GB total VRAM across the node, multi-GPU parallelism is possible
(run multiple SPSA probes in parallel on separate GPUs).

---

## Output files

```
output/bridge_sds/
├── renders/<run_name>/
│   └── iter_XXXXX.mp4          debug video every --video_interval iters
├── phis/<run_name>/
│   ├── phi_iter_XXXXX.npz      φ checkpoints (D, E, H + loss)
│   └── best_phi.json           best φ found during optimisation
└── logs/<run_name>/
    └── metrics.jsonl           per-iteration loss / grad_norm / phi values
```
