# MPMAvatar VDS — Changes & Additions

## Goal

Replace the MSE geometric loss in `train_material_params.py` with **Score Distillation
Sampling (SDS)** via a frozen **Wan 2.2 I2V** (14B) model as the loss signal.
Friction is added as a fourth differentiable parameter alongside D, E, H.
Target: 100 SPSA iterations on ActorsHQ Actor 1 / Sequence 1, frames 460–469.

---

## Files Added (new, did not exist in original MPMAvatar)

| File | Purpose |
|------|---------|
| `train_sds_physics.py` | Main entry point. Subclasses `Trainer` from `train_material_params.py`, overrides `train_one_step()` to use Wan SDS loss instead of MSE. Adds friction as SPSA parameter. Logs full CSV trajectory. |
| `bridge_sds/wan22_i2v_guidance.py` | Wan 2.2 I2V wrapper. Loads VAE + low-noise expert, encodes video to latent, computes flow-prediction SDS loss at a random timestep. |
| `bridge_sds/physical_regularizers.py` | Penalty losses: `L_pen` (penetration), `L_stretch` (stretch), `L_temporal_smooth` (temporal smoothness). |
| `bridge_sds/runner_mpmavatar.py` | Simulation + rendering runner used by the bridge. |
| `bridge_sds/scorer_phase3.py` | Wan I2V SDS scorer stub (raises if no checkpoint). |
| `bridge_sds/optimize_phi.py` | SPSA optimizer + main loop. |
| `bridge_sds/utils_video_io.py` | Video I/O helpers. |
| `bridge_sds/configs/sds_test.yaml` | SDS config (n_iterations, SPSA perturbation sizes, lr, param bounds). |
| `sds_lightning_run.ipynb` | Lightning AI notebook. Orchestrates setup, data prep, and runs `train_sds_physics.py`. |

### Synthetic data generators (needed because real preprocessed data was not in the released zip for all files)

| File | What it generates |
|------|------------------|
| `gen_cam_info.py` | `DATASET_DIR/a1_s1/cam_info.json` — 160 synthetic hemispherical cameras derived from `params_460.npz` intrinsics |
| `gen_uv_obj.py` | `data/a1_s1/a1s1_uv.obj` — placeholder UV OBJ with grid UV layout (one UV per face) |
| `gen_smplx_fitted.py` | `DATASET_DIR/a1_s1/smplx_fitted/<frame>/smplx_icp.obj` — body-only mesh per frame extracted from full tracking mesh using `split_idx.npz` |
| `gen_smplx_params.py` | `DATASET_DIR/a1_s1/smplx_fitted/<frame>/smplx_icp_param.pth` — zero-pose SMPLX params with translation from body centroid |

---

## Files Modified (original MPMAvatar files, patched)

### `scene/mesh_gaussian_model.py`
- `load_ply()`: slice `verts_offset` to `[:n]` to match `verts_orig` timestep count (pretrained offset covers 200 frames, scene loads only 10–14).
- `load_ply()`: `torch.load(..., weights_only=False)` to allow numpy-based `.pt` files under PyTorch 2.8.

### `scene/actorshq_dataset.py`
- `_load_image_dataset()`: wrapped `Image.open()` in `try/except` — returns zero arrays when image files are missing (synthetic camera setup has no real images).

### `train_material_params.py`
- All `torch.load(smplx_param_path)` calls: added `weights_only=False` (PyTorch 2.8 compatibility).

### `utils/smplx_deformer.py`
- `smplx_forward()` line 125: `scale = smplx_param['scale'].reshape(-1)` — normalises scale to 1-D before unsqueeze, fixing broadcast error when `torch.stack` produces shape `[1,1]` instead of `[1]` for single-frame test set.

### `warp_mpm/warp_utils.py`
- `from_torch_safe()`: removed `owner=False` from `wp.types.array()` — parameter dropped in warp ≥ 1.0.

### `warp_mpm/mpm_utils.py`
- All `wp.mat33(vec3, vec3, vec3)` calls replaced with `wp.matrix_from_rows(vec3, vec3, vec3)` — API change in warp ≥ 1.0 (passing vectors to `wp.matrix()` no longer supported).
  - Patterns fixed: `w = wp.mat33(...)`, `dw = wp.mat33(...)`, `new_C = wp.mat33(new_v, new_v, new_v)`, `I33 = wp.mat33(I33_1, I33_2, I33_3)`.

### `warp_mpm/mpm_solver.py`
- Same `wp.mat33(vec3, vec3, vec3)` → `wp.matrix_from_rows(...)` fix in `compute_mesh` kernel (4 occurrences).

### `bridge_sds/wan22_i2v_guidance.py`
- Stub `dashscope` (Alibaba cloud SDK, not needed) before importing `wan` so `wan/__init__.py` doesn't crash.
- `high_noise_model` not loaded to GPU (saves ~28 GB VRAM); timestep sampling capped to `[0, boundary*T)` so only the low-noise expert is ever needed.
- `torch.randn_like(x0, generator=g)` → `torch.randn(x0.shape, generator=g, device=..., dtype=...)` (`randn_like` does not accept `generator`).
- `video[:, :, 0] = img_m11[:, :, 0]` → `video[:, :, 0] = img_m11` (img_m11 is `[B,C,H,W]`, not a video; `[:,:,0]` was slicing H dimension).

---

## Data: Downloaded vs Generated

### Downloaded (real, from MPMAvatar paper / Google Drive)

| Item | Source | Path on Lightning |
|------|--------|-------------------|
| Tracking output (200 frames) | `MPMAvatar_processed_data.zip` (paper Google Drive) | `pretrained_models/output/tracking/a1_s1_460_200/params_*.npz` |
| AO maps | Same zip | `pretrained_models/output/tracking/a1_s1_460_200/aomap/mesh_cloth_*.png` |
| Trained Gaussian checkpoint | Paper Google Drive (gdown) | `pretrained_models/model/a1_s1/point_cloud/timestep_030000/` |
| `verts_offset.npy`, `cams.npz`, `shadow_net.pt` | Paper Google Drive | Inside `timestep_030000/` |
| `optimized_weights.npy` | `MPMAvatar_processed_data.zip` → `data/a1_s1/` | Copied to `DATASET_DIR/a1_s1/` |
| `split_idx.npz` | Same zip | `data/a1_s1/split_idx.npz` (also at `mpm_avatar_vds/data/a1_s1/`) |
| SMPLX body model (`SMPLX_NEUTRAL.npz` etc.) | Paper Google Drive | `DATASET_DIR/body_models/smplx/` |
| VPoser weights (`TR00_E096.pt`) | Paper Google Drive | `DATASET_DIR/body_models/` |
| **Wan 2.2 I2V A14B** | HuggingFace `Wan-AI/Wan2.2-I2V-A14B` (~126 GB) | `wan_checkpoints_22/` |
| Wan 2.2 source code | GitHub `Wan-Video/Wan2.2` | `Wan2.2/` |

### Generated / Synthetic (created because not in released data)

| Item | Generator script | Path |
|------|-----------------|------|
| `cam_info.json` (160 synthetic cameras) | `gen_cam_info.py` | `DATASET_DIR/a1_s1/cam_info.json` |
| `a1s1_uv.obj` (grid UV) | `gen_uv_obj.py` | `mpm_avatar_vds/data/a1_s1/a1s1_uv.obj` |
| `smplx_icp.obj` per frame | `gen_smplx_fitted.py` | `DATASET_DIR/a1_s1/smplx_fitted/<frame>/smplx_icp.obj` |
| `smplx_icp_param.pth` per frame | `gen_smplx_params.py` | `DATASET_DIR/a1_s1/smplx_fitted/<frame>/smplx_icp_param.pth` |

> **Note:** synthetic SMPLX params use zero pose + translation from body centroid.
> These are sufficient for physics simulation (test-frame LBS deformation) but
> do not capture the real body pose sequence — replacing them with real ICP-fitted
> params would improve cloth deformation quality.

---

## Dependencies Added to Environment

```
librosa       # Wan2.2 speech2video module import chain
easydict      # Wan2.2 configs
peft          # Wan2.2 animate module + diffusers
decord        # Wan2.2 speech2video
scikit-learn  # transformers candidate_generator (force-reinstall for numpy 1.x ABI)
```

numpy is pinned to `>=1.26.0,<2.0` throughout (required by wan + pandas; opencv warning is harmless).
