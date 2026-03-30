"""
train_sds_physics.py
====================
SDS-guided MPM physics parameter optimisation for MPMAvatar.

HYPOTHESIS TEST
---------------
Replace the MSE geometric loss in train_material_params.py with
Score Distillation Sampling (SDS) via frozen Wan 5B (I2V).

Pipeline each iteration
  1. Randomly pick a camera from test cameras
  2. Run MPM simulation with current params {D, E, H, friction}
     - Uses pretrained Gaussians & SMPLX motion from MPMAvatar
  3. Render each frame with full quality pipeline:
       shadow_net(AO) → shadow-modulated SH colors → 3DGS rasterize
       → camera exposure (cam_m / cam_c) → mask composite
  4. Stack frames into video tensor [1, 3, T, H, W]
  5. Resize to SDS target resolution → Wan 5B VAE → flow-prediction MSE
  6. SPSA finite-differences across {D, E, H, friction} → update params
  7. Log everything — params, all perturbation losses, gradients,
     per-step timing, rendered GIF, CSV trajectory

Render quality
--------------
  Uses the identical rendering pipeline as train_material_params.eval():
    - shadow_net(mean_AO) instead of per-frame baked AO
      (mean AO is a good static approximation; no Blender needed inline)
    - Full cam_m / cam_c per-camera exposure correction
    - Mask compositing with correct background colour
    - SH evaluated from current viewpoint (view-dependent appearance)

Usage
-----
  python train_sds_physics.py \\
    --trained_model_path ./output/tracking/a1_s1_460_200 \\
    --model_path         ./model/a1_s1 \\
    --dataset_dir        ./data \\
    --split_idx_path     ./data/a1_s1/split_idx.npz \\
    --actor 1 --sequence 1 \\
    --train_frame_start_num 460 10 \\
    --verts_start_idx 460 \\
    --wan_ckpt_dir  /path/to/wan_5b_model \\
    --sds_cfg bridge_sds/configs/sds_test.yaml \\
    --iterations 100 \\
    --use_wandb --wandb_project MPMAvatar_SDS
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
import math
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import warp as wp

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from train_material_params import Trainer, convert_SH
from bridge_sds.physical_regularizers import compute_all_regularizers
from warp_mpm.warp_utils import from_torch_safe

try:
    from bridge_sds.wan22_i2v_guidance import Wan22I2VConfig, Wan22I2VGuidance
except ImportError as _wan_err:
    raise ImportError(
        "Failed to import Wan22I2VGuidance. "
        "Ensure bridge_sds/ is present and diffusers is installed "
        "(pip install git+https://github.com/huggingface/diffusers).\n"
        f"Original error: {_wan_err}"
    ) from _wan_err


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(val: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, val)))


def _load_sds_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _uniform_clip_starts(total_frames: int, clip_len: int, num_clips: int) -> List[int]:
    """Choose `num_clips` start indices spread over a rollout."""
    if total_frames <= 0:
        return [0] * num_clips

    max_start = max(total_frames - clip_len, 0)
    if num_clips <= 1:
        return [max_start // 2]
    if max_start == 0:
        return [0] * num_clips

    return [int(round(i * max_start / (num_clips - 1))) for i in range(num_clips)]


def _slice_or_pad_frames(frames: List[torch.Tensor], start: int, count: int) -> List[torch.Tensor]:
    if not frames:
        raise ValueError("Cannot slice from an empty frame list.")
    start = max(0, start)
    sliced = frames[start:start + count]
    if not sliced:
        sliced = [frames[-1]]
    while len(sliced) < count:
        sliced.append(sliced[-1])
    return sliced


def _sample_taped_substeps(num_substeps: int, taped_substeps: int) -> List[int]:
    if num_substeps <= 0:
        return []
    taped_substeps = max(1, min(taped_substeps, num_substeps))
    if taped_substeps == 1:
        return [num_substeps - 1]
    if taped_substeps == num_substeps:
        return list(range(num_substeps))

    interior_count = taped_substeps - 1
    interior = random.sample(range(num_substeps - 1), interior_count)
    sampled = sorted(interior + [num_substeps - 1])
    return sampled


# ---------------------------------------------------------------------------
# SDSPhysicsTrainer
# ---------------------------------------------------------------------------

class SDSPhysicsTrainer(Trainer):
    """
    Inherits all Gaussian / SMPLX / MPM setup from Trainer, but overrides
    train_one_step() to use SDS (Wan 5B) instead of MSE vertex loss.

    Render quality improvements over the naive version
    --------------------------------------------------
    • Shadow pass:  shadow_net(mean_AO) → per-face shadow multiplier
    • Exposure:     per-camera cam_m (multiplicative) & cam_c (additive)
    • Compositing:  render_pkg["mask"] + scene background colour
    • Multi-camera: random camera sampled each iteration for diverse SDS signal

    Additional trainable parameter: friction (SPSA + simple GD).
    """

    def __init__(
        self,
        args,
        opt,
        pipe,
        run_eval: bool,
        sds_cfg: dict,
        wan_ckpt_dir: str,
        wan_repo_root: Optional[str],
        resume_ckpt: Optional[str] = None,
    ):
        # ── Parent setup (loads Gaussians, SMPLX, MPM, Adam optimizer) ──────
        print("\n[SDS] Loading base Trainer (Gaussians + SMPLX + MPM) …")
        super().__init__(args, opt, pipe, run_eval)

        # SDS loss is >> 1  →  reset sentinel so best-param tracking works
        self.best_params["loss"] = float("inf")

        self.sds_cfg = sds_cfg

        # ── Friction: fixed simulation parameter (not optimized) ─────────────
        friction_cfg = sds_cfg.get("phi", {}).get("friction", {})
        self.friction_val = float(friction_cfg.get("init", args.mesh_friction_coeff))
        self.friction_init = self.friction_val
        self.friction_min = float(friction_cfg.get("min",  0.01))
        self.friction_max = float(friction_cfg.get("max",  1.0))
        self.param_ranges["friction"] = [self.friction_min, self.friction_max]
        self.torch_param["friction"] = torch.tensor(
            float(self.friction_val),
            requires_grad=True,
            device=self.torch_param["D"].device,
        )
        friction_lr = float(sds_cfg.get("lr", {}).get("friction", opt.lr_H))
        self.optimizer.add_param_group(
            {"params": [self.torch_param["friction"]], "lr": friction_lr}
        )
        print(f"[SDS] Initial friction = {self.friction_val:.3f}  "
              f"range [{self.friction_min}, {self.friction_max}]")

        # ── Random initialisation of H only ──────────────────────────────────
        # D and E are randomised by the parent Trainer when --random_init_params
        # is passed. Here we randomise H only.
        if sds_cfg.get("random_init", False):
            H_cfg = sds_cfg.get("phi", {}).get("H", {})
            H_rnd = float(np.random.uniform(
                float(H_cfg.get("min", 0.8)),
                float(H_cfg.get("max", 1.2)),
            ))
            self.torch_param["H"].data.fill_(H_rnd)
            print(
                f"[SDS] Random init φ: "
                f"D={self.torch_param['D'].item():.4f}  "
                f"E={self.torch_param['E'].item()*100:.1f}Pa  "
                f"H={H_rnd:.4f}  friction={self.friction_val:.4f}"
            )

        # ── Freeze shadow_net + set to eval mode ─────────────────────────────
        # Default PyTorch state is .train().  BatchNorm in training mode uses
        # batch statistics, which is wrong for single-image [1,H,W] inference.
        # Weights are frozen (we never want to update them here).
        self.gaussians.shadow_net.eval()
        self.gaussians.shadow_net.requires_grad_(False)

        # ── Precompute mean AO map for shadow pass ────────────────────────────
        # ao_maps: [N_frames, 1, H, W]  →  mean over frames: [1, H, W]
        # Matches exactly what eval() passes to shadow_net per frame.
        # Static approximation: good enough for video dynamics signal.
        self._ao_approx = self.gaussians.ao_maps.mean(dim=0)   # [1, H, W]
        print(f"[SDS] AO approx shape: {self._ao_approx.shape}  "
              f"(mean over {self.gaussians.ao_maps.shape[0]} frames)")

        # ── Build (camera, camera_idx) list for random multi-view sampling ────
        # Start with test cameras (test_camera_index = list of actual cam ids)
        self._cameras: List[Tuple] = list(zip(
            self.scene.test_dataset.camera_list,
            self.scene.test_camera_index,
        ))
        # Expand pool with train cameras for diverse SDS signal.
        # Train dataset holds ALL cameras in order → position i == actual cam id i,
        # which matches the indexing of gaussians.cam_m / cam_c.
        train_cam_list = self.scene.train_dataset.camera_list
        n_train = len(train_cam_list)
        max_extra = int(sds_cfg.get("training", {}).get("max_extra_cameras", 16))
        if n_train <= max_extra:
            extra_cam_ids = list(range(n_train))
        else:
            extra_cam_ids = [
                round(i * (n_train - 1) / (max_extra - 1))
                for i in range(max_extra)
            ]
        existing_idxs = {cidx for _, cidx in self._cameras}
        extra_cameras = [
            (train_cam_list[i], i)
            for i in extra_cam_ids
            if i not in existing_idxs
        ]
        self._cameras = self._cameras + extra_cameras
        print(
            f"[SDS] Camera pool: {len(self._cameras)} total "
            f"({len(self._cameras) - len(extra_cameras)} test "
            f"+ {len(extra_cameras)} train)"
        )

        requested_cond_camera_idx = getattr(args, "condition_camera_idx", None)
        if requested_cond_camera_idx is None:
            requested_cond_camera_idx = self.scene.test_camera_index[0] if len(self.scene.test_camera_index) > 0 else 0
        self.condition_camera_idx = int(requested_cond_camera_idx)
        cond_matches = [(cam, cidx) for cam, cidx in self._cameras if int(cidx) == self.condition_camera_idx]
        if not cond_matches:
            raise ValueError(
                f"Condition camera index {self.condition_camera_idx} is not available in the camera pool."
            )
        self._condition_camera = cond_matches[0]
        print(f"[SDS] Condition camera idx: {self.condition_camera_idx}")

        # ── Load Wan 5B guidance (frozen) ────────────────────────────────────
        print(f"\n[SDS] Loading Wan 5B from {wan_ckpt_dir} …")
        ckpt_dir = Path(wan_ckpt_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Wan checkpoint dir not found: {ckpt_dir}")

        sds_prompt = str(sds_cfg.get("sds", {}).get("prompt", ""))
        wan_cfg = Wan22I2VConfig(
            wan_repo_root=Path(wan_repo_root) if wan_repo_root else None,
            ckpt_dir=ckpt_dir,
            device="cuda",
            dtype=torch.float16,
            prompt=sds_prompt,
            negative_prompt="",
            use_cfg=False,
        )
        print(f"[SDS] Wan prompt: '{sds_prompt}'")
        self.wan_guidance = Wan22I2VGuidance(wan_cfg)
        self.wan_guidance.eval()
        print("[SDS] Wan 5B loaded and frozen.")

        # ── Precompute I2V conditioning image (first GT frame, camera 0) ─────
        print("[SDS] Precomputing fixed front-facing GSplat conditioning image …")
        cam0, cam0_idx = self._condition_camera
        self.cond_image = self._render_frame(
            verts=self.train_frame_verts[0],
            camera=cam0,
            camera_idx=cam0_idx,
        )   # [3, H, W] in [0, 1]
        print(f"[SDS] Conditioning image shape: {self.cond_image.shape}")

        # ── Full checkpoint resume ────────────────────────────────────────────
        # Restores: params, friction, optimizer state, scheduler state, step.
        # Must happen after super().__init__() (creates optimizer/scheduler)
        # and after Wan is loaded (nothing else to initialise).
        if resume_ckpt and os.path.exists(resume_ckpt):
            print(f"[SDS] Resuming full checkpoint from {resume_ckpt} …")
            ckpt = torch.load(resume_ckpt, map_location="cpu")
            self.torch_param["D"].data.fill_(float(ckpt["D"]))
            self.torch_param["E"].data.fill_(float(ckpt["E"]))
            self.torch_param["H"].data.fill_(float(ckpt["H"]))
            self.friction_val = float(ckpt["friction"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.step        = int(ckpt["step"])
            self.best_params = ckpt["best_params"]
            print(
                f"[SDS] Resumed at step {self.step}  "
                f"D={self.torch_param['D'].item():.4f}  "
                f"E={self.torch_param['E'].item()*100:.1f}Pa  "
                f"H={self.torch_param['H'].item():.4f}  "
                f"friction={self.friction_val:.4f}  "
                f"best_loss={self.best_params.get('loss', float('inf')):.6f}"
            )
        elif resume_ckpt:
            print(f"[SDS] Warning: resume_ckpt not found: {resume_ckpt} — starting fresh")

        # ── Param-trajectory CSV (main process only) ─────────────────────────
        if self.accelerator.is_main_process:
            csv_path = os.path.join(self.output_path, "param_trajectory.csv")
            # Append mode when resuming so we don't lose previous steps
            csv_mode = "a" if (resume_ckpt and os.path.exists(resume_ckpt)) else "w"
            self._csv_file = open(csv_path, csv_mode, newline="")
            self._csv_writer = csv.writer(self._csv_file)
            if csv_mode == "w":
                self._csv_writer.writerow([
                    "step", "D", "E_Pa", "H", "friction",
                    "sds_loss_base",
                    "sds_loss_dD", "sds_loss_dE", "sds_loss_dH", "sds_loss_dfriction",
                    "grad_D", "grad_E", "grad_H", "grad_friction", "grad_norm",
                    "camera_idx",
                ])
        else:
            self._csv_file   = None
            self._csv_writer = None

        # Curriculum over fixed 4.8s segments (120 frames at 25 FPS).
        curriculum_cfg = sds_cfg.get("curriculum", {})
        self.curriculum_clip_frames = int(curriculum_cfg.get("clip_frames", sds_cfg.get("clip", {}).get("frame_num", 120)))
        self.curriculum_first_stage_iterations = int(curriculum_cfg.get("first_stage_iterations", 40))
        self.curriculum_stage_iterations = int(curriculum_cfg.get("stage_iterations", 40))
        self.curriculum_final_random_iterations = int(curriculum_cfg.get("final_random_iterations", 100))
        self.curriculum_final_random_batch_clips = int(curriculum_cfg.get("final_random_batch_clips", 8))
        self.curriculum_total_rollout_frames = max(int(self.scene.train_frame_num) - 1, 0)
        self.curriculum_num_segments = max(
            1,
            math.ceil(self.curriculum_total_rollout_frames / max(self.curriculum_clip_frames, 1)),
        )
        self.curriculum_sequential_iterations = (
            self.curriculum_first_stage_iterations
            + max(0, self.curriculum_num_segments - 1) * self.curriculum_stage_iterations
        )
        self.curriculum_total_iterations = (
            self.curriculum_sequential_iterations + self.curriculum_final_random_iterations
        )
        self.iterations = self.curriculum_total_iterations

        zero_C = torch.zeros_like(self.particle_init_dir)
        self.curriculum_stage_snapshots: List[Dict[str, torch.Tensor]] = [
            {
                "x": self.particle_init_position.detach().clone(),
                "v": self.particle_init_velo.detach().clone(),
                "d": self.particle_init_dir.detach().clone(),
                "C": zero_C.detach().clone(),
            }
        ]
        self.curriculum_stage_initial_positions: List[np.ndarray] = [
            self.particle_init_position.detach().cpu().numpy()
        ]
        self.curriculum_positions_path = os.path.join(
            self.output_path, "curriculum_stage_initial_positions.npy"
        )
        np.save(
            self.curriculum_positions_path,
            np.stack(self.curriculum_stage_initial_positions, axis=0),
        )
        print(
            "[SDS] Curriculum: "
            f"{self.curriculum_num_segments} fixed segments, "
            f"sequential iters={self.curriculum_sequential_iterations}, "
            f"final random iters={self.curriculum_final_random_iterations}, "
            f"total={self.iterations}"
        )

        print("\n[SDS] SDSPhysicsTrainer ready.\n")

    # -----------------------------------------------------------------------
    # Friction helpers
    # -----------------------------------------------------------------------

    def _set_friction(self, val) -> None:
        """Update friction coefficient in every MPM mesh collider."""
        for collider in self.mpm_solver.mesh_collider_params:
            if isinstance(val, torch.Tensor):
                friction_tensor = val.reshape(1).contiguous()
                collider.friction = from_torch_safe(
                    friction_tensor,
                    dtype=wp.float32,
                    requires_grad=bool(friction_tensor.requires_grad),
                )
            else:
                collider.friction = wp.from_numpy(
                    np.asarray([float(val)], dtype=np.float32),
                    dtype=wp.float32,
                    device="cuda:0",
                    requires_grad=False,
                )

    def _segment_bounds(self, segment_idx: int) -> Tuple[int, int]:
        frame_start = int(segment_idx) * self.curriculum_clip_frames
        frame_end = min(frame_start + self.curriculum_clip_frames, self.curriculum_total_rollout_frames)
        return frame_start, max(frame_end, frame_start)

    def _curriculum_stage_for_step(self, step: int) -> Optional[int]:
        if step < self.curriculum_first_stage_iterations:
            return 0
        remaining = step - self.curriculum_first_stage_iterations
        stage_offset = remaining // max(self.curriculum_stage_iterations, 1)
        stage_idx = 1 + stage_offset
        if stage_idx >= self.curriculum_num_segments:
            return None
        return stage_idx

    def _curriculum_phase(self, step: int) -> Tuple[str, List[int]]:
        stage_idx = self._curriculum_stage_for_step(step)
        if stage_idx is not None:
            return "sequential", [stage_idx]

        num_segments = max(self.curriculum_num_segments, 1)
        batch_clips = min(self.curriculum_final_random_batch_clips, num_segments)
        if batch_clips <= 0:
            batch_clips = 1
        if batch_clips >= num_segments:
            return "random_fixed_batch", list(range(num_segments))
        return "random_fixed_batch", random.sample(range(num_segments), batch_clips)

    def _stage_iteration_range(self, stage_idx: int) -> Tuple[int, int]:
        if stage_idx <= 0:
            return 0, self.curriculum_first_stage_iterations
        stage_start = self.curriculum_first_stage_iterations + (stage_idx - 1) * self.curriculum_stage_iterations
        return stage_start, stage_start + self.curriculum_stage_iterations

    def _capture_sim_snapshot(self) -> Dict[str, torch.Tensor]:
        return {
            "x": wp.to_torch(self.mpm_state.particle_x).detach().clone(),
            "v": wp.to_torch(self.mpm_state.particle_v).detach().clone(),
            "d": wp.to_torch(self.mpm_state.particle_d).detach().clone(),
            "C": wp.to_torch(self.mpm_state.particle_C).detach().clone(),
        }

    def _persist_curriculum_stage_positions(self) -> None:
        np.save(
            self.curriculum_positions_path,
            np.stack(self.curriculum_stage_initial_positions, axis=0),
        )

    def _render_segment_video_for_logging(
        self,
        phi: Dict[str, float],
        segment_idx: int,
        camera_info: Tuple,
    ) -> torch.Tensor:
        device = "cuda"
        self._ensure_curriculum_snapshots(int(segment_idx))
        start_snapshot = self.curriculum_stage_snapshots[int(segment_idx)]
        frame_start, frame_end = self._segment_bounds(int(segment_idx))
        segment_len = max(frame_end - frame_start, 0)
        render_stride_steps = int(self.sds_cfg.get("clip", {}).get("frame_stride_steps", 1))
        render_stride_steps = max(1, render_stride_steps)
        cam, cidx = camera_info

        particle_R_inv = self.compute_rest_dir_inv_from_vf(
            torch.stack([
                self.vertices_init_position[:, 0],
                self.vertices_init_position[:, 1] * float(phi["H"]),
                self.vertices_init_position[:, 2],
            ], dim=1),
            self.new_cloth_faces,
        )

        self.mpm_state.continue_from_torch(
            start_snapshot["x"].clone(),
            tensor_velocity=start_snapshot["v"].clone(),
            tensor_d=start_snapshot["d"].clone(),
            tensor_C=start_snapshot["C"].clone(),
            tensor_R_inv=particle_R_inv.clone(),
            device=device,
            requires_grad=False,
        )
        self.mpm_state.set_require_grad(False)
        self.mpm_model.set_require_grad(False)

        density = torch.ones_like(self.particle_init_position[..., 0]) * float(phi["D"])
        youngs = torch.ones_like(self.particle_init_position[..., 0]) * float(phi["E"]) * 100.0
        self.mpm_state.reset_density(
            density,
            None,
            device,
            requires_grad=False,
            update_mass=True,
        )
        self.mpm_solver.set_E_nu_from_torch(
            self.mpm_model,
            youngs,
            self.poisson_ratio.detach().clone(),
            self.gamma.detach().clone(),
            self.kappa.detach().clone(),
            device,
        )
        self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)
        self._set_friction(phi["friction"])
        self.mpm_solver.time = 0.0

        delta_time = 1.0 / 25.0
        substep_size = delta_time / self.args.substep
        num_substeps = int(delta_time / substep_size)

        frames: List[torch.Tensor] = []
        init_frame_idx = min(frame_start, self.scene.train_frame_num - 1)
        frames.append(
            self._render_frame(
                self.train_frame_verts[init_frame_idx].clone(),
                cam,
                cidx,
                requires_grad=False,
            ).cpu()
        )

        for i in range(frame_start, frame_start + segment_len):
            mesh_x = self.wld2sim(self.train_frame_smplx[i].clone())
            mesh_v = self.train_frame_smplx_velo[i].clone() * self.scale
            joint_verts_v = self.train_frame_verts_velo[i, self.joint_v_idx].clone() * self.scale
            joint_faces_v = joint_verts_v[self.new_cloth_faces[:self.num_joint_f]].mean(1).clone()

            for sub in range(num_substeps):
                mesh_x_curr = mesh_x + substep_size * sub * mesh_v
                self.mpm_solver.p2g2p(
                    self.mpm_model,
                    self.mpm_state,
                    substep_size,
                    mesh_x=mesh_x_curr,
                    mesh_v=mesh_v,
                    joint_traditional_v=None,
                    joint_verts_v=joint_verts_v,
                    joint_faces_v=joint_faces_v,
                    device=device,
                )

            if ((i - frame_start) + 1) % render_stride_steps == 0 or i == frame_start + segment_len - 1:
                particle_pos = wp.to_torch(self.mpm_state.particle_x).clone()
                cloth_verts = self.sim2wld(particle_pos[self.n_elements:])
                verts = self.train_frame_verts[i + 1].clone()
                verts[self.reordered_cloth_v_idx] = cloth_verts.to(verts.device)
                frames.append(
                    self._render_frame(verts, cam, cidx, requires_grad=False).cpu()
                )

        clip = torch.stack(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        self._restore_initial_sim_state(H=float(phi["H"]), friction=float(phi["friction"]), requires_grad=False)
        return clip

    def _rollout_segment_end_snapshot(
        self,
        start_snapshot: Dict[str, torch.Tensor],
        segment_idx: int,
        phi: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        device = "cuda"
        frame_start, frame_end = self._segment_bounds(segment_idx)
        segment_len = max(frame_end - frame_start, 0)
        particle_R_inv = self.compute_rest_dir_inv_from_vf(
            torch.stack([
                self.vertices_init_position[:, 0],
                self.vertices_init_position[:, 1] * float(phi["H"]),
                self.vertices_init_position[:, 2],
            ], dim=1),
            self.new_cloth_faces,
        )

        self.mpm_state.continue_from_torch(
            start_snapshot["x"].clone(),
            tensor_velocity=start_snapshot["v"].clone(),
            tensor_d=start_snapshot["d"].clone(),
            tensor_C=start_snapshot["C"].clone(),
            tensor_R_inv=particle_R_inv.clone(),
            device=device,
            requires_grad=False,
        )
        self.mpm_state.set_require_grad(False)
        self.mpm_model.set_require_grad(False)

        density = torch.ones_like(self.particle_init_position[..., 0]) * float(phi["D"])
        youngs = torch.ones_like(self.particle_init_position[..., 0]) * float(phi["E"]) * 100.0
        self.mpm_state.reset_density(
            density,
            None,
            device,
            requires_grad=False,
            update_mass=True,
        )
        self.mpm_solver.set_E_nu_from_torch(
            self.mpm_model,
            youngs,
            self.poisson_ratio.detach().clone(),
            self.gamma.detach().clone(),
            self.kappa.detach().clone(),
            device,
        )
        self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)
        self._set_friction(phi["friction"])
        self.mpm_solver.time = 0.0

        delta_time = 1.0 / 25.0
        substep_size = delta_time / self.args.substep
        num_substeps = int(delta_time / substep_size)

        for i in range(frame_start, frame_start + segment_len):
            mesh_x = self.wld2sim(self.train_frame_smplx[i].clone())
            mesh_v = self.train_frame_smplx_velo[i].clone() * self.scale
            joint_verts_v = self.train_frame_verts_velo[i, self.joint_v_idx].clone() * self.scale
            joint_faces_v = joint_verts_v[self.new_cloth_faces[:self.num_joint_f]].mean(1).clone()

            for sub in range(num_substeps):
                mesh_x_curr = mesh_x + substep_size * sub * mesh_v
                self.mpm_solver.p2g2p(
                    self.mpm_model,
                    self.mpm_state,
                    substep_size,
                    mesh_x=mesh_x_curr,
                    mesh_v=mesh_v,
                    joint_traditional_v=None,
                    joint_verts_v=joint_verts_v,
                    joint_faces_v=joint_faces_v,
                    device=device,
                )

        next_snapshot = self._capture_sim_snapshot()
        self._restore_initial_sim_state(H=float(phi["H"]), friction=float(phi["friction"]), requires_grad=False)
        return next_snapshot

    def _ensure_curriculum_snapshots(self, needed_stage_idx: int) -> None:
        while len(self.curriculum_stage_snapshots) <= needed_stage_idx:
            prev_stage_idx = len(self.curriculum_stage_snapshots) - 1
            phi = {
                "D": self.torch_param["D"].item(),
                "E": self.torch_param["E"].item(),
                "H": self.torch_param["H"].item(),
                "friction": self.friction_val,
            }
            next_snapshot = self._rollout_segment_end_snapshot(
                self.curriculum_stage_snapshots[-1],
                prev_stage_idx,
                phi,
            )
            self.curriculum_stage_snapshots.append(next_snapshot)
            self.curriculum_stage_initial_positions.append(
                next_snapshot["x"].detach().cpu().numpy()
            )
            self._persist_curriculum_stage_positions()

    def _restore_initial_sim_state(self, H: float, friction: float, requires_grad: bool = False) -> None:
        """Restore the simulator to the rollout start state and reset solver time."""
        device = "cuda"
        particle_R_inv = self.compute_rest_dir_inv_from_vf(
            torch.stack([
                self.vertices_init_position[:, 0],
                self.vertices_init_position[:, 1] * H,
                self.vertices_init_position[:, 2],
            ], dim=1),
            self.new_cloth_faces,
        )
        self.mpm_state.reset_state(
            self.n_vertices,
            self.particle_init_position.clone(),
            self.particle_init_dir.clone(),
            None,
            self.particle_init_velo.clone(),
            tensor_R_inv=particle_R_inv.clone(),
            device=device,
            requires_grad=requires_grad,
        )
        self.mpm_state.set_require_grad(requires_grad)
        self.mpm_model.set_require_grad(requires_grad)

        density = torch.ones_like(self.particle_init_position[..., 0]) * float(self.torch_param["D"].item())
        youngs = torch.ones_like(self.particle_init_position[..., 0]) * float(self.torch_param["E"].item()) * 100.0
        self.mpm_state.reset_density(
            density,
            None,
            device,
            requires_grad=requires_grad,
            update_mass=True,
        )
        self.mpm_solver.set_E_nu_from_torch(
            self.mpm_model,
            youngs,
            self.poisson_ratio.detach().clone(),
            self.gamma.detach().clone(),
            self.kappa.detach().clone(),
            device,
        )
        self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)
        self._set_friction(friction)
        self.mpm_solver.time = 0.0

    # -----------------------------------------------------------------------
    # Full-quality rendering  (matches train_material_params.eval() pipeline)
    # -----------------------------------------------------------------------

    def _render_frame(
        self,
        verts: torch.Tensor,
        camera,
        camera_idx: int,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """
        Render a single frame at full quality.

        Steps (identical to train_material_params.eval() lines 857-875)
        ---------------------------------------------------------------
        1. set_mesh_by_verts(verts)
        2. shadow_net(ao_approx) → per-face shadow multiplier
        3. shadow * convert_SH(viewpoint) → precomputed colours
        4. 3DGS rasterize
        5. cam_m / cam_c exposure correction
        6. mask compositing

        Parameters
        ----------
        verts       : full mesh vertices [N_verts, 3]
        camera      : Camera object (scene.cameras)
        camera_idx  : actual dataset camera id (for cam_m / cam_c lookup)

        Returns
        -------
        frame : [3, H, W] float tensor in [0, 1]
        """
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            # 1. Inject vertices into Gaussian model
            self.gaussians.set_mesh_by_verts(verts)

            # 2. Shadow pass: shadow_net expects [1, H, W] (matches eval())
            shadow_map = self.gaussians.shadow_net(self._ao_approx)["shadow_map"]
            shadow = F.grid_sample(
                shadow_map,
                self.gaussians.uv_coord,
                mode="bilinear",
                align_corners=False,
            ).squeeze()[..., None][self.gaussians.binding]

            # 3. Shadow-modulated SH colours
            colors_precomp = shadow * convert_SH(
                self.gaussians.get_features,
                camera,
                self.gaussians,
                self.gaussians.get_xyz,
            )

            # 4. 3DGS rasterize
            render_pkg = render(
                camera,
                self.gaussians,
                self.pipe,
                self.bg,
                override_color=colors_precomp,
            )

            # 5. Per-camera exposure correction
            rendering = (
                render_pkg["render"]
                * torch.exp(self.gaussians.cam_m[camera_idx])[:, None, None]
                + self.gaussians.cam_c[camera_idx][:, None, None]
            )

            # 6. Mask compositing
            rendering = rendering * render_pkg["mask"]
            if self.scene.white_bkgd:
                rendering = rendering + (1.0 - render_pkg["mask"])

            out = torch.cat([rendering, render_pkg["mask"]], dim=0).clamp(0.0, 1.0)
            return out if requires_grad else out.detach()

    # -----------------------------------------------------------------------

    def _simulate_and_render(
        self,
        D: float,
        E: float,
        H: float,
        friction: float,
        cameras: list,
        clip_starts: List[int],
        clip_len: int,
        requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        """
        Run one full rollout, render all frames for the chosen cameras, then
        slice it into a batch of fixed-length video clips.
        """
        device = "cuda"

        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            self._set_friction(friction)

            particle_R_inv = self.compute_rest_dir_inv_from_vf(
                torch.stack([
                    self.vertices_init_position[:, 0],
                    self.vertices_init_position[:, 1] * H,
                    self.vertices_init_position[:, 2],
                ], dim=1),
                self.new_cloth_faces,
            )
            self.mpm_state.reset_state(
                self.n_vertices,
                self.particle_init_position.clone(),
                self.particle_init_dir.clone(),
                None,
                self.particle_init_velo.clone(),
                tensor_R_inv=particle_R_inv.clone(),
                device=device,
                requires_grad=requires_grad,
            )
            self.mpm_state.set_require_grad(requires_grad)
            self.mpm_model.set_require_grad(requires_grad)

            density = torch.ones_like(self.particle_init_position[..., 0]) * float(D)
            youngs = torch.ones_like(self.particle_init_position[..., 0]) * float(E) * 100.0
            self.mpm_state.reset_density(
                density,
                None,
                device,
                requires_grad=requires_grad,
                update_mass=True,
            )
            self.mpm_solver.set_E_nu_from_torch(
                self.mpm_model,
                youngs,
                self.poisson_ratio.detach().clone(),
                self.gamma.detach().clone(),
                self.kappa.detach().clone(),
                device,
            )
            self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)

            delta_time = 1.0 / 25.0
            substep_size = delta_time / self.args.substep
            num_substeps = int(delta_time / substep_size)
            total_frames = self.scene.train_frame_num - 1
            render_stride_steps = int(self.sds_cfg.get("clip", {}).get("frame_stride_steps", 10))
            render_stride_steps = max(1, render_stride_steps)

            frames: List[List[torch.Tensor]] = [[] for _ in cameras]
            for cam_i, (cam, cidx) in enumerate(cameras):
                frame0 = self._render_frame(
                    self.train_frame_verts[0].clone(),
                    cam,
                    cidx,
                    requires_grad=requires_grad,
                )
                frames[cam_i].append(frame0 if requires_grad else frame0.cpu())

            cloth_verts_seq: List[torch.Tensor] = []
            body_verts_seq: List[torch.Tensor] = []
            cloth_vels_seq: List[torch.Tensor] = []

            for i in range(total_frames):
                mesh_x = self.wld2sim(self.train_frame_smplx[i].clone())
                mesh_v = self.train_frame_smplx_velo[i].clone() * self.scale
                joint_verts_v = self.train_frame_verts_velo[i, self.joint_v_idx].clone() * self.scale
                joint_faces_v = joint_verts_v[self.new_cloth_faces[:self.num_joint_f]].mean(1).clone()

                for sub in range(num_substeps):
                    mesh_x_curr = mesh_x + substep_size * sub * mesh_v
                    self.mpm_solver.p2g2p(
                        self.mpm_model,
                        self.mpm_state,
                        substep_size,
                        mesh_x=mesh_x_curr,
                        mesh_v=mesh_v,
                        joint_traditional_v=None,
                        joint_verts_v=joint_verts_v,
                        joint_faces_v=joint_faces_v,
                        device=device,
                    )

                particle_pos = wp.to_torch(self.mpm_state.particle_x)
                particle_vel = wp.to_torch(self.mpm_state.particle_v)
                if not requires_grad:
                    particle_pos = particle_pos.clone()
                    particle_vel = particle_vel.clone()
                cloth_verts = self.sim2wld(particle_pos[self.n_elements:])
                cloth_vels = particle_vel[self.n_elements:] / self.scale

                cloth_verts_seq.append(cloth_verts if requires_grad else cloth_verts.detach().cpu())
                cloth_vels_seq.append(cloth_vels if requires_grad else cloth_vels.detach().cpu())
                body_verts = self.train_frame_smplx[i + 1]
                body_verts_seq.append(body_verts if requires_grad else body_verts.detach().cpu())

                if (i + 1) % render_stride_steps == 0:
                    verts = self.train_frame_verts[i + 1].clone()
                    verts[self.reordered_cloth_v_idx] = cloth_verts.to(verts.device)

                    for c_i, (cam, cidx) in enumerate(cameras):
                        frame = self._render_frame(verts, cam, cidx, requires_grad=requires_grad)
                        frames[c_i].append(frame if requires_grad else frame.cpu())

            clip_batch: List[torch.Tensor] = []
            cond_batch: List[torch.Tensor] = []
            max_valid_start = max(len(frames[0]) - 1 - clip_len, 0)
            n_cams = max(len(cameras), 1)
            for clip_idx, start in enumerate(clip_starts):
                cam_idx = clip_idx % n_cams
                start = min(max(start, 0), max_valid_start)
                cond_batch.append(self.cond_image if requires_grad else self.cond_image.detach().cpu())
                clip_frames = _slice_or_pad_frames(frames[cam_idx], start + 1, clip_len)
                clip = torch.stack(clip_frames, dim=0).permute(1, 0, 2, 3)
                clip_batch.append(clip)

            cond_video = torch.stack(cond_batch, dim=0)
            video = torch.stack(clip_batch, dim=0)
            sim_data = {
                "cloth_verts_seq": cloth_verts_seq,
                "body_verts_seq": body_verts_seq,
                "cloth_vels_seq": cloth_vels_seq,
            }
            result = (cond_video, video, sim_data)

        self._restore_initial_sim_state(H=float(H), friction=float(friction), requires_grad=False)
        return result

    # -----------------------------------------------------------------------
    # SDS loss via Wan 5B
    # -----------------------------------------------------------------------

    def _compute_sds_loss(
        self,
        cond_video: torch.Tensor,
        video: torch.Tensor,
        generator: torch.Generator,
        requires_grad: bool = False,
    ) -> torch.Tensor | float:
        """
        Compute SDS (flow-prediction) loss for the given video.

        Parameters
        ----------
        cond_video: [B, 4, H, W] in [0, 1] on CPU
        video     : [B, 4, T, H, W] in [0, 1] on CPU
        generator : torch.Generator with a fixed seed for this step.
                    Must be reset to the same seed for paired SPSA probes.

        Returns
        -------
        scalar float
        """
        target_res = int(self.sds_cfg.get("sds", {}).get("target_resolution", 128))
        use_mask = getattr(self.args, "use_mask", False)
        use_attention_soft_mask = bool(
            getattr(self.args, "use_attention_soft_mask", False)
            or self.sds_cfg.get("sds", {}).get("use_attention_soft_mask", False)
        )
        
        video_cuda = video[:, :3, ...].cuda()   # [B, 3, T, H, W]
        mask_cuda = video[:, 3:4, ...].cuda()   # [B, 1, T, H, W]

        # Resize spatial dims to target resolution for Wan VAE efficiency
        B, C, T_frames, H, W = video_cuda.shape
        if H != target_res or W != target_res:
            # Fold T into batch, resize, unfold
            v = video_cuda.permute(0, 2, 1, 3, 4).reshape(B * T_frames, C, H, W)
            v = F.interpolate(v, size=(target_res, target_res),
                              mode="bilinear", align_corners=False)
            video_cuda = (v.reshape(B, T_frames, C, target_res, target_res)
                           .permute(0, 2, 1, 3, 4))
            
            # mask resize
            v_m = mask_cuda.permute(0, 2, 1, 3, 4).reshape(B * T_frames, 1, H, W)
            v_m = F.interpolate(v_m, size=(target_res, target_res),
                                mode="area")
            mask_cuda = (v_m.reshape(B, T_frames, 1, target_res, target_res)
                          .permute(0, 2, 1, 3, 4))
        
        # Conditioning image: resize to match video resolution
        cond = F.interpolate(
            cond_video[:, :3, ...].cuda(),
            size=(target_res, target_res),
            mode="bilinear",
            align_corners=False,
        )   # [B, 3, target_res, target_res]

        # Wan VAE encode + frozen flow prediction.
        # generator ensures same t and same noise ε across all perturbations
        # in one SPSA step — critical for meaningful finite differences.
        
        timesteps = None
        timesteps_max = int(getattr(self.wan_guidance.scheduler.config, "num_train_timesteps", 1000))
        t_min = int(self.sds_cfg.get("sds", {}).get("timestep_min", 0))
        t_max = int(self.sds_cfg.get("sds", {}).get("timestep_max", timesteps_max - 1))
        
        bias = getattr(self.args, "timestep_bias", "uniform")
        if bias != "uniform":
            u = torch.rand((B,), generator=generator, device=video_cuda.device)
            if bias == "clean_1":
                # slightly biased towards low noise (small t)
                t_sampled = t_min + (t_max - t_min) * (u ** 2)
            elif bias == "clean_2":
                # more biased towards low noise
                t_sampled = t_min + (t_max - t_min) * (u ** 3)
            elif bias == "transition_core":
                # Smooth center-focused warp over [0, 1]:
                # compresses the extremes toward the middle without the sharp
                # triangular peak from averaging uniforms.
                centered = 2.0 * u - 1.0
                soft_mid = 0.5 + 0.5 * torch.sign(centered) * centered.abs().pow(1.5)
                t_sampled = t_min + (t_max - t_min) * soft_mid
            else:
                t_sampled = t_min + (t_max - t_min) * u
            timesteps = t_sampled.long().clamp(0, timesteps_max - 1)
        else:
            timesteps = torch.randint(
                t_min, t_max + 1, (B,),
                generator=generator, device=video_cuda.device,
            ).long()

        score_mask = mask_cuda if use_mask else None
        if use_attention_soft_mask:
            with torch.no_grad():
                attention_mask_cuda = self.wan_guidance.build_condition_attention_mask(
                    cond,
                    num_frames=T_frames,
                    height=target_res,
                    width=target_res,
                )
            score_mask = attention_mask_cuda if score_mask is None else (score_mask * attention_mask_cuda)

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
            loss = self.wan_guidance.compute_loss(
                video_cuda,
                cond,
                timesteps=timesteps,
                generator=generator,
                mask_01=score_mask,
            )

        if getattr(self.args, "use_consistency_reg", False):
            # Frame consistency regularization: dampen high-frequency jitter across consecutive frames
            consistency_loss = F.mse_loss(video_cuda[:, :, 1:], video_cuda[:, :, :-1])
            weight = getattr(self.args, "consistency_weight", 0.1)
            loss = loss + weight * consistency_loss

        return loss if requires_grad else float(loss.item())

    def _compute_regularization_loss(
        self,
        sim_data: Dict[str, object],
        friction: float,
        requires_grad: bool = False,
    ) -> Tuple[torch.Tensor | float, Dict[str, torch.Tensor | float]]:
        loss_cfg = self.sds_cfg.get("loss", {})
        sim_proxy = type(
            "SimResultProxy",
            (),
            {
                "cloth_verts_seq": sim_data["cloth_verts_seq"],
                "body_verts_seq": sim_data["body_verts_seq"],
                "cloth_vels_seq": sim_data["cloth_vels_seq"],
            },
        )()
        regs = compute_all_regularizers(
            sim_proxy,
            cloth_faces=self.new_cloth_faces.detach().cpu(),
            rest_verts=self.vertices_init_position.detach().cpu(),
            margin=float(loss_cfg.get("penetration_margin", 0.005)),
            max_strain=float(loss_cfg.get("max_strain", 0.30)),
        )

        weighted = {
            "penetration": float(loss_cfg.get("lambda_penetration", 0.05)) * regs["penetration"],
            "stretch": float(loss_cfg.get("lambda_stretch", 0.02)) * regs["stretch"],
            "temporal_smooth": float(loss_cfg.get("lambda_temporal_smooth", 0.02)) * regs["temporal_smooth"],
            "friction_reg": torch.zeros_like(regs["penetration"]),
        }
        total = sum(weighted.values())
        if requires_grad:
            return total, weighted
        return float(total.item()), {k: float(v.item()) for k, v in weighted.items()}

    def _evaluate_loss(
        self,
        phi: Dict[str, float],
        cameras: List[Tuple],
        clip_starts: List[int],
        clip_len: int,
        step_seed: int,
        requires_grad: bool = False,
    ) -> Tuple[torch.Tensor | float, torch.Tensor, Dict[str, torch.Tensor | float], float, float]:
        t_sim = time.time()
        cond_video, video, sim_data = self._simulate_and_render(
            phi["D"], phi["E"], phi["H"], phi["friction"], cameras, clip_starts, clip_len, requires_grad=requires_grad
        )
        sim_dt = time.time() - t_sim

        t_wan = time.time()
        num_noise_samples = int(getattr(self.args, "num_noise_samples", self.sds_cfg.get("sds", {}).get("num_noise_samples", 4)))
        sds_loss = None
        sds_generator = torch.Generator(device="cuda")
        for n in range(num_noise_samples):
            sds_generator.manual_seed(step_seed + n * 1337)
            loss_n = self._compute_sds_loss(cond_video, video, sds_generator, requires_grad=requires_grad)
            sds_loss = loss_n if sds_loss is None else sds_loss + loss_n
        sds_loss = sds_loss / max(num_noise_samples, 1)
        wan_dt = time.time() - t_wan

        loss_cfg = self.sds_cfg.get("loss", {})
        reg_loss, reg_terms = self._compute_regularization_loss(sim_data, phi["friction"], requires_grad=requires_grad)
        total_loss = (
            float(loss_cfg.get("lambda_sds", 0.9)) * sds_loss
            + float(loss_cfg.get("lambda_regularization", 0.1)) * reg_loss
        )
        return total_loss, video, {"sds": sds_loss, "total": total_loss, **reg_terms}, sim_dt, wan_dt

    def _evaluate_autodiff_proxy_loss(
        self,
        phi: Dict[str, float],
        segment_indices: List[int],
    ) -> Tuple[float, Dict[str, float], float]:
        raise RuntimeError(
            "The Warp autodiff branch no longer uses GT cloth vertices. "
            "A prior-only end-to-end differentiable objective "
            "(Warp rollout -> GSplat render -> Wan loss -> gradients to D/E/H) "
            "has not been implemented yet, so there is currently no valid "
            "gradient source for the Warp-only trainer."
        )

    def _set_param_grad_from_scalar(self, key: str, grad_value: float) -> None:
        """Assign a Warp-derived scalar gradient onto a torch parameter safely."""
        param = self.torch_param[key]
        param.grad = param.detach().new_tensor(float(grad_value))

    # -----------------------------------------------------------------------
    # Core training step  (overrides parent)
    # -----------------------------------------------------------------------

    def train_one_step(self) -> None:
        """
        One Warp autodiff optimization step using the taped proxy loss.
        """
        step_t0 = time.time()

        optim_mode = getattr(self.args, "optim", "warp_autodiff")

        D = self.torch_param["D"].item()
        E = self.torch_param["E"].item()
        H = self.torch_param["H"].item()
        friction = self.torch_param["friction"].item()

        num_cams = getattr(self.args, "num_cams", 1)
        sampled_cams_info = random.sample(self._cameras, min(num_cams, len(self._cameras)))
        camera_idx = sampled_cams_info[0][1]

        phase_name, segment_indices = self._curriculum_phase(self.step)
        needed_stage_idx = self.curriculum_num_segments if phase_name == "random_fixed_batch" else max(segment_indices)
        self._ensure_curriculum_snapshots(needed_stage_idx)
        clip_len = self.curriculum_clip_frames
        num_batch_videos = len(segment_indices)

        base_phi = {"D": D, "E": E, "H": H, "friction": friction}
        print(f"\n{'='*72}")
        print(f"  {optim_mode.upper()} PHYSICS STEP {self.step:4d} / {self.iterations}  |  camera_idx={camera_idx}")
        print(f"{'-'*72}")
        print(f"  Current:  D={D:.4f}  E={E*100:.1f}Pa  H={H:.4f}  friction={friction:.4f}")
        print(f"  Phase:    {phase_name}  segments={segment_indices}")
        print(f"  Batch:    {num_batch_videos} fixed clips x up to {clip_len} frames")
        print(f"{'-'*72}")

        if optim_mode != "warp_autodiff":
            raise RuntimeError(
                f"Unsupported optimization mode '{optim_mode}'. "
                "This trainer is Warp-autodiff-only."
            )

        self.optimizer.zero_grad(set_to_none=True)
        for key in ("D", "E", "H", "friction"):
            self.torch_param[key].grad = None

        base_loss, proxy_grads, base_sim_dt = self._evaluate_autodiff_proxy_loss(base_phi, segment_indices)
        base_video = None
        base_terms = {
            "proxy": base_loss,
            "total": base_loss,
            "penetration": 0.0,
            "stretch": 0.0,
            "temporal_smooth": 0.0,
            "friction_reg": 0.0,
        }
        grad_D = proxy_grads["D"]
        grad_E = proxy_grads["E"]
        grad_H = proxy_grads["H"]
        grad_friction = float(proxy_grads.get("friction", 0.0))

        if max(abs(grad_D), abs(grad_E), abs(grad_H), abs(grad_friction)) == 0.0:
            raise RuntimeError(
                "Warp autodiff produced zero gradients for D/E/H/friction. "
                "Check the Warp/Torch bridge and taped simulation path."
            )

        self._set_param_grad_from_scalar("D", grad_D)
        self._set_param_grad_from_scalar("E", grad_E)
        self._set_param_grad_from_scalar("H", grad_H)
        self._set_param_grad_from_scalar("friction", grad_friction)
        grad_norm = math.sqrt(
            grad_D * grad_D + grad_E * grad_E + grad_H * grad_H + grad_friction * grad_friction
        )
        self.optimizer.step()
        self.scheduler.step()

        base_wan_dt = 0.0

        param_ranges = self.param_ranges
        for key in ("D", "E", "H", "friction"):
            lo, hi = param_ranges[key]
            self.torch_param[key].data.clamp_(lo, hi)

        self.friction_val = self.torch_param["friction"].item()

        D_new = self.torch_param["D"].item()
        E_new = self.torch_param["E"].item()
        H_new = self.torch_param["H"].item()
        f_new = self.friction_val

        step_dt = time.time() - step_t0
        print(f"{'-'*72}")
        print(f"  Updated:  D={D_new:.4f}  E={E_new*100:.1f}Pa  H={H_new:.4f}  friction={f_new:.4f}")
        print(f"  Grads:    D={grad_D:+.5f}  E={grad_E:+.5f}  H={grad_H:+.6f}  friction={grad_friction:+.5f}  norm={grad_norm:.6f}")
        print(f"  Proxy:    loss={base_loss:.6f}")
        print(
            f"  Terms:    proxy={base_terms['proxy']:.6f}  pen={base_terms['penetration']:.6f}  "
            f"stretch={base_terms['stretch']:.6f}  temp={base_terms['temporal_smooth']:.6f}  "
            f"freg={base_terms['friction_reg']:.6f}"
        )
        print(f"  Timing:   total={step_dt:.1f}s  sim={base_sim_dt:.1f}s  wan={base_wan_dt:.1f}s")
        print(f"{'='*72}\n")

        self.last_params = {
            "D": D_new,
            "E": E_new * 100,
            "H": H_new,
            "friction": f_new,
            "loss": base_loss,
            "step": self.step,
        }

        if base_loss < self.best_params.get("loss", float("inf")):
            self.best_params = {k: v for k, v in self.last_params.items()}
            print(f"  * New best  D={D_new:.4f}  E={E_new*100:.1f}Pa  H={H_new:.4f}  friction={f_new:.4f}  loss={base_loss:.6f}\n")

        if self.accelerator.is_main_process and self._csv_writer is not None:
            self._csv_writer.writerow([
                self.step,
                D_new, E_new * 100, H_new, f_new,
                base_loss,
                base_loss, base_loss, base_terms["proxy"], base_terms["temporal_smooth"],
                grad_D, grad_E, grad_H, grad_friction, grad_norm,
                camera_idx,
            ])
            self._csv_file.flush()

        if self.use_wandb and self.accelerator.is_main_process:
            wd: Dict = {
                "params/D": D_new,
                "params/E_Pa": E_new * 100,
                "params/H": H_new,
                "params/friction": f_new,
                "loss/proxy_base": base_loss,
                "loss/proxy": base_terms["proxy"],
                "loss/reg_penetration": base_terms["penetration"],
                "loss/reg_stretch": base_terms["stretch"],
                "loss/reg_temporal": base_terms["temporal_smooth"],
                "loss/reg_friction": base_terms["friction_reg"],
                "gradients/D": grad_D,
                "gradients/E": grad_E,
                "gradients/H": grad_H,
                "gradients/friction": grad_friction,
                "gradients/norm": grad_norm,
                "warp/taped_substeps_per_frame": int(getattr(self.args, "taped_substeps_per_frame", 4)),
                "curriculum/phase": 0 if phase_name == "sequential" else 1,
                "curriculum/batch_clip_count": len(segment_indices),
                "curriculum/segment_min": min(segment_indices),
                "curriculum/segment_max": max(segment_indices),
                "best/D": float(self.best_params.get("D", D_new)),
                "best/E_Pa": float(self.best_params.get("E", E_new * 100)),
                "best/H": float(self.best_params.get("H", H_new)),
                "best/friction": float(self.best_params.get("friction", f_new)),
                "best/proxy_loss": float(self.best_params.get("loss", base_loss)),
                "best/step": int(self.best_params.get("step", self.step)),
                "timing/step_total_s": step_dt,
                "timing/sim_s": float(base_sim_dt),
                "timing/wan_s": float(base_wan_dt),
                "lr/D": self.optimizer.param_groups[0]["lr"],
                "lr/E": self.optimizer.param_groups[1]["lr"],
                "lr/H": self.optimizer.param_groups[2]["lr"],
                "info/camera_idx": camera_idx,
                "info/n_cameras": len(self._cameras),
            }

            video_every = getattr(self.args, "video_every", 30)
            if self.step > 0 and self.step % video_every == 0:
                preview_segment_idx = int(segment_indices[0])
                preview_video = self._render_segment_video_for_logging(
                    base_phi,
                    preview_segment_idx,
                    sampled_cams_info[0],
                )
                v = preview_video[0, :3, ...]
                v = v.permute(1, 0, 2, 3)
                v_np = (v.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                try:
                    wd["video/simulation_video"] = wandb.Video(v_np, fps=10, format="gif")
                except Exception as e:
                    print(f"[SDS] Could not log wandb Video/Gif: {e}")
                    T_v = v.shape[0]
                    for fi, frame_idx in enumerate([0, T_v // 2, T_v - 1]):
                        f_np = v_np[frame_idx].transpose(1, 2, 0)
                        wd[f"video/frame_{fi}"] = wandb.Image(f_np, caption=f"frame {frame_idx}")

                n_preview = min(4, len(self._cameras))
                preview_cams = [
                    self._cameras[round(i * (len(self._cameras) - 1) / max(1, n_preview - 1))]
                    for i in range(n_preview)
                ]
                for pcam, pcam_idx in preview_cams:
                    frame = self._render_frame(
                        verts=self.train_frame_verts[0],
                        camera=pcam,
                        camera_idx=pcam_idx,
                    )
                    img_np = (frame[:3].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    wd[f"frames/cam_{pcam_idx:03d}"] = wandb.Image(img_np)

            wandb.log(wd, step=self.step)

    # -----------------------------------------------------------------------
    # Override train() to add CSV cleanup + final summary
    # -----------------------------------------------------------------------

    def train(self) -> None:
        print(f"\n[SDS] Starting hypothesis test: {self.iterations} iterations.\n")
        
        save_every = getattr(self.args, "save_every", 30)
        try:
            from tqdm import tqdm
            for index in tqdm(range(self.step, self.iterations), desc="Training progress"):
                self.train_one_step()
                if self.step > 0 and self.step % save_every == 0:
                    if self.accelerator.is_main_process:
                        self.save()
                self.accelerator.wait_for_everyone()
                self.step += 1
        finally:
            if self._csv_file is not None:
                self._csv_file.close()

            print("\n[SDS] Training complete.")
            if self.best_params.get("step", -1) >= 0:
                print(f"  Best params at step {self.best_params['step']}:")
                print(f"    D        = {self.best_params['D']:.4f}")
                print(f"    E        = {self.best_params['E']:.2f} Pa")
                print(f"    H        = {self.best_params['H']:.4f}")
                print(f"    friction = {self.best_params['friction']:.4f}")
                print(f"    SDS loss = {self.best_params['loss']:.6f}")

    # -----------------------------------------------------------------------
    # Override save() to also persist friction
    # -----------------------------------------------------------------------

    def save(self) -> None:
        super().save()
        # Persist best params as .npz (legacy format)
        np.savez(
            os.path.join(self.output_path, f"sds_best_param_{self.step:05d}.npz"),
            **{k: v for k, v in self.best_params.items()},
        )
        # Full resume checkpoint — params + optimizer + scheduler + step
        resume_data = {
            "step":        self.step,
            "D":           self.torch_param["D"].item(),
            "E":           self.torch_param["E"].item(),
            "H":           self.torch_param["H"].item(),
            "friction":    self.friction_val,
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "best_params": self.best_params,
        }
        torch.save(resume_data, os.path.join(self.output_path, f"resume_ckpt_{self.step:05d}.pt"))
        torch.save(resume_data, os.path.join(self.output_path, "resume_latest.pt"))
        print(f"[SDS] Resume checkpoint saved (step {self.step})")
        # Save rendered PNG frames from a few cameras for visual inspection
        ckpt_dir = os.path.join(self.output_path, f"checkpoint_{self.step:05d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        n_save = min(4, len(self._cameras))
        save_cams = [
            self._cameras[round(i * (len(self._cameras) - 1) / max(1, n_save - 1))]
            for i in range(n_save)
        ]
        try:
            from PIL import Image as _PILImg
            for cam, cidx in save_cams:
                frame = self._render_frame(
                    verts=self.train_frame_verts[0],
                    camera=cam,
                    camera_idx=cidx,
                )
                img_np = (frame.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                _PILImg.fromarray(img_np).save(os.path.join(ckpt_dir, f"cam_{cidx:03d}.png"))
            print(f"[SDS] Checkpoint frames saved → {ckpt_dir}")
        except Exception as _e:
            print(f"[SDS] Frame save skipped ({_e})")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SDS-guided MPM physics parameter optimisation (hypothesis test)"
    )
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument(
        "--wan_ckpt_dir", type=str, required=True,
        help="Local path to snapshot_download of Wan-AI/Wan2.2-TI2V-5B-Diffusers "
             "(diffusers layout: transformer/, vae/, text_encoder/, tokenizer/, model_index.json).",
    )
    parser.add_argument(
        "--wan_repo_root", type=str, default=None,
        help="[UNUSED] Legacy arg kept for backwards compat. No custom repo needed with diffusers.",
    )
    parser.add_argument(
        "--sds_cfg", type=str,
        default="bridge_sds/configs/sds_test.yaml",
        help="SDS YAML config path.",
    )
    parser.add_argument("--min_friction",  type=float, default=None)
    parser.add_argument("--max_friction",  type=float, default=None)
    parser.add_argument(
        "--resume_ckpt", type=str, default=None,
        help="Path to resume_latest.pt or resume_ckpt_NNNNN.pt — restores "
             "params, optimizer, scheduler, and step counter exactly.",
    )
    parser.add_argument("--run_eval",    action="store_true", default=False)
    parser.add_argument("--skip_sim",    action="store_true", default=False)
    parser.add_argument("--skip_render", action="store_true", default=False)
    parser.add_argument("--skip_video",  action="store_true", default=False)
    parser.add_argument("--local_rank",  type=int, default=-1)
    parser.add_argument("--num_cams",  type=int, default=1, help="Number of cameras/views to render per SPSA step.")
    parser.add_argument("--timestep_bias",  type=str, default="uniform", choices=["uniform", "clean_1", "clean_2", "transition_core"], help="Bias timestep sampling logic.")
    parser.add_argument("--save_every", type=int, default=50, help="Iterations between saving checkpoints.")
    parser.add_argument("--video_every", type=int, default=100, help="Iterations between logging full videos.")
    parser.add_argument("--use_mask", action="store_true", default=False, help="Compute SDS loss only on foreground via rendered mask.")
    parser.add_argument("--use_attention_soft_mask", action="store_true", default=False, help="Compute a no-grad soft human mask from averaged conditioning-image attention maps and apply it in the SDS score.")
    parser.add_argument("--condition_camera_idx", type=int, default=None, help="Actual dataset camera index to use for the fixed front-facing GSplat conditioning render.")
    parser.add_argument("--taped_substeps_per_frame", type=int, default=4, help="Number of random MPM substeps per frame to keep in Warp tape for autodiff.")
    parser.add_argument(
        "--optim",
        type=str,
        default="warp_autodiff",
        choices=["warp_autodiff"],
        help="Optimization method for physics parameters. Only Warp autodiff is supported.",
    )
    parser.add_argument("--use_consistency_reg", action="store_true", default=False, help="Add temporal consistency loss to physics trajectory.")
    parser.add_argument("--consistency_weight", type=float, default=0.1, help="Weight multiplier for frame consistency regularization.")
    parser.add_argument("--num_noise_samples", type=int, default=4, help="Number of random noise samples injected for computing expected SDS score.")

    raw = parser.parse_args(sys.argv[1:])
    env_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_rank != -1 and env_rank != raw.local_rank:
        raw.local_rank = env_rank

    return (
        lp.extract(raw), op.extract(raw), pp.extract(raw),
        raw.run_eval, raw.skip_sim, raw.skip_render, raw.skip_video,
        raw.wan_ckpt_dir, raw.wan_repo_root, raw.sds_cfg,
        raw.min_friction, raw.max_friction, raw.resume_ckpt,
        raw.num_cams, raw.timestep_bias, raw.save_every, raw.video_every, raw.use_mask, raw.use_attention_soft_mask, raw.condition_camera_idx, raw.taped_substeps_per_frame, raw.optim,
        raw.use_consistency_reg, raw.consistency_weight, raw.num_noise_samples,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    (
        args, opt, pipe,
        run_eval, skip_sim, skip_render, skip_video,
        wan_ckpt_dir, wan_repo_root, sds_cfg_path,
        min_friction, max_friction, resume_ckpt,
        num_cams, timestep_bias, save_every, video_every, use_mask, use_attention_soft_mask, condition_camera_idx, taped_substeps_per_frame, optim_mode,
        use_consistency_reg, consistency_weight, num_noise_samples,
    ) = parse_args()
    
    args.num_cams = num_cams
    args.timestep_bias = timestep_bias
    args.save_every = save_every
    args.video_every = video_every
    args.use_mask = use_mask
    args.use_attention_soft_mask = use_attention_soft_mask
    args.condition_camera_idx = condition_camera_idx
    args.taped_substeps_per_frame = taped_substeps_per_frame
    args.optim = optim_mode
    args.use_adjoint = True
    args.use_consistency_reg = use_consistency_reg
    args.consistency_weight = consistency_weight
    args.num_noise_samples = num_noise_samples

    sds_cfg = _load_sds_cfg(sds_cfg_path)

    # CLI overrides for friction bounds
    if min_friction is not None:
        sds_cfg.setdefault("phi", {}).setdefault("friction", {})["min"] = min_friction
    if max_friction is not None:
        sds_cfg.setdefault("phi", {}).setdefault("friction", {})["max"] = max_friction

    # Override iterations from YAML only when CLI is at its default
    yaml_iters = sds_cfg.get("training", {}).get("n_iterations", None)
    if yaml_iters is not None and opt.iterations == 30_000:
        opt.iterations = int(yaml_iters)

    # Override MPM sim settings from YAML
    mpm_cfg = sds_cfg.get("mpm", {})
    if "substep"   in mpm_cfg: args.substep   = int(mpm_cfg["substep"])
    if "grid_size" in mpm_cfg: args.grid_size = int(mpm_cfg["grid_size"])

    # Override initial param values from YAML
    phi_cfg = sds_cfg.get("phi", {})
    if "D" in phi_cfg and "init" in phi_cfg["D"]:
        args.init_D = float(phi_cfg["D"]["init"])
    if "E" in phi_cfg and "init" in phi_cfg["E"]:
        args.init_E = float(phi_cfg["E"]["init"])

    # Override gamma / kappa / friction_angle from YAML
    mpm_fixed = sds_cfg.get("mpm_fixed", {})
    if "gamma"          in mpm_fixed: args.init_gamma     = float(mpm_fixed["gamma"])
    if "kappa"          in mpm_fixed: args.init_kappa     = float(mpm_fixed["kappa"])
    if "friction_angle" in mpm_fixed: args.friction_angle = float(mpm_fixed["friction_angle"])

    print("\n" + "═"*72)
    print("  MPMAvatar — SDS Physics Hypothesis Test")
    print("═"*72)
    print(f"  Wan checkpoint : {wan_ckpt_dir}")
    print(f"  SDS config     : {sds_cfg_path}")
    print(f"  Iterations     : {opt.iterations}")
    print(f"  Optimizer      : {args.optim}")
    print(f"  Substep        : {args.substep}")
    print(f"  Grid size      : {args.grid_size}")
    print(f"  Init D / E / H : {args.init_D} / {args.init_E} Pa / 1.0")
    print("═"*72 + "\n")

    trainer = SDSPhysicsTrainer(
        args, opt, pipe, run_eval,
        sds_cfg=sds_cfg,
        wan_ckpt_dir=wan_ckpt_dir,
        wan_repo_root=wan_repo_root,
        resume_ckpt=resume_ckpt,
    )

    if run_eval:
        trainer.eval(skip_sim, skip_render, skip_video)
    else:
        trainer.train()

