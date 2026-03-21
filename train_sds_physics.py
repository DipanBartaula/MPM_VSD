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

        # ── Friction: SPSA parameter, managed separately from Adam ───────────
        friction_cfg = sds_cfg.get("phi", {}).get("friction", {})
        self.friction_val = float(friction_cfg.get("init", args.mesh_friction_coeff))
        self.friction_min = float(friction_cfg.get("min",  0.01))
        self.friction_max = float(friction_cfg.get("max",  1.0))
        self.param_ranges["friction"] = [self.friction_min, self.friction_max]
        print(f"[SDS] Initial friction = {self.friction_val:.3f}  "
              f"range [{self.friction_min}, {self.friction_max}]")

        # ── Random initialisation of H and friction ──────────────────────────
        # D and E are randomised by the parent Trainer when --random_init_params
        # is passed.  Here we randomise H and friction so the full φ vector
        # starts from a uniformly random point in the search space.
        if sds_cfg.get("random_init", False):
            H_cfg = sds_cfg.get("phi", {}).get("H", {})
            H_rnd = float(np.random.uniform(
                float(H_cfg.get("min", 0.8)),
                float(H_cfg.get("max", 1.2)),
            ))
            self.torch_param["H"].data.fill_(H_rnd)
            f_rnd = float(np.random.uniform(self.friction_min, self.friction_max))
            self.friction_val = f_rnd
            print(
                f"[SDS] Random init φ: "
                f"D={self.torch_param['D'].item():.4f}  "
                f"E={self.torch_param['E'].item()*100:.1f}Pa  "
                f"H={H_rnd:.4f}  friction={f_rnd:.4f}"
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
        print("[SDS] Precomputing I2V conditioning image …")
        cam0, cam0_idx = self._cameras[0]
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
                    "grad_D", "grad_E", "grad_H", "grad_friction",
                    "camera_idx",
                ])
        else:
            self._csv_file   = None
            self._csv_writer = None

        print("\n[SDS] SDSPhysicsTrainer ready.\n")

    # -----------------------------------------------------------------------
    # Friction helpers
    # -----------------------------------------------------------------------

    def _set_friction(self, val: float) -> None:
        """Update friction coefficient in every MPM mesh collider."""
        for collider in self.mpm_solver.mesh_collider_params:
            collider.friction = float(val)

    # -----------------------------------------------------------------------
    # Full-quality rendering  (matches train_material_params.eval() pipeline)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _render_frame(
        self,
        verts: torch.Tensor,
        camera,
        camera_idx: int,
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
        # 1. Inject vertices into Gaussian model
        self.gaussians.set_mesh_by_verts(verts)

        # 2. Shadow pass: shadow_net expects [1, H, W] (matches eval())
        shadow_map = self.gaussians.shadow_net(self._ao_approx)["shadow_map"]
        # shadow_map: [1, 1, uv_size, uv_size]
        # uv_coord:   [1, 1, N_faces, 2]
        shadow = F.grid_sample(
            shadow_map,
            self.gaussians.uv_coord,
            mode="bilinear",
            align_corners=False,
        ).squeeze()[..., None][self.gaussians.binding]
        # shadow: [N_faces, 1]

        # 3. Shadow-modulated SH colours
        colors_precomp = shadow * convert_SH(
            self.gaussians.get_features,
            camera,
            self.gaussians,
            self.gaussians.get_xyz,
        )   # [N_faces, 3]

        # 4. 3DGS rasterize
        render_pkg = render(camera, self.gaussians, self.pipe, self.bg,
                            override_color=colors_precomp)

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

        out = torch.cat([rendering, render_pkg["mask"]], dim=0) # [4, H, W]
        return out.clamp(0.0, 1.0).detach()   # [4, H, W]

    # -----------------------------------------------------------------------

    def _simulate_and_render(
        self,
        D: torch.Tensor,
        E: torch.Tensor,
        H: torch.Tensor,
        friction: float,
        cameras: list,
        req_grad: bool = False,
        slice_start: int = 0,
        slice_len: int = 31,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MPM simulation with params {D, E, H, friction} and render
        each frame with the full-quality 3DGS pipeline for all given cameras.

        Returns
        -------
        cond_video: torch.Tensor [B, 4, H, W] in [0, 1] on CPU/CUDA
        video : torch.Tensor  [B, 4, T, H, W] in [0, 1] on CPU (if req_grad=False) or CUDA with grads
        """
        device = "cuda"
        context = torch.enable_grad() if req_grad else torch.no_grad()
        
        with context:
            # 1. Update friction in all MPM mesh colliders
            self._set_friction(friction)

            # 2. Reset MPM state with height scale H
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
                requires_grad=req_grad,
            )

            # 3. Set material parameters
            density = torch.ones_like(self.particle_init_position[..., 0]) * D
            youngs  = torch.ones_like(self.particle_init_position[..., 0]) * E * 100.0
            self.mpm_state.reset_density(density, None, device, update_mass=True)
            self.mpm_solver.set_E_nu_from_torch(
                self.mpm_model,
                youngs,
                self.poisson_ratio.detach().clone(),
                self.gamma.detach().clone(),
                self.kappa.detach().clone(),
                device,
            )
            self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)

            # 4. Simulate frame-by-frame and render each frame
            delta_time   = 1.0 / 25.0
            substep_size = 1.0 / 25.0 / self.args.substep
            num_substeps = int(delta_time / substep_size)
            n_frames     = min(self.scene.train_frame_num - 1, slice_start + slice_len)

            frames: List[List[torch.Tensor]] = [[] for _ in cameras]
            cond_frames: List[torch.Tensor] = []

            if slice_start == 0:
                for cam, cidx in cameras:
                    frame_rgba = self._render_frame(self.train_frame_verts[0].clone(), cam, cidx)
                    cond_frames.append(frame_rgba if req_grad else frame_rgba.cpu())

            for i in range(n_frames):
                mesh_x = self.wld2sim(self.train_frame_smplx[i].clone())
                mesh_v = self.train_frame_smplx_velo[i].clone() * self.scale
                joint_verts_v = (
                    self.train_frame_verts_velo[i, self.joint_v_idx].clone() * self.scale
                )
                joint_faces_v = (
                    joint_verts_v[self.new_cloth_faces[:self.num_joint_f]].mean(1).clone()
                )

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

                if i >= slice_start - 1:
                    particle_pos = wp.to_torch(self.mpm_state.particle_x).clone()
                    cloth_verts  = self.sim2wld(particle_pos[self.n_elements:])

                    # Build full-mesh verts: GT human body + simulated cloth
                    verts = self.train_frame_verts[i + 1].clone()
                    verts[self.reordered_cloth_v_idx] = cloth_verts

                    if i == slice_start - 1:
                        for cam, cidx in cameras:
                            frame_rgba = self._render_frame(verts, cam, cidx)
                            cond_frames.append(frame_rgba if req_grad else frame_rgba.cpu())
                    elif i >= slice_start:
                        for c_i, (cam, cidx) in enumerate(cameras):
                            frame_rgba = self._render_frame(verts, cam, cidx)
                            frames[c_i].append(frame_rgba if req_grad else frame_rgba.cpu())   # [4, H, W]

            # Stack: [T, 4, H, W] → permute → [4, T, H, W] for each cam, then concat
            videos = []
            for lst in frames:
                videos.append(torch.stack(lst, dim=0).permute(1, 0, 2, 3).unsqueeze(0))
            video = torch.cat(videos, dim=0)
            
            cond_video = torch.stack(cond_frames, dim=0)
            return cond_video, video   # cond_video: [B, 4, H, W], video: [B, 4, T, H, W]

    # -----------------------------------------------------------------------
    # SDS loss via Wan 5B
    # -----------------------------------------------------------------------

    def _compute_sds_loss(
        self,
        video: torch.Tensor,
        generator: torch.Generator,
    ) -> float:
        """
        Compute SDS (flow-prediction) loss for the given video.

        Parameters
        ----------
        video     : [1, 3, T, H, W] in [0, 1] on CPU
        generator : torch.Generator with a fixed seed for this step.
                    MUST be the same generator (same seed) across all
                    perturbations within one SPSA step so that timestep t
                    and noise ε are identical — otherwise the finite-
                    difference gradients measure timestep randomness, not
                    the effect of the parameter perturbation.

        Returns
        -------
        scalar float
        """
        target_res = int(self.sds_cfg.get("sds", {}).get("target_resolution", 128))
        use_mask = getattr(self.args, "use_mask", False)
        
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
            self.cond_image[:, :3, ...] if self.cond_image.ndim == 4 else self.cond_image[:3, ...].unsqueeze(0),
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
                # more biased
                t_sampled = t_min + (t_max - t_min) * (u ** 3)
            else:
                t_sampled = t_min + (t_max - t_min) * u
            timesteps = t_sampled.long().clamp(0, timesteps_max - 1)
        else:
            timesteps = torch.randint(
                t_min, t_max + 1, (B,),
                generator=generator, device=video_cuda.device,
            ).long()

        context = torch.enable_grad() if getattr(self.args, "use_adjoint", False) else torch.no_grad()
        with context:
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                loss = self.wan_guidance.compute_loss(
                    video_cuda, cond, timesteps=timesteps, generator=generator, mask_01=(mask_cuda if use_mask else None)
                )

        if getattr(self.args, "use_consistency_reg", False):
            # Frame consistency regularization: dampen high-frequency jitter across consecutive frames
            consistency_loss = F.mse_loss(video_cuda[:, :, 1:], video_cuda[:, :, :-1])
            weight = getattr(self.args, "consistency_weight", 0.1)
            loss = loss + weight * consistency_loss

        # NOTE: Return tensor without breaking computational graph if doing analytical diff
        return loss if getattr(self.args, "use_adjoint", False) else loss.item()

    # -----------------------------------------------------------------------
    # Core training step  (overrides parent)
    # -----------------------------------------------------------------------

    def train_one_step(self) -> None:
        """
        One SPSA step with SDS loss from Wan 5B.

        For each of 5 perturbations (base + 4 one-sided):
          - Simulate MPM with {D±δ, E±δ, H±δ, friction±δ}
          - Render T frames at full quality
          - Compute Wan 5B SDS loss
        Then finite-difference gradient + Adam/GD update.
        """
        step_t0 = time.time()
        
        use_adjoint = getattr(self.args, "use_adjoint", False)

        # ── Current parameter values ─────────────────────────────────────
        D_tens   = self.torch_param["D"].cuda()
        E_tens   = self.torch_param["E"].cuda()
        H_tens   = self.torch_param["H"].cuda()
        friction = self.friction_val
        
        D = D_tens.item()
        E = E_tens.item()
        H = H_tens.item()
        
        # Determine randomized sliding window over the timeline
        slice_len = 31
        max_start = max(0, (self.scene.train_frame_num - 1) - slice_len)
        step_slice_start = random.randint(0, max_start)
        
        if use_adjoint:
            D_tens.requires_grad_(True)
            E_tens.requires_grad_(True)
            H_tens.requires_grad_(True)
            
            self.optimizer.zero_grad()
            
            # Setup generator
            num_cams = getattr(self.args, "num_cams", 1)
            sampled_cams_info = random.sample(self._cameras, min(num_cams, len(self._cameras)))
            
            _step_seed = self.step * 10_000 + 42
            _sds_generator = torch.Generator(device="cuda")
            _sds_generator.manual_seed(_step_seed)
            
            # 1. Forward simulation WITH gradients via Warp BPTT / Adjoint
            t_sim = time.time()
            cond_img_tens, video = self._simulate_and_render(
                D_tens, E_tens, H_tens, friction, sampled_cams_info, req_grad=True, slice_start=step_slice_start, slice_len=slice_len
            )
            self.cond_image = cond_img_tens
            sim_dt = time.time() - t_sim
            
            t_wan = time.time()
            num_noise_samples = getattr(self.args, "num_noise_samples", 1)
            loss_tensor = 0.0
            for n in range(num_noise_samples):
                _sds_generator.manual_seed(_step_seed + n * 1337)
                loss_n = self._compute_sds_loss(video, _sds_generator)
                if n == 0:
                    loss_tensor = loss_n
                else:
                    loss_tensor = loss_tensor + loss_n
            loss_tensor = loss_tensor / num_noise_samples
            wan_dt = time.time() - t_wan
            
            # 3. Analytical Differential Backprop
            loss_tensor.backward()
            
            # Extrapolate analytical gradients into classical buffer slots for step logs
            grad_D = D_tens.grad.item() if D_tens.grad is not None else 0.0
            grad_E = E_tens.grad.item() if E_tens.grad is not None else 0.0
            grad_H = H_tens.grad.item() if H_tens.grad is not None else 0.0
            grad_friction = 0.0 # BPTT logic doesn't cover mesh friction trivially right now
            base_loss = loss_tensor.item()
            sds_losses = [base_loss]
            sim_times = [sim_dt]
            wan_times = [wan_dt]
            
            self.torch_param["D"].grad = torch.tensor(grad_D).float()
            self.torch_param["E"].grad = torch.tensor(grad_E).float()
            self.torch_param["H"].grad = torch.tensor(grad_H).float()
            
            D_tens.requires_grad_(False)
            E_tens.requires_grad_(False)
            H_tens.requires_grad_(False)
            
            self.optimizer.step()
            self.scheduler.step()
            
            base_video = video[0:1].detach().cpu()
            
            pdD = pdE = pdH = pdf = 0.0
            dD = dE = dH = df = 0.0
            camera_idx = sampled_cams_info[0][1] # Use first camera for logging
        else:
            # ── ORIGINAL SPSA APPROXIMATION ──────────────────────────────────

            # ── SPSA perturbation sizes (with optional cosine annealing) ─────
            spsa = self.sds_cfg.get("spsa", {})
            delta_pct = getattr(self.args, "delta_percent", 0.005)

            # Perturbation is delta_percent (e.g. 0.5%) of the base/initial value in *this* iteration
            dD = float(max(abs(D * delta_pct), 1e-6))
            dE = float(max(abs(E * delta_pct), 1e-6))
            dH = float(max(abs(H * delta_pct), 1e-6))
            
            # Friction can occasionally be very close to 0. Handle fallback safely.
            f_base = friction if abs(friction) > 1e-3 else 0.01
            df = float(max(abs(f_base * delta_pct), 1e-6))

            # Cosine annealing: shrink perturbations as training converges.
            # factor: 1.0 at step=0 → cosine_min_factor at step=iterations-1
            if spsa.get("cosine_decay", False):
                progress = self.step / max(1, self.iterations - 1)
                min_fac  = float(spsa.get("cosine_min_factor", 0.1))
                cosine_factor = min_fac + (1.0 - min_fac) * 0.5 * (1.0 + np.cos(np.pi * progress))
                dD *= cosine_factor
                dE *= cosine_factor
                dH *= cosine_factor
                df *= cosine_factor
            else:
                cosine_factor = 1.0

            # One-sided perturbations: base + one param nudged at a time
            perturbations = [
                (0.0, 0.0, 0.0, 0.0),   # [0] base
                (dD,  0.0, 0.0, 0.0),   # [1] +D
                (0.0, dE,  0.0, 0.0),   # [2] +E
                (0.0, 0.0, dH,  0.0),   # [3] +H
                (0.0, 0.0, 0.0, df),    # [4] +friction
            ]

            # ── Random cameras for this step ──────────────────────────────────
            num_cams = getattr(self.args, "num_cams", 1)
            sampled_cams_info = random.sample(self._cameras, min(num_cams, len(self._cameras)))
            camera_idx = sampled_cams_info[0][1] # Use first camera for logging

            # ── Fixed generator for this step ─────────────────────────────────
            # All 5 perturbations use the same seed so that Wan samples the
            # same timestep t and the same noise ε.  The only source of
            # loss difference is then the parameter perturbation itself,
            # making finite-difference gradients meaningful.
            _step_seed = self.step * 10_000 + 42
            _sds_generator = torch.Generator(device="cuda")

            param_ranges  = self.param_ranges
            sds_losses:   List[float]                 = []
            sim_times:    List[float]                 = []
            wan_times:    List[float]                 = []
            base_video:   Optional[torch.Tensor]      = None

            # ── Console header ────────────────────────────────────────────────
            print(f"\n{'═'*72}")
            print(f"  SDS PHYSICS STEP {self.step:4d} / {self.iterations}"
                  f"  │  camera_idx={camera_idx}")
            print(f"{'─'*72}")
            print(f"  Current:  D={D:.4f}  E={E*100:.1f}Pa  "
                  f"H={H:.4f}  friction={friction:.4f}")
            print(f"{'─'*72}")

            # ── SPSA perturbation loop ────────────────────────────────────────
            for idx, (pdD, pdE, pdH, pdf) in enumerate(perturbations):
                D_  = _clamp(D + pdD,        *param_ranges["D"])
                E_  = _clamp(E + pdE,        *param_ranges["E"])
                H_  = _clamp(H + pdH,        *param_ranges["H"])
                f_  = _clamp(friction + pdf, *param_ranges["friction"])

                # Simulate + render
                t_sim = time.time()
                cond_img_tens, video = self._simulate_and_render(
                    D_, E_, H_, f_, sampled_cams_info, req_grad=False, slice_start=step_slice_start, slice_len=slice_len
                )
                if idx == 0:
                    self.cond_image = cond_img_tens # Assign canonical condition representation 
                    
                sim_dt = time.time() - t_sim
                sim_times.append(sim_dt)

                if idx == 0:
                    base_video = video[0:1]   # keep for wandb gif (only one view)

                # SDS loss — reset generator to same seed every perturbation
                # so timestep t and noise ε are identical across all 5 runs
                t_wan = time.time()
                num_noise_samples = getattr(self.args, "num_noise_samples", 1)
                loss_val = 0.0
                for n in range(num_noise_samples):
                    _sds_generator.manual_seed(_step_seed + n * 1337)
                    loss_n = self._compute_sds_loss(video, _sds_generator)
                    loss_val += loss_n
                loss_val /= num_noise_samples
                wan_dt = time.time() - t_wan
                wan_times.append(wan_dt)

                sds_losses.append(loss_val)

                print(
                    f"  perm[{idx}] dD={pdD:+.3f} dE={pdE:+.3f} "
                    f"dH={pdH:+.4f} df={pdf:+.3f}"
                    f" → SDS={loss_val:.6f}  sim={sim_dt:.1f}s  wan={wan_dt:.1f}s"
                )

            # ── SPSA finite-difference gradients (one-sided) ──────────────────
            base_loss     = sds_losses[0]
            grad_D        = (sds_losses[1] - base_loss) / dD
            grad_E        = (sds_losses[2] - base_loss) / dE
            grad_H        = (sds_losses[3] - base_loss) / dH
            grad_friction = (sds_losses[4] - base_loss) / df

            # ── Update D, E, H via Adam ───────────────────────────────────────
            self.optimizer.zero_grad()
            self.torch_param["D"].grad = torch.tensor(grad_D).float()
            self.torch_param["E"].grad = torch.tensor(grad_E).float()
            self.torch_param["H"].grad = torch.tensor(grad_H).float()
            self.optimizer.step()
            self.scheduler.step()

        param_ranges  = self.param_ranges
        # Clamp D, E, H to physical bounds
        for key in ("D", "E", "H"):
            lo, hi = param_ranges[key]
            self.torch_param[key].data.clamp_(lo, hi)

        # ── Update friction via simple gradient descent ───────────────────
        lr_f = float(self.sds_cfg.get("lr", {}).get("friction", 0.01))
        self.friction_val = _clamp(
            friction - lr_f * grad_friction,
            self.friction_min,
            self.friction_max,
        )

        # ── Read updated values ───────────────────────────────────────────
        D_new = self.torch_param["D"].item()
        E_new = self.torch_param["E"].item()
        H_new = self.torch_param["H"].item()
        f_new = self.friction_val

        # ── Rich console log ──────────────────────────────────────────────
        step_dt = time.time() - step_t0
        print(f"{'─'*72}")
        print(f"  Updated:  D={D_new:.4f}  E={E_new*100:.1f}Pa  "
              f"H={H_new:.4f}  friction={f_new:.4f}")
        print(f"  Grads:    D={grad_D:+.5f}  E={grad_E:+.5f}  "
              f"H={grad_H:+.6f}  friction={grad_friction:+.5f}")
        print(f"  SDS:      base={base_loss:.6f}  "
              f"all={[f'{l:.5f}' for l in sds_losses]}")
        print(f"  Timing:   total={step_dt:.1f}s  "
              f"sim_avg={np.mean(sim_times):.1f}s  "
              f"wan_avg={np.mean(wan_times):.1f}s")
        print(f"{'═'*72}\n")

        # ── Update last_params (used by parent save()) ────────────────────
        self.last_params = {
            "D":        D_new,
            "E":        E_new * 100,
            "H":        H_new,
            "friction": f_new,
            "loss":     base_loss,
            "step":     self.step,
        }

        # ── Track best ────────────────────────────────────────────────────
        if base_loss < self.best_params.get("loss", float("inf")):
            self.best_params = {k: v for k, v in self.last_params.items()}
            print(f"  ★ New best  D={D_new:.4f}  E={E_new*100:.1f}Pa  "
                  f"H={H_new:.4f}  friction={f_new:.4f}  loss={base_loss:.6f}\n")

        # ── CSV trajectory log ────────────────────────────────────────────
        if self.accelerator.is_main_process and self._csv_writer is not None:
            self._csv_writer.writerow([
                self.step,
                D_new, E_new * 100, H_new, f_new,
                base_loss,
                sds_losses[1], sds_losses[2], sds_losses[3], sds_losses[4],
                grad_D, grad_E, grad_H, grad_friction,
                camera_idx,
            ])
            self._csv_file.flush()

        # ── WandB log ─────────────────────────────────────────────────────
        if self.use_wandb and self.accelerator.is_main_process:
            wd: Dict = {
                # Parameters
                "params/D":                D_new,
                "params/E_Pa":             E_new * 100,
                "params/H":                H_new,
                "params/friction":         f_new,
                # SDS losses — base + each perturbation
                "loss/sds_base":           base_loss,
                "loss/sds_dD":             sds_losses[1],
                "loss/sds_dE":             sds_losses[2],
                "loss/sds_dH":             sds_losses[3],
                "loss/sds_dfriction":      sds_losses[4],
                # SPSA gradients
                "gradients/D":             grad_D,
                "gradients/E":             grad_E,
                "gradients/H":             grad_H,
                "gradients/friction":      grad_friction,
                # SPSA perturbation sizes (after cosine decay)
                "spsa/dD":                 dD,
                "spsa/dE":                 dE,
                "spsa/dH":                 dH,
                "spsa/dfriction":          df,
                "spsa/cosine_factor":      cosine_factor,
                # Best params so far
                "best/D":                  float(self.best_params.get("D",        D_new)),
                "best/E_Pa":               float(self.best_params.get("E",        E_new * 100)),
                "best/H":                  float(self.best_params.get("H",        H_new)),
                "best/friction":           float(self.best_params.get("friction", f_new)),
                "best/sds_loss":           float(self.best_params.get("loss",     base_loss)),
                "best/step":               int(  self.best_params.get("step",     self.step)),
                # Timing
                "timing/step_total_s":     step_dt,
                "timing/sim_avg_s":        float(np.mean(sim_times)),
                "timing/wan_avg_s":        float(np.mean(wan_times)),
                # Learning rates
                "lr/D":                    self.optimizer.param_groups[0]["lr"],
                "lr/E":                    self.optimizer.param_groups[1]["lr"],
                "lr/H":                    self.optimizer.param_groups[2]["lr"],
                # Camera used this step
                "info/camera_idx":         camera_idx,
                "info/n_cameras":          len(self._cameras),
            }

            video_every = getattr(self.args, "video_every", 30)
            if (self.step > 0 and self.step % video_every == 0) and base_video is not None:
                # Log fully rendered MPM simulation as a video block (gif natively supports 4D tensor T,3,H,W or T,H,W,3)
                v = base_video[0, :3, ...] # [3, T, H, W]
                # rearrange to [T, 3, H, W]
                v = v.permute(1, 0, 2, 3) 
                v_np = (v.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                try:
                    wd["video/simulation_video"] = wandb.Video(v_np, fps=10, format="gif")
                except Exception as e:
                    print(f"[SDS] Could not log wandb Video/Gif: {e}")
                    # Fallback to images
                    T_v = v.shape[0]
                    for fi, frame_idx in enumerate([0, T_v // 2, T_v - 1]):
                        f_np = v_np[frame_idx].transpose(1, 2, 0)
                        wd[f"video/frame_{fi}"] = wandb.Image(f_np, caption=f"frame {frame_idx}")

                # Multi-camera snapshots: uniformly spaced across pool
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
    parser.add_argument("--timestep_bias",  type=str, default="uniform", choices=["uniform", "clean_1", "clean_2"], help="Bias timestep sampling logic.")
    parser.add_argument("--save_every", type=int, default=50, help="Iterations between saving checkpoints.")
    parser.add_argument("--video_every", type=int, default=50, help="Iterations between logging full videos.")
    parser.add_argument("--delta_percent", type=float, default=0.005, help="Delta (0.5 percent = 0.005) for finite difference w.r.t initial param values.")
    parser.add_argument("--use_mask", action="store_true", default=False, help="Compute SDS loss only on foreground via rendered mask.")
    parser.add_argument("--use_adjoint", action="store_true", default=False, help="Use Adjoint method (analytical backprop) instead of SPSA.")
    parser.add_argument("--use_consistency_reg", action="store_true", default=False, help="Add temporal consistency loss to physics trajectory.")
    parser.add_argument("--consistency_weight", type=float, default=0.1, help="Weight multiplier for frame consistency regularization.")
    parser.add_argument("--num_noise_samples", type=int, default=1, help="Number of random noise samples injected for computing expected SDS score.")

    raw = parser.parse_args(sys.argv[1:])
    env_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_rank != -1 and env_rank != raw.local_rank:
        raw.local_rank = env_rank

    return (
        lp.extract(raw), op.extract(raw), pp.extract(raw),
        raw.run_eval, raw.skip_sim, raw.skip_render, raw.skip_video,
        raw.wan_ckpt_dir, raw.wan_repo_root, raw.sds_cfg,
        raw.min_friction, raw.max_friction, raw.resume_ckpt,
        raw.num_cams, raw.timestep_bias, raw.save_every, raw.video_every, raw.delta_percent, raw.use_mask, raw.use_adjoint,
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
        num_cams, timestep_bias, save_every, video_every, delta_percent, use_mask, use_adjoint,
        use_consistency_reg, consistency_weight, num_noise_samples,
    ) = parse_args()
    
    args.num_cams = num_cams
    args.timestep_bias = timestep_bias
    args.save_every = save_every
    args.video_every = video_every
    args.delta_percent = delta_percent
    args.use_mask = use_mask
    args.use_adjoint = use_adjoint
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
