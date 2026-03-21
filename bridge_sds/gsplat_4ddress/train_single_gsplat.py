"""
bridge_sds/gsplat_4ddress/train_single_gsplat.py
=================================================
Train a 3D Gaussian Splatting model from multi-view images extracted by
extract_frame.py.

Initialises Gaussians from the SMPLX mesh (if available) or a random
point cloud sampled on a sphere.  Uses MPMAvatar's GaussianModel +
gaussian_renderer pipeline — no external gsplat library required.

Usage
-----
    python bridge_sds/gsplat_4ddress/train_single_gsplat.py \\
        --data_dir ./output/gsplat_4ddress/extracted/s00170_t1_f00021 \\
        --output_dir ./output/gsplat_4ddress/models/s00170_t1_f00021 \\
        --iterations 7000

    # From targets.yaml (called by batch_gsplat.py):
    python train_single_gsplat.py \\
        --data_dir <extracted_frame_dir> \\
        --output_dir <model_output_dir> \\
        --iterations 7000 --sh_degree 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── MPMAvatar path ────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent
BRIDGE_ROOT    = SCRIPT_DIR.parent
MPMAVATAR_ROOT = BRIDGE_ROOT.parent
if str(MPMAVATAR_ROOT) not in sys.path:
    sys.path.insert(0, str(MPMAVATAR_ROOT))

from scene.gaussian_model import GaussianModel
from scene.cameras        import Camera
from gaussian_renderer    import render as gs_render
from utils.graphics_utils import BasicPointCloud
from utils.loss_utils     import l1_loss, ssim
from utils.general_utils  import get_expon_lr_func
from arguments            import PipelineParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gsplat_train")


# =============================================================================
# Camera loading
# =============================================================================

def load_cameras(data_dir: Path, downscale: float = 1.0) -> List[Camera]:
    """Load Camera objects from cameras.pkl in the extracted frame directory."""
    cam_pkl = data_dir / "cameras.pkl"
    if not cam_pkl.exists():
        raise FileNotFoundError(f"cameras.pkl not found in {data_dir}")

    with open(cam_pkl, "rb") as f:
        cam_data = pickle.load(f)

    cameras = []
    for cam_id, cam_info in sorted(cam_data.items()):
        # Load image to get resolution
        img_path = data_dir / "images" / f"{cam_id}.png"
        if not img_path.exists():
            logger.warning(f"Image missing for camera {cam_id}: {img_path}")
            continue

        img = Image.open(img_path)
        w_raw, h_raw = img.size
        w = round(w_raw / downscale)
        h = round(h_raw / downscale)

        k = cam_info["intrinsics"].copy().astype(np.float32)  # 3×3
        # Scale intrinsics with downsampling
        sx = w / w_raw
        sy = h / h_raw
        k[0, 0] *= sx; k[0, 2] *= sx   # fx, cx
        k[1, 1] *= sy; k[1, 2] *= sy   # fy, cy

        w2c = cam_info["extrinsics"].astype(np.float32)        # 3×4
        w2c = np.vstack([w2c, [0, 0, 0, 1]])                   # → 4×4

        cam = Camera(
            camera_id=str(cam_id),
            w=w, h=h, k=k, w2c=w2c,
            near=0.1, far=20.0,
            data_device="cuda",
        )
        cameras.append(cam)

    logger.info(f"Loaded {len(cameras)} cameras from {cam_pkl.name}")
    return cameras


# =============================================================================
# Image loading
# =============================================================================

def load_gt_image(
    data_dir:   Path,
    cam_id:     str,
    downscale:  float = 1.0,
    use_mask:   bool  = True,
    bg_white:   bool  = True,
    device:     str   = "cuda",
) -> torch.Tensor:
    """
    Load one ground-truth image as a [3, H, W] float32 tensor in [0, 1].
    If a mask is available, alpha-composite the image over the background.
    """
    img_path = data_dir / "images" / f"{cam_id}.png"
    msk_path = data_dir / "masks"  / f"{cam_id}.png"

    img = Image.open(img_path).convert("RGB")
    if downscale != 1.0:
        w = round(img.width  / downscale)
        h = round(img.height / downscale)
        img = img.resize((w, h), Image.LANCZOS)
    img_t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)  # [H,W,3]

    # Apply mask (composite over background colour)
    if use_mask and msk_path.exists():
        msk = Image.open(msk_path).convert("L")
        if downscale != 1.0:
            msk = msk.resize((img.width, img.height), Image.NEAREST)
        msk_t = torch.from_numpy(
            np.array(msk).astype(np.float32) / 255.0
        ).unsqueeze(-1)   # [H, W, 1]
        bg = torch.ones_like(img_t) if bg_white else torch.zeros_like(img_t)
        img_t = img_t * msk_t + bg * (1.0 - msk_t)

    return img_t.permute(2, 0, 1).to(device)   # [3, H, W]


# =============================================================================
# Point cloud initialisation
# =============================================================================

def init_from_smplx(smplx_ply: Path, n_points: int = 10_000) -> BasicPointCloud:
    """
    Sample `n_points` from the SMPLX mesh surface and return as BasicPointCloud.
    Uses random vertex sampling (fast, no triangle-area weighting needed here).
    """
    from plyfile import PlyData

    ply = PlyData.read(str(smplx_ply))
    verts = np.stack([
        ply.elements[0]["x"],
        ply.elements[0]["y"],
        ply.elements[0]["z"],
    ], axis=-1).astype(np.float32)

    # Random subsample
    idx = np.random.choice(len(verts), size=min(n_points, len(verts)), replace=False)
    pts = verts[idx]
    # Assign neutral grey colour (will be learned)
    colors = np.ones((len(pts), 3), dtype=np.float32) * 0.5

    return BasicPointCloud(points=pts, colors=colors, normals=np.zeros_like(pts))


def init_sphere(n_points: int = 10_000, radius: float = 1.0) -> BasicPointCloud:
    """Fallback: random points on a sphere."""
    pts = np.random.randn(n_points, 3).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=-1, keepdims=True)
    pts *= radius
    colors = np.random.rand(n_points, 3).astype(np.float32)
    return BasicPointCloud(points=pts, colors=colors, normals=np.zeros_like(pts))


# =============================================================================
# Training args namespace
# =============================================================================

class TrainArgs:
    """Simple namespace matching what GaussianModel.training_setup() expects."""
    def __init__(self, cfg: dict):
        self.percent_dense          = cfg.get("percent_dense",          0.01)
        self.position_lr_init       = cfg.get("position_lr_init",       0.00016)
        self.position_lr_final      = cfg.get("position_lr_final",      0.0000016)
        self.position_lr_delay_mult = cfg.get("position_lr_delay_mult", 0.01)
        self.position_lr_max_steps  = cfg.get("position_lr_max_steps",  7000)
        self.feature_lr             = cfg.get("feature_lr",             0.0025)
        self.opacity_lr             = cfg.get("opacity_lr",             0.05)
        self.scaling_lr             = cfg.get("scaling_lr",             0.005)
        self.rotation_lr            = cfg.get("rotation_lr",            0.001)


class PipeArgs:
    """Minimal pipeline params for gaussian_renderer."""
    compute_cov3D_python = False
    convert_SHs_python   = False
    debug                = False


# =============================================================================
# Main training loop
# =============================================================================

def train(
    data_dir:    Path,
    output_dir:  Path,
    iterations:  int   = 7000,
    sh_degree:   int   = 3,
    downscale:   float = 1.0,
    bg_white:    bool  = True,
    cfg:         dict  = None,
    device:      str   = "cuda",
) -> Path:
    """
    Train a 3D Gaussian Splatting model from extracted multi-view images.

    Returns the path to the saved .ply model.
    """
    cfg = cfg or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cameras ──────────────────────────────────────────────────────────
    cameras = load_cameras(data_dir, downscale=downscale)
    if not cameras:
        raise RuntimeError(f"No cameras loaded from {data_dir}")
    n_cams = len(cameras)

    # ── Background colour ─────────────────────────────────────────────────────
    bg = torch.ones(3, device=device) if bg_white else torch.zeros(3, device=device)

    # ── Initialise Gaussians ──────────────────────────────────────────────────
    smplx_ply = data_dir / "smplx_mesh.ply"
    if smplx_ply.exists():
        logger.info(f"Initialising from SMPLX mesh: {smplx_ply}")
        pcd = init_from_smplx(smplx_ply, n_points=cfg.get("n_init_points", 50_000))
    else:
        logger.warning("No SMPLX mesh found; initialising from sphere.")
        # Estimate radius from camera centres
        cam_centers = np.array([
            c.camera_center.detach().cpu().numpy() for c in cameras
        ])
        radius = float(np.linalg.norm(cam_centers, axis=-1).mean()) * 0.3
        pcd = init_sphere(n_points=cfg.get("n_init_points", 20_000), radius=radius)

    # Estimate scene radius for learning rate scaling
    scene_extent = float(np.linalg.norm(pcd.points, axis=-1).max())

    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.create_from_pcd(pcd, spatial_lr_scale=scene_extent)

    train_args = TrainArgs(cfg)
    gaussians.training_setup(train_args)

    pipe = PipeArgs()

    # Densification thresholds
    densify_from       = cfg.get("densify_from_iter",   500)
    densify_until      = cfg.get("densify_until_iter",  5000)
    densify_grad_thr   = cfg.get("densify_grad_thresh", 0.0002)
    opacity_reset_itvl = cfg.get("opacity_reset_interval", 3000)
    min_opacity        = cfg.get("min_opacity",         0.005)
    max_screen_size    = cfg.get("max_screen_size",     20)
    lambda_dssim       = cfg.get("lambda_dssim",        0.2)
    log_interval       = cfg.get("log_interval",        500)
    save_interval      = cfg.get("save_interval",       2000)

    logger.info(
        f"Training: {iterations} iters | "
        f"{n_cams} cameras | "
        f"{gaussians.get_xyz.shape[0]} initial Gaussians | "
        f"scene_extent={scene_extent:.3f}"
    )

    ema_loss = 0.0
    t0 = time.time()

    for iteration in range(1, iterations + 1):
        # ── LR update ─────────────────────────────────────────────────────────
        gaussians.update_learning_rate(iteration)

        # Up SH degree every 1000 iters
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ── Pick random camera ─────────────────────────────────────────────────
        cam = cameras[random.randint(0, n_cams - 1)]
        gt  = load_gt_image(
            data_dir, cam.camera_id,
            downscale=downscale, bg_white=bg_white, device=device,
        )

        # ── Render ────────────────────────────────────────────────────────────
        render_pkg = gs_render(
            viewpoint_camera=cam,
            pc=gaussians,
            pipe=pipe,
            bg_color=bg,
        )
        rendered = render_pkg["render"]      # [3, H, W]
        viewspace_pts = render_pkg["viewspace_points"]
        visibility     = render_pkg["visibility_filter"]
        radii          = render_pkg["radii"]

        # ── Resize GT to rendered resolution if needed ────────────────────────
        if gt.shape[-2:] != rendered.shape[-2:]:
            gt = F.interpolate(
                gt.unsqueeze(0),
                size=rendered.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        # ── Loss ──────────────────────────────────────────────────────────────
        Ll1 = l1_loss(rendered, gt)
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(rendered, gt))

        loss.backward()

        # ── Densification ──────────────────────────────────────────────────────
        with torch.no_grad():
            # Accumulate viewspace gradients for densification
            if iteration < densify_until:
                gaussians.max_radii2D[visibility] = torch.max(
                    gaussians.max_radii2D[visibility],
                    radii[visibility],
                )
                gaussians.add_densification_stats(viewspace_pts, visibility)

            if (densify_from < iteration < densify_until
                    and iteration % 100 == 0):
                size_thr = max_screen_size if iteration > opacity_reset_itvl else None
                gaussians.densify_and_prune(
                    densify_grad_thr, min_opacity, scene_extent, size_thr
                )

            if iteration % opacity_reset_itvl == 0:
                gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        # ── Logging ────────────────────────────────────────────────────────────
        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
        if iteration % log_interval == 0:
            elapsed = time.time() - t0
            n_pts   = gaussians.get_xyz.shape[0]
            logger.info(
                f"  iter {iteration:5d}/{iterations} | "
                f"loss={ema_loss:.5f} | "
                f"n_pts={n_pts:,} | "
                f"{elapsed:.0f}s"
            )

        # ── Periodic model save ────────────────────────────────────────────────
        if iteration % save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_{iteration:05d}.ply"
            gaussians.save_ply(str(ckpt_path))
            logger.info(f"  Checkpoint saved: {ckpt_path.name}")

    # ── Final save ─────────────────────────────────────────────────────────────
    final_ply = output_dir / "point_cloud_final.ply"
    gaussians.save_ply(str(final_ply))

    # Also save a viewer-ready version (absolute Gaussian positions)
    viewer_ply = output_dir / "point_cloud_viewer.ply"
    gaussians.save_ply(str(viewer_ply), for_viewer=True)

    logger.info(
        f"\nTraining complete.\n"
        f"  Final model : {final_ply}\n"
        f"  Viewer ply  : {viewer_ply}\n"
        f"  Gaussians   : {gaussians.get_xyz.shape[0]:,}\n"
        f"  Total time  : {(time.time()-t0)/60:.1f} min"
    )

    # Save a small summary JSON
    with open(output_dir / "train_summary.json", "w") as f:
        json.dump({
            "iterations":  iterations,
            "sh_degree":   sh_degree,
            "n_cameras":   n_cams,
            "n_gaussians": gaussians.get_xyz.shape[0],
            "final_loss":  float(ema_loss),
            "final_ply":   str(final_ply),
            "viewer_ply":  str(viewer_ply),
        }, f, indent=2)

    return final_ply


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Train 3DGS from an extracted 4D-DRESS single-frame directory."
    )
    p.add_argument("--data_dir",   required=True,
                   help="Directory produced by extract_frame.py.")
    p.add_argument("--output_dir", required=True,
                   help="Where to save the trained model.")
    p.add_argument("--iterations", type=int,   default=7000)
    p.add_argument("--sh_degree",  type=int,   default=3)
    p.add_argument("--downscale",  type=float, default=1.0,
                   help="Image downscale factor (>1 = lower res, faster).")
    p.add_argument("--bg_white",   action="store_true", default=True)
    p.add_argument("--bg_black",   dest="bg_white", action="store_false")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--seed",       type=int, default=42)

    # Override individual training hyper-parameters
    p.add_argument("--position_lr_init",   type=float, default=0.00016)
    p.add_argument("--position_lr_final",  type=float, default=0.0000016)
    p.add_argument("--feature_lr",         type=float, default=0.0025)
    p.add_argument("--opacity_lr",         type=float, default=0.05)
    p.add_argument("--scaling_lr",         type=float, default=0.005)
    p.add_argument("--rotation_lr",        type=float, default=0.001)
    p.add_argument("--densify_from_iter",  type=int,   default=500)
    p.add_argument("--densify_until_iter", type=int,   default=5000)
    p.add_argument("--densify_grad_thresh",type=float, default=0.0002)
    p.add_argument("--lambda_dssim",       type=float, default=0.2)
    p.add_argument("--log_interval",       type=int,   default=500)
    p.add_argument("--save_interval",      type=int,   default=2000)
    p.add_argument("--n_init_points",      type=int,   default=50_000)

    args = p.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = {
        "percent_dense":          0.01,
        "position_lr_init":       args.position_lr_init,
        "position_lr_final":      args.position_lr_final,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps":  args.iterations,
        "feature_lr":             args.feature_lr,
        "opacity_lr":             args.opacity_lr,
        "scaling_lr":             args.scaling_lr,
        "rotation_lr":            args.rotation_lr,
        "densify_from_iter":      args.densify_from_iter,
        "densify_until_iter":     args.densify_until_iter,
        "densify_grad_thresh":    args.densify_grad_thresh,
        "opacity_reset_interval": 3000,
        "min_opacity":            0.005,
        "max_screen_size":        20,
        "lambda_dssim":           args.lambda_dssim,
        "log_interval":           args.log_interval,
        "save_interval":          args.save_interval,
        "n_init_points":          args.n_init_points,
    }

    train(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        iterations=args.iterations,
        sh_degree=args.sh_degree,
        downscale=args.downscale,
        bg_white=args.bg_white,
        cfg=cfg,
        device=args.device,
    )


if __name__ == "__main__":
    main()
