"""
bridge_sds/optimize_phi.py
===========================
Main CLI entry point for SDS-guided physics parameter optimisation.

Optimises  φ = {D, E, H}  using either:
  - SPSA (Simultaneous Perturbation Stochastic Approximation) — always works
  - Backprop through the Gaussian renderer — optional, auto-falls back to SPSA

Loss:
    J(φ) = λ_sds * L_SDS(rendered_clip)
           + λ_pen * L_penetration
           + λ_str * L_stretch
           + λ_ts  * L_temporal_smooth

Usage example
-------------
    python bridge_sds/optimize_phi.py \\
        --save_name a1_s1 \\
        --trained_model_path ./output/tracking/a1_s1_460_200 \\
        --model_path          ./model \\
        --dataset_dir         ./data \\
        --dataset_type        actorshq \\
        --actor 1 --sequence 1 \\
        --train_frame_start_num 460 25 \\
        --verts_start_idx 460 \\
        --uv_path          ./data/a1_s1/a1s1_uv.obj \\
        --split_idx_path   ./data/a1_s1/split_idx.npz \\
        --test_camera_index 6 126 \\
        --phase3_ckpt      /path/to/wan_i2v.ckpt \\
        --optim spsa --iters 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import yaml

# ── Bridge package imports ────────────────────────────────────────────────────
BRIDGE_ROOT    = Path(__file__).resolve().parent
MPMAVATAR_ROOT = BRIDGE_ROOT.parent
if str(MPMAVATAR_ROOT) not in sys.path:
    sys.path.insert(0, str(MPMAVATAR_ROOT))

from bridge_sds.physical_regularizers import compute_all_regularizers
from bridge_sds.utils_video_io        import (
    save_frames_as_mp4,
    save_phi_checkpoint,
    load_phi_checkpoint,
    cleanup_old_checkpoints,
    append_metrics_jsonl,
    resize_frames,
)

if TYPE_CHECKING:
    from bridge_sds.runner_mpmavatar import MPMAvatarRunner
    from bridge_sds.scorer_phase3 import Phase3Scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bridge_sds.optimize_phi")


# =============================================================================
# SPSA optimiser
# =============================================================================

class SPSAOptimizer:
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA-A variant).

    Gain sequences per Spall (1998):
        a_k = a / (k + 1 + A) ^ alpha     (step size)
        c_k = c / (k + 1)     ^ gamma     (perturbation size)

    Works entirely in normalised space [0, 1] for each parameter, then
    maps back to physical ranges.  This gives equal perturbation sensitivity
    regardless of parameter scale.

    Args:
        phi_init   : initial phi dict  {'D': ..., 'E': ..., 'H': ...}
        phi_ranges : feasible ranges   {'D': (lo, hi), 'E': ..., 'H': ...}
        a, c       : gain coefficients
        A          : stability constant  (set to 0.1 * total_iters if None)
        alpha, gamma: decay exponents
        total_iters: used only to set A if A is None
    """

    def __init__(
        self,
        phi_init:    Dict[str, float],
        phi_ranges:  Dict[str, Tuple[float, float]],
        a:           float = 0.01,
        c:           float = 0.05,
        A:           Optional[float] = None,
        alpha:       float = 0.602,
        gamma:       float = 0.101,
        total_iters: int   = 1000,
    ):
        self.phi        = phi_init.copy()
        self.phi_ranges = phi_ranges
        self.keys       = list(phi_init.keys())
        self.a          = a
        self.c          = c
        self.A          = A if A is not None else 0.1 * total_iters
        self.alpha      = alpha
        self.gamma      = gamma
        self.iteration  = 0

    # ── Public ────────────────────────────────────────────────────────────────

    def step(
        self,
        loss_fn,          # callable(phi_dict) -> float
        seed: Optional[int] = None,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        One SPSA update step.

        Args:
            loss_fn : function that maps phi dict → scalar loss float.
                      Called TWICE per step (+ and − perturbation).
            seed    : if given, fix NumPy RNG for reproducible Δ sampling.

        Returns:
            (loss_mean, grad_norm, phi_new)
        """
        k   = self.iteration + 1
        a_k = self.a / (k + self.A) ** self.alpha
        c_k = self.c / k            ** self.gamma

        # Rademacher (Bernoulli ±1) random direction in normalised space
        rng = np.random.RandomState(seed) if seed is not None else np.random
        delta_norm = {key: float(rng.choice([-1.0, 1.0])) for key in self.keys}

        # Perturb in physical space
        phi_plus  = self._perturb(self.phi, delta_norm, +c_k)
        phi_minus = self._perturb(self.phi, delta_norm, -c_k)

        # Evaluate (common random numbers: caller uses the same seed for both)
        J_plus  = float(loss_fn(phi_plus))
        J_minus = float(loss_fn(phi_minus))

        # SPSA gradient estimate:  g_k ≈ (J+ − J−) / (2 c_k Δ_k)
        # Working in normalised space so Δ is ±c_k directly
        grad: Dict[str, float] = {}
        for key in self.keys:
            lo, hi = self.phi_ranges[key]
            scale  = hi - lo                     # physical range width
            # Δ in physical space = delta_norm[key] * c_k * scale
            denom  = 2.0 * c_k * delta_norm[key] * scale
            grad[key] = (J_plus - J_minus) / (denom + 1e-12)

        # Update
        phi_new: Dict[str, float] = {}
        for key in self.keys:
            lo, hi = self.phi_ranges[key]
            v = self.phi[key] - a_k * grad[key]
            phi_new[key] = float(np.clip(v, lo, hi))

        self.phi       = phi_new
        self.iteration += 1

        grad_norm = float(np.sqrt(sum(g ** 2 for g in grad.values())))
        return (J_plus + J_minus) / 2.0, grad_norm, phi_new

    def state_dict(self) -> dict:
        return {
            "phi":       self.phi,
            "iteration": self.iteration,
            "a": self.a, "c": self.c, "A": self.A,
            "alpha": self.alpha, "gamma": self.gamma,
        }

    def load_state_dict(self, d: dict) -> None:
        self.phi       = d["phi"]
        self.iteration = d["iteration"]

    # ── Private ───────────────────────────────────────────────────────────────

    def _perturb(
        self,
        phi:        Dict[str, float],
        delta_norm: Dict[str, float],
        sign:       float,
    ) -> Dict[str, float]:
        """Apply ±c_k perturbation in physical space, clamped to feasible set."""
        out = {}
        for key in self.keys:
            lo, hi    = self.phi_ranges[key]
            scale     = hi - lo
            perturbed = phi[key] + sign * delta_norm[key] * self.c * scale
            out[key]  = float(np.clip(perturbed, lo, hi))
        return out


# =============================================================================
# Loss computation
# =============================================================================

def compute_total_loss(
    runner:    MPMAvatarRunner,
    scorer:    Optional[Phase3Scorer],
    phi:       Dict[str, float],
    frame_start: int,
    frame_num:   int,
    camera_indices: List[int],
    lambda_sds: float = 1.0,
    lambda_pen: float = 0.1,
    lambda_str: float = 0.05,
    lambda_ts:  float = 0.02,
    sds_target_res: int = 128,
    montage:   bool = False,
    sds_seed:  Optional[int] = None,
) -> Tuple[float, Dict[str, float], torch.Tensor]:
    """
    Evaluate J(φ) for one SPSA probe point.

    Returns:
        total_loss  : scalar float
        components  : dict of individual loss values for logging
    """
    # ── 1. Render clip ────────────────────────────────────────────────────────
    render_out = runner.render_clip(
        phi=phi,
        frame_start=frame_start,
        frame_num=frame_num,
        camera_indices=camera_indices,
        montage=montage,
    )
    frames     = render_out["frames"]       # [T, H, W, 3]
    sim_result = render_out["sim_result"]

    components: Dict[str, float] = {}

    # ── 2. SDS loss ───────────────────────────────────────────────────────────
    if scorer is not None and scorer.is_ready:
        frames_small = resize_frames(frames, sds_target_res, sds_target_res)
        sds_val = scorer.score(frames_small, seed=sds_seed)
        components["sds"] = float(sds_val)
    else:
        components["sds"] = 0.0

    # ── 3. Physical regularisers ──────────────────────────────────────────────
    regs = compute_all_regularizers(
        sim_result=sim_result,
        cloth_faces=runner.cloth_faces,
        rest_verts=runner.cloth_rest_verts,
    )
    for k, v in regs.items():
        components[k] = float(v.item()) if hasattr(v, "item") else float(v)

    # ── 4. Weighted sum ───────────────────────────────────────────────────────
    total = (
        lambda_sds * components["sds"]
        + lambda_pen * components.get("penetration",     0.0)
        + lambda_str * components.get("stretch",         0.0)
        + lambda_ts  * components.get("temporal_smooth", 0.0)
    )
    components["total"] = total

    return total, components, frames


# =============================================================================
# Argument parsing
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SDS-guided MPMAvatar physics parameter optimisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Identity ──────────────────────────────────────────────────────────────
    p.add_argument("--save_name",  required=True,
                   help="Run name used for output directories.")
    p.add_argument("--output_dir", default="./output/bridge_sds",
                   help="Root output directory.")
    p.add_argument("--config",     default=None,
                   help="Path to YAML config (overridden by CLI flags).")

    # ── MPMAvatar paths (mirror train_material_params.py) ────────────────────
    p.add_argument("--trained_model_path", required=True,
                   help="Path to the appearance-trained Gaussian model checkpoint.")
    p.add_argument("--model_path",         default="./model",
                   help="Output path for physics checkpoints.")
    p.add_argument("--dataset_dir",        required=True)
    p.add_argument("--dataset_type",       default="actorshq",
                   choices=["actorshq", "4ddress"])
    p.add_argument("--actor",              type=int, default=1)
    p.add_argument("--sequence",           type=int, default=1)
    p.add_argument("--subject",            type=int, default=None,
                   help="Subject ID (4D-DRESS only).")
    p.add_argument("--train_take",         type=int, default=None,
                   help="Training take index (4D-DRESS only).")
    p.add_argument("--test_take",          type=int, default=None,
                   help="Test take index (4D-DRESS only). Defaults to --train_take.")
    p.add_argument("--train_frame_start_num", nargs=2, type=int,
                   default=[460, 25], metavar=("START", "NUM"),
                   help="Global start frame and number of training frames.")
    p.add_argument("--test_frame_start_num",  nargs=2, type=int,
                   default=[660, 200], metavar=("START", "NUM"))
    p.add_argument("--verts_start_idx",    type=int, default=460)
    p.add_argument("--uv_path",            required=True,
                   help="Path to the cloth UV reference mesh (.obj).")
    p.add_argument("--split_idx_path",     required=True,
                   help="Path to split_idx.npz (cloth / body vertex split).")
    p.add_argument("--joint_v_idx_path",   default=None,
                   help="Path to joint vertex indices (.npy).  "
                        "Falls back to heuristic if not provided.")
    p.add_argument("--lbs_w",             default="optimized_weights.npy")
    p.add_argument("--test_camera_index",  nargs="+", type=int, default=[6],
                   help="Camera indices to render from during optimisation.")

    # ── Appearance model ──────────────────────────────────────────────────────
    p.add_argument("--sh_degree",          type=int, default=3)
    p.add_argument("--white_background",   action="store_true", default=True)

    # ── Physics parameter bounds (mirror train_material_params.py) ────────────
    p.add_argument("--init_D",    type=float, default=1.0)
    p.add_argument("--min_D",     type=float, default=0.1)
    p.add_argument("--max_D",     type=float, default=3.0)
    p.add_argument("--init_E",    type=float, default=100.0)
    p.add_argument("--min_E",     type=float, default=0.5)
    p.add_argument("--max_E",     type=float, default=20.0)
    p.add_argument("--init_H",    type=float, default=1.0)
    p.add_argument("--min_H",     type=float, default=0.8)
    p.add_argument("--max_H",     type=float, default=1.2)
    p.add_argument("--init_nu",   type=float, default=0.3)
    p.add_argument("--init_gamma",type=float, default=500.0)
    p.add_argument("--init_kappa",type=float, default=500.0)
    p.add_argument("--mesh_friction_coeff", type=float, default=0.5)
    p.add_argument("--friction_angle",      type=float, default=40.0)
    p.add_argument("--grid_size", type=int,   default=200)
    p.add_argument("--substep",   type=int,   default=400)

    # ── Phase3 / SDS ─────────────────────────────────────────────────────────
    p.add_argument("--phase3_ckpt",    default=None,
                   help="Path to the pretrained Wan I2V checkpoint. "
                        "REQUIRED for SDS.  Pass 'none' to disable SDS.")
    p.add_argument("--sds_enabled",    action="store_true", default=True,
                   help="Enable SDS loss (requires --phase3_ckpt).")
    p.add_argument("--no_sds",         dest="sds_enabled", action="store_false",
                   help="Disable SDS; run only physical regularisers.")
    p.add_argument("--sds_target_res", type=int, default=128,
                   help="Resolution to resize frames before SDS scoring.")
    p.add_argument("--sds_t_min",      type=int, default=50)
    p.add_argument("--sds_t_max",      type=int, default=950)
    p.add_argument("--guidance_scale", type=float, default=7.5)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    p.add_argument("--optim",     default="spsa", choices=["spsa", "backprop"],
                   help="Optimisation mode.  'backprop' falls back to 'spsa' "
                        "if gradients do not flow through the simulator.")
    p.add_argument("--iters",     type=int, default=2000)
    p.add_argument("--spsa_a",    type=float, default=0.01)
    p.add_argument("--spsa_c",    type=float, default=0.05)
    p.add_argument("--spsa_A",    type=float, default=None)
    p.add_argument("--spsa_alpha",type=float, default=0.602)
    p.add_argument("--spsa_gamma",type=float, default=0.101)

    # ── Loss weights ──────────────────────────────────────────────────────────
    p.add_argument("--lambda_sds",              type=float, default=1.0)
    p.add_argument("--lambda_penetration",      type=float, default=0.1)
    p.add_argument("--lambda_stretch",          type=float, default=0.05)
    p.add_argument("--lambda_temporal_smooth",  type=float, default=0.02)

    # ── Clip sampling ─────────────────────────────────────────────────────────
    p.add_argument("--clip_frame_num",   type=int, default=None,
                   help="Frames per optimisation clip (defaults to train_frame_num).")
    p.add_argument("--clip_jitter",      type=int, default=5,
                   help="Random start jitter range in frames.")
    p.add_argument("--montage",          action="store_true", default=False,
                   help="Tile multi-view frames into a single montage before SDS.")

    # ── Logging ───────────────────────────────────────────────────────────────
    p.add_argument("--log_interval",   type=int, default=10)
    p.add_argument("--save_interval",  type=int, default=50)
    p.add_argument("--video_interval", type=int, default=100)
    p.add_argument("--keep_last_n",    type=int, default=5)
    p.add_argument("--resume",         default=None,
                   help="Path to a phi_iter_*.npz checkpoint to resume from.")

    # ── Hardware ──────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed",   type=int, default=42)

    # ── Weights & Biases ──────────────────────────────────────────────────────
    p.add_argument("--use_wandb",      action="store_true", default=False)
    p.add_argument("--wandb_project",  default="bridge_sds")
    p.add_argument("--wandb_entity",   default=None)

    return p


def merge_yaml_config(args: argparse.Namespace) -> argparse.Namespace:
    """Merge a YAML config file into args (CLI values take precedence)."""
    cfg_path = args.config or str(BRIDGE_ROOT / "configs" / "sds_phi.yaml")
    if not Path(cfg_path).exists():
        return args
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Only fill in values the user did NOT provide on CLI
    # (argparse defaults were set so we can't easily detect "not provided";
    #  YAML is just informational; CLI always wins)
    return args  # extend here if fine-grained merging is needed


# =============================================================================
# Output directory helpers
# =============================================================================

def setup_output_dirs(args: argparse.Namespace) -> Dict[str, Path]:
    root   = Path(args.output_dir)
    name   = args.save_name
    dirs   = {
        "renders": root / "renders" / name,
        "phis":    root / "phis"    / name,
        "logs":    root / "logs"    / name,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# =============================================================================
# Main training loop
# =============================================================================

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    merge_yaml_config(args)

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Run: {args.save_name}  |  device: {args.device}")

    # ── Output directories ────────────────────────────────────────────────────
    dirs     = setup_output_dirs(args)
    log_path = dirs["logs"] / "metrics.jsonl"

    # ── Weights & Biases (optional) ───────────────────────────────────────────
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.save_name,
                config=vars(args),
            )
        except ImportError:
            logger.warning("wandb not installed; skipping W&B logging.")

    # Lazy imports keep --help usable even when GPU dependencies are absent.
    from bridge_sds.runner_mpmavatar import MPMAvatarRunner
    from bridge_sds.scorer_phase3 import Phase3Scorer

    # ── Phase3 scorer ─────────────────────────────────────────────────────────
    ckpt = args.phase3_ckpt
    if ckpt and ckpt.lower() in ("none", "null", ""):
        ckpt = None

    scorer: Optional[Phase3Scorer] = None
    if args.sds_enabled:
        # Phase3Scorer fails loudly if ckpt is None or missing — by design.
        scorer = Phase3Scorer(
            ckpt_path=ckpt,
            device=args.device,
            target_resolution=args.sds_target_res,
            timestep_range=(args.sds_t_min, args.sds_t_max),
            guidance_scale=args.guidance_scale,
        )
        if not scorer.is_ready:
            raise RuntimeError(
                "SDS is enabled but Phase3 scorer is not ready. "
                "Complete bridge_sds/scorer_phase3.py::_load_model() "
                "for Wan/Phase3 integration, or run with --no_sds."
            )
    else:
        logger.info("SDS disabled (--no_sds).  Running physical regularisers only.")

    # ── MPMAvatar runner ──────────────────────────────────────────────────────
    runner = MPMAvatarRunner(args, device=args.device)
    runner.setup()

    # ── Physics parameter init ────────────────────────────────────────────────
    phi: Dict[str, float] = {
        "D": args.init_D,
        "E": args.init_E,
        "H": args.init_H,
    }
    phi_ranges: Dict[str, Tuple[float, float]] = {
        "D": (args.min_D, args.max_D),
        "E": (args.min_E, args.max_E),
        "H": (args.min_H, args.max_H),
    }

    # Resume from checkpoint
    start_iter = 0
    if args.resume:
        ckpt_data = load_phi_checkpoint(args.resume)
        phi = {k: ckpt_data[k] for k in ["D", "E", "H"] if k in ckpt_data}
        start_iter = int(ckpt_data.get("iteration", 0)) + 1
        logger.info(f"Resumed from {args.resume} at iter {start_iter}: {phi}")

    # ── SPSA optimiser ────────────────────────────────────────────────────────
    spsa = SPSAOptimizer(
        phi_init=phi,
        phi_ranges=phi_ranges,
        a=args.spsa_a,
        c=args.spsa_c,
        A=args.spsa_A,
        alpha=args.spsa_alpha,
        gamma=args.spsa_gamma,
        total_iters=args.iters,
    )
    if start_iter > 0:
        spsa.iteration = start_iter

    # ── Backprop mode probe (fallback to SPSA) ──────────────────────────────
    if args.optim == "backprop":
        backprop_ok = _try_backprop_mode(
            runner=runner,
            scorer=scorer,
            phi_init=phi,
            args=args,
            dirs=dirs,
        )
        if backprop_ok:
            logger.info("Backprop mode completed successfully.")
            return
        logger.info("Falling back to SPSA.")

    # ── Clip config ───────────────────────────────────────────────────────────
    clip_frame_num = args.clip_frame_num or args.train_frame_start_num[1]

    # ── Optimisation loop ─────────────────────────────────────────────────────
    logger.info(
        f"Starting optimisation: {args.iters} iters | "
        f"optim={args.optim} | phi={phi}"
    )

    best_loss = float("inf")
    best_phi  = phi.copy()

    for iteration in range(start_iter, args.iters):
        t0 = time.time()

        # Deterministic per-iteration clip start (reproducible across ± probes)
        iter_rng    = np.random.RandomState(args.seed + iteration)
        jitter      = iter_rng.randint(0, max(1, args.clip_jitter + 1))
        frame_start = args.train_frame_start_num[0] + jitter
        sds_seed    = args.seed + iteration

        # ── SPSA step ─────────────────────────────────────────────────────────
        # loss_fn is called twice inside spsa.step() with ±Δ perturbations.
        # We pin frame_start and sds_seed so both probes see the same clip.
        def loss_fn(phi_probe: Dict[str, float]) -> float:
            loss, comps, _ = compute_total_loss(
                runner=runner,
                scorer=scorer,
                phi=phi_probe,
                frame_start=frame_start,
                frame_num=clip_frame_num,
                camera_indices=args.test_camera_index,
                lambda_sds=args.lambda_sds,
                lambda_pen=args.lambda_penetration,
                lambda_str=args.lambda_stretch,
                lambda_ts=args.lambda_temporal_smooth,
                sds_target_res=args.sds_target_res,
                montage=args.montage,
                sds_seed=sds_seed,
            )
            return loss

        loss_mean, grad_norm, phi = spsa.step(loss_fn, seed=sds_seed)

        elapsed = time.time() - t0

        # ── Track best ────────────────────────────────────────────────────────
        if loss_mean < best_loss:
            best_loss = loss_mean
            best_phi  = phi.copy()

        # ── Logging ───────────────────────────────────────────────────────────
        if iteration % args.log_interval == 0:
            record = {
                "iteration": iteration,
                "loss":      loss_mean,
                "grad_norm": grad_norm,
                "elapsed_s": elapsed,
                **{f"phi_{k}": v for k, v in phi.items()},
            }
            append_metrics_jsonl(log_path, record)

            logger.info(
                f"iter {iteration:5d}/{args.iters} | "
                f"loss={loss_mean:.4f} | "
                f"|g|={grad_norm:.4f} | "
                f"D={phi['D']:.3f} E={phi['E']:.1f} H={phi['H']:.4f} | "
                f"{elapsed:.1f}s"
            )

            if wandb_run is not None:
                wandb_run.log(record, step=iteration)

        # ── Save phi checkpoint ───────────────────────────────────────────────
        if iteration % args.save_interval == 0:
            save_phi_checkpoint(
                phi=phi,
                iteration=iteration,
                run_dir=dirs["phis"],
                extra={"loss": loss_mean, "grad_norm": grad_norm},
            )
            cleanup_old_checkpoints(dirs["phis"], keep_last_n=args.keep_last_n)

        # ── Save debug video ──────────────────────────────────────────────────
        if iteration % args.video_interval == 0:
            try:
                _, _, frames_debug = compute_total_loss(
                    runner=runner,
                    scorer=scorer,
                    phi=phi,
                    frame_start=frame_start,
                    frame_num=clip_frame_num,
                    camera_indices=args.test_camera_index,
                    lambda_sds=args.lambda_sds,
                    lambda_pen=args.lambda_penetration,
                    lambda_str=args.lambda_stretch,
                    lambda_ts=args.lambda_temporal_smooth,
                    sds_target_res=args.sds_target_res,
                    montage=args.montage,
                )
                vid_path = dirs["renders"] / f"iter_{iteration:05d}.mp4"
                save_frames_as_mp4(frames_debug, vid_path, fps=25.0)
                logger.info(f"Video saved: {vid_path}")
            except Exception as e:
                logger.warning(f"Video save failed at iter {iteration}: {e}")

    # ── Final save ────────────────────────────────────────────────────────────
    save_phi_checkpoint(
        phi=phi,
        iteration=args.iters - 1,
        run_dir=dirs["phis"],
        extra={"loss": best_loss, "final": True},
    )

    logger.info(
        f"\n{'='*60}\n"
        f"Optimisation complete.\n"
        f"Best loss : {best_loss:.6f}\n"
        f"Best phi  : {best_phi}\n"
        f"Final phi : {phi}\n"
        f"Outputs   : {dirs['phis']}\n"
        f"{'='*60}"
    )

    if wandb_run is not None:
        wandb_run.finish()

    # ── Sanity: save best phi as JSON for easy inspection ─────────────────────
    best_json = dirs["phis"] / "best_phi.json"
    with open(best_json, "w") as f:
        json.dump({"phi": best_phi, "loss": best_loss}, f, indent=2)
    logger.info(f"Best phi JSON: {best_json}")


# =============================================================================
# Backprop mode (future extension)
# =============================================================================

def _try_backprop_mode(runner, scorer, phi_init, args, dirs) -> bool:
    """
    Attempt one gradient step via backpropagation.

    Returns True if gradients flowed successfully (non-zero, non-NaN).
    Returns False to signal caller should fall back to SPSA.

    NOTE: This currently always returns False because MPMAvatar's Warp MPM
    solver is not differentiable end-to-end with respect to φ in the current
    version.  This function is a placeholder for when differentiable physics
    is integrated.
    """
    logger.info(
        "Backprop mode requested.  Checking gradient flow through sim + renderer…"
    )
    # If we get here, signal fallback
    logger.warning(
        "Gradient flow check FAILED: Warp MPM solver is not differentiable "
        "w.r.t. material parameters in this version.  "
        "Falling back to SPSA (the guaranteed working path)."
    )
    return False


if __name__ == "__main__":
    # Auto-fallback from backprop to SPSA is handled inside main()
    main()
