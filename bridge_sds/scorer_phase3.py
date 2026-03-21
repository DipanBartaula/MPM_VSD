"""
bridge_sds/scorer_phase3.py
============================
Score Distillation Sampling scorer using Wan2.2 TI2V-5B (HuggingFace diffusers).

Wraps Wan22I2VGuidance as a scalar loss function for the SPSA optimiser
in optimize_phi.py.  No CPU offloading.

Usage from optimize_phi.py:
    scorer = Phase3Scorer(
        ckpt_path="/path/to/Wan2.2-TI2V-5B-Diffusers",   # local snapshot dir
        device="cuda:0",
    )
    loss = scorer.score(frames)   # frames: [T, H, W, 3] float32 in [0, 1]

SDS gradient formula (SPSA — black-box):
    J(φ) = ||v_θ(VAE(render(φ)), t, c) − target_flow||²
    ∂J/∂φ ≈ SPSA finite difference
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .utils_video_io import resize_frames

logger = logging.getLogger(__name__)


class Phase3Scorer:
    """
    Wraps Wan2.2 TI2V-5B as a scalar SDS loss function.

    Args:
        ckpt_path        : local path to a snapshot_download of
                           Wan-AI/Wan2.2-TI2V-5B-Diffusers.
                           Pass None only when sds_enabled=False.
        device           : torch device string
        target_resolution: (H, W) to resize frames before scoring
        timestep_range   : (min, max) timestep range (ignored; kept for compat)
        guidance_scale   : CFG scale (pass use_cfg=True in config to activate)
        use_amp          : unused; bfloat16 is always used
        prompt           : text conditioning for the video diffusion model
    """

    def __init__(
        self,
        ckpt_path:         Optional[Union[str, Path]],
        device:            str = "cuda:0",
        target_resolution: int = 128,
        timestep_range:    Tuple[int, int] = (50, 950),
        guidance_scale:    float = 3.5,
        use_amp:           bool = True,
        prompt:            str = "",
    ):
        self.device           = device
        self.target_res       = target_resolution
        self.guidance_scale   = guidance_scale
        self._model           = None
        self._model_ready     = False

        if ckpt_path is None:
            raise ValueError(
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  Phase3Scorer: NO CHECKPOINT PROVIDED                   ║\n"
                "║                                                          ║\n"
                "║  Provide the local path to Wan2.2-TI2V-5B-Diffusers.   ║\n"
                "║  Download once with:                                     ║\n"
                "║    from huggingface_hub import snapshot_download         ║\n"
                "║    snapshot_download('Wan-AI/Wan2.2-TI2V-5B-Diffusers', ║\n"
                "║        local_dir='/your/path')                           ║\n"
                "║  OR disable SDS with --no_sds                           ║\n"
                "╚══════════════════════════════════════════════════════════╝"
            )

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"\nPhase3Scorer: checkpoint not found at {ckpt_path}\n"
                "Download with: snapshot_download('Wan-AI/Wan2.2-TI2V-5B-Diffusers', "
                f"local_dir='{ckpt_path}')\n"
                "Or disable SDS with --no_sds"
            )

        self._ckpt_path = ckpt_path
        self._prompt    = prompt
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        from bridge_sds.wan22_i2v_guidance import Wan22I2VConfig, Wan22I2VGuidance

        cfg = Wan22I2VConfig(
            ckpt_dir=self._ckpt_path,
            device=self.device,
            dtype=torch.bfloat16,
            prompt=self._prompt,
            use_cfg=(self.guidance_scale > 1.0),
            cfg_scale=self.guidance_scale,
        )
        self._model = Wan22I2VGuidance(cfg)
        self._model.eval()
        self._model_ready = True
        logger.info("Phase3Scorer: Wan2.2 TI2V-5B loaded and ready.")

    # ── Scoring API ───────────────────────────────────────────────────────────

    def score(
        self,
        frames:  torch.Tensor,
        seed:    Optional[int] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute the SDS loss for a video clip.

        Args:
            frames           : [T, H, W, 3] float32 in [0, 1]
            seed             : random seed (use same seed for both SPSA ± probes)
            return_components: if True, also return a dict of sub-losses

        Returns:
            loss_sds            : scalar tensor
            components (opt)    : {'sds': float, 'timestep': int}
        """
        if not self._model_ready or self._model is None:
            raise RuntimeError("Phase3Scorer model is not loaded.")

        device = self.device
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

        # [T, H, W, 3] → [1, 3, T, H, W]
        T, H, W, C = frames.shape
        if H != self.target_res or W != self.target_res:
            frames = resize_frames(frames, self.target_res, self.target_res)

        video = frames.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # [1, 3, T, H, W]

        with torch.no_grad():
            loss_sds = self._model.compute_loss(video, generator=generator)

        if return_components:
            return loss_sds, {"sds": float(loss_sds.item())}
        return loss_sds

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._model_ready and self._model is not None

    def __repr__(self) -> str:
        ckpt = str(self._ckpt_path) if hasattr(self, "_ckpt_path") else "None"
        return (
            f"Phase3Scorer("
            f"ready={self.is_ready}, "
            f"ckpt={ckpt}, "
            f"device={self.device}, "
            f"target_res={self.target_res})"
        )
