"""
Wan2.2 TI2V-5B guidance via HuggingFace diffusers.

Uses WanPipeline (Wan-AI/Wan2.2-TI2V-5B-Diffusers) loaded from a local
snapshot_download directory.  No CPU offloading.  bfloat16 throughout.

Architecture (Wan2.2 TI2V-5B):
  - WanTransformer3DModel  — flow-prediction denoiser (bfloat16, frozen)
  - AutoencoderKLWan       — 4D VAE, spatial ×8 + temporal ×4 compression (float32)
  - T5EncoderModel         — umt5-xxl text encoder, 4096-dim embeddings (bfloat16, frozen)
  - UniPCMultistepScheduler — flow-matching, prediction_type='flow_prediction', T=1000

SDS objective (rectified-flow / flow-matching):
    σ = t / T  ∈ [0, 1]
    x_t = (1 − σ) · x₀ + σ · ε
    v_target = ε − x₀          (direction from data to noise)
    L_SDS = ‖v_θ(x_t, t, c) − v_target‖²

x₀ = VAE.encode(simulated_video) · scaling_factor
v_θ is the frozen Wan transformer.
SPSA probes the scalar loss with ±Δφ and estimates ∂L/∂φ numerically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Wan22I2VConfig:
    # Local path to a snapshot_download of Wan-AI/Wan2.2-TI2V-5B-Diffusers
    ckpt_dir: Path

    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16  # for transformer; VAE always float32

    # Text conditioning (empty → unconditional baseline works fine for SDS)
    prompt: str = ""
    negative_prompt: str = ""
    use_cfg: bool = False
    cfg_scale: float = 3.5

    # Flow-matching schedule length — must match model (Wan uses 1000)
    num_train_timesteps: int = 1000

    # ── Legacy fields kept for backwards-compat with train_sds_physics.py ──────
    # These are accepted but unused in the diffusers-based implementation.
    wan_repo_root: Optional[Path] = None
    boundary: float = 0.900
    use_first_frame_mask: bool = True
    t5_checkpoint_name: str = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer: str = "google/umt5-xxl"
    vae_checkpoint_name: str = "Wan2.1_VAE.pth"
    low_noise_subfolder: str = "low_noise_model"
    high_noise_subfolder: str = "high_noise_model"


class Wan22I2VGuidance(nn.Module):
    """
    Wraps Wan2.2-TI2V-5B (via HuggingFace diffusers) as a frozen SDS prior.

    All components live on device at all times — no CPU offloading.

    GPU memory layout (after __init__):
      - self.vae         : AutoencoderKLWan (float32)        ~1.5 GB
      - self.transformer : WanTransformer3DModel (bfloat16)  ~10  GB
      - self.scheduler   : UniPCMultistepScheduler (tiny)
      - self._context    : pre-encoded text embedding        ~negligible
      NOTE: text_encoder (~9.4 GB) and tokenizer are deleted after prompt
      pre-encoding to free GPU memory. Total on-device: ~13 GB → fits on L4.

    Public API:
      guidance.compute_loss(video_01, cond_image_01=None, timesteps=None, generator=None)
        → scalar SDS loss (float32 tensor)
    """

    def __init__(self, config: Wan22I2VConfig):
        super().__init__()
        self.cfg    = config
        self.device = torch.device(config.device)
        # Always bfloat16 for transformer (more stable than float16 for Wan)
        self._dtype = torch.bfloat16

        try:
            from diffusers import AutoencoderKLWan  # type: ignore
            from diffusers.models import WanTransformer3DModel  # type: ignore
            from diffusers.schedulers import UniPCMultistepScheduler  # type: ignore
            from transformers import AutoTokenizer, UMT5EncoderModel, CLIPVisionModel, CLIPImageProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "diffusers and transformers are required for Wan 5B guidance.\n"
                "Install: pip install git+https://github.com/huggingface/diffusers\n"
                f"Error: {exc}"
            ) from exc

        model_path = str(config.ckpt_dir)
        print(f"[Wan5B] Loading from {model_path} …")

        # ── Load transformer directly to GPU (bfloat16, ~10 GB) ──────────────
        print("[Wan5B]   loading transformer …")
        self.transformer: nn.Module = WanTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=self._dtype
        ).to(self.device).eval().requires_grad_(False)

        # ── Load VAE directly to GPU (float32, ~1.5 GB) ───────────────────────
        print("[Wan5B]   loading vae …")
        self.vae: nn.Module = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.float32
        ).to(self.device).eval().requires_grad_(False)

        # ── Scheduler (no GPU memory) ─────────────────────────────────────────
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        # Cache scaling_factor — defaults to the Wan value 0.13235
        self._scaling_factor: float = float(
            getattr(getattr(self.vae, "config", None), "scaling_factor", None) or 0.13235
        )

        # ── Text encoder: CPU-only encoding (T5 never goes to GPU) ──────────────
        # transformer+VAE+MPM+Gaussians already fill the L4's 22 GB.
        # T5 (9.4 GB) is only needed once — run it on CPU in float32, then
        # cast the embeddings to bfloat16 and move to GPU.
        print("[Wan5B]   loading text_encoder on CPU …")
        tokenizer    = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
        ).eval().requires_grad_(False)  # stays on CPU — bfloat16 halves RAM (~4.7 GB vs 9.4 GB)

        print("[Wan5B]   loading image_encoder …")
        self.image_encoder = CLIPVisionModel.from_pretrained(
            model_path, subfolder="image_encoder", torch_dtype=torch.float32
        ).to(self.device).eval().requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path, subfolder="image_processor")

        print("[Wan5B]   encoding prompts on CPU …")
        self.tokenizer    = tokenizer
        self.text_encoder = text_encoder          # temporarily store for _encode_text

        with torch.no_grad():
            self._context      = self._encode_text(config.prompt)
            self._context_null = self._encode_text(config.negative_prompt or "")

        # Move embeddings to GPU
        self._context      = self._context.to(self.device)
        self._context_null = self._context_null.to(self.device)

        # Free T5 + tokenizer — they are no longer needed
        del self.text_encoder
        del self.tokenizer
        del text_encoder
        del tokenizer
        torch.cuda.empty_cache()

        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(
            f"[Wan5B] Ready — transformer=bfloat16, vae=float32, "
            f"scaling_factor={self._scaling_factor}, device={self.device}, "
            f"VRAM after load={vram_gb:.1f} GB"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_train_timesteps(self) -> int:
        return int(getattr(self.scheduler.config, "num_train_timesteps", 1000))

    # ── Text encoding ─────────────────────────────────────────────────────────

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Tokenise + encode text → (1, seq_len, 4096) bfloat16.

        Passes attention_mask so padding tokens are not attended to.
        """
        max_len = min(int(getattr(self.tokenizer, "model_max_length", 512)), 512)
        tokens = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        # Run on whichever device text_encoder lives on (CPU during init, GPU if called later)
        enc_device     = next(self.text_encoder.parameters()).device
        input_ids      = tokens.input_ids.to(enc_device)
        attention_mask = tokens.attention_mask.to(enc_device)

        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return out.last_hidden_state.to(self._dtype)   # (1, seq_len, 4096)

    # ── Image encoding ─────────────────────────────────────────────────────────

    def _encode_image(self, cond_image_01: torch.Tensor) -> torch.Tensor:
        """
        cond_image_01: (3, H, W) or (B, 3, H, W) float32 in [0, 1]
        """
        if cond_image_01.ndim == 3:
            cond_image_01 = cond_image_01.unsqueeze(0)
        
        # Convert tensor to something CLIPImageProcessor easily parses
        import numpy as np
        # CLIP image processor expects [0, 255] numpy HWC array or similar
        imgs_np = cond_image_01.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0
        imgs_np = imgs_np.astype(np.uint8)
        imgs_list = [img for img in imgs_np]
        
        inputs = self.image_processor(images=imgs_list, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.image_encoder(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-2].to(self._dtype)  # (B, seq_len, dim)

    def build_condition_attention_mask(
        self,
        cond_image_01: torch.Tensor,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Build a soft spatial mask from averaged conditioning-image attention maps.

        The mask is computed fully under `torch.no_grad()` so it does not become
        part of the optimization graph. The resulting mask is broadcast over the
        video time dimension and can be passed to `mask_01` in `compute_loss()`.
        """
        if cond_image_01.ndim == 3:
            cond_image_01 = cond_image_01.unsqueeze(0)

        import numpy as np

        imgs_np = cond_image_01.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0
        imgs_np = imgs_np.astype(np.uint8)
        imgs_list = [img for img in imgs_np]

        inputs = self.image_processor(images=imgs_list, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.image_encoder(**inputs, output_attentions=True)

        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            raise RuntimeError(
                "Wan image encoder did not return attention maps; cannot build "
                "conditioning attention mask."
            )

        layer_maps = []
        for attn in attentions:
            # attn: [B, heads, tokens, tokens]
            attn = attn.float().abs().mean(dim=1)
            if attn.shape[-1] <= 1:
                continue
            cls_to_patch = attn[:, 0, 1:]
            patch_to_cls = attn[:, 1:, 0]
            patch_energy = 0.5 * (cls_to_patch + patch_to_cls)
            layer_maps.append(patch_energy)

        if not layer_maps:
            raise RuntimeError("No usable attention maps were returned by the image encoder.")

        patch_map = torch.stack(layer_maps, dim=0).mean(dim=0)
        num_patches = patch_map.shape[-1]
        side = int(math.isqrt(num_patches))
        if side * side != num_patches:
            raise RuntimeError(
                f"Conditioning attention patch count {num_patches} is not square."
            )

        spatial = patch_map.reshape(patch_map.shape[0], 1, side, side)
        spatial = spatial - spatial.amin(dim=(-2, -1), keepdim=True)
        spatial = spatial / spatial.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        spatial = F.interpolate(
            spatial,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return spatial.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()

    # ── VAE helpers ───────────────────────────────────────────────────────────

    def _vae_encode(
        self,
        video_m11: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Encode video [-1, 1] → scaled latents.

        video_m11: (B, 3, T, H, W) float32 in [-1, 1]
        Returns:   (B, 16, T/4, H/8, W/8) float32, scaled by scaling_factor
        """
        posterior = self.vae.encode(video_m11)
        latent_dist = posterior.latent_dist

        # DiagonalGaussianDistribution.sample() signature varies across
        # diffusers versions — handle both with and without generator arg.
        try:
            x0 = latent_dist.sample(generator=generator)
        except TypeError:
            # Fallback: set manual seed on torch global RNG if generator given
            if generator is not None:
                torch.manual_seed(generator.initial_seed())
            x0 = latent_dist.sample()

        return x0 * self._scaling_factor

    # ── Transformer forward ───────────────────────────────────────────────────

    def _denoise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.LongTensor,
        context: torch.Tensor,
        image_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single frozen transformer forward pass.

        x_t       : (B, 16, T', H', W') bfloat16 — noisy latents
        timesteps : (B,) long
        context   : (B, seq_len, 4096) bfloat16 — text embeddings
        image_embeds: (B, seq_len, dim) bfloat16 — image embeddings

        Returns:  (B, 16, T', H', W') float32 — predicted flow
        """
        # return_dict=False → returns tuple; [0] is the output tensor.
        # This avoids any potential Transformer2DModelOutput attribute issues.
        out = self.transformer(
            hidden_states=x_t,
            timestep=timesteps,
            encoder_hidden_states=context,
            encoder_hidden_states_image=image_embeds,
            return_dict=False,
        )
        pred = out[0]  # (B, 16, T', H', W') bfloat16

        # Optional classifier-free guidance
        if self.cfg.use_cfg:
            context_null = self._context_null.expand(x_t.shape[0], -1, -1)
            out_uncond = self.transformer(
                hidden_states=x_t,
                timestep=timesteps,
                encoder_hidden_states=context_null,
                encoder_hidden_states_image=image_embeds,
                return_dict=False,
            )
            pred_uncond = out_uncond[0]
            pred = pred_uncond + float(self.cfg.cfg_scale) * (pred - pred_uncond)

        return pred.float()

    # ── SDS loss ──────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        video_01: torch.Tensor,
        cond_image_01: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        generator: Optional[torch.Generator] = None,
        mask_01: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One-step rectified-flow SDS loss.

        Args:
            video_01      : (B, 3, T, H, W) float32 in [0, 1] — simulated video
            cond_image_01 : (3, H, W) or (B, 3, H, W) — optional first-frame
                            conditioning image for image-to-video guidance.
            timesteps     : (B,) long in [0, num_train_timesteps).
                            Sampled uniformly at random if None.
            generator     : torch.Generator for reproducible noise across SPSA
                            ± perturbations.  Pass the same generator (reset to
                            the same seed) for every probe in one SPSA step.
            mask_01       : (B, 1, T, H, W) float32 in [0, 1] — foreground mask.
                            If provided, limits the SDS loss purely to the 
                            region where garment/human are present.

        Returns:
            Scalar SDS loss tensor (float32).
            For SPSA, detach() is not needed — the caller wraps in no_grad().
        """
        if video_01.ndim != 5 or video_01.shape[1] != 3:
            raise ValueError(
                f"video_01 must be (B, 3, T, H, W), got shape {tuple(video_01.shape)}"
            )
        b, _, T_frames, H, W = video_01.shape
        # Wan VAE compresses temporal ×4 — need at least 4 frames for 1 latent frame
        if T_frames < 4:
            raise ValueError(
                f"video_01 has only {T_frames} frames. "
                "Wan VAE requires ≥4 frames (temporal compression ×4). "
                "Set TRAIN_FRAME_NUM ≥ 16 in your config."
            )

        # ── 1. Encode video to scaled latents ────────────────────────────────
        video_m11 = (video_01 * 2.0 - 1.0).to(device=self.device, dtype=torch.float32)  # [-1, 1]
        x0 = self._vae_encode(video_m11, generator=generator)
        # x0: (B, 16, T/4, H/8, W/8) float32

        # ── 2. Sample diffusion timestep ─────────────────────────────────────
        if timesteps is None:
            timesteps = torch.randint(
                0, self.num_train_timesteps, (b,),
                generator=generator, device=self.device,
            ).long()
        timesteps = timesteps.to(self.device)

        # ── 3. Rectified-flow noising ─────────────────────────────────────────
        #   x_t = (1 − σ) x₀ + σ ε,    σ = t / T ∈ [0, 1]
        if generator is not None:
            noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
        else:
            noise = torch.randn_like(x0)
        sigma = (timesteps.float() / float(self.num_train_timesteps))
        # Reshape for broadcasting: (B,) → (B, 1, 1, 1, 1)
        sigma = sigma.view(b, 1, 1, 1, 1)
        x_t = (1.0 - sigma) * x0 + sigma * noise

        # Target flow: model should predict direction from x₀ to noise
        target = noise - x0  # (B, 16, T', H', W') float32

        # ── 4. Frozen denoiser forward ────────────────────────────────────────
        context = self._context.expand(b, -1, -1)  # (B, 512, 4096) bfloat16
        
        image_embeds = None
        if cond_image_01 is not None:
            image_embeds = self._encode_image(cond_image_01)
            if image_embeds.shape[0] == 1 and b > 1:
                image_embeds = image_embeds.expand(b, -1, -1)
                
        pred = self._denoise(x_t.to(self._dtype), timesteps, context, image_embeds)
        # pred: (B, 16, T', H', W') float32

        # ── 5. MSE flow loss ──────────────────────────────────────────────────
        if mask_01 is not None:
            # Mask is [B, 1, T, H, W]. VAE latents are [B, 16, T/4, H/8, W/8].
            b_m, c_m, T_m, H_m, W_m = mask_01.shape
            
            # 1. Spatial downsample
            # Note target.shape[-2:] is H/8, W/8
            mask_spatial = F.interpolate(
                mask_01.permute(0, 2, 1, 3, 4).reshape(b_m * T_m, 1, H_m, W_m),
                size=(target.shape[-2], target.shape[-1]),
                mode="area"
            ).reshape(b_m, T_m, 1, target.shape[-2], target.shape[-1]).permute(0, 2, 1, 3, 4)
            # mask_spatial: [B, 1, T, H/8, W/8]
            
            # 2. Temporal downsample (VAE compresses time by 4x)
            mask_latent = F.avg_pool3d(
                mask_spatial,
                kernel_size=(4, 1, 1),
                stride=(4, 1, 1)
            ) # [B, 1, T/4, H/8, W/8]
            
            # Calculate element-wise loss then apply soft mask weighting
            mse = F.mse_loss(pred, target, reduction="none") # [B, 16, T/4, H/8, W/8]
            
            # Prevent divide by zero if mask is completely empty
            mask_sum = mask_latent.mean() + 1e-8
            return (mse * mask_latent).mean() / mask_sum
        else:
            return F.mse_loss(pred, target, reduction="mean")
