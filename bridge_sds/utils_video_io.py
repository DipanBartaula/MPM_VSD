"""
bridge_sds/utils_video_io.py
============================
Video / frame I/O utilities used by the bridge pipeline.

Responsibilities
----------------
- Save a [T, H, W, 3] float32 tensor as an .mp4
- Load an .mp4 back to a tensor
- Normalise / un-normalise frame tensors
- Tile a list of views into a montage grid
- Resize frames to a target resolution
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── mp4 writers ──────────────────────────────────────────────────────────────

def save_frames_as_mp4(
    frames: torch.Tensor,
    path: Union[str, Path],
    fps: float = 25.0,
    quality: int = 8,
) -> Path:
    """
    Save a  [T, H, W, 3]  float32 tensor (values in [0, 1]) as an .mp4 file.

    Tries imageio/ffmpeg first, falls back to cv2, then raises if neither
    is available.

    Args:
        frames  : [T, H, W, 3] float32 in [0, 1]
        path    : output file path (parent dirs are created automatically)
        fps     : output frame rate
        quality : FFMPEG quality 1-10 (higher = better)

    Returns:
        Resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to uint8 numpy [T, H, W, 3]
    frames_np = (frames.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

    try:
        import imageio
        writer = imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            quality=quality,
            pixelformat="yuv420p",
        )
        for frame in frames_np:
            writer.append_data(frame)
        writer.close()
        logger.debug(f"Saved mp4 ({len(frames_np)} frames) → {path}")
        return path

    except ImportError:
        pass

    try:
        import cv2
        H, W = frames_np.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, float(fps), (W, H))
        for frame in frames_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        logger.debug(f"Saved mp4 (cv2, {len(frames_np)} frames) → {path}")
        return path

    except ImportError:
        pass

    raise RuntimeError(
        "Neither `imageio` nor `cv2` is available. "
        "Install one of them:  pip install imageio[ffmpeg]  or  pip install opencv-python"
    )


def load_video_as_frames(
    path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Load an .mp4 and return a [T, H, W, 3] float32 tensor in [0, 1].

    Args:
        path        : path to the .mp4 file
        target_size : optional (H, W) to resize each frame

    Returns:
        [T, H, W, 3] float32 in [0, 1]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        import imageio
        reader = imageio.get_reader(str(path), "ffmpeg")
        frames = [f for f in reader]
        reader.close()
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [T, H, W, 3]
    frames_t = torch.from_numpy(frames_np)

    if target_size is not None:
        H, W = target_size
        frames_t = resize_frames(frames_t, H, W)

    return frames_t


# ── resizing ─────────────────────────────────────────────────────────────────

def resize_frames(
    frames: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Resize  [T, H, W, C]  frames to  [T, height, width, C].

    Uses torch.nn.functional.interpolate internally (NCHW convention).
    """
    T, H, W, C = frames.shape
    # → [T, C, H, W]
    x = frames.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(height, width), mode=mode, align_corners=False)
    # → [T, H, W, C]
    return x.permute(0, 2, 3, 1).contiguous()


# ── montage tiling ────────────────────────────────────────────────────────────

def tile_montage(
    views: List[torch.Tensor],
    n_cols: Optional[int] = None,
) -> torch.Tensor:
    """
    Tile a list of per-view  [H, W, C]  tensors into a single montage image.

    The grid is laid out row-major.  If n_cols is None, we use the smallest
    square grid that fits all views (ceiling of sqrt(len(views))).

    Args:
        views  : list of [H, W, C] float32 tensors, all the same resolution
        n_cols : number of columns in the grid

    Returns:
        [rows*H, cols*W, C] float32 tensor
    """
    if not views:
        raise ValueError("views list is empty")

    n = len(views)
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))

    H, W, C = views[0].shape
    device = views[0].device

    # Pad with black tiles if needed
    pad = [torch.zeros(H, W, C, device=device)] * (n_rows * n_cols - n)
    tiles = views + pad

    rows = [
        torch.cat(tiles[r * n_cols : (r + 1) * n_cols], dim=1)
        for r in range(n_rows)
    ]
    return torch.cat(rows, dim=0)  # [rows*H, cols*W, C]


def tile_montage_sequence(
    views_per_frame: List[List[torch.Tensor]],
    n_cols: Optional[int] = None,
) -> torch.Tensor:
    """
    Tile multi-view sequences.

    Args:
        views_per_frame : T × K list of [H, W, C] tensors
        n_cols          : grid columns

    Returns:
        [T, rows*H, cols*W, C]
    """
    tiled = [tile_montage(views, n_cols) for views in views_per_frame]
    return torch.stack(tiled, dim=0)


# ── normalisation ─────────────────────────────────────────────────────────────

def normalize_frames(
    frames: torch.Tensor,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std:  Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """
    Normalise  [*, H, W, 3]  frames from [0, 1] to a mean/std range.
    Default mean=0.5, std=0.5 maps [0,1] → [-1, 1].
    """
    mean_t = torch.tensor(mean, dtype=frames.dtype, device=frames.device)
    std_t  = torch.tensor(std,  dtype=frames.dtype, device=frames.device)
    return (frames - mean_t) / std_t


def unnormalize_frames(
    frames: torch.Tensor,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std:  Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Inverse of normalize_frames.  Output is clamped to [0, 1]."""
    mean_t = torch.tensor(mean, dtype=frames.dtype, device=frames.device)
    std_t  = torch.tensor(std,  dtype=frames.dtype, device=frames.device)
    return (frames * std_t + mean_t).clamp(0, 1)


# ── metrics logging ───────────────────────────────────────────────────────────

def append_metrics_jsonl(
    path: Union[str, Path],
    record: dict,
) -> None:
    """Append a single metrics dict as a JSON line to a .jsonl log file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_metrics_jsonl(path: Union[str, Path]) -> List[dict]:
    """Load all records from a .jsonl metrics file."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_phi_checkpoint(
    phi: dict,
    iteration: int,
    run_dir: Union[str, Path],
    extra: Optional[dict] = None,
) -> Path:
    """
    Save φ = {D, E, H} to  <run_dir>/phi_iter_XXXXX.npz

    Args:
        phi       : {'D': float, 'E': float, 'H': float}
        iteration : current iteration number
        run_dir   : directory for this run's φ checkpoints
        extra     : optional additional scalars to store (e.g., loss)

    Returns:
        Path to saved file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / f"phi_iter_{iteration:05d}.npz"
    payload = {**phi}
    if extra:
        payload.update(extra)
    payload["iteration"] = iteration
    np.savez(str(save_path), **{k: np.float32(v) for k, v in payload.items()})
    return save_path


def load_phi_checkpoint(path: Union[str, Path]) -> dict:
    """Load φ from an .npz checkpoint.  Returns {'D', 'E', 'H', ...}."""
    data = np.load(str(path))
    return {k: float(data[k]) for k in data.files}


def cleanup_old_checkpoints(
    run_dir: Union[str, Path],
    keep_last_n: int = 5,
) -> None:
    """
    Keep only the `keep_last_n` most recent phi_iter_*.npz checkpoints.
    """
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("phi_iter_*.npz"))
    to_delete = ckpts[: max(0, len(ckpts) - keep_last_n)]
    for p in to_delete:
        p.unlink()
        logger.debug(f"Removed old checkpoint: {p.name}")
