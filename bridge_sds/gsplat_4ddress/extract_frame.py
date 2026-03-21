"""
bridge_sds/gsplat_4ddress/extract_frame.py
==========================================
Extract multi-view images, masks, and camera parameters for a single frame
from 4D-DRESS into a clean self-contained directory.

Output layout
-------------
    <output_dir>/
    ├── cameras.pkl            copy of the original cameras.pkl
    ├── cameras_info.json      human-readable camera summary
    ├── smplx_mesh.ply         SMPLX body mesh for this frame
    ├── images/
    │   ├── cam_0.png          RGB image from camera 0
    │   ├── cam_1.png          ...
    │   └── ...
    └── masks/
        ├── cam_0.png          foreground mask from camera 0
        ├── cam_1.png          ...
        └── ...

Usage
-----
    python bridge_sds/gsplat_4ddress/extract_frame.py \\
        --root  /iopsstor/scratch/cscs/dbartaula/4D-Dress \\
        --subject 170 --take 1 --frame 21 \\
        --output_dir ./output/gsplat_4ddress/extracted/s170_t1_f021

    # Or let the script choose the output path automatically:
    python bridge_sds/gsplat_4ddress/extract_frame.py \\
        --root  /iopsstor/scratch/cscs/dbartaula/4D-Dress \\
        --subject 170 --take 1 --frame 21 \\
        --output_base ./output/gsplat_4ddress/extracted
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_4ddress_root(root: Path) -> Path:
    """Accept both parent-of-4D-DRESS and direct 4D-DRESS roots.

    Handles two known layouts:
      - <root>/4D-DRESS/<subj>_Inner/Inner/Take<N>/  (original)
      - <root>/4D-DRESS/<subj>/Inner/Take<N>/         (Bristen/CSCS layout)
    """
    candidates = [
        root / "4D-DRESS",
        root / "4D-Dress",
        root,
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        # Match either <subj>_Inner or plain numeric <subj> dirs
        if any(cand.glob("*_Inner")) or any(cand.glob("[0-9][0-9][0-9][0-9][0-9]")):
            return cand
    raise FileNotFoundError(
        "Could not locate 4D-DRESS root.\n"
        f"Tried:\n  {root / '4D-DRESS'}\n"
        f"  {root / '4D-Dress'}\n"
        f"  {root}"
    )


def take_dir(root: Path, subject: int, take: int) -> Path:
    data_root = resolve_4ddress_root(root)
    # Try both layouts: <subj>_Inner/Inner/Take<N> and <subj>/Inner/Take<N>
    for subj_dirname in (f"{subject:05d}_Inner", f"{subject:05d}"):
        candidate = data_root / subj_dirname / "Inner" / f"Take{take}"
        if candidate.exists():
            return candidate
    # Return the expected path so the validator gives a clear error
    return data_root / f"{subject:05d}" / "Inner" / f"Take{take}"


def validate_take(tdir: Path, subject: int, take: int, frame: int) -> None:
    """Raise FileNotFoundError if expected paths are missing."""
    cam_pkl = tdir / "Capture" / "cameras.pkl"
    if not cam_pkl.exists():
        raise FileNotFoundError(
            f"cameras.pkl not found: {cam_pkl}\n"
            f"Subject {subject} Take {take} may not be preprocessed yet."
        )


def extract(
    root:       Path,
    subject:    int,
    take:       int,
    frame:      int,
    output_dir: Path,
    copy:       bool = True,        # True = copy files; False = symlink (faster)
) -> dict:
    """
    Extract one frame from 4D-DRESS.

    Args:
        root       : 4D-DRESS dataset root
        subject    : subject ID (e.g. 170)
        take       : take ID   (e.g. 1)
        frame      : frame index (e.g. 21)
        output_dir : where to write the extracted data
        copy       : if True copy files; if False create symlinks

    Returns:
        info dict with camera count, image paths, etc.
    """
    tdir = take_dir(root, subject, take)
    validate_take(tdir, subject, take, frame)

    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks" ).mkdir(parents=True, exist_ok=True)

    # ── Load cameras ──────────────────────────────────────────────────────────
    cam_pkl_src = tdir / "Capture" / "cameras.pkl"
    with open(cam_pkl_src, "rb") as f:
        cam_data = pickle.load(f)
    camera_ids = sorted(cam_data.keys())

    # Copy cameras.pkl
    _transfer(cam_pkl_src, output_dir / "cameras.pkl", copy)

    # ── Write human-readable camera JSON ─────────────────────────────────────
    cam_info = {}
    for cid, cval in cam_data.items():
        cam_info[str(cid)] = {
            "extrinsics": cval["extrinsics"].tolist(),  # 3×4
            "intrinsics": cval["intrinsics"].tolist(),  # 3×3
        }
    with open(output_dir / "cameras_info.json", "w") as f:
        json.dump(
            {
                "subject": subject,
                "take":    take,
                "frame":   frame,
                "n_cameras": len(camera_ids),
                "camera_ids": camera_ids,
                "cameras": cam_info,
            },
            f,
            indent=2,
        )

    # ── Extract images and masks ──────────────────────────────────────────────
    image_paths = []
    mask_paths  = []
    missing     = []

    for idx, cid in enumerate(camera_ids):
        img_src  = tdir / "Capture" / cid / "images" / f"capture-f{frame:05d}.png"
        msk_src  = tdir / "Capture" / cid / "masks"  / f"mask-f{frame:05d}.png"
        img_dst  = output_dir / "images" / f"{cid}.png"
        msk_dst  = output_dir / "masks"  / f"{cid}.png"

        if img_src.exists():
            _transfer(img_src, img_dst, copy)
            image_paths.append(str(img_dst.relative_to(output_dir)))
        else:
            missing.append(str(img_src))
            print(f"  [warn] Missing image: {img_src}")

        if msk_src.exists():
            _transfer(msk_src, msk_dst, copy)
            mask_paths.append(str(msk_dst.relative_to(output_dir)))
        # masks are optional; don't warn if missing

    if missing:
        print(
            f"  [warn] {len(missing)} images missing for frame {frame}. "
            "This frame may not exist in the dataset."
        )

    # ── Copy SMPLX mesh ───────────────────────────────────────────────────────
    smplx_ply = tdir / "SMPLX" / f"mesh-f{frame:05d}_smplx.ply"
    smplx_dst = output_dir / "smplx_mesh.ply"
    if smplx_ply.exists():
        _transfer(smplx_ply, smplx_dst, copy)
    else:
        print(f"  [warn] SMPLX mesh not found: {smplx_ply}")

    smplx_pkl = tdir / "SMPLX" / f"mesh-f{frame:05d}_smplx.pkl"
    if smplx_pkl.exists():
        _transfer(smplx_pkl, output_dir / "smplx_params.pkl", copy)

    # ── Write metadata ─────────────────────────────────────────────────────────
    meta = {
        "subject":    subject,
        "take":       take,
        "frame":      frame,
        "source_dir": str(tdir),
        "n_cameras":  len(camera_ids),
        "n_images":   len(image_paths),
        "has_masks":  bool(mask_paths),
        "has_smplx":  smplx_ply.exists(),
        "image_paths": image_paths,
        "mask_paths":  mask_paths,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  Extracted s{subject} Take{take} frame {frame}: "
        f"{len(image_paths)} images → {output_dir}"
    )
    return meta


def _transfer(src: Path, dst: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(src, dst)
    else:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract multi-view images for a single 4D-DRESS frame."
    )
    p.add_argument("--root",        required=True,
                   help="4D-DRESS dataset root (contains 4D-DRESS/ subfolder).")
    p.add_argument("--subject",     type=int, required=True)
    p.add_argument("--take",        type=int, required=True)
    p.add_argument("--frame",       type=int, required=True)
    p.add_argument("--output_dir",  default=None,
                   help="Explicit output directory.")
    p.add_argument("--output_base", default="./output/gsplat_4ddress/extracted",
                   help="Base dir; actual dir = <base>/s{subj:05d}_t{take}_f{frame:05d}/")
    p.add_argument("--symlink",     action="store_true", default=False,
                   help="Use symlinks instead of copying (saves disk space, faster).")
    args = p.parse_args()

    root = Path(args.root)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = (
            Path(args.output_base)
            / f"s{args.subject:05d}_t{args.take}_f{args.frame:05d}"
        )

    extract(
        root=root,
        subject=args.subject,
        take=args.take,
        frame=args.frame,
        output_dir=out_dir,
        copy=not args.symlink,
    )


if __name__ == "__main__":
    main()
