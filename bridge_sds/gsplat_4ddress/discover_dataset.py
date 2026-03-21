"""
bridge_sds/gsplat_4ddress/discover_dataset.py
==============================================
Scan a 4D-DRESS dataset root and report every available
(subject, take, frame-range) combination.

Usage
-----
    python bridge_sds/gsplat_4ddress/discover_dataset.py \\
        --root /iopsstor/scratch/cscs/dbartaula/4D-Dress \\
        [--out discovered.yaml]

Output YAML format
------------------
    subjects:
      - id: 170
        takes:
          - id: 1
            n_cameras: 4
            frame_range: [21, 150]
            n_frames: 130
            example_image: Capture/cam_0/images/capture-f00021.png
      ...
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def scan_take(take_dir: Path) -> Optional[dict]:
    """Scan one Take directory and return summary dict, or None if incomplete."""
    take_dir = Path(take_dir)

    # Must have cameras.pkl
    cam_pkl = take_dir / "Capture" / "cameras.pkl"
    if not cam_pkl.exists():
        return None

    # Load camera list
    try:
        with open(cam_pkl, "rb") as f:
            cam_data = pickle.load(f)
        camera_ids = sorted(cam_data.keys())
        n_cameras  = len(camera_ids)
    except Exception as e:
        print(f"  [warn] Could not read cameras.pkl in {take_dir}: {e}")
        return None

    # Discover frame range from the first camera's images folder
    frames = []
    for cam_id in camera_ids:
        img_dir = take_dir / "Capture" / cam_id / "images"
        if img_dir.exists():
            pngs = sorted(img_dir.glob("capture-f*.png"))
            if pngs:
                for p in pngs:
                    # filename: capture-f00021.png  → frame 21
                    try:
                        frame_num = int(p.stem.split("capture-f")[-1])
                        frames.append(frame_num)
                    except ValueError:
                        pass
            break  # only need one camera to get frame list

    if not frames:
        return None

    frames = sorted(set(frames))
    example_img = (
        f"Capture/{camera_ids[0]}/images/capture-f{frames[0]:05d}.png"
    )

    # Check SMPLX dir
    smplx_dir = take_dir / "SMPLX"
    has_smplx = smplx_dir.exists() and any(smplx_dir.glob("*.ply"))

    return {
        "n_cameras":   n_cameras,
        "camera_ids":  camera_ids,
        "frame_range": [frames[0], frames[-1]],
        "n_frames":    len(frames),
        "frames":      frames,
        "has_smplx":   has_smplx,
        "example_image": example_img,
    }


def discover(root: Path) -> List[dict]:
    """Walk root and return list of subject dicts."""
    root = Path(root)
    candidates = [root / "4D-DRESS", root / "4D-Dress", root]
    fourd_dir = None
    for cand in candidates:
        if cand.exists():
            fourd_dir = cand
            break
    if fourd_dir is None:
        raise FileNotFoundError(f"4D-DRESS root not found: {root}")

    subjects = []

    for subj_dir in sorted(fourd_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        # Handle both layouts:
        #   <subj>_Inner  (original)
        #   <subj>        (Bristen/CSCS layout, e.g. 00170)
        name = subj_dir.name
        if name.endswith("_Inner"):
            try:
                subj_id = int(name.split("_")[0])
            except ValueError:
                continue
        elif name.isdigit():
            subj_id = int(name)
        else:
            continue

        inner_dir = subj_dir / "Inner"
        if not inner_dir.exists():
            continue

        takes = []
        for take_dir in sorted(inner_dir.iterdir()):
            if not take_dir.is_dir() or not take_dir.name.startswith("Take"):
                continue
            try:
                take_id = int(take_dir.name.replace("Take", ""))
            except ValueError:
                continue

            info = scan_take(take_dir)
            if info is None:
                print(f"  [skip] {take_dir.relative_to(root)} — incomplete data")
                continue

            print(
                f"  Found s{subj_id:05d} Take{take_id}: "
                f"{info['n_cameras']} cameras, "
                f"frames {info['frame_range'][0]}–{info['frame_range'][1]} "
                f"({info['n_frames']} total)"
                + ("  [SMPLX ✓]" if info["has_smplx"] else "  [no SMPLX]")
            )
            takes.append({"id": take_id, **info})

        if takes:
            subjects.append({"id": subj_id, "takes": takes})

    return subjects


def main() -> None:
    p = argparse.ArgumentParser(description="Discover 4D-DRESS dataset contents.")
    p.add_argument("--root", required=True,
                   help="Path to 4D-DRESS dataset root "
                        "(should contain a '4D-DRESS/' subfolder).")
    p.add_argument("--out",  default=None,
                   help="Optional path to write discovered.yaml.")
    args = p.parse_args()

    print(f"\nScanning 4D-DRESS at: {args.root}\n{'─'*60}")
    subjects = discover(Path(args.root))

    print(f"\n{'='*60}")
    print(f"Summary: {len(subjects)} subjects found.")
    total_takes  = sum(len(s["takes"]) for s in subjects)
    total_frames = sum(
        t["n_frames"] for s in subjects for t in s["takes"]
    )
    print(f"         {total_takes} takes total, ~{total_frames} frames total.")

    # Build YAML-serialisable output (remove 'frames' list if too long)
    out_data = {"dataset_root": str(args.root), "subjects": []}
    for s in subjects:
        s_entry = {"id": s["id"], "takes": []}
        for t in s["takes"]:
            t_copy = {k: v for k, v in t.items() if k != "frames"}
            # Keep only a representative subset of frames for the YAML
            all_f = t["frames"]
            t_copy["sample_frames"] = all_f[::max(1, len(all_f) // 10)][:10]
            s_entry["takes"].append(t_copy)
        out_data["subjects"].append(s_entry)

    if args.out:
        with open(args.out, "w") as f:
            yaml.dump(out_data, f, sort_keys=False, default_flow_style=False)
        print(f"\nDiscovery output written to: {args.out}")
    else:
        print("\n--- YAML preview ---")
        print(yaml.dump(out_data, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    main()
