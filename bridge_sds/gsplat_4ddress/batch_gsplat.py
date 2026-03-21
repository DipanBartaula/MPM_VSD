"""
bridge_sds/gsplat_4ddress/batch_gsplat.py
==========================================
Orchestrate extraction + training of 10-12 Gaussian Splatting models
from 4D-DRESS targets defined in targets.yaml.

For each target:
  1. discover_dataset.py  (skipped — targets already defined)
  2. extract_frame.py     → extracted/<name>/
  3. train_single_gsplat.py → models/<name>/point_cloud_final.ply

Usage
-----
    # Run all 12 targets sequentially:
    python bridge_sds/gsplat_4ddress/batch_gsplat.py

    # Run only targets whose names match a prefix:
    python bridge_sds/gsplat_4ddress/batch_gsplat.py --filter s170

    # Dry run (show what would be done):
    python bridge_sds/gsplat_4ddress/batch_gsplat.py --dry_run

    # Skip already-completed targets:
    python bridge_sds/gsplat_4ddress/batch_gsplat.py --skip_done

    # Override dataset root (without editing the YAML):
    python bridge_sds/gsplat_4ddress/batch_gsplat.py \\
        --root /iopsstor/scratch/cscs/dbartaula/4D-Dress

Clariden / SLURM note
---------------------
To run each target as a separate SLURM job, set --slurm and each job will
be submitted with sbatch using the template in slurm_job.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Load targets
# =============================================================================

def load_targets(yaml_path: Path, filter_prefix: Optional[str] = None) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    targets = cfg.get("targets", [])
    if filter_prefix:
        targets = [t for t in targets if t["name"].startswith(filter_prefix)]

    return cfg, targets


# =============================================================================
# Run one target
# =============================================================================

def run_target(
    target:      dict,
    dataset_root: str,
    output_dir:  Path,
    train_cfg:   dict,
    skip_done:   bool = False,
    dry_run:     bool = False,
    symlink:     bool = False,
) -> dict:
    """
    Extract + train for one target.  Returns a status dict.
    """
    name    = target["name"]
    subject = target["subject"]
    take    = target["take"]
    frame   = target["frame"]

    extract_dir = output_dir / "extracted" / name
    model_dir   = output_dir / "models"    / name
    final_ply   = model_dir  / "point_cloud_final.ply"

    # ── Skip check ────────────────────────────────────────────────────────────
    if skip_done and final_ply.exists():
        print(f"  [skip]  {name} — already done ({final_ply})")
        return {"name": name, "status": "skipped", "ply": str(final_ply)}

    print(f"\n{'='*60}")
    print(f"  Target : {name}  (s{subject} Take{take} frame {frame})")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [dry]  Would extract to:  {extract_dir}")
        print(f"  [dry]  Would train  to:   {model_dir}")
        return {"name": name, "status": "dry_run"}

    t_start = time.time()

    # ── Step 1: Extract frame ─────────────────────────────────────────────────
    extract_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "extract_frame.py"),
        "--root",       dataset_root,
        "--subject",    str(subject),
        "--take",       str(take),
        "--frame",      str(frame),
        "--output_dir", str(extract_dir),
    ]
    if symlink:
        extract_cmd.append("--symlink")

    print(f"  [1/2]  Extracting frame…")
    result = subprocess.run(extract_cmd, check=False)
    if result.returncode != 0:
        print(f"  [FAIL] Extraction failed for {name} (exit {result.returncode})")
        return {"name": name, "status": "extract_failed"}

    # Check we have something
    if not (extract_dir / "cameras.pkl").exists():
        print(f"  [FAIL] cameras.pkl not found after extraction: {extract_dir}")
        return {"name": name, "status": "extract_failed"}

    # ── Step 2: Train gsplat ──────────────────────────────────────────────────
    train_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_single_gsplat.py"),
        "--data_dir",   str(extract_dir),
        "--output_dir", str(model_dir),
        "--iterations", str(train_cfg.get("iterations", 7000)),
        "--sh_degree",  str(train_cfg.get("sh_degree",   3)),
        "--downscale",  str(train_cfg.get("downscale_ratio", 1.0)),
        "--lambda_dssim",        str(train_cfg.get("lambda_dssim",        0.2)),
        "--position_lr_init",    str(train_cfg.get("position_lr_init",    0.00016)),
        "--position_lr_final",   str(train_cfg.get("position_lr_final",   0.0000016)),
        "--feature_lr",          str(train_cfg.get("feature_lr",          0.0025)),
        "--opacity_lr",          str(train_cfg.get("opacity_lr",          0.05)),
        "--scaling_lr",          str(train_cfg.get("scaling_lr",          0.005)),
        "--rotation_lr",         str(train_cfg.get("rotation_lr",         0.001)),
        "--densify_from_iter",   str(train_cfg.get("densify_from_iter",   500)),
        "--densify_until_iter",  str(train_cfg.get("densify_until_iter",  5000)),
        "--densify_grad_thresh", str(train_cfg.get("densify_grad_thresh", 0.0002)),
        "--log_interval",        str(train_cfg.get("log_interval",        500)),
        "--save_interval",       str(train_cfg.get("save_interval",       2000)),
    ]
    if not train_cfg.get("white_background", True):
        train_cmd.append("--bg_black")

    print(f"  [2/2]  Training gsplat ({train_cfg.get('iterations',7000)} iters)…")
    result = subprocess.run(train_cmd, check=False)

    elapsed = time.time() - t_start

    if result.returncode != 0:
        print(f"  [FAIL] Training failed for {name} (exit {result.returncode})")
        return {"name": name, "status": "train_failed", "elapsed_s": elapsed}

    if not final_ply.exists():
        print(f"  [WARN] Training exited OK but .ply not found: {final_ply}")
        return {"name": name, "status": "no_ply", "elapsed_s": elapsed}

    print(f"  [DONE] {name}  ✓  ({elapsed/60:.1f} min) → {final_ply}")
    return {
        "name":      name,
        "status":    "done",
        "ply":       str(final_ply),
        "elapsed_s": elapsed,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch Gaussian Splatting from 4D-DRESS targets."
    )
    p.add_argument("--config",    default=str(SCRIPT_DIR / "targets.yaml"),
                   help="Path to targets.yaml.")
    p.add_argument("--root",      default=None,
                   help="Override dataset_root from YAML.")
    p.add_argument("--output_dir", default=None,
                   help="Override output_dir from YAML.")
    p.add_argument("--filter",    default=None,
                   help="Only run targets whose name starts with this prefix.")
    p.add_argument("--skip_done", action="store_true", default=False,
                   help="Skip targets that already have a final .ply.")
    p.add_argument("--dry_run",   action="store_true", default=False,
                   help="Print what would be done without running anything.")
    p.add_argument("--symlink",   action="store_true", default=False,
                   help="Symlink images instead of copying (saves disk).")
    p.add_argument("--iterations", type=int, default=None,
                   help="Override iteration count from YAML.")
    args = p.parse_args()

    cfg, targets = load_targets(Path(args.config), filter_prefix=args.filter)

    dataset_root = args.root      or cfg.get("dataset_root", "")
    output_dir   = Path(args.output_dir or cfg.get("output_dir", "./output/gsplat_4ddress"))
    train_cfg    = cfg.get("train", {})

    if args.iterations is not None:
        train_cfg["iterations"] = args.iterations

    if not dataset_root:
        print("ERROR: dataset_root not set. Use --root or set it in targets.yaml.")
        sys.exit(1)

    print(f"\n4D-DRESS Batch Gaussian Splatting")
    print(f"  Dataset root : {dataset_root}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Targets      : {len(targets)}")
    print(f"  Iterations   : {train_cfg.get('iterations', 7000)}")
    if args.dry_run:
        print(f"  [DRY RUN]")

    results = []
    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}]", end=" ")
        status = run_target(
            target=target,
            dataset_root=dataset_root,
            output_dir=output_dir,
            train_cfg=train_cfg,
            skip_done=args.skip_done,
            dry_run=args.dry_run,
            symlink=args.symlink,
        )
        results.append(status)

    # ── Summary ────────────────────────────────────────────────────────────────
    done    = [r for r in results if r["status"] == "done"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed  = [r for r in results if r["status"] not in ("done","skipped","dry_run")]

    print(f"\n{'='*60}")
    print(f"Batch complete.")
    print(f"  Done    : {len(done)}")
    print(f"  Skipped : {len(skipped)}")
    print(f"  Failed  : {len(failed)}")
    if failed:
        print(f"  Failed names: {[r['name'] for r in failed]}")

    # Save summary
    summary_path = output_dir / "batch_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Summary : {summary_path}")

    # Print final table
    print(f"\n{'─'*60}")
    print(f"{'Name':<25} {'Status':<12} {'Time'}")
    print(f"{'─'*60}")
    for r in results:
        elapsed = f"{r.get('elapsed_s',0)/60:.1f} min" if 'elapsed_s' in r else "–"
        print(f"  {r['name']:<23} {r['status']:<12} {elapsed}")


if __name__ == "__main__":
    main()
