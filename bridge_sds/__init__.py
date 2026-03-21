"""
bridge_sds
==========
Score Distillation Sampling bridge for MPMAvatar physics parameter optimisation.

Uses Dreamcloth Phase3 video diffusion prior (Wan I2V) to optimise cloth
simulation parameters  φ = {D, E, H}  via SPSA or backpropagation.

Package layout
--------------
runner_mpmavatar.py   – given φ + args → rendered RGB clip + sim aux data
scorer_phase3.py      – given RGB clip → SDS scalar loss  (Phase3 / Wan)
physical_regularizers.py – L_penetration, L_stretch, L_temporal_smooth
optimize_phi.py       – main SPSA / backprop training loop  (CLI entry point)
utils_video_io.py     – frame I/O, mp4 writing, montage tiling
configs/sds_phi.yaml  – default hyper-parameters

Usage
-----
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
        --optim spsa \\
        --iters 2000
"""

__version__ = "0.1.0"
__all__ = [
    "MPMAvatarRunner",
    "Phase3Scorer",
    "SPSAOptimizer",
    "compute_all_regularizers",
]
