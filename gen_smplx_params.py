"""
Generate synthetic smplx_icp_param.pth for all training+test frames.
Keys match SmplxDeformer.smplx_forward() signature.
Zero-pose SMPLX (neutral T-pose) — only trans is estimated from body vertices.
This is sufficient for SDS training (test_frame_verts from LBS is not used in train_one_step).

Run on Lightning:
    python gen_smplx_params.py
"""
import numpy as np
import torch
import os
import glob

PRETRAINED   = "/teamspace/studios/this_studio/pretrained_models"
TRACKING_DIR = f"{PRETRAINED}/output/tracking/a1_s1_460_200"
OUT_DIR      = f"{TRACKING_DIR}/a1_s1/smplx_fitted"

# split_idx.npz not required — use centroid of all vertices as translation estimate
human_idx = None

params_files = sorted(glob.glob(f"{TRACKING_DIR}/params_*.npz"))
print(f"Generating smplx params for {len(params_files)} frames...")

for pf in params_files:
    frame = int(os.path.basename(pf).replace("params_", "").replace(".npz", ""))
    data  = np.load(pf)
    verts = data["vertices"]           # [N_verts, 3]

    # Estimate translation from centroid of all vertices
    trans = verts.mean(0)   # [3]

    # Save as numpy arrays — train_material_params.py line 288 does torch.from_numpy(v)
    param = {
        "trans":           trans[None].astype(np.float32),          # [1, 3]
        "orient":          np.zeros((1, 3),   dtype=np.float32),    # [1, 3]
        "body_pose":       np.zeros((1, 63),  dtype=np.float32),    # [1, 63]
        "beta":            np.zeros((1, 300), dtype=np.float32),    # [1, 300]
        "left_hand_pose":  np.zeros((1, 45),  dtype=np.float32),   # [1, 45]
        "right_hand_pose": np.zeros((1, 45),  dtype=np.float32),   # [1, 45]
        "expr":            np.zeros((1, 100), dtype=np.float32),    # [1, 100]
        "jaw_pose":        np.zeros((1, 3),   dtype=np.float32),    # [1, 3]
        "left_eye_pose":   np.zeros((1, 3),   dtype=np.float32),    # [1, 3]
        "right_eye_pose":  np.zeros((1, 3),   dtype=np.float32),    # [1, 3]
        "scale":           np.ones(1,         dtype=np.float32),    # [1]
    }

    out_path = os.path.join(OUT_DIR, f"{frame:06d}", "smplx_icp_param.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(param, out_path)

print(f"Done. Written to {OUT_DIR}/<frame>/smplx_icp_param.pth")
