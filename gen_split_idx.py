"""
Generate a synthetic split_idx.npz that treats the entire mesh as cloth.
This is sufficient for SDS hypothesis testing — no real cloth/body segmentation needed.

Keys produced (match what train_material_params.load_gaussians() expects):
  num_joint_v            — 0  (no seam vertices)
  num_joint_f            — 0
  reordered_cloth_v_idx  — [0 .. N_verts-1]   all vertices → cloth
  reordered_cloth_f_idx  — [0 .. N_faces-1]   all Gaussian faces → cloth
  reordered_human_v_idx  — []  (empty)
  reordered_human_f_idx  — []  (empty)
  new_cloth_faces        — faces from params file (local == global since all verts are cloth)
  new_human_faces        — []  (empty)

Run on Lightning:
    python gen_split_idx.py
"""
import numpy as np
import os
import glob

PRETRAINED   = "/teamspace/studios/this_studio/pretrained_models"
TRACKING_DIR = f"{PRETRAINED}/output/tracking/a1_s1_460_200"
REPO_DIR     = "/teamspace/studios/this_studio/mpm_avatar_vds"
OUT_PATH     = f"{REPO_DIR}/data/a1_s1/split_idx.npz"

# Load any params file to get mesh topology
params_files = sorted(glob.glob(f"{TRACKING_DIR}/params_*.npz"))
if not params_files:
    raise FileNotFoundError(f"No params_*.npz found in {TRACKING_DIR}")

data  = np.load(params_files[0])
verts = data["vertices"]   # [N_verts, 3]
faces = data["faces"]      # [N_faces, 3]

N_verts = verts.shape[0]
N_faces = faces.shape[0]
print(f"Mesh: {N_verts} verts, {N_faces} faces")

# Treat everything as cloth, nothing as human body
split = {
    "num_joint_v":           np.int64(0),
    "num_joint_f":           np.int64(0),
    "reordered_cloth_v_idx": np.arange(N_verts, dtype=np.int64),
    "reordered_cloth_f_idx": np.arange(N_faces, dtype=np.int64),
    "reordered_human_v_idx": np.array([], dtype=np.int64),
    "reordered_human_f_idx": np.array([], dtype=np.int64),
    "new_cloth_faces":       faces.astype(np.int64),
    "new_human_faces":       np.zeros((0, 3), dtype=np.int64),
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.savez(OUT_PATH, **split)
print(f"Saved → {OUT_PATH}")
