"""
Generate per-frame smplx_fitted OBJ files from params_*.npz + split_idx.npz.
The SMPLX body vertices are extracted from the full mesh using split_idx.

Run on Lightning:
    python gen_smplx_fitted.py
"""
import numpy as np
import os
import glob

PRETRAINED   = "/teamspace/studios/this_studio/pretrained_models"
TRACKING_DIR = f"{PRETRAINED}/output/tracking/a1_s1_460_200"
OUT_DIR      = f"{TRACKING_DIR}/a1_s1/smplx_fitted"

# split_idx.npz not available — use all vertices/faces (full mesh as body proxy)
# This is sufficient for SDS training since we only need a rough body shape for LBS
human_v_idx = None
human_f_idx = None

# Find all params files
params_files = sorted(glob.glob(f"{TRACKING_DIR}/params_*.npz"))
print(f"Found {len(params_files)} params files")

for pf in params_files:
    frame = int(os.path.basename(pf).replace("params_", "").replace(".npz", ""))
    data  = np.load(pf)
    verts = data["vertices"]          # [N_verts, 3]
    faces = data["faces"]             # [N_faces, 3]

    # Use full mesh as body proxy (no split_idx available)
    body_verts      = verts
    body_faces_local = faces

    # Write OBJ
    out_path = os.path.join(OUT_DIR, f"{frame:06d}", "smplx_icp.obj")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for v in body_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in body_faces_local:
            if np.all(tri >= 0):
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    print(f"  frame {frame:06d} → {out_path}  ({len(body_verts)} verts)")

print("Done.")
