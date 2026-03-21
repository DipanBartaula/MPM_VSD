"""
Generate a synthetic cam_info.json for ActorsHQ Actor1 Sequence1.
Run this on Lightning before training when cam_info.json is missing.

Usage:
    python gen_cam_info.py
"""
import numpy as np
import json
import os
import math

PRETRAINED  = "/teamspace/studios/this_studio/pretrained_models"
DATASET_DIR = f"{PRETRAINED}/output/tracking/a1_s1_460_200"
OUT_PATH    = f"{DATASET_DIR}/a1_s1/cam_info.json"

# Read N cameras from params file
params  = np.load(f"{DATASET_DIR}/params_460.npz")
N       = params['cam_m'].shape[0]
print(f"N cameras from params: {N}")

cam_info = {}
for i in range(N):
    # Distribute cameras evenly in a horizontal ring, slightly elevated
    theta = 2 * math.pi * i / N
    phi   = math.radians(30)   # ~30 deg elevation (typical capture dome)
    r     = 3.5                # ~3.5 m radius

    pos = np.array([
        r * math.cos(phi) * math.cos(theta),
        r * math.sin(phi) + 0.8,           # 0.8 m above ground (waist height)
        r * math.cos(phi) * math.sin(theta),
    ])

    look_at = np.array([0.0, 0.8, 0.0])   # looking at human centre of mass
    fwd  = look_at - pos;  fwd  /= np.linalg.norm(fwd)
    rgt  = np.cross(fwd, [0, 1, 0]); rgt /= np.linalg.norm(rgt)
    up   = np.cross(rgt, fwd)

    c2w = np.eye(4)
    c2w[:3, 0] = rgt
    c2w[:3, 1] = up
    c2w[:3, 2] = -fwd   # camera looks along -Z in OpenCV convention
    c2w[:3, 3] = pos

    W, H = 1920, 1080
    cam_info[f"Cam{i:03d}"] = {
        "W":  W,
        "H":  H,
        "K":  [[1000.0, 0.0, W/2], [0.0, 1000.0, H/2], [0.0, 0.0, 1.0]],
        "RT": c2w.tolist(),
    }

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(cam_info, f)

print(f"Written {N} synthetic cameras → {OUT_PATH}")
