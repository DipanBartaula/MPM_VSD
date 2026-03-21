"""
Generate a placeholder a1s1_uv.obj for Actor1 Sequence1.
Packs faces into a regular grid UV layout — good enough for shadow net inference.

Run on Lightning:
    python gen_uv_obj.py
"""
import numpy as np
import os

PRETRAINED = "/teamspace/studios/this_studio/pretrained_models"
PARAMS_NPZ = f"{PRETRAINED}/output/tracking/a1_s1_460_200/params_460.npz"
OUT_OBJ    = "/teamspace/studios/this_studio/mpm_avatar_vds/data/a1_s1/a1s1_uv.obj"

params  = np.load(PARAMS_NPZ)
faces   = params['faces']       # [N_faces, 3]  0-indexed vertex indices
verts   = params['vertices']    # [N_verts, 3]
N_faces = len(faces)
N_verts = len(verts)

print(f"N_faces={N_faces}  N_verts={N_verts}")

# Grid dimensions for UV packing
GRID = int(np.ceil(np.sqrt(N_faces)))

lines = []

# Vertex positions (needed so face indices are valid)
for v in verts:
    lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

# UV coordinates: one per face, tightly packed in a GRID×GRID layout
# Each face gets 3 identical UVs (centroid of a small cell)
for i in range(N_faces):
    row = i // GRID
    col = i %  GRID
    u = (col + 0.5) / GRID
    v = (row + 0.5) / GRID
    lines.append(f"vt {u:.6f} {v:.6f}")   # vt index = i+1

# Faces: v/vt (all 3 corners of face i share the same UV index i+1)
for i, face in enumerate(faces):
    v1, v2, v3 = int(face[0])+1, int(face[1])+1, int(face[2])+1
    vt = i + 1
    lines.append(f"f {v1}/{vt} {v2}/{vt} {v3}/{vt}")

os.makedirs(os.path.dirname(OUT_OBJ), exist_ok=True)
with open(OUT_OBJ, "w") as f:
    f.write("\n".join(lines))

print(f"Written → {OUT_OBJ}")
