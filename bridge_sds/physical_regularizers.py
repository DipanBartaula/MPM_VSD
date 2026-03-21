"""
bridge_sds/physical_regularizers.py
=====================================
Lightweight physical regularisers computed directly from MPM simulation output.

None of these require re-running the simulator; they operate purely on the
vertex trajectories produced by SimResult.

Regularisers
------------
L_penetration      – cloth vertices that penetrate the body mesh
L_stretch          – edges stretched / compressed beyond a strain limit
L_temporal_smooth  – frame-to-frame cloth-vertex acceleration (jitter penalty)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── L_penetration ─────────────────────────────────────────────────────────────

def penetration_loss(
    cloth_verts_seq: List[torch.Tensor],
    body_verts_seq:  List[torch.Tensor],
    margin: float = 0.005,
    n_body_sample: int = 512,
) -> torch.Tensor:
    """
    Penalise cloth vertices that are closer than `margin` to the body surface.

    This is a proximity proxy rather than a true signed-distance check: for
    every cloth vertex we find the nearest sampled body vertex and penalise
    by relu(margin - dist)².  It is fast and differentiable.

    Args:
        cloth_verts_seq : T × [n_cloth, 3]  simulated cloth vertices per frame
        body_verts_seq  : T × [n_body, 3]   GT body vertices per frame
        margin          : safe-distance threshold (metres)
        n_body_sample   : number of body vertices to sample for efficiency

    Returns:
        Scalar loss tensor.
    """
    if not cloth_verts_seq:
        return torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu").squeeze()

    device = cloth_verts_seq[0].device
    total  = torch.zeros(1, device=device)

    for cloth_v, body_v in zip(cloth_verts_seq, body_verts_seq):
        body_v = body_v.to(device)

        # Sub-sample body vertices to avoid O(n_cloth × n_body) OOM
        n_body = body_v.shape[0]
        if n_body > n_body_sample:
            idx = torch.linspace(0, n_body - 1, n_body_sample, device=device).long()
            body_v = body_v[idx]

        # Pairwise L2 distances: [n_cloth, n_body_sample]
        dists    = torch.cdist(cloth_v, body_v, p=2)          # [n_cloth, n_sample]
        min_dist = dists.min(dim=1).values                     # [n_cloth]

        penetration = F.relu(margin - min_dist)                # [n_cloth]
        total = total + penetration.pow(2).mean()

    return total / max(len(cloth_verts_seq), 1)


# ── L_stretch ─────────────────────────────────────────────────────────────────

def stretch_loss(
    cloth_verts_seq: List[torch.Tensor],
    rest_verts:      torch.Tensor,
    cloth_faces:     torch.Tensor,
    max_strain:      float = 0.3,
) -> torch.Tensor:
    """
    Penalise triangle edges whose strain exceeds `max_strain`.

    strain_e = (current_length_e - rest_length_e) / rest_length_e
    loss      = mean  relu( |strain_e| - max_strain )²

    Args:
        cloth_verts_seq : T × [n_cloth, 3]  simulated cloth per frame
        rest_verts      : [n_cloth, 3]      rest-pose cloth vertices
        cloth_faces     : [n_faces, 3]      face connectivity (0-indexed)
        max_strain      : allowable relative strain before penalty kicks in

    Returns:
        Scalar loss tensor.
    """
    if not cloth_verts_seq:
        return torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu").squeeze()

    device = cloth_verts_seq[0].device
    rest   = rest_verts.to(device)
    faces  = cloth_faces.to(device)

    # Rest-pose edge lengths — computed once
    r0 = rest[faces[:, 0]]
    r1 = rest[faces[:, 1]]
    r2 = rest[faces[:, 2]]
    rest_len = torch.stack([
        (r1 - r0).norm(dim=-1),
        (r2 - r1).norm(dim=-1),
        (r0 - r2).norm(dim=-1),
    ], dim=-1)                        # [n_faces, 3]

    total = torch.zeros(1, device=device)

    for cloth_v in cloth_verts_seq:
        v = cloth_v.to(device)
        v0 = v[faces[:, 0]]
        v1 = v[faces[:, 1]]
        v2 = v[faces[:, 2]]
        cur_len = torch.stack([
            (v1 - v0).norm(dim=-1),
            (v2 - v1).norm(dim=-1),
            (v0 - v2).norm(dim=-1),
        ], dim=-1)                    # [n_faces, 3]

        strain = (cur_len - rest_len) / (rest_len + 1e-8)
        excess = F.relu(strain.abs() - max_strain)
        total  = total + excess.pow(2).mean()

    return total / max(len(cloth_verts_seq), 1)


# ── L_temporal_smooth ─────────────────────────────────────────────────────────

def temporal_smooth_loss(
    cloth_vels_seq: List[torch.Tensor],
) -> torch.Tensor:
    """
    Penalise frame-to-frame changes in cloth vertex velocity (acceleration).

    loss = mean || v_{t+1} - v_t ||²   (averaged over time and vertices)

    This discourages jittery motion that could fool the diffusion prior.

    Args:
        cloth_vels_seq : T × [n_cloth, 3]  cloth vertex velocities per frame

    Returns:
        Scalar loss tensor.
    """
    if len(cloth_vels_seq) < 2:
        return torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu").squeeze()

    device = cloth_vels_seq[0].device
    total  = torch.zeros(1, device=device)

    for i in range(1, len(cloth_vels_seq)):
        v_prev = cloth_vels_seq[i - 1]
        v_curr = cloth_vels_seq[i].to(device)
        accel  = v_curr - v_prev                        # [n_cloth, 3]
        total  = total + accel.pow(2).sum(dim=-1).mean()

    return total / (len(cloth_vels_seq) - 1)


# ── Combined interface ────────────────────────────────────────────────────────

def compute_all_regularizers(
    sim_result,
    cloth_faces:     torch.Tensor,
    rest_verts:      torch.Tensor,
    margin:          float = 0.005,
    max_strain:      float = 0.3,
) -> dict:
    """
    Compute all three regulariser losses from a SimResult.

    Args:
        sim_result  : SimResult dataclass from runner_mpmavatar.py
        cloth_faces : [n_faces, 3] cloth face connectivity
        rest_verts  : [n_cloth, 3] cloth rest-pose vertices
        margin      : penetration margin (metres)
        max_strain  : stretch strain threshold

    Returns:
        dict with keys 'penetration', 'stretch', 'temporal_smooth'
        (all scalar Python floats, detached from graph)
    """
    pen = penetration_loss(
        sim_result.cloth_verts_seq,
        sim_result.body_verts_seq,
        margin=margin,
    )

    str_l = stretch_loss(
        sim_result.cloth_verts_seq,
        rest_verts,
        cloth_faces,
        max_strain=max_strain,
    )

    ts = temporal_smooth_loss(sim_result.cloth_vels_seq)

    return {
        "penetration":     pen,
        "stretch":         str_l,
        "temporal_smooth": ts,
    }
