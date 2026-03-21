"""
bridge_sds/runner_mpmavatar.py
================================
MPMAvatar simulation + Gaussian rendering runner.

Given physics parameters  φ = {D, E, H}  and a frame window, this module:
  1. Resets the MPM state to the initial cloth configuration.
  2. Runs the cloth simulation for `frame_num` frames using MPMAvatar's
     Warp-based p2g2p solver (mirroring train_material_params.py).
  3. After each simulated frame, injects the simulated cloth vertices into
     the pretrained MeshGaussianModel and renders a photorealistic RGB image.
  4. Returns the rendered frames as a  [T, H, W, 3]  float32 tensor in [0,1]
     together with simulation aux data for regulariser computation.

Design principles
-----------------
- No MPMAvatar core files are modified.
- All MPMAvatar modules are imported by adding the parent directory to
  sys.path at the top of this file.
- The Gaussian model is loaded from the appearance-training checkpoint and
  kept frozen (eval mode, no grad).
- Vertex injection uses a save/restore context manager so the frozen model
  is never permanently modified.

Coordinate convention
---------------------
MPM operates in a normalised sim space ≈ [0,1]³.
World space is whatever coordinate system the dataset uses.
Transform:  sim = wld * scale + shift
            wld = (sim - shift) / scale
"""

from __future__ import annotations

import contextlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── Add MPMAvatar root to Python path ────────────────────────────────────────
BRIDGE_ROOT   = Path(__file__).resolve().parent
MPMAVATAR_ROOT = BRIDGE_ROOT.parent
if str(MPMAVATAR_ROOT) not in sys.path:
    sys.path.insert(0, str(MPMAVATAR_ROOT))

# ── MPMAvatar imports ─────────────────────────────────────────────────────────
import warp as wp  # NVIDIA Warp GPU physics

from scene import Scene
from scene.mesh_gaussian_model import MeshGaussianModel
from gaussian_renderer import render as gs_render
from warp_mpm.mpm_solver import MPMSolver
from warp_mpm.mpm_data_structure import MPMModelStruct, MPMStateStruct
from utils.general_utils import (
    read_obj,
    read_ply,
    rotation_activation,
    scaling_activation,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """
    Output of a single  simulate_clip()  call.

    Attributes
    ----------
    cloth_verts_seq : per-frame simulated cloth vertices  [n_cloth, 3]  (world)
    body_verts_seq  : per-frame GT body vertices          [n_body,  3]  (world)
    cloth_vels_seq  : per-frame cloth vertex velocities   [n_cloth, 3]
    frame_indices   : global frame indices that were simulated
    phi             : the physics parameters used
    n_frames        : number of frames actually simulated (may be < frame_num
                      if sequence runs short)
    """
    cloth_verts_seq: List[torch.Tensor] = field(default_factory=list)
    body_verts_seq:  List[torch.Tensor] = field(default_factory=list)
    cloth_vels_seq:  List[torch.Tensor] = field(default_factory=list)
    frame_indices:   List[int]          = field(default_factory=list)
    phi:             Dict[str, float]   = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.cloth_verts_seq)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class MPMAvatarRunner:
    """
    Bridge between MPMAvatar's physics + appearance pipeline and the SDS loop.

    Call order
    ----------
    1.  runner = MPMAvatarRunner(args, device)
    2.  runner.setup()                              # one-time initialisation
    3.  result = runner.render_clip(phi, ...)       # per-iteration
        or
        sim    = runner.simulate_clip(phi, ...)     # if you only need sim data

    Thread safety
    -------------
    The runner is NOT thread-safe because it mutates shared Warp arrays and
    temporarily modifies the Gaussian model during rendering.  Run one runner
    per GPU process.
    """

    def __init__(self, args, device: str = "cuda:0"):
        self.args   = args
        self.device = device

        # Set after setup()
        self._setup_done  = False
        self.scene:     Optional[Scene]              = None
        self.gaussians: Optional[MeshGaussianModel]  = None
        self.pipe:      Optional[Any]                = None
        self.bg_color:  Optional[torch.Tensor]       = None
        self.cameras:   List[Any]                    = []

        # MPM solver components
        self.mpm_solver: Optional[MPMSolver]         = None
        self.mpm_model:  Optional[MPMModelStruct]    = None
        self.mpm_state:  Optional[MPMStateStruct]    = None

        # Simulation geometry (set during setup)
        self.scale: float                         = 1.0
        self.shift: Optional[torch.Tensor]        = None
        self.n_elements:    int                   = 0   # element (face-centre) particles
        self.n_cloth_verts: int                   = 0   # vertex particles
        self._n_particles:  int                   = 0

        # Cloth mesh
        self.cloth_faces:      Optional[torch.Tensor] = None  # [n_faces, 3]
        self._cloth_rest_verts: Optional[torch.Tensor] = None  # [n_cloth, 3] (unscaled)

        # Index mappings
        self.cloth_v_idx:  Optional[torch.Tensor] = None  # cloth verts in full mesh
        self.body_v_idx:   Optional[torch.Tensor] = None  # body  verts in full mesh
        self.joint_v_idx:  Optional[torch.Tensor] = None  # cloth verts attached to body
        self.reordered_cloth_v_idx: Optional[np.ndarray] = None

        # Per-frame GT data loaded at setup
        self.train_frame_smplx:      Optional[torch.Tensor] = None  # [T, n_body, 3]
        self.train_frame_smplx_velo: Optional[torch.Tensor] = None  # [T, n_body, 3]
        self.train_frame_verts:      Optional[torch.Tensor] = None  # [T, n_cloth, 3]
        self.train_frame_verts_velo: Optional[torch.Tensor] = None  # [T, n_cloth, 3]

        # Initial MPM particle state (numpy, for fast reset)
        self._init_particle_x: Optional[np.ndarray] = None
        self._init_particle_v: Optional[np.ndarray] = None

    # =========================================================================
    # Public API
    # =========================================================================

    def setup(self) -> None:
        """
        Load all MPMAvatar components.  Must be called once before any
        simulate_clip() or render_clip() calls.
        """
        logger.info("MPMAvatarRunner.setup() starting …")
        self._setup_gaussians()
        self._setup_physics_data()
        self._setup_mpm_solver()
        self._setup_cameras()
        self._setup_done = True
        logger.info("MPMAvatarRunner.setup() complete.")

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate_clip(
        self,
        phi:         Dict[str, float],
        frame_start: int,
        frame_num:   int,
    ) -> SimResult:
        """
        Simulate `frame_num` frames starting from global frame `frame_start`
        using physics parameters  φ = {D, E, H}.

        Returns a SimResult with per-frame cloth/body vertex positions and
        cloth vertex velocities.
        """
        self._assert_setup()
        self._apply_phi(phi)
        self._reset_mpm_state(phi.get("H", 1.0))

        args        = self.args
        device      = self.device
        substeps    = getattr(args, "substep", 400)
        fps         = getattr(args, "fps", 25.0)
        delta_time  = 1.0 / fps
        substep_dt  = delta_time / substeps
        n_substeps  = substeps  # by definition

        # Map global frame_start to index into loaded training data
        train_start = self.args.train_frame_start_num[0]
        local_start = max(0, frame_start - train_start)
        local_end   = min(local_start + frame_num, self.train_frame_smplx.shape[0])
        actual_num  = local_end - local_start

        if actual_num < frame_num:
            logger.warning(
                f"Requested {frame_num} frames but only {actual_num} available "
                f"(local_start={local_start}, data_len={self.train_frame_smplx.shape[0]}). "
                "Simulating truncated clip."
            )

        result = SimResult(phi=dict(phi))

        for i in range(actual_num):
            li = local_start + i  # local frame index

            # Body pose and velocity (sim space)
            mesh_x_wld = self.train_frame_smplx[li]          # [n_body, 3]
            mesh_v_wld = self.train_frame_smplx_velo[li]     # [n_body, 3]
            mesh_x = self._wld2sim(mesh_x_wld)               # normalise to sim space
            mesh_v = mesh_v_wld * self.scale

            # Constrained cloth vertices: those attached to the body
            joint_verts_v = (
                self.train_frame_verts_velo[li][self.joint_v_idx] * self.scale
            )                                                  # [n_joint, 3]

            # Joint faces: average face velocity for constrained faces
            joint_faces_v = self._compute_joint_faces_v(joint_verts_v)

            # ── Substep loop ──────────────────────────────────────────────
            for sub in range(n_substeps):
                mesh_x_curr = mesh_x + substep_dt * sub * mesh_v
                self.mpm_solver.p2g2p(
                    self.mpm_model,
                    self.mpm_state,
                    substep_dt,
                    mesh_x=mesh_x_curr,
                    mesh_v=mesh_v,
                    joint_traditional_v=None,
                    joint_verts_v=joint_verts_v,
                    joint_faces_v=joint_faces_v,
                    device=device,
                )

            # ── Extract results ───────────────────────────────────────────
            particle_x = wp.to_torch(self.mpm_state.particle_x).clone()
            particle_v = wp.to_torch(self.mpm_state.particle_v).clone()

            # Cloth vertex particles start at index n_elements
            cloth_v_sim = particle_x[self.n_elements:]           # [n_cloth, 3] sim
            cloth_vel   = particle_v[self.n_elements:]           # [n_cloth, 3] sim

            cloth_v_wld = self._sim2wld(cloth_v_sim)            # back to world
            cloth_vel_wld = cloth_vel / self.scale               # sim vel → world vel

            result.cloth_verts_seq.append(cloth_v_wld)
            result.body_verts_seq.append(mesh_x_wld.clone())
            result.cloth_vels_seq.append(cloth_vel_wld)
            result.frame_indices.append(frame_start + i)

        return result

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_clip(
        self,
        phi:            Dict[str, float],
        frame_start:    int,
        frame_num:      int,
        camera_indices: List[int],
        montage:        bool = False,
    ) -> Dict[str, Any]:
        """
        Simulate a clip and render it using the pretrained Gaussian model.

        Args:
            phi            : {'D': float, 'E': float, 'H': float}
            frame_start    : global frame index to begin from
            frame_num      : number of frames to simulate/render
            camera_indices : list of camera indices (from test_camera_index)
            montage        : tile multiple views into a single frame if True

        Returns dict:
            'frames'     : [T, H, W, 3]  float32 in [0, 1]
            'masks'      : [T, H, W, 1]  float32
            'sim_result' : SimResult (for regulariser computation)
        """
        self._assert_setup()

        sim_result = self.simulate_clip(phi, frame_start, frame_num)

        cams = self._get_cameras(camera_indices)
        if not cams:
            raise RuntimeError(
                f"No valid cameras for indices {camera_indices}. "
                f"Available: {len(self.cameras)}. "
                "Check test_camera_index and that scene was loaded correctly."
            )

        all_rgb:   List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []

        for cloth_v, body_v in zip(
            sim_result.cloth_verts_seq, sim_result.body_verts_seq
        ):
            if montage and len(cams) > 1:
                rgb, mask = self._render_montage(cloth_v, body_v, cams)
            else:
                rgb, mask = self._render_single(cloth_v, body_v, cams[0])

            all_rgb.append(rgb)
            all_masks.append(mask)

        frames = torch.stack(all_rgb,   dim=0)  # [T, H, W, 3]
        masks  = torch.stack(all_masks, dim=0)  # [T, H, W, 1]

        return {"frames": frames, "masks": masks, "sim_result": sim_result}

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def cloth_rest_verts(self) -> torch.Tensor:
        """Rest-pose cloth vertices [n_cloth, 3] (unscaled, world space)."""
        return self._cloth_rest_verts

    @property
    def n_training_frames(self) -> int:
        if self.train_frame_smplx is None:
            return 0
        return self.train_frame_smplx.shape[0]

    # =========================================================================
    # Setup helpers (private)
    # =========================================================================

    def _normalize_scene_args(self) -> None:
        """Populate argument defaults expected by MPMAvatar Scene."""
        defaults = {
            "white_bkgd": bool(getattr(self.args, "white_background", True)),
            "image_downscale_ratio": 1.0,
            "test_take": getattr(self.args, "train_take", 1),
        }
        for key, value in defaults.items():
            if not hasattr(self.args, key) or getattr(self.args, key) is None:
                setattr(self.args, key, value)

    def _setup_gaussians(self) -> None:
        """Load the appearance-trained MeshGaussianModel (frozen)."""
        self._normalize_scene_args()

        # gaussian_renderer only needs these three flags.
        self.pipe = SimpleNamespace(
            compute_cov3D_python=False,
            convert_SHs_python=False,
            debug=False,
        )

        self.gaussians = MeshGaussianModel(
            sh_degree=self.args.sh_degree,
            args=self.args,
        )

        trained_path = Path(self.args.trained_model_path)
        if not trained_path.exists():
            raise FileNotFoundError(
                f"Appearance model not found: {trained_path}\n"
                "Run train_appearance.py first and pass --trained_model_path."
            )

        self.scene = Scene(
            args=self.args,
            gaussians=self.gaussians,
            return_type="image",
            device=self.device,
        )

        # Freeze the Gaussian model — we only render, never train it here
        self.gaussians.eval()
        for p in self.gaussians.parameters():
            p.requires_grad_(False)

        white_bg = bool(getattr(self.args, "white_bkgd", True))
        self.bg_color = torch.ones(3, device=self.device) if white_bg \
                   else torch.zeros(3, device=self.device)

        logger.info(f"Gaussians loaded from: {trained_path}")

    def _setup_physics_data(self) -> None:
        """
        Load SMPLX body meshes and tracked cloth vertices for all training
        frames.  Mirrors PhysicsTrainer.__init__ in train_material_params.py.
        """
        args   = self.args
        device = self.device

        # ── Split indices (cloth / body separation) ───────────────────────
        split = np.load(args.split_idx_path)
        self.cloth_v_idx = torch.from_numpy(split["cloth_idx"]).long().to(device)
        self.body_v_idx  = torch.from_numpy(split["body_idx" ]).long().to(device)

        if "reordered_cloth_v_idx" in split:
            self.reordered_cloth_v_idx = split["reordered_cloth_v_idx"]
        else:
            self.reordered_cloth_v_idx = np.arange(len(self.cloth_v_idx))

        # ── Load per-frame data ────────────────────────────────────────────
        train_start, train_num = args.train_frame_start_num
        logger.info(
            f"Loading training frames {train_start} … "
            f"{train_start + train_num} ({train_num + 1} total incl. overlap)."
        )

        smplx_list:  List[torch.Tensor] = []
        cloth_list:  List[torch.Tensor] = []

        for i in range(train_num + 1):  # +1 for finite-difference velocity
            frame_idx = train_start + i
            smplx_v, cloth_v = self._load_frame(frame_idx)
            smplx_list.append(smplx_v)
            cloth_list.append(cloth_v)

        smplx_t = torch.stack(smplx_list)  # [T+1, n_body,  3]
        cloth_t = torch.stack(cloth_list)  # [T+1, n_cloth, 3]

        fps = getattr(args, "fps", 25.0)
        smplx_velo = (smplx_t[1:] - smplx_t[:-1]) * fps   # [T, n_body,  3]
        cloth_velo = (cloth_t[1:] - cloth_t[:-1]) * fps    # [T, n_cloth, 3]

        self.train_frame_smplx      = smplx_t[:-1].to(device)
        self.train_frame_smplx_velo = smplx_velo.to(device)
        self.train_frame_verts      = cloth_t[:-1].to(device)
        self.train_frame_verts_velo = cloth_velo.to(device)

        logger.info(
            f"Loaded {train_num} frames | "
            f"body_verts={smplx_t.shape[1]} | "
            f"cloth_verts={cloth_t.shape[1]}"
        )

    def _load_frame(self, frame_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single frame's SMPLX body verts and cloth verts.

        Returns:
            smplx_v : [n_body,  3]  world space
            cloth_v : [n_cloth, 3]  world space
        """
        args = self.args
        dtype = torch.float32

        if args.dataset_type.lower() in ("actorshq", "actorshq_a"):
            # Primary path: {frame:06d}/smplx_icp.obj
            base = (
                Path(args.dataset_dir)
                / f"a{args.actor}_s{args.sequence}"
                / "smplx_fitted"
            )
            obj_path = base / f"{frame_idx:06d}" / "smplx_icp.obj"
            if not obj_path.exists():
                # Fallback: flat file
                obj_path = base / f"{frame_idx:06d}.obj"
            if not obj_path.exists():
                primary = base / f"{frame_idx:06d}" / "smplx_icp.obj"
                fallback = base / f"{frame_idx:06d}.obj"
                raise FileNotFoundError(
                    f"SMPLX mesh not found for frame {frame_idx}. "
                    f"Tried:\n  {primary}\n"
                    f"  {fallback}"
                )

        elif args.dataset_type.lower() in ("4ddress", "4d-dress"):
            dress_root = self._resolve_4ddress_root(Path(args.dataset_dir))
            obj_path = (
                dress_root
                / f"{args.subject:05d}_Inner"
                / "Inner"
                / f"Take{args.train_take}"
                / "SMPLX"
                / f"mesh-f{frame_idx:05d}_smplx.ply"
            )
            if not obj_path.exists():
                raise FileNotFoundError(
                    f"SMPLX mesh not found for frame {frame_idx}: {obj_path}"
                )

        else:
            raise ValueError(
                f"Unknown dataset_type='{args.dataset_type}'. "
                "Supported: 'actorshq', '4ddress'."
            )

        if obj_path.suffix.lower() == ".ply":
            verts, _ = read_ply(str(obj_path))
        else:
            verts, _ = read_obj(str(obj_path))
        verts_t  = torch.from_numpy(verts).to(dtype)

        # NOTE: the full mesh has body+cloth.  Use split indices correctly.
        body_v  = verts_t[self.body_v_idx.cpu()]
        cloth_v = verts_t[self.cloth_v_idx.cpu()]

        return body_v, cloth_v

    def _resolve_4ddress_root(self, dataset_dir: Path) -> Path:
        """
        Resolve 4D-DRESS root robustly.

        Supports:
        - <dataset_dir>/4D-DRESS
        - <dataset_dir>/4D-Dress
        - <dataset_dir> itself being the 4D-DRESS folder
        """
        candidates = [
            dataset_dir / "4D-DRESS",
            dataset_dir / "4D-Dress",
            dataset_dir,
        ]
        for cand in candidates:
            if cand.exists() and any(cand.glob("*_Inner")):
                return cand
        raise FileNotFoundError(
            "Could not locate 4D-DRESS root from --dataset_dir.\n"
            f"Tried:\n  {dataset_dir / '4D-DRESS'}\n"
            f"  {dataset_dir / '4D-Dress'}\n"
            f"  {dataset_dir}"
        )

    def _setup_mpm_solver(self) -> None:
        """
        Initialise MPM solver, model, and state.
        Mirrors PhysicsTrainer.setup_simulation() in train_material_params.py.
        """
        args   = self.args
        device = self.device

        # Load cloth UV (rest-pose) mesh
        uv_verts_np, uv_faces_np = read_obj(args.uv_path)
        uv_verts = torch.from_numpy(uv_verts_np).float().to(device)
        uv_faces = torch.from_numpy(uv_faces_np).long().to(device)

        self.cloth_faces        = uv_faces
        self._cloth_rest_verts  = uv_verts.clone()
        self.n_cloth_verts      = uv_verts.shape[0]

        # World → sim normalisation: map cloth bounding box to [0.1, 0.9]³
        self._compute_sim_transform(uv_verts)

        cloth_sim = self._wld2sim(uv_verts)                   # [n_cloth, 3]

        # Element particles: face centres in sim space
        face_verts  = uv_verts[uv_faces]                      # [n_faces, 3, 3]
        elem_centres = face_verts.mean(dim=1)                 # [n_faces, 3]
        elem_sim    = self._wld2sim(elem_centres)

        self.n_elements  = elem_sim.shape[0]
        self._n_particles = self.n_elements + self.n_cloth_verts

        all_particles = torch.cat([elem_sim, cloth_sim], dim=0)   # [N, 3]

        # Joint vertices (cloth verts constrained to follow body at attachment)
        self._init_joint_verts()

        # Initialise Warp solver
        grid_size = getattr(args, "grid_size", 200)
        self.mpm_solver = MPMSolver(
            n_particles=self._n_particles,
            n_grid=grid_size,
            device=device,
        )

        self.mpm_model = MPMModelStruct()
        self.mpm_solver.initialize_model(
            self.mpm_model,
            material=7,                          # cloth material in MPMAvatar
            n_particles=self._n_particles,
            n_elements=self.n_elements,
            n_vertices=self.n_cloth_verts,
            faces=uv_faces.cpu().numpy(),
            device=device,
        )

        # Set default body mesh collider geometry
        body_verts_init = self.train_frame_smplx[0]           # [n_body, 3]
        body_verts_sim  = self._wld2sim(body_verts_init)
        self.mpm_solver.set_mesh_collider(
            self.mpm_model,
            body_verts_sim.cpu().numpy(),
            mesh_friction_coeff=getattr(args, "mesh_friction_coeff", 0.5),
            device=device,
        )

        # Initial particle state (numpy for fast CPU→GPU copying on reset)
        init_x_np = all_particles.cpu().numpy().astype(np.float32)
        init_v_np = np.zeros_like(init_x_np)

        self.mpm_state = MPMStateStruct()
        self.mpm_solver.initialize_state(
            self.mpm_model,
            self.mpm_state,
            init_x_np,
            init_v_np,
            device=device,
        )

        self._init_particle_x = init_x_np.copy()
        self._init_particle_v = init_v_np.copy()

        # Apply default material params
        self._apply_phi({
            "D": getattr(args, "init_D", 1.0),
            "E": getattr(args, "init_E", 100.0),
            "H": getattr(args, "init_H", 1.0),
        })

        logger.info(
            f"MPM solver ready | "
            f"n_particles={self._n_particles} | "
            f"n_elements={self.n_elements} | "
            f"n_cloth_verts={self.n_cloth_verts} | "
            f"grid={grid_size}"
        )

    def _init_joint_verts(self) -> None:
        """
        Identify cloth vertices that are constrained to follow the body
        (attachment joints).

        Tries to load from args.joint_v_idx_path.  If not provided, falls back
        to a simple heuristic: the cloth vertices closest to the first
        training frame's body, up to `n_joint_fallback` vertices.
        """
        args   = self.args
        device = self.device

        jpath = getattr(args, "joint_v_idx_path", None)
        if jpath and Path(jpath).exists():
            self.joint_v_idx = (
                torch.from_numpy(np.load(jpath)).long().to(device)
            )
            logger.info(
                f"Joint vertices loaded from: {jpath} "
                f"({len(self.joint_v_idx)} verts)"
            )
            return

        # ── Heuristic fallback ─────────────────────────────────────────────
        n_joint = max(8, self.n_cloth_verts // 20)  # ~5 % of cloth verts
        logger.warning(
            f"joint_v_idx_path not provided or not found.  "
            f"Using heuristic: top-{n_joint} cloth vertices nearest to body "
            "for joint constraints.  Pass --joint_v_idx_path for best accuracy."
        )

        cloth_v  = self._cloth_rest_verts                   # [n_cloth, 3]
        body_v   = self.train_frame_smplx[0].to(cloth_v.device)  # [n_body, 3]

        # Pairwise distances cloth → body; pick closest n_joint cloth verts
        # Sub-sample body to keep memory bounded
        n_sample = min(512, body_v.shape[0])
        samp_idx = torch.linspace(0, body_v.shape[0] - 1, n_sample).long().to(device)
        body_sub = body_v[samp_idx]

        dists     = torch.cdist(cloth_v, body_sub)          # [n_cloth, n_sample]
        min_dists = dists.min(dim=1).values                 # [n_cloth]
        topk      = min_dists.topk(n_joint, largest=False)  # smallest distances
        self.joint_v_idx = topk.indices.to(device)

    def _setup_cameras(self) -> None:
        """Load Camera objects for the test_camera_index list."""
        if self.scene is None:
            logger.warning("Scene not loaded; cannot load cameras.")
            return

        test_cams_all: List[Any] = []
        try:
            test_cams_all = list(self.scene.test_dataset.camera_list)
        except Exception:
            logger.warning("Could not retrieve cameras from scene.test_dataset.camera_list.")

        self.cameras = test_cams_all
        logger.info(f"Cameras available: {len(self.cameras)}")

    # =========================================================================
    # Physics parameter control (private)
    # =========================================================================

    def _apply_phi(self, phi: Dict[str, float]) -> None:
        """
        Push φ = {D, E, H} into the live MPM model.

        Mirrors the parameter update logic in train_material_params.py.
        """
        if self.mpm_model is None or self.mpm_solver is None:
            return  # Called before setup; silently skip

        D = float(phi.get("D", 1.0))
        E = float(phi.get("E", 100.0))

        nu    = float(getattr(self.args, "init_nu",    0.3))
        gamma = float(getattr(self.args, "init_gamma", 500.0))
        kappa = float(getattr(self.args, "init_kappa", 500.0))

        # MPMAvatar stores E as (E * 100); here E is already in that unit
        youngs = E * 100.0

        self.mpm_solver.set_E_nu_from_torch(
            self.mpm_model,
            torch.tensor(youngs, device=self.device),
            torch.tensor(nu,     device=self.device),
            torch.tensor(gamma,  device=self.device),
            torch.tensor(kappa,  device=self.device),
            self.device,
        )
        self.mpm_solver.prepare_mu_lam(
            self.mpm_model, self.mpm_state, self.device
        )

        self.mpm_state.reset_density(
            torch.tensor(D, device=self.device),
            self.device,
            update_mass=True,
        )

        self._current_H = float(phi.get("H", 1.0))

    def _reset_mpm_state(self, H: float = 1.0) -> None:
        """
        Reset the MPM particle state to the rest configuration, applying
        the height-scale parameter H to the cloth y-coordinates.
        """
        init_x = self._init_particle_x.copy()
        init_v = self._init_particle_v.copy()

        # H scales the vertical (y) coordinate of cloth vertex particles
        if H != 1.0:
            init_x[self.n_elements:, 1] *= H   # cloth verts start at n_elements

        self.mpm_state.reset_state(
            wp.array(init_x, dtype=wp.vec3, device=self.device),
            wp.array(init_v, dtype=wp.vec3, device=self.device),
            device=self.device,
        )

    # =========================================================================
    # Coordinate transforms (private)
    # =========================================================================

    def _compute_sim_transform(self, verts: torch.Tensor) -> None:
        """
        Compute scale + shift to map world-space cloth into MPM sim space.
        Target: all cloth vertices map to [0.1, 0.9]³.

        Matches the normalisation convention in train_material_params.py.
        """
        v_min   = verts.min(dim=0).values
        v_max   = verts.max(dim=0).values
        extent  = (v_max - v_min).max().item()
        self.scale = 0.8 / extent
        self.shift = (0.5 - (v_min + v_max) * 0.5 * self.scale).to(verts.device)

    def _wld2sim(self, x: torch.Tensor) -> torch.Tensor:
        """World → simulation space."""
        return x * self.scale + self.shift.to(x.device)

    def _sim2wld(self, x: torch.Tensor) -> torch.Tensor:
        """Simulation → world space."""
        return (x - self.shift.to(x.device)) / self.scale

    # =========================================================================
    # Rendering helpers (private)
    # =========================================================================

    def _render_single(
        self,
        cloth_verts: torch.Tensor,   # [n_cloth, 3] world
        body_verts:  torch.Tensor,   # [n_body,  3] world
        camera: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render one frame from one camera.  Returns (rgb [H,W,3], mask [H,W,1])."""
        with self._inject_sim_verts(cloth_verts, body_verts):
            with torch.no_grad():
                pkg = gs_render(
                    viewpoint_camera=camera,
                    pc=self.gaussians,
                    pipe=self.pipe,
                    bg_color=self.bg_color,
                )

        rgb  = pkg["render"].permute(1, 2, 0).clamp(0, 1)   # [H, W, 3]
        mask = pkg["mask" ].permute(1, 2, 0).clamp(0, 1)    # [H, W, 1]
        return rgb, mask

    def _render_montage(
        self,
        cloth_verts: torch.Tensor,
        body_verts:  torch.Tensor,
        cameras: List[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render from multiple cameras and tile into a single montage frame."""
        rgbs:  List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        for cam in cameras:
            rgb, mask = self._render_single(cloth_verts, body_verts, cam)
            rgbs.append(rgb)
            masks.append(mask)

        # Square grid tiling
        n    = len(rgbs)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        H, W, C = rgbs[0].shape
        dev = self.device

        while len(rgbs) < rows * cols:
            rgbs.append(torch.zeros(H, W, C,  device=dev))
            masks.append(torch.zeros(H, W, 1, device=dev))

        row_rgb = [torch.cat(rgbs [r*cols:(r+1)*cols], dim=1) for r in range(rows)]
        row_msk = [torch.cat(masks[r*cols:(r+1)*cols], dim=1) for r in range(rows)]
        return torch.cat(row_rgb, dim=0), torch.cat(row_msk, dim=0)

    @contextlib.contextmanager
    def _inject_sim_verts(
        self,
        cloth_verts: torch.Tensor,
        body_verts:  torch.Tensor,
    ):
        """
        Context manager: temporarily replace Gaussian model vertices with
        the simulated cloth + GT body vertices so we can render them.

        The Gaussian model stores `verts_orig[timestep]` for all timesteps.
        We use slot 0 as a scratch buffer:
          1. Save verts_orig[0] and the derived face geometry tensors.
          2. Build a combined vertex array replacing cloth part with sim verts.
          3. Recompute face centres/orientations/scales from the new verts.
          4. Render (caller's code runs here).
          5. Restore everything.

        If the Gaussian model does not have the expected attributes (e.g. API
        has changed), we fall back to rendering without vertex injection and
        log a warning.
        """
        gs = self.gaussians
        if gs is None or not hasattr(gs, "verts_orig"):
            logger.warning(
                "Gaussian model missing 'verts_orig' attribute — "
                "rendering with GT/default vertices (sim injection skipped)."
            )
            yield
            return

        # ── Save state ────────────────────────────────────────────────────
        saved_v0    = gs.verts_orig[0].clone()
        saved_fc    = gs.face_center.clone()   if hasattr(gs, "face_center")     else None
        saved_om    = gs.face_orien_mat.clone()if hasattr(gs, "face_orien_mat")  else None
        saved_oq    = gs.face_orien_quat.clone()if hasattr(gs,"face_orien_quat") else None
        saved_fs    = gs.face_scaling.clone()  if hasattr(gs, "face_scaling")    else None
        saved_ts    = getattr(gs, "current_timestep", 0)

        # ── Inject ────────────────────────────────────────────────────────
        combined = saved_v0.clone()
        dev      = combined.device

        # Replace cloth and body parts with current-frame data
        # cloth_v_idx / body_v_idx index into the full mesh vertex array
        combined[self.cloth_v_idx.to(dev)] = cloth_verts.to(dev)
        combined[self.body_v_idx.to(dev)]  = body_verts.to(dev)

        gs.verts_orig[0] = combined
        if hasattr(gs, "current_timestep"):
            gs.current_timestep = 0

        # Recompute face geometry so Gaussian positions are correct
        self._update_gs_face_geometry(combined)

        try:
            yield
        finally:
            # ── Restore ───────────────────────────────────────────────────
            gs.verts_orig[0] = saved_v0
            if saved_fc is not None: gs.face_center     = saved_fc
            if saved_om is not None: gs.face_orien_mat  = saved_om
            if saved_oq is not None: gs.face_orien_quat = saved_oq
            if saved_fs is not None: gs.face_scaling    = saved_fs
            if hasattr(gs, "current_timestep"):
                gs.current_timestep = saved_ts

    def _update_gs_face_geometry(self, verts: torch.Tensor) -> None:
        """
        Recompute face centres, orientations, and scales in the Gaussian model
        from the given vertex positions.

        Called inside _inject_sim_verts to ensure the renderer sees the
        correct geometry.
        """
        gs    = self.gaussians
        faces = gs.faces.to(verts.device)          # [n_faces, 3]

        # ── Face centres ──────────────────────────────────────────────────
        if hasattr(gs, "face_center"):
            face_v = verts[faces]                   # [n_faces, 3, 3]
            gs.face_center = face_v.mean(dim=1)     # [n_faces, 3]

        # ── Face orientations ─────────────────────────────────────────────
        if hasattr(gs, "face_orien_mat") and gs.face_orien_mat is not None:
            try:
                oq = rotation_activation(verts, faces)  # [n_faces, 4] quaternions
                from utils.graphics_utils import compute_face_orientation
                om, oq2 = compute_face_orientation(verts, faces)[:2]
                gs.face_orien_mat  = om.to(verts.device)
                gs.face_orien_quat = oq2.to(verts.device)
            except Exception as e:
                logger.debug(f"Face orientation update skipped: {e}")

        # ── Face scales ───────────────────────────────────────────────────
        if hasattr(gs, "face_scaling") and gs.face_scaling is not None:
            try:
                gs.face_scaling = scaling_activation(verts, faces).to(verts.device)
            except Exception as e:
                logger.debug(f"Face scaling update skipped: {e}")

    # =========================================================================
    # Misc helpers (private)
    # =========================================================================

    def _compute_joint_faces_v(
        self, joint_verts_v: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute per-face velocity for constrained cloth faces.

        For each cloth face that has a constrained vertex, the face velocity
        is the mean of its vertex velocities.  Mirrors the `joint_faces_v`
        computation in train_material_params.py.
        """
        if self.cloth_faces is None or len(self.cloth_faces) == 0:
            return None

        # Build a velocity array for all cloth vertices
        # (only joint_v_idx positions are known; others set to zero)
        cloth_vel = torch.zeros(
            self.n_cloth_verts, 3, device=self.device
        )
        cloth_vel[self.joint_v_idx] = joint_verts_v

        # Average vertex velocities per face
        f = self.cloth_faces                                 # [n_faces, 3]
        v0 = cloth_vel[f[:, 0]]
        v1 = cloth_vel[f[:, 1]]
        v2 = cloth_vel[f[:, 2]]
        face_vel = (v0 + v1 + v2) / 3.0                     # [n_faces, 3]
        return face_vel

    def _get_cameras(self, camera_indices: List[int]) -> List[Any]:
        """Return camera objects for the given indices."""
        if not self.cameras:
            return []
        return [
            self.cameras[i]
            for i in camera_indices
            if 0 <= i < len(self.cameras)
        ]

    def _assert_setup(self) -> None:
        if not self._setup_done:
            raise RuntimeError("Call MPMAvatarRunner.setup() before use.")
