"""Backend-agnostic Warp RGB-D camera over triangle meshes.

**Role:** nvdiffrast alternative for depth-compositing dynamic meshes (e.g. robot)
over 3DGS in active-adaptation ``FvdbGaussianWorld``. Scene sensors use
``LambertRaycastCameraSensor`` + :class:`~active_adaptation.envs.mesh_registry.MeshRegistry`.

Inspired by ``mujoco_warp`` ``render.py`` / ``compute_ray``, but unbound from
``MjModel`` / ``MjData``: bind ``wp.Mesh`` lists (or a
:class:`MultiMeshRaycaster`) and pass camera + mesh poses each frame.

Example::

    cam = RaycastCamera(64, 48, fov_y_deg=70.0, convention="opencv")
    cam.bind_trimeshes(body_trimeshes, albedos=albedos)
    rgb, depth, mask = cam.render(cam_pos_w, cam_quat_wxyz,
                                  mesh_pos_w=..., mesh_quat_w=...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import torch
import trimesh
import warp as wp

from .helpers import trimesh2wp, workspace_tensor
from .kernels import raycast_camera_kernel

DepthMode = Literal["planar", "ray"]
CameraConvention = Literal["opencv", "opengl"]
CameraOutput = Literal["depth", "rgb", "normal", "seg"]


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole parameters (shared across envs)."""

    width: int
    height: int
    fov_y_deg: float
    near: float = 0.05
    far: float = 50.0
    convention: CameraConvention = "opencv"
    depth_mode: DepthMode = "planar"
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None

    def resolved_k(self) -> tuple[float, float, float, float]:
        """Return ``(fx, fy, cx, cy)``."""
        w, h = self.width, self.height
        cx = float(self.cx) if self.cx is not None else 0.5 * w
        cy = float(self.cy) if self.cy is not None else 0.5 * h
        if self.fx is not None and self.fy is not None:
            return float(self.fx), float(self.fy), cx, cy
        fy = h / (2.0 * math.tan(math.radians(self.fov_y_deg) * 0.5))
        fx = fy if self.fx is None else float(self.fx)
        return fx, fy, cx, cy


class RaycastCamera:
    """Warp megakernel RGB-D camera over posed triangle meshes."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        fov_y_deg: float = 70.0,
        near: float = 0.05,
        far: float = 50.0,
        convention: CameraConvention = "opencv",
        depth_mode: DepthMode = "planar",
        outputs: Sequence[CameraOutput] = ("rgb", "depth"),
        device: str | torch.device = "cuda",
        ambient: float = 0.30,
        diffuse: float = 0.80,
        light_dir: tuple[float, float, float] = (0.45, -0.35, 0.82),
        default_albedo: tuple[float, float, float] = (0.65, 0.65, 0.65),
    ) -> None:
        self.intrinsics = CameraIntrinsics(
            width=int(width),
            height=int(height),
            fov_y_deg=float(fov_y_deg),
            near=float(near),
            far=float(far),
            convention=convention,
            depth_mode=depth_mode,
        )
        self.outputs = tuple(outputs)
        if isinstance(device, torch.device):
            self.torch_device = device
            self.device = str(device)
        else:
            self.device = device
            self.torch_device = torch.device(device)
        self.ambient = float(ambient)
        self.diffuse = float(diffuse)
        self.light_dir = tuple(float(x) for x in light_dir)
        self.default_albedo = tuple(float(x) for x in default_albedo)

        self.meshes_wp: list[wp.Mesh] = []
        self.meshes_array: wp.array | None = None
        self._albedo_wp: wp.array | None = None
        self._albedo_t: torch.Tensor | None = None
        self.initialized = False

        self.bvh_constructor: str | None = None
        self.bvh_leaf_size: int | None = None
        self.block_dim = 128
        self._work: dict = {}

    @property
    def n_meshes(self) -> int:
        return len(self.meshes_wp)

    def bind_trimeshes(
        self,
        meshes: Sequence[trimesh.Trimesh],
        *,
        albedos: Sequence | torch.Tensor | None = None,
    ) -> None:
        """Upload body-local trimeshes as ``wp.Mesh`` (skips empty entries)."""
        kept: list[trimesh.Trimesh] = []
        for mesh in meshes:
            if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                continue
            kept.append(mesh)

        if not kept:
            raise ValueError("bind_trimeshes: no non-empty meshes")

        self.meshes_wp = [
            trimesh2wp(
                m,
                self.device,
                bvh_constructor=getattr(self, "bvh_constructor", None),
                bvh_leaf_size=getattr(self, "bvh_leaf_size", None),
            )
            for m in kept
        ]
        self.meshes_array = None
        self.initialized = False

        if albedos is None:
            alb = np.tile(np.asarray(self.default_albedo, dtype=np.float32), (len(kept), 1))
        else:
            alb = np.asarray(albedos, dtype=np.float32).reshape(-1, 3)
            if alb.shape[0] != len(kept):
                raise ValueError(
                    f"albedos length {alb.shape[0]} != kept meshes {len(kept)}"
                )
        self._albedo_t = torch.as_tensor(alb, device=self.torch_device, dtype=torch.float32)
        self._albedo_wp = None

    def bind_meshes(self, raycaster: Any) -> None:
        """Share meshes from a :class:`MultiMeshRaycaster` or mesh bind view."""
        if not hasattr(raycaster, "meshes_wp"):
            raise TypeError("raycaster must expose meshes_wp")
        self.meshes_wp = list(raycaster.meshes_wp)
        self.meshes_array = None
        self.initialized = False
        n = len(self.meshes_wp)
        src_alb = getattr(raycaster, "mesh_albedos", None)
        if src_alb is not None and len(src_alb) == n:
            self._albedo_t = torch.as_tensor(
                src_alb, device=self.torch_device, dtype=torch.float32
            ).reshape(n, 3).contiguous()
        else:
            self._albedo_t = (
                torch.tensor(self.default_albedo, device=self.torch_device, dtype=torch.float32)
                .expand(n, 3)
                .contiguous()
                .clone()
            )
        self._albedo_wp = None

    def initialize(self) -> None:
        if self.initialized:
            return
        if self.n_meshes == 0:
            raise RuntimeError("RaycastCamera has no meshes; call bind_* first")
        wp.init()
        self.meshes_array = wp.array(
            [m.id for m in self.meshes_wp], device=self.device, dtype=wp.uint64
        )
        assert self._albedo_t is not None
        self._albedo_wp = wp.from_torch(self._albedo_t.contiguous(), dtype=wp.vec3)
        self.initialized = True

    @torch.no_grad()
    def render(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        *,
        mesh_pos_w: torch.Tensor,
        mesh_quat_w: torch.Tensor,
        light_dir: torch.Tensor | tuple[float, float, float] | None = None,
        enabled: torch.Tensor | None = None,
        mesh_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render mesh RGB-D → ``rgb[N,H,W,3]``, ``depth[N,H,W]``, ``mask[N,H,W]``.

        Output buffers are reused across calls with the same ``N`` and resolution
        (overwritten on the next ``render``).
        """
        del enabled
        if mesh_indices is not None:
            raise NotImplementedError(
                "RaycastCamera megakernel does not yet support mesh_indices; "
                "bind a subset of meshes instead"
            )

        if not self.initialized:
            self.initialize()

        n = cam_pos_w.shape[0]
        w = self.intrinsics.width
        h = self.intrinsics.height
        p = w * h
        fx, fy, cx, cy = self.intrinsics.resolved_k()
        convention = 0 if self.intrinsics.convention == "opencv" else 1
        depth_mode = 0 if self.intrinsics.depth_mode == "planar" else 1

        mesh_pos, mesh_quat = mesh_pos_w, mesh_quat_w
        if mesh_pos.shape[0] != n or mesh_pos.shape[1] != self.n_meshes:
            raise ValueError(
                f"mesh_pos_w shape {tuple(mesh_pos.shape)} incompatible with "
                f"N={n}, n_meshes={self.n_meshes}"
            )
        if mesh_quat.shape != mesh_pos.shape[:-1] + (4,):
            raise ValueError(
                f"mesh_quat_w shape {tuple(mesh_quat.shape)} incompatible with "
                f"mesh_pos_w shape {tuple(mesh_pos.shape)}"
            )
        mesh_pos = mesh_pos.contiguous()
        mesh_quat = mesh_quat.contiguous()

        if light_dir is None:
            light = torch.tensor(self.light_dir, device=self.torch_device, dtype=torch.float32)
        elif isinstance(light_dir, torch.Tensor):
            light = light_dir.to(device=self.torch_device, dtype=torch.float32).reshape(3)
        else:
            light = torch.tensor(light_dir, device=self.torch_device, dtype=torch.float32)

        rgb_flat = workspace_tensor(self._work, "rgb", (n, p, 3), torch.float32, self.torch_device)
        depth_flat = workspace_tensor(self._work, "depth", (n, p), torch.float32, self.torch_device)
        mask_flat = workspace_tensor(self._work, "mask", (n, p), torch.bool, self.torch_device)

        wp.launch(
            raycast_camera_kernel,
            dim=(n, p),
            inputs=[
                self.meshes_array,
                wp.from_torch(mesh_pos, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(mesh_quat, dtype=wp.vec4, return_ctype=True),
                self._albedo_wp,
                wp.from_torch(cam_pos_w.contiguous(), dtype=wp.vec3, return_ctype=True),
                wp.from_torch(cam_quat_wxyz.contiguous(), dtype=wp.vec4, return_ctype=True),
                w,
                h,
                float(fx),
                float(fy),
                float(cx),
                float(cy),
                float(self.intrinsics.near),
                float(self.intrinsics.far),
                convention,
                depth_mode,
                wp.vec3(float(light[0]), float(light[1]), float(light[2])),
                self.ambient,
                self.diffuse,
            ],
            outputs=[
                wp.from_torch(rgb_flat, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(depth_flat, dtype=wp.float32, return_ctype=True),
                wp.from_torch(mask_flat, dtype=wp.bool, return_ctype=True),
            ],
            device=self.device,
            record_tape=False,
            block_dim=self.block_dim,
        )

        rgb = rgb_flat.view(n, h, w, 3)
        rgb.clamp_(0.0, 1.0)
        depth = depth_flat.view(n, h, w)
        mask = mask_flat.view(n, h, w)
        return rgb, depth, mask


__all__ = [
    "CameraIntrinsics",
    "CameraConvention",
    "CameraOutput",
    "DepthMode",
    "RaycastCamera",
]
