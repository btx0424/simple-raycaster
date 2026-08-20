"""Standalone PBR RGB-D camera. Does not modify :class:`RaycastCamera`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import trimesh
import warp as wp

from ..helpers import workspace_tensor
from ..raycast_camera import CameraIntrinsics, RaycastCamera
from .hdri import EnvironmentHDRI, default_hdri_path
from .kernels import raycast_gbuffer_kernel
from .shade import aces_tonemap, shade_pbr, ssao


class RaycastPBRCamera:
    """Closest-hit G-buffer + PyTorch PBR / HDRI / SSAO.

    Bind geometry the same way as :class:`RaycastCamera`. ``render`` returns
    the same ``(rgb, depth, mask)`` shapes so it can be A/B'd against Lambert.
    RGB is ACES-tonemapped LDR; linear HDR is available via ``return_hdr=True``.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        fov_y_deg: float = 70.0,
        near: float = 0.05,
        far: float = 50.0,
        convention: str = "opencv",
        depth_mode: str = "planar",
        device: str | torch.device = "cuda",
        hdri: str | Path | EnvironmentHDRI | None = None,
        default_albedo: tuple[float, float, float] = (0.65, 0.65, 0.65),
        default_roughness: float = 0.45,
        default_metallic: float = 0.05,
        sun_dir: tuple[float, float, float] = (0.45, -0.35, 0.82),
        sun_intensity: float = 2.5,
        exposure: float = 0.85,
        ssao_enabled: bool = True,
        ssao_radius: float = 0.12,
    ) -> None:
        self.geom = RaycastCamera(
            width,
            height,
            fov_y_deg=fov_y_deg,
            near=near,
            far=far,
            convention=convention,  # type: ignore[arg-type]
            depth_mode=depth_mode,  # type: ignore[arg-type]
            device=device,
            default_albedo=default_albedo,
        )
        self.default_roughness = float(default_roughness)
        self.default_metallic = float(default_metallic)
        self.sun_dir = tuple(float(x) for x in sun_dir)
        self.sun_intensity = float(sun_intensity)
        self.exposure = float(exposure)
        self.ssao_enabled = bool(ssao_enabled)
        self.ssao_radius = float(ssao_radius)
        self._work: dict = {}
        self._rough_t: torch.Tensor | None = None
        self._metal_t: torch.Tensor | None = None
        self._rough_wp = None
        self._metal_wp = None
        if isinstance(hdri, EnvironmentHDRI):
            self.env = hdri
        else:
            self.env = EnvironmentHDRI.from_path(
                hdri if hdri is not None else default_hdri_path(),
                device=self.geom.torch_device,
            )

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self.geom.intrinsics

    @property
    def n_meshes(self) -> int:
        return self.geom.n_meshes

    @property
    def device(self):
        return self.geom.device

    def set_resolution(self, width: int, height: int, *, fov_y_deg: float | None = None) -> None:
        self.geom.set_resolution(width, height, fov_y_deg=fov_y_deg)

    def bind_trimeshes(
        self,
        meshes: Sequence[trimesh.Trimesh],
        *,
        albedos: Sequence | torch.Tensor | None = None,
        roughness: Sequence | torch.Tensor | float | None = None,
        metallic: Sequence | torch.Tensor | float | None = None,
        pose_entity: Any | None = None,
        pose_body_ids: Sequence[int] | None = None,
    ) -> None:
        self.geom.bind_trimeshes(
            meshes, albedos=albedos, pose_entity=pose_entity, pose_body_ids=pose_body_ids
        )
        self._set_materials(self.geom.n_meshes, roughness, metallic)

    def bind_meshes(self, raycaster: Any) -> None:
        self.geom.bind_meshes(raycaster)
        self._set_materials(self.geom.n_meshes, None, None)

    def _set_materials(
        self,
        n: int,
        roughness: Sequence | torch.Tensor | float | None,
        metallic: Sequence | torch.Tensor | float | None,
    ) -> None:
        device = self.geom.torch_device

        def _vec(val, default: float) -> torch.Tensor:
            if val is None:
                t = torch.full((n,), default, device=device, dtype=torch.float32)
            elif torch.is_tensor(val):
                t = val.to(device=device, dtype=torch.float32).reshape(-1)
            elif np.isscalar(val):
                t = torch.full((n,), float(val), device=device, dtype=torch.float32)
            else:
                t = torch.as_tensor(val, device=device, dtype=torch.float32).reshape(-1)
            if t.numel() != n:
                raise ValueError(f"material length {t.numel()} != n_meshes {n}")
            return t.contiguous()

        self._rough_t = _vec(roughness, self.default_roughness)
        self._metal_t = _vec(metallic, self.default_metallic)
        self._rough_wp = None
        self._metal_wp = None
        self.geom.initialized = False

    def initialize(self) -> None:
        self.geom.initialize()
        assert self._rough_t is not None and self._metal_t is not None
        self._rough_wp = wp.from_torch(self._rough_t, dtype=wp.float32)
        self._metal_wp = wp.from_torch(self._metal_t, dtype=wp.float32)

    def _cached(self, key: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        return workspace_tensor(self._work, key, shape, dtype, self.geom.torch_device)

    @torch.no_grad()
    def render_gbuffer(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        *,
        mesh_pos_w: torch.Tensor | None = None,
        mesh_quat_w: torch.Tensor | None = None,
        origin_w: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        width: int | None = None,
        height: int | None = None,
        fov_y_deg: float | None = None,
    ) -> dict[str, torch.Tensor]:
        if width is not None or height is not None or fov_y_deg is not None:
            self.set_resolution(
                width or self.intrinsics.width,
                height or self.intrinsics.height,
                fov_y_deg=fov_y_deg,
            )
        if not self.geom.initialized or self._rough_wp is None:
            self.initialize()

        n = cam_pos_w.shape[0]
        w = self.intrinsics.width
        h = self.intrinsics.height
        p = w * h
        fx, fy, cx, cy = self.intrinsics.resolved_k()
        convention = 0 if self.intrinsics.convention == "opencv" else 1
        depth_mode = 0 if self.intrinsics.depth_mode == "planar" else 1
        mesh_pos, mesh_quat = self.geom._gather_mesh_poses(
            n,
            mesh_pos_w=mesh_pos_w,
            mesh_quat_w=mesh_quat_w,
            origin_w=origin_w,
            env_ids=env_ids,
        )

        depth = self._cached("depth", (n, p), torch.float32)
        mask = self._cached("mask", (n, p), torch.bool)
        normal = self._cached("nrm", (n, p, 3), torch.float32)
        albedo = self._cached("alb", (n, p, 3), torch.float32)
        rough = self._cached("rough", (n, p), torch.float32)
        metal = self._cached("metal", (n, p), torch.float32)
        view = self._cached("view", (n, p, 3), torch.float32)

        wp.launch(
            raycast_gbuffer_kernel,
            dim=(n, p),
            inputs=[
                self.geom.meshes_array,
                wp.from_torch(mesh_pos, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(mesh_quat, dtype=wp.vec4, return_ctype=True),
                self.geom._albedo_wp,
                self._rough_wp,
                self._metal_wp,
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
            ],
            outputs=[
                wp.from_torch(depth, dtype=wp.float32, return_ctype=True),
                wp.from_torch(mask, dtype=wp.bool, return_ctype=True),
                wp.from_torch(normal, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(albedo, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(rough, dtype=wp.float32, return_ctype=True),
                wp.from_torch(metal, dtype=wp.float32, return_ctype=True),
                wp.from_torch(view, dtype=wp.vec3, return_ctype=True),
            ],
            device=self.geom.device,
            record_tape=False,
            block_dim=self.geom.block_dim,
        )
        return {
            "depth": depth.view(n, h, w),
            "mask": mask.view(n, h, w),
            "normal": normal.view(n, h, w, 3),
            "albedo": albedo.view(n, h, w, 3),
            "roughness": rough.view(n, h, w),
            "metallic": metal.view(n, h, w),
            "view": view.view(n, h, w, 3),
        }

    @torch.no_grad()
    def render(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        *,
        mesh_pos_w: torch.Tensor | None = None,
        mesh_quat_w: torch.Tensor | None = None,
        origin_w: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        width: int | None = None,
        height: int | None = None,
        fov_y_deg: float | None = None,
        light_dir: torch.Tensor | tuple[float, float, float] | None = None,
        return_hdr: bool = False,
        return_gbuffer: bool = False,
    ):
        gb = self.render_gbuffer(
            cam_pos_w,
            cam_quat_wxyz,
            mesh_pos_w=mesh_pos_w,
            mesh_quat_w=mesh_quat_w,
            origin_w=origin_w,
            env_ids=env_ids,
            width=width,
            height=height,
            fov_y_deg=fov_y_deg,
        )
        if light_dir is None:
            sun = torch.tensor(self.sun_dir, device=self.geom.torch_device, dtype=torch.float32)
        elif isinstance(light_dir, torch.Tensor):
            sun = light_dir.to(device=self.geom.torch_device, dtype=torch.float32).reshape(3)
        else:
            sun = torch.tensor(light_dir, device=self.geom.torch_device, dtype=torch.float32)

        hdr = shade_pbr(
            gb["albedo"],
            gb["normal"],
            gb["view"],
            gb["roughness"],
            gb["metallic"],
            self.env,
            sun_dir_w=sun,
            sun_intensity=self.sun_intensity,
        )
        bg = self.env.sample(-gb["view"], roughness=0.0)
        hdr = torch.where(gb["mask"].unsqueeze(-1), hdr, bg)

        if self.ssao_enabled:
            fx, fy, cx, cy = self.intrinsics.resolved_k()
            ao = ssao(
                gb["depth"],
                gb["mask"],
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                radius=self.ssao_radius,
            )
            hdr = torch.where(gb["mask"].unsqueeze(-1), hdr * ao.unsqueeze(-1), hdr)

        rgb = aces_tonemap(hdr, exposure=self.exposure)
        if return_hdr and return_gbuffer:
            return rgb, gb["depth"], gb["mask"], hdr, gb
        if return_gbuffer:
            return rgb, gb["depth"], gb["mask"], gb
        if return_hdr:
            return rgb, gb["depth"], gb["mask"], hdr
        return rgb, gb["depth"], gb["mask"]
