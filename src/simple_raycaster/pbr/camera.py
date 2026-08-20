"""Standalone PBR RGB-D camera. Does not modify :class:`RaycastCamera`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import trimesh
import warp as wp

from ..helpers import quat_rotate, quat_rotate_inverse, workspace_tensor
from ..raycast_camera import CameraIntrinsics, RaycastCamera
from .hdri import EnvironmentHDRI, default_hdri_path
from .kernels import raycast_gbuffer_kernel, raycast_ortho_depth_kernel, shadow_ray_kernel
from .materials import materials_for_names, pack_vertex_attrs, pack_vertex_attrs_from_wp
from .shade import aces_tonemap, contact_shadows, fxaa, sample_shadow_map, shade_pbr, ssao
from .tiled_filters import fxaa_tiled, ssao_tiled


class RaycastPBRCamera:
    """Closest-hit G-buffer + PyTorch PBR / HDRI.

    Bind geometry the same way as :class:`RaycastCamera`. ``render`` returns
    the same ``(rgb, depth, mask)`` shapes so it can be A/B'd against Lambert.
    RGB is ACES-tonemapped LDR; linear HDR is available via ``return_hdr=True``.

    Default ``quality="fast"`` is GGX + HDRI + ACES + FXAA (no SSAO / shadows).
    Use ``quality="pretty"`` for dumps / viewers (SSAO + 128² sun shadow map).
    Contact shadows and shadow rays stay opt-in; contact alone is a poor
    shadow proxy (see ``_pbr_camera.md`` probe).

    ``fxaa_impl`` / ``ssao_impl``: ``"tiled"`` (Warp shared-memory tiles) or
    ``"torch"``. Default FXAA is tiled (~1.7× vs Torch @ N=256); default SSAO
    stays Torch (tiled SSAO is correct but slower at 128×96 / 4 taps).
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
        sun_color: tuple[float, float, float] = (1.0, 0.98, 0.92),
        exposure: float = 0.85,
        quality: str = "fast",
        ssao_enabled: bool | None = None,
        ssao_radius: float = 0.08,
        contact_shadows_enabled: bool | None = None,
        contact_shadow_length: float = 0.08,
        fxaa_enabled: bool = True,
        fxaa_impl: str = "tiled",
        ssao_impl: str = "torch",
        smooth_normals: bool = True,
        shadow_map_enabled: bool | None = None,
        shadow_map_size: int = 128,
        shadow_map_extent: float = 2.0,
        shadow_map_catch: float = 4.0,
        shadow_rays: bool = False,
        shadow_ray_tmax: float = 8.0,
    ) -> None:
        q = str(quality).lower().strip()
        if q not in ("fast", "pretty"):
            raise ValueError(f"quality must be 'fast' or 'pretty', got {quality!r}")
        pretty = q == "pretty"
        if ssao_enabled is None:
            ssao_enabled = pretty
        if contact_shadows_enabled is None:
            # Contact ≠ real shadows; keep off unless explicitly requested.
            contact_shadows_enabled = False
        if shadow_map_enabled is None:
            shadow_map_enabled = pretty

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
        self.quality = q
        self.default_roughness = float(default_roughness)
        self.default_metallic = float(default_metallic)
        self.sun_dir = tuple(float(x) for x in sun_dir)
        self.sun_intensity = float(sun_intensity)
        self.sun_color = tuple(float(x) for x in sun_color)
        self.exposure = float(exposure)
        self.ssao_enabled = bool(ssao_enabled)
        self.ssao_radius = float(ssao_radius)
        self.contact_shadows_enabled = bool(contact_shadows_enabled)
        self.contact_shadow_length = float(contact_shadow_length)
        self.fxaa_enabled = bool(fxaa_enabled)
        fxaa_i = str(fxaa_impl).lower().strip()
        ssao_i = str(ssao_impl).lower().strip()
        if fxaa_i not in ("tiled", "torch"):
            raise ValueError(f"fxaa_impl must be 'tiled' or 'torch', got {fxaa_impl!r}")
        if ssao_i not in ("tiled", "torch"):
            raise ValueError(f"ssao_impl must be 'tiled' or 'torch', got {ssao_impl!r}")
        self.fxaa_impl = fxaa_i
        self.ssao_impl = ssao_i
        self.smooth_normals = bool(smooth_normals)
        self.shadow_map_enabled = bool(shadow_map_enabled)
        self.shadow_map_size = int(shadow_map_size)
        self.shadow_map_extent = float(shadow_map_extent)
        self.shadow_map_catch = float(shadow_map_catch)
        self.shadow_rays = bool(shadow_rays)
        self.shadow_ray_tmax = float(shadow_ray_tmax)
        self._work: dict = {}
        self._rough_t: torch.Tensor | None = None
        self._metal_t: torch.Tensor | None = None
        self._vert_color_t: torch.Tensor | None = None
        self._vert_normal_t: torch.Tensor | None = None
        self._vert_offset_t: torch.Tensor | None = None
        self._rough_wp = None
        self._metal_wp = None
        self._vert_color_wp = None
        self._vert_normal_wp = None
        self._vert_offset_wp = None
        self._mesh_names: list[str] = []
        self._last_mesh_pos: torch.Tensor | None = None
        self._last_mesh_quat: torch.Tensor | None = None
        self._last_cam_pos: torch.Tensor | None = None
        self._last_cam_quat: torch.Tensor | None = None
        if isinstance(hdri, EnvironmentHDRI):
            self.env = hdri
        else:
            self.env = EnvironmentHDRI.from_path(
                hdri if hdri is not None else default_hdri_path(),
                device=self.geom.torch_device,
            )
        # Cached for CUDA-graph-safe shade (no host torch.tensor during capture).
        self._sun_dir_t = torch.tensor(
            self.sun_dir, device=self.geom.torch_device, dtype=torch.float32
        )
        self._sun_color_t = torch.tensor(
            self.sun_color, device=self.geom.torch_device, dtype=torch.float32
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
        names: Sequence[str] | None = None,
        pose_entity: Any | None = None,
        pose_body_ids: Sequence[int] | None = None,
    ) -> None:
        kept: list[trimesh.Trimesh] = []
        kept_idx: list[int] = []
        for i, mesh in enumerate(meshes):
            if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                continue
            kept.append(mesh)
            kept_idx.append(i)
        if names is not None:
            if len(names) == len(meshes):
                self._mesh_names = [str(names[i]) for i in kept_idx]
            elif len(names) == len(kept):
                self._mesh_names = [str(n) for n in names]
            else:
                raise ValueError(
                    f"names length {len(names)} != meshes {len(meshes)} or kept {len(kept)}"
                )
        else:
            self._mesh_names = [""] * len(kept)

        self.geom.bind_trimeshes(
            meshes, albedos=albedos, pose_entity=pose_entity, pose_body_ids=pose_body_ids
        )
        self._pack_from_trimeshes(kept)
        if roughness is None and metallic is None and any(self._mesh_names):
            r, m = materials_for_names(
                self._mesh_names,
                default_roughness=self.default_roughness,
                default_metallic=self.default_metallic,
            )
            self._set_materials(self.geom.n_meshes, r, m)
        else:
            self._set_materials(self.geom.n_meshes, roughness, metallic)

    def bind_meshes(
        self,
        raycaster: Any,
        *,
        names: Sequence[str] | None = None,
        albedos: Sequence | torch.Tensor | None = None,
        roughness: Sequence | torch.Tensor | float | None = None,
        metallic: Sequence | torch.Tensor | float | None = None,
    ) -> None:
        self.geom.bind_meshes(raycaster)
        n = self.geom.n_meshes
        if albedos is None:
            src_alb = getattr(raycaster, "mesh_albedos", None)
            if src_alb is not None and len(src_alb) == n:
                albedos = src_alb
        if albedos is not None:
            alb = torch.as_tensor(albedos, device=self.geom.torch_device, dtype=torch.float32)
            alb = alb.reshape(-1, 3)
            if alb.shape[0] != n:
                raise ValueError(f"albedos length {alb.shape[0]} != n_meshes {n}")
            self.geom._albedo_t = alb.contiguous()
            self.geom._albedo_wp = None
            self.geom.initialized = False
        if names is not None:
            if len(names) != n:
                raise ValueError(f"names length {len(names)} != n_meshes {n}")
            self._mesh_names = [str(x) for x in names]
        else:
            src = getattr(raycaster, "mesh_names", None)
            self._mesh_names = [str(x) for x in src] if src is not None and len(src) == n else [""] * n
        self._pack_from_wp(self.geom.meshes_wp)
        if roughness is None and metallic is None and any(self._mesh_names):
            r, m = materials_for_names(
                self._mesh_names,
                default_roughness=self.default_roughness,
                default_metallic=self.default_metallic,
            )
            self._set_materials(n, r, m)
        else:
            self._set_materials(n, roughness, metallic)

    def _pack_from_trimeshes(self, meshes: Sequence[trimesh.Trimesh]) -> None:
        assert self.geom._albedo_t is not None
        alb = self.geom._albedo_t.detach().float().cpu().numpy()
        colors, normals, offsets = pack_vertex_attrs(meshes, alb)
        self._set_vertex_attrs(colors, normals, offsets)

    def _pack_from_wp(self, meshes_wp: Sequence) -> None:
        assert self.geom._albedo_t is not None
        alb = self.geom._albedo_t.detach().float().cpu().numpy()
        colors, normals, offsets = pack_vertex_attrs_from_wp(meshes_wp, alb)
        self._set_vertex_attrs(colors, normals, offsets)

    def _set_vertex_attrs(
        self, colors: np.ndarray, normals: np.ndarray, offsets: np.ndarray
    ) -> None:
        device = self.geom.torch_device
        self._vert_color_t = torch.as_tensor(colors, device=device, dtype=torch.float32)
        self._vert_normal_t = torch.as_tensor(normals, device=device, dtype=torch.float32)
        self._vert_offset_t = torch.as_tensor(offsets, device=device, dtype=torch.int32)
        self._vert_color_wp = None
        self._vert_normal_wp = None
        self._vert_offset_wp = None
        self.geom.initialized = False

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
        assert self._vert_color_t is not None and self._vert_normal_t is not None
        assert self._vert_offset_t is not None
        self._rough_wp = wp.from_torch(self._rough_t, dtype=wp.float32)
        self._metal_wp = wp.from_torch(self._metal_t, dtype=wp.float32)
        self._vert_color_wp = wp.from_torch(self._vert_color_t, dtype=wp.vec3)
        self._vert_normal_wp = wp.from_torch(self._vert_normal_t, dtype=wp.vec3)
        self._vert_offset_wp = wp.from_torch(self._vert_offset_t, dtype=wp.int32)

    def _cached(self, key: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        return workspace_tensor(self._work, key, shape, dtype, self.geom.torch_device)

    def _planar_depth(self, gb: dict[str, torch.Tensor]) -> torch.Tensor:
        """Camera-Z for SSAO / contact shadows (converts ray depth if needed)."""
        depth = gb["depth"]
        if self.intrinsics.depth_mode == "planar":
            return depth
        n, h, w = depth.shape
        fx, fy, cx, cy = self.intrinsics.resolved_k()
        ys, xs = torch.meshgrid(
            torch.arange(h, device=depth.device, dtype=torch.float32),
            torch.arange(w, device=depth.device, dtype=torch.float32),
            indexing="ij",
        )
        u = (xs + 0.5 - cx) / fx
        v = (ys + 0.5 - cy) / fy
        if self.intrinsics.convention == "opencv":
            dir_c = torch.stack([u, v, torch.ones_like(u)], dim=-1)
            optical_z = dir_c[..., 2] / dir_c.norm(dim=-1).clamp_min(1e-8)
        else:
            dir_c = torch.stack([u, -v, -torch.ones_like(u)], dim=-1)
            optical_z = -dir_c[..., 2] / dir_c.norm(dim=-1).clamp_min(1e-8)
        return depth * optical_z.unsqueeze(0).expand(n, -1, -1)

    def _hit_positions_w(
        self,
        gb: dict[str, torch.Tensor],
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
    ) -> torch.Tensor:
        n, h, w = gb["depth"].shape
        fx, fy, cx, cy = self.intrinsics.resolved_k()
        device = gb["depth"].device
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        u = (xs + 0.5 - cx) / fx
        v = (ys + 0.5 - cy) / fy
        if self.intrinsics.convention == "opencv":
            dir_c = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        else:
            dir_c = torch.stack([u, -v, -torch.ones_like(u)], dim=-1)
        dir_c = dir_c / dir_c.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dir_w = quat_rotate(
            cam_quat_wxyz[:, None, None, :].expand(n, h, w, 4),
            dir_c.unsqueeze(0).expand(n, -1, -1, -1),
        )
        if self.intrinsics.depth_mode == "planar":
            optical_z = dir_c[..., 2] if self.intrinsics.convention == "opencv" else -dir_c[..., 2]
            t = gb["depth"] / optical_z.unsqueeze(0).clamp_min(1e-8)
        else:
            t = gb["depth"]
        return cam_pos_w[:, None, None, :] + dir_w * t.unsqueeze(-1)

    def _sun_basis(self, sun: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forward = -sun / sun.norm().clamp_min(1e-8)
        world_up = torch.tensor([0.0, 0.0, 1.0], device=sun.device, dtype=sun.dtype)
        if forward[2].abs() > 0.95:
            world_up = torch.tensor([0.0, 1.0, 0.0], device=sun.device, dtype=sun.dtype)
        right = torch.linalg.cross(world_up, forward)
        right = right / right.norm().clamp_min(1e-8)
        up = torch.linalg.cross(forward, right)
        up = up / up.norm().clamp_min(1e-8)
        return right, up, forward

    def _render_shadow_map(self, n: int, sun: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._last_mesh_pos is not None and self._last_mesh_quat is not None
        size = self.shadow_map_size
        right, up, forward = self._sun_basis(sun)
        center = torch.zeros(n, 3, device=self.geom.torch_device, dtype=torch.float32)
        depth = self._cached("smap", (n, size * size), torch.float32)
        wp.launch(
            raycast_ortho_depth_kernel,
            dim=(n, size * size),
            inputs=[
                self.geom.meshes_array,
                wp.from_torch(self._last_mesh_pos, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self._last_mesh_quat, dtype=wp.vec4, return_ctype=True),
                wp.from_torch(center, dtype=wp.vec3, return_ctype=True),
                wp.vec3(float(right[0]), float(right[1]), float(right[2])),
                wp.vec3(float(up[0]), float(up[1]), float(up[2])),
                wp.vec3(float(forward[0]), float(forward[1]), float(forward[2])),
                float(self.shadow_map_extent),
                float(self.shadow_map_catch),
                int(size),
                float(self.intrinsics.near),
                float(self.intrinsics.far),
            ],
            outputs=[wp.from_torch(depth, dtype=wp.float32, return_ctype=True)],
            device=self.geom.device,
            record_tape=False,
            block_dim=self.geom.block_dim,
        )
        return depth.view(n, size, size), center, right, up, forward

    def _render_shadow_rays(
        self,
        gb: dict[str, torch.Tensor],
        hit_w: torch.Tensor,
        sun: torch.Tensor,
    ) -> torch.Tensor:
        assert self._last_mesh_pos is not None and self._last_mesh_quat is not None
        n, h, w = gb["mask"].shape
        p = h * w
        vis = self._cached("sray", (n, p), torch.float32)
        wp.launch(
            shadow_ray_kernel,
            dim=(n, p),
            inputs=[
                self.geom.meshes_array,
                wp.from_torch(self._last_mesh_pos, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(self._last_mesh_quat, dtype=wp.vec4, return_ctype=True),
                wp.from_torch(hit_w.reshape(n, p, 3).contiguous(), dtype=wp.vec3, return_ctype=True),
                wp.from_torch(gb["mask"].reshape(n, p).contiguous(), dtype=wp.bool, return_ctype=True),
                wp.vec3(float(sun[0]), float(sun[1]), float(sun[2])),
                2.0e-3,
                float(self.shadow_ray_tmax),
            ],
            outputs=[wp.from_torch(vis, dtype=wp.float32, return_ctype=True)],
            device=self.geom.device,
            record_tape=False,
            block_dim=self.geom.block_dim,
        )
        return vis.view(n, h, w)

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
        if not self.geom.initialized or self._rough_wp is None or self._vert_color_wp is None:
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
        self._last_mesh_pos = mesh_pos
        self._last_mesh_quat = mesh_quat
        self._last_cam_pos = cam_pos_w.contiguous()
        self._last_cam_quat = cam_quat_wxyz.contiguous()

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
                self._rough_wp,
                self._metal_wp,
                self._vert_color_wp,
                self._vert_normal_wp,
                self._vert_offset_wp,
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
                1 if self.smooth_normals else 0,
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
            sun = self._sun_dir_t
        elif isinstance(light_dir, torch.Tensor):
            sun = light_dir.to(device=self.geom.torch_device, dtype=torch.float32).reshape(3)
        else:
            sun = torch.tensor(light_dir, device=self.geom.torch_device, dtype=torch.float32)

        vis = torch.ones_like(gb["depth"])
        fx, fy, cx, cy = self.intrinsics.resolved_k()
        planar = self._planar_depth(gb)
        need_hits = self.shadow_map_enabled or self.shadow_rays
        hit_w = None
        if need_hits:
            hit_w = self._hit_positions_w(gb, cam_pos_w.contiguous(), cam_quat_wxyz.contiguous())
            gb["position"] = hit_w

        if self.contact_shadows_enabled:
            sun_cam = quat_rotate_inverse(
                cam_quat_wxyz.contiguous(),
                sun.view(1, 3).expand_as(cam_pos_w),
            )
            vis = vis * contact_shadows(
                planar,
                gb["mask"],
                sun_cam,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                length=self.contact_shadow_length,
            )

        if self.shadow_map_enabled:
            assert hit_w is not None
            sm, center, right, up, forward = self._render_shadow_map(cam_pos_w.shape[0], sun)
            vis = vis * sample_shadow_map(
                hit_w,
                gb["mask"],
                sm,
                center_w=center,
                right_w=right,
                up_w=up,
                forward_w=forward,
                half_extent=self.shadow_map_extent,
                catch_dist=self.shadow_map_catch,
            )

        if self.shadow_rays:
            assert hit_w is not None
            vis = vis * self._render_shadow_rays(gb, hit_w, sun)

        hdr = shade_pbr(
            gb["albedo"],
            gb["normal"],
            gb["view"],
            gb["roughness"],
            gb["metallic"],
            self.env,
            sun_dir_w=sun,
            sun_intensity=self.sun_intensity,
            sun_color=self._sun_color_t,
            sun_visibility=vis,
        )
        bg = self.env.sample(-gb["view"], roughness=0.0)
        hdr = torch.where(gb["mask"].unsqueeze(-1), hdr, bg)

        hdr = self._compose_indirect(
            hdr,
            gb,
            planar,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            cam_quat_wxyz=cam_quat_wxyz.contiguous(),
        )

        rgb = aces_tonemap(hdr, exposure=self.exposure)
        if self.fxaa_enabled:
            if self.fxaa_impl == "tiled":
                rgb = fxaa_tiled(rgb)
            else:
                rgb = fxaa(rgb)
        gb["sun_visibility"] = vis
        if return_hdr and return_gbuffer:
            return rgb, gb["depth"], gb["mask"], hdr, gb
        if return_gbuffer:
            return rgb, gb["depth"], gb["mask"], gb
        if return_hdr:
            return rgb, gb["depth"], gb["mask"], hdr
        return rgb, gb["depth"], gb["mask"]

    def _compose_indirect(
        self,
        hdr: torch.Tensor,
        gb: dict[str, torch.Tensor],
        planar: torch.Tensor,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        cam_quat_wxyz: torch.Tensor,
    ) -> torch.Tensor:
        """Optional screen-space occlusion / GI after direct shade. Default: SSAO."""
        del cam_quat_wxyz
        if not self.ssao_enabled:
            return hdr
        if self.ssao_impl == "tiled":
            ao = ssao_tiled(
                planar,
                gb["mask"],
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                radius=self.ssao_radius,
            )
        else:
            ao = ssao(
                planar,
                gb["mask"],
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                radius=self.ssao_radius,
            )
        return torch.where(gb["mask"].unsqueeze(-1), hdr * ao.unsqueeze(-1), hdr)
