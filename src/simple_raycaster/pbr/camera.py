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
from .kernels import (
    raycast_gbuffer_kernel,
    raycast_gbuffer_sparse_kernel,
    raycast_ortho_depth_kernel,
    shadow_ray_kernel,
)
from .materials import (
    materials_for_names,
    pack_vertex_normals,
    pack_vertex_normals_from_wp,
)
from .shade import aces_tonemap, contact_shadows, fxaa, sample_shadow_map, shade_pbr, ssao
from .textures import (
    load_albedo_texture,
    pack_vertex_uvs_from_wp,
)
from .tiled_filters import fxaa_tiled, ssao_tiled


class RaycastPBRCamera:
    """Closest-hit G-buffer + PyTorch PBR / HDRI.

    Bind geometry the same way as :class:`RaycastCamera`. ``render`` returns
    the same ``(rgb, depth, mask)`` shapes so it can be A/B'd against Lambert.
    RGB is ACES-tonemapped LDR; linear HDR is available via ``return_hdr=True``.

    ``render`` / ``render_gbuffer`` take a **pose ticket** (camera + mesh poses)
    and optional per-call material/light tables. Resolution, quality, HDRI, and
    textures are camera state (:meth:`set_resolution`, constructor,
    :meth:`set_ground_albedo`) — not ``render`` kwargs.

    :meth:`render_sparse` renders a subset of ``tile_size`` tiles
    (``tiles [B,T,2]`` indices) to packed ``[B,T,S,S,…]`` NHWC outputs (no FXAA/SSAO).

    Default ``quality="fast"`` is GGX + HDRI + ACES + FXAA + **sun shadow rays**
    (second closest-hit cast). Use ``quality="pretty"`` for dumps / viewers
    (SSAO + 128² sun shadow map; no shadow rays — do not stack both).
    Contact shadows stay opt-in; contact alone is a poor shadow proxy
    (see ``_pbr_camera.md`` probe).

    Round-7 defaults (@ 192×144, N=256): ``compile_mode="max-autotune-no-cudagraphs"``
    with Torch FXAA/SSAO (compiled shade+FXAA ~2× vs eager; tiled FXAA loses to
    compile). Pass ``compile_mode=None`` and ``fxaa_impl="tiled"`` for the
    no-compile full-frame path (still beats eager Torch FXAA).

    Materials/lights support mujoco_warp-style batched tables via
    :meth:`set_materials` / :meth:`set_lights` (``[Nw, M, …]`` with ``Nw==1``
    broadcast). G-buffer albedo/rough/metal come from those tables, not baked
    vertex colors, so DR is a tensor write without rebinding meshes.

    Optional UV albedo: :meth:`set_ground_albedo` (or :meth:`set_albedo_textures`
    + planar UVs) samples a diffuse map when ``mesh_tex_id[m] >= 0``, multiplied
    by the table albedo tint.
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
        fxaa_impl: str = "torch",
        ssao_impl: str = "torch",
        compile_mode: str | None = "max-autotune-no-cudagraphs",
        smooth_normals: bool = True,
        shadow_map_enabled: bool | None = None,
        shadow_map_size: int = 128,
        shadow_map_extent: float = 2.0,
        shadow_map_catch: float = 4.0,
        shadow_rays: bool | None = None,
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
            # Pretty / viewers: soft-looking ortho map. Fast RL: off (use sray).
            shadow_map_enabled = pretty
        if shadow_rays is None:
            # Fast RL: hard sun visibility via a second ray cast. Pretty: off
            # so we do not stack sray + smap by default.
            shadow_rays = not pretty

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
        if compile_mode is not None:
            compile_mode = str(compile_mode).strip()
            if not compile_mode:
                compile_mode = None
        self.compile_mode = compile_mode
        self.smooth_normals = bool(smooth_normals)
        self.shadow_map_enabled = bool(shadow_map_enabled)
        self.shadow_map_size = int(shadow_map_size)
        self.shadow_map_extent = float(shadow_map_extent)
        self.shadow_map_catch = float(shadow_map_catch)
        self.shadow_rays = bool(shadow_rays)
        self.shadow_ray_tmax = float(shadow_ray_tmax)
        self._work: dict = {}
        self._compiled_shade_fxaa = None
        self._compiled_shade_fxaa_key: tuple | None = None
        self._albedo_t: torch.Tensor | None = None  # [Nw, M, 3]
        self._rough_t: torch.Tensor | None = None  # [Nw, M]
        self._metal_t: torch.Tensor | None = None  # [Nw, M]
        self._vert_normal_t: torch.Tensor | None = None
        self._vert_offset_t: torch.Tensor | None = None
        self._vert_uv_t: torch.Tensor | None = None  # [V, 2]
        self._mesh_tex_id_t: torch.Tensor | None = None  # [Nw, M] int32; -1 = flat
        self._albedo_tex_t: torch.Tensor | None = None  # [T, H, W, 3]
        self._albedo_wp = None
        self._rough_wp = None
        self._metal_wp = None
        self._vert_normal_wp = None
        self._vert_offset_wp = None
        self._vert_uv_wp = None
        self._mesh_tex_id_wp = None
        self._albedo_tex_wp = None
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
        # Leading dim 1 broadcasts across worlds (mujoco_warp-style).
        device = self.geom.torch_device
        self._sun_dir_t = torch.tensor(
            self.sun_dir, device=device, dtype=torch.float32
        ).view(1, 3)
        self._sun_color_t = torch.tensor(
            self.sun_color, device=device, dtype=torch.float32
        ).view(1, 3)
        self._sun_intensity_t = torch.tensor(
            [self.sun_intensity], device=device, dtype=torch.float32
        )
        self._exposure_t = torch.tensor(
            [self.exposure], device=device, dtype=torch.float32
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
        n = self.geom.n_meshes
        base_alb = self.geom._albedo_t
        assert base_alb is not None
        if roughness is None and metallic is None and any(self._mesh_names):
            r, m = materials_for_names(
                self._mesh_names,
                default_roughness=self.default_roughness,
                default_metallic=self.default_metallic,
            )
            self.set_materials(albedo=base_alb, roughness=r, metallic=m)
        else:
            self.set_materials(albedo=base_alb, roughness=roughness, metallic=metallic)

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
        base_alb = self.geom._albedo_t
        if base_alb is None:
            base_alb = torch.tensor(
                self.geom.default_albedo, device=self.geom.torch_device, dtype=torch.float32
            ).view(1, 3).expand(n, 3).contiguous()
            self.geom._albedo_t = base_alb
        if roughness is None and metallic is None and any(self._mesh_names):
            r, m = materials_for_names(
                self._mesh_names,
                default_roughness=self.default_roughness,
                default_metallic=self.default_metallic,
            )
            self.set_materials(albedo=base_alb, roughness=r, metallic=m)
        else:
            self.set_materials(albedo=base_alb, roughness=roughness, metallic=metallic)

    def _pack_from_trimeshes(self, meshes: Sequence[trimesh.Trimesh]) -> None:
        normals, offsets = pack_vertex_normals(meshes)
        self._set_vertex_normals(normals, offsets)

    def _pack_from_wp(self, meshes_wp: Sequence) -> None:
        normals, offsets = pack_vertex_normals_from_wp(meshes_wp)
        self._set_vertex_normals(normals, offsets)

    def _set_vertex_normals(self, normals: np.ndarray, offsets: np.ndarray) -> None:
        device = self.geom.torch_device
        self._vert_normal_t = torch.as_tensor(normals, device=device, dtype=torch.float32)
        self._vert_offset_t = torch.as_tensor(offsets, device=device, dtype=torch.int32)
        # Flat albedo until set_ground_albedo / set_albedo_textures.
        n_verts = int(normals.shape[0])
        n_meshes = int(offsets.shape[0])
        self._vert_uv_t = torch.zeros(n_verts, 2, device=device, dtype=torch.float32)
        self._mesh_tex_id_t = torch.full(
            (1, n_meshes), -1, device=device, dtype=torch.int32
        )
        self._albedo_tex_t = torch.ones(1, 1, 1, 3, device=device, dtype=torch.float32)
        self._vert_normal_wp = None
        self._vert_offset_wp = None
        self._vert_uv_wp = None
        self._mesh_tex_id_wp = None
        self._albedo_tex_wp = None
        self.geom.initialized = False

    @staticmethod
    def _normalize_mesh_table(
        val: Sequence | torch.Tensor | float | None,
        *,
        n_meshes: int,
        default: float | tuple[float, ...],
        device: torch.device,
        n_feat: int | None = None,
    ) -> torch.Tensor:
        """Return a contiguous ``[Nw, M]`` or ``[Nw, M, F]`` table."""
        if val is None:
            if n_feat is None:
                t = torch.full((1, n_meshes), float(default), device=device, dtype=torch.float32)
            else:
                base = torch.as_tensor(default, device=device, dtype=torch.float32).reshape(n_feat)
                t = base.view(1, 1, n_feat).expand(1, n_meshes, n_feat).contiguous()
            return t

        t = torch.as_tensor(val, device=device, dtype=torch.float32)
        if n_feat is None:
            if t.ndim == 0:
                t = t.view(1, 1).expand(1, n_meshes)
            elif t.ndim == 1:
                if t.numel() != n_meshes:
                    raise ValueError(f"material length {t.numel()} != n_meshes {n_meshes}")
                t = t.view(1, n_meshes)
            elif t.ndim == 2:
                if t.shape[-1] != n_meshes:
                    raise ValueError(f"material shape {tuple(t.shape)} last dim != {n_meshes}")
            else:
                raise ValueError(f"expected scalar/[M]/[Nw,M], got {tuple(t.shape)}")
            return t.contiguous()

        if t.ndim == 1:
            if t.numel() != n_feat:
                raise ValueError(f"feature length {t.numel()} != {n_feat}")
            t = t.view(1, 1, n_feat).expand(1, n_meshes, n_feat)
        elif t.ndim == 2:
            if t.shape == (n_meshes, n_feat):
                t = t.unsqueeze(0)
            elif t.shape == (1, n_feat):
                t = t.view(1, 1, n_feat).expand(1, n_meshes, n_feat)
            else:
                raise ValueError(f"albedo shape {tuple(t.shape)} incompatible with M={n_meshes}")
        elif t.ndim == 3:
            if t.shape[-2:] != (n_meshes, n_feat):
                raise ValueError(f"albedo shape {tuple(t.shape)} incompatible with M={n_meshes}")
        else:
            raise ValueError(f"expected [M,3]/[Nw,M,3], got {tuple(t.shape)}")
        return t.contiguous()

    def set_materials(
        self,
        *,
        albedo: Sequence | torch.Tensor | None = None,
        roughness: Sequence | torch.Tensor | float | None = None,
        metallic: Sequence | torch.Tensor | float | None = None,
    ) -> None:
        """Update mesh albedo/roughness/metallic tables without rebinding geometry.

        Shapes: ``[M]`` / ``[M,3]``, or batched ``[Nw, M]`` / ``[Nw, M, 3]``.
        ``Nw==1`` broadcasts across camera batch rows (like mujoco_warp).
        """
        n = int(self.geom.n_meshes)
        if n <= 0:
            raise RuntimeError("bind meshes before set_materials")
        device = self.geom.torch_device
        if albedo is not None or self._albedo_t is None:
            src = albedo
            if src is None:
                src = self.geom._albedo_t
            if src is None:
                src = self.geom.default_albedo
            self._albedo_t = self._normalize_mesh_table(
                src, n_meshes=n, default=tuple(self.geom.default_albedo), device=device, n_feat=3
            )
        if roughness is not None or self._rough_t is None:
            self._rough_t = self._normalize_mesh_table(
                roughness if roughness is not None else self._rough_t,
                n_meshes=n,
                default=self.default_roughness,
                device=device,
            )
        if metallic is not None or self._metal_t is None:
            self._metal_t = self._normalize_mesh_table(
                metallic if metallic is not None else self._metal_t,
                n_meshes=n,
                default=self.default_metallic,
                device=device,
            )
        # Tables feed the G-buffer kernel; shade compile takes G-buffer tensors as
        # inputs, so table writes must not drop the compiled post graph.
        self._albedo_wp = None
        self._rough_wp = None
        self._metal_wp = None
        self.geom.initialized = False

    def set_albedo_textures(
        self,
        textures: Sequence | torch.Tensor | np.ndarray,
        *,
        mesh_tex_id: Sequence | torch.Tensor | None = None,
    ) -> None:
        """Bind albedo texture stack ``[T,H,W,3]`` and optional per-mesh tex ids.

        ``mesh_tex_id`` is ``[M]`` or ``[Nw,M]``; ``-1`` keeps flat table albedo.
        Sampled RGB is multiplied by the mesh albedo tint.
        """
        n = int(self.geom.n_meshes)
        if n <= 0:
            raise RuntimeError("bind meshes before set_albedo_textures")
        device = self.geom.torch_device
        tex = torch.as_tensor(textures, device=device, dtype=torch.float32)
        if tex.ndim == 3:
            tex = tex.unsqueeze(0)
        if tex.ndim != 4 or tex.shape[-1] != 3:
            raise ValueError(f"textures shape {tuple(tex.shape)}, expected [T,H,W,3]")
        self._albedo_tex_t = tex.contiguous().clamp(0.0, 1.0)
        if mesh_tex_id is not None:
            ids = torch.as_tensor(mesh_tex_id, device=device, dtype=torch.int32)
            if ids.ndim == 1:
                if ids.numel() != n:
                    raise ValueError(f"mesh_tex_id length {ids.numel()} != n_meshes {n}")
                ids = ids.view(1, n)
            elif ids.ndim == 2:
                if ids.shape[-1] != n:
                    raise ValueError(f"mesh_tex_id shape {tuple(ids.shape)} last != {n}")
            else:
                raise ValueError(f"mesh_tex_id expected [M] or [Nw,M], got {tuple(ids.shape)}")
            self._mesh_tex_id_t = ids.contiguous()
        self._albedo_tex_wp = None
        self._mesh_tex_id_wp = None
        self.geom.initialized = False

    def set_vertex_uvs(self, uvs: Sequence | torch.Tensor | np.ndarray) -> None:
        """Replace packed vertex UVs ``[V, 2]`` (must match packed normal count)."""
        if self._vert_normal_t is None:
            raise RuntimeError("bind meshes before set_vertex_uvs")
        device = self.geom.torch_device
        t = torch.as_tensor(uvs, device=device, dtype=torch.float32)
        if t.ndim != 2 or t.shape[-1] != 2:
            raise ValueError(f"uvs shape {tuple(t.shape)}, expected [V,2]")
        if t.shape[0] != self._vert_normal_t.shape[0]:
            raise ValueError(
                f"uvs V={t.shape[0]} != packed verts {self._vert_normal_t.shape[0]}"
            )
        self._vert_uv_t = t.contiguous()
        self._vert_uv_wp = None
        self.geom.initialized = False

    def set_ground_albedo(
        self,
        name_or_path: str | Path = "forest_ground",
        *,
        mesh_name: str = "ground",
        mesh_index: int | None = None,
        tile_m: float = 1.0,
        tint: Sequence | torch.Tensor | None = None,
    ) -> None:
        """Minimal ground path: planar XZ UVs + one diffuse albedo on a named mesh.

        Looks up ``mesh_name`` in bound names (or uses ``mesh_index``). Table albedo
        for that mesh becomes ``tint`` (default white) so the texture shows through.
        """
        n = int(self.geom.n_meshes)
        if n <= 0:
            raise RuntimeError("bind meshes before set_ground_albedo")
        if mesh_index is None:
            key = str(mesh_name).lower()
            hits = [i for i, nm in enumerate(self._mesh_names) if key in str(nm).lower()]
            if not hits:
                raise ValueError(
                    f"no mesh matching name {mesh_name!r} in {self._mesh_names!r}; "
                    "pass mesh_index=…"
                )
            mesh_index = int(hits[0])
        mi = int(mesh_index)
        if mi < 0 or mi >= n:
            raise ValueError(f"mesh_index {mi} out of range [0, {n})")

        device = self.geom.torch_device
        ids = torch.full((n,), -1, device=device, dtype=torch.int32)
        ids[mi] = 0
        tex = load_albedo_texture(name_or_path)
        self.set_albedo_textures(tex[None, ...], mesh_tex_id=ids)

        if not self.geom.meshes_wp:
            raise RuntimeError("set_ground_albedo requires Warp meshes after bind")
        uvs = pack_vertex_uvs_from_wp(
            self.geom.meshes_wp,
            textured_mesh_ids=[mi],
            scale=float(tile_m),
        )
        self.set_vertex_uvs(uvs)

        if self._albedo_t is None:
            alb = torch.ones(1, n, 3, device=device, dtype=torch.float32)
        else:
            alb = self._albedo_t.clone()
        if tint is None:
            alb[:, mi] = 1.0
        else:
            t = torch.as_tensor(tint, device=device, dtype=torch.float32).reshape(3)
            alb[:, mi] = t
        self.set_materials(albedo=alb)

    def set_lights(
        self,
        *,
        sun_dir: Sequence | torch.Tensor | None = None,
        sun_color: Sequence | torch.Tensor | None = None,
        sun_intensity: Sequence | torch.Tensor | float | None = None,
        exposure: Sequence | torch.Tensor | float | None = None,
    ) -> None:
        """Update sun / exposure. Accepts shared ``[3]``/scalar or batched ``[N,…]``."""
        device = self.geom.torch_device

        def _vec3(val, fallback: torch.Tensor) -> torch.Tensor:
            if val is None:
                return fallback
            t = torch.as_tensor(val, device=device, dtype=torch.float32)
            if t.ndim == 1:
                if t.numel() != 3:
                    raise ValueError(f"expected 3-vector, got {tuple(t.shape)}")
                t = t.view(1, 3)
            elif t.ndim == 2:
                if t.shape[-1] != 3:
                    raise ValueError(f"expected [N,3], got {tuple(t.shape)}")
            else:
                raise ValueError(f"expected [3] or [N,3], got {tuple(t.shape)}")
            return t.contiguous()

        def _scalar(val, fallback: torch.Tensor) -> torch.Tensor:
            if val is None:
                return fallback
            t = torch.as_tensor(val, device=device, dtype=torch.float32).reshape(-1)
            return t.contiguous()

        self._sun_dir_t = _vec3(sun_dir, self._sun_dir_t)
        self._sun_color_t = _vec3(sun_color, self._sun_color_t)
        self._sun_intensity_t = _scalar(sun_intensity, self._sun_intensity_t)
        self._exposure_t = _scalar(exposure, self._exposure_t)
        # Keep Python scalar mirrors for the leading entry (debug / repr).
        self.sun_dir = tuple(float(x) for x in self._sun_dir_t[0].tolist())
        self.sun_color = tuple(float(x) for x in self._sun_color_t[0].tolist())
        self.sun_intensity = float(self._sun_intensity_t[0].item())
        self.exposure = float(self._exposure_t[0].item())
        # Sun/exposure are compile graph inputs — no need to drop the cache.

    def initialize(self) -> None:
        self.geom.initialize()
        assert self._albedo_t is not None and self._rough_t is not None and self._metal_t is not None
        assert self._vert_normal_t is not None and self._vert_offset_t is not None
        assert self._vert_uv_t is not None and self._mesh_tex_id_t is not None
        assert self._albedo_tex_t is not None
        alb = self._albedo_t.contiguous()
        self._albedo_wp = wp.from_torch(alb, dtype=wp.vec3)
        self._rough_wp = wp.from_torch(self._rough_t.contiguous(), dtype=wp.float32)
        self._metal_wp = wp.from_torch(self._metal_t.contiguous(), dtype=wp.float32)
        self._vert_normal_wp = wp.from_torch(self._vert_normal_t, dtype=wp.vec3)
        self._vert_offset_wp = wp.from_torch(self._vert_offset_t, dtype=wp.int32)
        self._vert_uv_wp = wp.from_torch(self._vert_uv_t.contiguous(), dtype=wp.vec2)
        self._mesh_tex_id_wp = wp.from_torch(self._mesh_tex_id_t.contiguous(), dtype=wp.int32)
        self._albedo_tex_wp = wp.from_torch(self._albedo_tex_t.contiguous(), dtype=wp.float32)

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
    ) -> dict[str, torch.Tensor]:
        """Closest-hit G-buffer for a pose ticket.

        Resolution is camera state (:meth:`set_resolution`). Materials come from
        the current tables (:meth:`set_materials` or ``render(..., albedo=…)``).
        """
        if (
            not self.geom.initialized
            or self._rough_wp is None
            or self._albedo_wp is None
            or self._vert_normal_wp is None
            or self._vert_uv_wp is None
            or self._mesh_tex_id_wp is None
            or self._albedo_tex_wp is None
        ):
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
                self._albedo_wp,
                self._rough_wp,
                self._metal_wp,
                self._mesh_tex_id_wp,
                self._vert_normal_wp,
                self._vert_uv_wp,
                self._vert_offset_wp,
                self._albedo_tex_wp,
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
        albedo: Sequence | torch.Tensor | None = None,
        roughness: Sequence | torch.Tensor | float | None = None,
        metallic: Sequence | torch.Tensor | float | None = None,
        sun_dir: Sequence | torch.Tensor | None = None,
        sun_color: Sequence | torch.Tensor | None = None,
        sun_intensity: Sequence | torch.Tensor | float | None = None,
        exposure: Sequence | torch.Tensor | float | None = None,
        return_hdr: bool = False,
        return_gbuffer: bool = False,
    ):
        """Render a pose (+ optional DR) ticket.

        Per-call: camera/mesh poses and optional material/light tables.
        Not per-call: resolution (:meth:`set_resolution`), quality/HDRI/textures.
        """
        if any(x is not None for x in (albedo, roughness, metallic)):
            self.set_materials(albedo=albedo, roughness=roughness, metallic=metallic)
        if any(x is not None for x in (sun_dir, sun_color, sun_intensity, exposure)):
            self.set_lights(
                sun_dir=sun_dir,
                sun_color=sun_color,
                sun_intensity=sun_intensity,
                exposure=exposure,
            )

        gb = self.render_gbuffer(
            cam_pos_w,
            cam_quat_wxyz,
            mesh_pos_w=mesh_pos_w,
            mesh_quat_w=mesh_quat_w,
            origin_w=origin_w,
            env_ids=env_ids,
        )
        sun = self._sun_dir_t

        # Broadcast sun rows to camera batch (Nw==1 or Nw==N).
        n_cam = cam_pos_w.shape[0]
        if sun.shape[0] == 1 and n_cam > 1:
            sun_b = sun.expand(n_cam, 3)
        elif sun.shape[0] == n_cam:
            sun_b = sun
        else:
            raise ValueError(f"sun_dir batch {sun.shape[0]} != cameras {n_cam}")

        vis = torch.ones_like(gb["depth"])
        fx, fy, cx, cy = self.intrinsics.resolved_k()
        planar = self._planar_depth(gb)
        # Prefer hard sray over the ortho map when both are on (do not stack).
        use_sray = bool(self.shadow_rays)
        use_smap = bool(self.shadow_map_enabled) and not use_sray
        need_hits = use_smap or use_sray
        hit_w = None
        if need_hits:
            hit_w = self._hit_positions_w(gb, cam_pos_w.contiguous(), cam_quat_wxyz.contiguous())
            gb["position"] = hit_w

        # Shadow map / shadow rays currently use a shared sun direction (row 0).
        sun0 = sun_b[0]

        if self.contact_shadows_enabled:
            sun_cam = quat_rotate_inverse(
                cam_quat_wxyz.contiguous(),
                sun_b,
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

        if use_smap:
            assert hit_w is not None
            sm, center, right, up, forward = self._render_shadow_map(cam_pos_w.shape[0], sun0)
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

        if use_sray:
            assert hit_w is not None
            vis = vis * self._render_shadow_rays(gb, hit_w, sun0)

        rgb, hdr = self._shade_tonemap_fxaa(
            gb,
            planar,
            vis,
            sun_b,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            cam_quat_wxyz=cam_quat_wxyz.contiguous(),
        )
        gb["sun_visibility"] = vis
        if return_hdr and return_gbuffer:
            return rgb, gb["depth"], gb["mask"], hdr, gb
        if return_gbuffer:
            return rgb, gb["depth"], gb["mask"], gb
        if return_hdr:
            return rgb, gb["depth"], gb["mask"], hdr
        return rgb, gb["depth"], gb["mask"]

    def _validate_sparse_tiles(
        self, tiles: torch.Tensor, tile_size: int
    ) -> tuple[torch.Tensor, int, int, int]:
        w = self.intrinsics.width
        h = self.intrinsics.height
        s = int(tile_size)
        if s <= 0:
            raise ValueError(f"tile_size must be > 0, got {tile_size}")
        if w % s != 0 or h % s != 0:
            raise ValueError(
                f"tile_size {s} must divide resolution {w}x{h} "
                f"(got remainders {w % s}, {h % s})"
            )
        t = torch.as_tensor(tiles, device=self.geom.torch_device, dtype=torch.int32)
        if t.ndim != 3 or t.shape[-1] != 2:
            raise ValueError(f"tiles expected [B, T, 2], got {tuple(t.shape)}")
        n_tx, n_ty = w // s, h // s
        if bool((t[..., 0] < 0).any() or (t[..., 0] >= n_tx).any()):
            raise ValueError(f"tile x must be in [0, {n_tx})")
        if bool((t[..., 1] < 0).any() or (t[..., 1] >= n_ty).any()):
            raise ValueError(f"tile y must be in [0, {n_ty})")
        return t.contiguous(), s, n_tx, n_ty

    def _hit_positions_sparse(
        self,
        gb: dict[str, torch.Tensor],
        tiles: torch.Tensor,
        tile_size: int,
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
    ) -> torch.Tensor:
        """World hit positions for sparse tiles. ``gb`` fields are ``[N,T,S,S,…]``."""
        depth = gb["depth"]
        n, n_tiles, s, _ = depth.shape
        device = depth.device
        fx, fy, cx, cy = self.intrinsics.resolved_k()
        tx = tiles[:, :, 0].to(dtype=torch.float32)  # [N,T]
        ty = tiles[:, :, 1].to(dtype=torch.float32)
        ys = torch.arange(s, device=device, dtype=torch.float32)
        xs = torch.arange(s, device=device, dtype=torch.float32)
        # local (ly, lx); pixel = tile*S + local
        ly, lx = torch.meshgrid(ys, xs, indexing="ij")
        px = tx[:, :, None, None] * float(s) + lx[None, None, :, :]
        py = ty[:, :, None, None] * float(s) + ly[None, None, :, :]
        u = (px + 0.5 - cx) / fx
        v = (py + 0.5 - cy) / fy
        if self.intrinsics.convention == "opencv":
            dir_c = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        else:
            dir_c = torch.stack([u, -v, -torch.ones_like(u)], dim=-1)
        dir_c = dir_c / dir_c.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        quat = cam_quat_wxyz[:, None, None, None, :].expand(n, n_tiles, s, s, 4)
        dir_w = quat_rotate(quat, dir_c)
        if self.intrinsics.depth_mode == "planar":
            optical_z = dir_c[..., 2] if self.intrinsics.convention == "opencv" else -dir_c[..., 2]
            t = depth / optical_z.clamp_min(1e-8)
        else:
            t = depth
        return cam_pos_w[:, None, None, None, :] + dir_w * t.unsqueeze(-1)

    @torch.no_grad()
    def render_gbuffer_sparse(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        *,
        tiles: torch.Tensor,
        tile_size: int,
        mesh_pos_w: torch.Tensor | None = None,
        mesh_quat_w: torch.Tensor | None = None,
        origin_w: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Closest-hit G-buffer for selected tiles only.

        ``tiles`` is ``[B, T, 2]`` integer ``(tx, ty)`` in tile-grid units.
        Returns fields shaped ``[B, T, S, S, …]`` (``S=tile_size``).
        """
        tiles_i, s, _, _ = self._validate_sparse_tiles(tiles, tile_size)
        if (
            not self.geom.initialized
            or self._rough_wp is None
            or self._albedo_wp is None
            or self._vert_normal_wp is None
            or self._vert_uv_wp is None
            or self._mesh_tex_id_wp is None
            or self._albedo_tex_wp is None
        ):
            self.initialize()

        n = cam_pos_w.shape[0]
        if tiles_i.shape[0] != n:
            raise ValueError(f"tiles batch {tiles_i.shape[0]} != cameras {n}")
        n_tiles = int(tiles_i.shape[1])
        p = n_tiles * s * s
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

        depth = self._cached("sp_depth", (n, p), torch.float32)
        mask = self._cached("sp_mask", (n, p), torch.bool)
        normal = self._cached("sp_nrm", (n, p, 3), torch.float32)
        albedo = self._cached("sp_alb", (n, p, 3), torch.float32)
        rough = self._cached("sp_rough", (n, p), torch.float32)
        metal = self._cached("sp_metal", (n, p), torch.float32)
        view = self._cached("sp_view", (n, p, 3), torch.float32)

        wp.launch(
            raycast_gbuffer_sparse_kernel,
            dim=(n, p),
            inputs=[
                self.geom.meshes_array,
                wp.from_torch(mesh_pos, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(mesh_quat, dtype=wp.vec4, return_ctype=True),
                self._albedo_wp,
                self._rough_wp,
                self._metal_wp,
                self._mesh_tex_id_wp,
                self._vert_normal_wp,
                self._vert_uv_wp,
                self._vert_offset_wp,
                self._albedo_tex_wp,
                wp.from_torch(cam_pos_w.contiguous(), dtype=wp.vec3, return_ctype=True),
                wp.from_torch(cam_quat_wxyz.contiguous(), dtype=wp.vec4, return_ctype=True),
                wp.from_torch(tiles_i, dtype=wp.int32, return_ctype=True),
                int(s),
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
            "depth": depth.view(n, n_tiles, s, s),
            "mask": mask.view(n, n_tiles, s, s),
            "normal": normal.view(n, n_tiles, s, s, 3),
            "albedo": albedo.view(n, n_tiles, s, s, 3),
            "roughness": rough.view(n, n_tiles, s, s),
            "metallic": metal.view(n, n_tiles, s, s),
            "view": view.view(n, n_tiles, s, s, 3),
            "tiles": tiles_i,
            "tile_size": s,
        }

    @torch.no_grad()
    def render_sparse(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        *,
        tiles: torch.Tensor,
        tile_size: int,
        mesh_pos_w: torch.Tensor | None = None,
        mesh_quat_w: torch.Tensor | None = None,
        origin_w: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
        albedo: Sequence | torch.Tensor | None = None,
        roughness: Sequence | torch.Tensor | float | None = None,
        metallic: Sequence | torch.Tensor | float | None = None,
        sun_dir: Sequence | torch.Tensor | None = None,
        sun_color: Sequence | torch.Tensor | None = None,
        sun_intensity: Sequence | torch.Tensor | float | None = None,
        exposure: Sequence | torch.Tensor | float | None = None,
        return_hdr: bool = False,
        return_gbuffer: bool = False,
    ):
        """Render selected image tiles (no FXAA / SSAO).

        Args:
            tiles: ``[B, T, 2]`` int tile indices ``(tx, ty)`` in
                ``[0, W/S) × [0, H/S)``.
            tile_size: ``S``; must divide camera width and height.

        Returns:
            ``(rgb, depth, mask)`` with shapes
            ``[B, T, S, S, 3]``, ``[B, T, S, S]``, ``[B, T, S, S]`` (NHWC).
            Optional HDR / G-buffer like :meth:`render`.
        """
        if any(x is not None for x in (albedo, roughness, metallic)):
            self.set_materials(albedo=albedo, roughness=roughness, metallic=metallic)
        if any(x is not None for x in (sun_dir, sun_color, sun_intensity, exposure)):
            self.set_lights(
                sun_dir=sun_dir,
                sun_color=sun_color,
                sun_intensity=sun_intensity,
                exposure=exposure,
            )

        gb = self.render_gbuffer_sparse(
            cam_pos_w,
            cam_quat_wxyz,
            tiles=tiles,
            tile_size=tile_size,
            mesh_pos_w=mesh_pos_w,
            mesh_quat_w=mesh_quat_w,
            origin_w=origin_w,
            env_ids=env_ids,
        )
        sun = self._sun_dir_t
        n_cam = cam_pos_w.shape[0]
        if sun.shape[0] == 1 and n_cam > 1:
            sun_b = sun.expand(n_cam, 3)
        elif sun.shape[0] == n_cam:
            sun_b = sun
        else:
            raise ValueError(f"sun_dir batch {sun.shape[0]} != cameras {n_cam}")

        n, n_tiles, s, _ = gb["depth"].shape
        p = n_tiles * s * s
        vis = torch.ones_like(gb["depth"])
        use_sray = bool(self.shadow_rays)
        use_smap = bool(self.shadow_map_enabled) and not use_sray
        hit_w = None
        if use_smap or use_sray:
            hit_w = self._hit_positions_sparse(
                gb, gb["tiles"], int(gb["tile_size"]), cam_pos_w.contiguous(), cam_quat_wxyz.contiguous()
            )
            gb["position"] = hit_w

        sun0 = sun_b[0]
        # Contact / SSAO need a dense neighborhood — skipped in sparse path.
        if use_smap:
            assert hit_w is not None
            sm, center, right, up, forward = self._render_shadow_map(n_cam, sun0)
            # Flatten tiles to a fake [N,1,P] grid for the dense sampler.
            vis = sample_shadow_map(
                hit_w.reshape(n, 1, p, 3),
                gb["mask"].reshape(n, 1, p),
                sm,
                center_w=center,
                right_w=right,
                up_w=up,
                forward_w=forward,
                half_extent=self.shadow_map_extent,
                catch_dist=self.shadow_map_catch,
            ).view(n, n_tiles, s, s)

        if use_sray:
            assert hit_w is not None
            vis_flat = self._cached("sp_sray", (n, p), torch.float32)
            wp.launch(
                shadow_ray_kernel,
                dim=(n, p),
                inputs=[
                    self.geom.meshes_array,
                    wp.from_torch(self._last_mesh_pos, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(self._last_mesh_quat, dtype=wp.vec4, return_ctype=True),
                    wp.from_torch(hit_w.reshape(n, p, 3).contiguous(), dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(gb["mask"].reshape(n, p).contiguous(), dtype=wp.bool, return_ctype=True),
                    wp.vec3(float(sun0[0]), float(sun0[1]), float(sun0[2])),
                    2.0e-3,
                    float(self.shadow_ray_tmax),
                ],
                outputs=[wp.from_torch(vis_flat, dtype=wp.float32, return_ctype=True)],
                device=self.geom.device,
                record_tape=False,
                block_dim=self.geom.block_dim,
            )
            vis = vis_flat.view(n, n_tiles, s, s)

        # Shade without FXAA/SSAO (tile patches have no full-frame neighborhood).
        sun_color, sun_intensity, exposure = self._broadcast_light_rows(n)
        hdr = shade_pbr(
            gb["albedo"],
            gb["normal"],
            gb["view"],
            gb["roughness"],
            gb["metallic"],
            self.env,
            sun_dir_w=sun_b,
            sun_intensity=sun_intensity,
            sun_color=sun_color,
            sun_visibility=vis,
        )
        bg = self.env.sample(-gb["view"], roughness=0.0)
        hdr = torch.where(gb["mask"].unsqueeze(-1), hdr, bg)
        rgb = aces_tonemap(hdr, exposure=exposure)
        gb["sun_visibility"] = vis

        if return_hdr and return_gbuffer:
            return rgb, gb["depth"], gb["mask"], hdr, gb
        if return_gbuffer:
            return rgb, gb["depth"], gb["mask"], gb
        if return_hdr:
            return rgb, gb["depth"], gb["mask"], hdr
        return rgb, gb["depth"], gb["mask"]

    def _fold_torch_post(self) -> bool:
        """True when shade+SSAO+tonemap+FXAA can share one ``torch.compile`` graph."""
        if not self.compile_mode:
            return False
        if self.fxaa_enabled and self.fxaa_impl != "torch":
            return False
        if self.ssao_enabled and self.ssao_impl != "torch":
            return False
        # Subclasses that replace SSAO (e.g. SSGI) keep ``_compose_indirect``.
        return type(self)._compose_indirect is RaycastPBRCamera._compose_indirect

    def _ensure_compiled_torch_post(self):
        key = (
            self.compile_mode,
            self.ssao_enabled,
            self.ssao_radius,
            self.fxaa_enabled,
        )
        if self._compiled_shade_fxaa is not None and self._compiled_shade_fxaa_key == key:
            return self._compiled_shade_fxaa

        env = self.env
        do_ssao = bool(self.ssao_enabled)
        do_fxaa = bool(self.fxaa_enabled)
        ssao_radius = float(self.ssao_radius)

        def torch_post(
            albedo: torch.Tensor,
            normal: torch.Tensor,
            view: torch.Tensor,
            roughness: torch.Tensor,
            metallic: torch.Tensor,
            mask: torch.Tensor,
            planar: torch.Tensor,
            vis: torch.Tensor,
            sun: torch.Tensor,
            sun_color: torch.Tensor,
            sun_intensity: torch.Tensor,
            exposure: torch.Tensor,
            fx: float,
            fy: float,
            cx: float,
            cy: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            hdr = shade_pbr(
                albedo,
                normal,
                view,
                roughness,
                metallic,
                env,
                sun_dir_w=sun,
                sun_intensity=sun_intensity,
                sun_color=sun_color,
                sun_visibility=vis,
            )
            bg = env.sample(-view, roughness=0.0)
            hdr = torch.where(mask.unsqueeze(-1), hdr, bg)
            if do_ssao:
                ao = ssao(
                    planar,
                    mask,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    radius=ssao_radius,
                )
                hdr = torch.where(mask.unsqueeze(-1), hdr * ao.unsqueeze(-1), hdr)
            rgb = aces_tonemap(hdr, exposure=exposure)
            if do_fxaa:
                rgb = fxaa(rgb)
            return rgb, hdr

        compiled = torch.compile(torch_post, mode=self.compile_mode)
        self._compiled_shade_fxaa = compiled
        self._compiled_shade_fxaa_key = key
        return compiled

    def _broadcast_light_rows(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def _row(t: torch.Tensor) -> torch.Tensor:
            if t.shape[0] == 1 and n > 1:
                return t.expand(n, *t.shape[1:])
            if t.shape[0] == n:
                return t
            raise ValueError(f"light batch {t.shape[0]} != cameras {n}")

        return (
            _row(self._sun_color_t),
            _row(self._sun_intensity_t),
            _row(self._exposure_t),
        )

    def _shade_tonemap_fxaa(
        self,
        gb: dict[str, torch.Tensor],
        planar: torch.Tensor,
        vis: torch.Tensor,
        sun: torch.Tensor,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        cam_quat_wxyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Direct shade → optional AO/GI → tonemap → FXAA. Returns ``(rgb, hdr)``."""
        n = gb["albedo"].shape[0]
        sun_color, sun_intensity, exposure = self._broadcast_light_rows(n)
        if self._fold_torch_post():
            compiled = self._ensure_compiled_torch_post()
            return compiled(
                gb["albedo"],
                gb["normal"],
                gb["view"],
                gb["roughness"],
                gb["metallic"],
                gb["mask"],
                planar,
                vis,
                sun,
                sun_color,
                sun_intensity,
                exposure,
                fx,
                fy,
                cx,
                cy,
            )

        hdr = shade_pbr(
            gb["albedo"],
            gb["normal"],
            gb["view"],
            gb["roughness"],
            gb["metallic"],
            self.env,
            sun_dir_w=sun,
            sun_intensity=sun_intensity,
            sun_color=sun_color,
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
            cam_quat_wxyz=cam_quat_wxyz,
        )
        rgb = aces_tonemap(hdr, exposure=exposure)
        if self.fxaa_enabled:
            if self.fxaa_impl == "tiled":
                rgb = fxaa_tiled(rgb)
            else:
                rgb = fxaa(rgb)
        return rgb, hdr

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
