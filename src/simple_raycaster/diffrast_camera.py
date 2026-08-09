"""nvdiffrast RGB-D camera over cached body-local triangle meshes.

OpenCV camera poses (``+Z`` forward, ``+Y`` down) are converted to OpenGL for
rasterization. Depth is **metric** meters along the optical axis (not clip
``z/w``), matching :class:`RaycastCamera` planar depth for GS compositing.

Requires optional dependency ``nvdiffrast``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import trimesh

from .helpers import matrix_from_quat
from .mesh_cache import (
    CachedEntityMesh,
    body_link_pose_to_mat,
    build_cached_entity_mesh,
    merge_rgbd,
    pose_verts,
)


def opencv_w2c_to_opengl(w2c_cv: torch.Tensor) -> torch.Tensor:
    """OpenCV w2c (+Z forward, +Y down) → OpenGL w2c (−Z forward, +Y up)."""
    flip = w2c_cv.new_tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return flip @ w2c_cv


def perspective_opengl(
    fov_y_deg: float,
    aspect: float,
    near: float,
    far: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch: int,
) -> torch.Tensor:
    """OpenGL perspective ``[B, 4, 4]``."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    proj = torch.zeros(batch, 4, 4, device=device, dtype=dtype)
    proj[:, 0, 0] = f / aspect
    proj[:, 1, 1] = f
    proj[:, 2, 2] = (far + near) / (near - far)
    proj[:, 2, 3] = (2.0 * far * near) / (near - far)
    proj[:, 3, 2] = -1.0
    return proj


def shade_lambert(
    colors: torch.Tensor,
    normals: torch.Tensor,
    light_dir_world: torch.Tensor,
    *,
    ambient: float = 0.30,
    diffuse: float = 0.80,
) -> torch.Tensor:
    """``colors`` ``[V,3]`` or ``[B,V,3]``; ``normals`` ``[B,V,3]`` → ``[B,V,3]``."""
    l = light_dir_world / light_dir_world.norm().clamp_min(1e-8)
    ndotl = (normals @ l).clamp(0.0, 1.0).unsqueeze(-1)
    if colors.dim() == 2:
        colors = colors.unsqueeze(0)
    return (colors * (ambient + diffuse * ndotl)).clamp(0.0, 1.0)


def _pos_quat_to_w2c(pos_w: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """OpenCV camera pose (world) → world-to-camera ``(N, 4, 4)``."""
    r_wc = matrix_from_quat(quat_wxyz)
    r_cw = r_wc.transpose(-1, -2)
    t = -(r_cw @ pos_w.unsqueeze(-1)).squeeze(-1)
    n = pos_w.shape[0]
    w2c = torch.eye(4, device=pos_w.device, dtype=pos_w.dtype).unsqueeze(0).repeat(n, 1, 1)
    w2c[:, :3, :3] = r_cw
    w2c[:, :3, 3] = t
    return w2c


@torch.no_grad()
def render_posed(
    glctx,
    mesh: CachedEntityMesh,
    body_T: torch.Tensor,
    mvp: torch.Tensor,
    w2c_gl: torch.Tensor,
    *,
    width: int,
    height: int,
    light_dir: torch.Tensor,
    ambient: float = 0.30,
    diffuse: float = 0.80,
    far_fill: float = 1.0e3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pose with ``body_T[B,L,4,4]``, rasterize RGB + metric depth + mask."""
    import nvdiffrast.torch as dr

    verts_w, normals_w = pose_verts(mesh, body_T)
    shaded = shade_lambert(
        mesh.colors, normals_w, light_dir, ambient=ambient, diffuse=diffuse
    )

    ones = torch.ones(
        verts_w.shape[0],
        verts_w.shape[1],
        1,
        device=verts_w.device,
        dtype=verts_w.dtype,
    )
    homo = torch.cat([verts_w, ones], dim=-1)
    pos = torch.einsum("bij,bvj->bvi", mvp, homo).contiguous()
    view = torch.einsum("bij,bvj->bvi", w2c_gl, homo)
    depth_attr = (-view[..., 2:3]).contiguous()
    attrs = torch.cat([shaded, depth_attr], dim=-1).contiguous()

    rast, _ = dr.rasterize(glctx, pos, mesh.faces, resolution=[height, width])
    interp, _ = dr.interpolate(attrs, rast, mesh.faces)
    rgb_aa = dr.antialias(interp[..., :3].contiguous(), rast, pos, mesh.faces)

    mask = rast[..., 3] > 0
    rgb = torch.where(mask.unsqueeze(-1), rgb_aa, torch.zeros_like(rgb_aa)).clamp(0.0, 1.0)
    depth = torch.where(mask, interp[..., 3], torch.full_like(interp[..., 3], far_fill))
    return rgb.flip(1), depth.flip(1), mask.flip(1)


class DiffrastCamera:
    """nvdiffrast RGB-D camera over one or more cached entity meshes."""

    def __init__(
        self,
        width: int = 64,
        height: int = 48,
        *,
        fov_y_deg: float = 70.0,
        near: float = 0.05,
        far: float = 50.0,
        device: str | torch.device = "cuda",
        ambient: float = 0.30,
        diffuse: float = 0.80,
        light_dir: tuple[float, float, float] = (0.45, -0.35, 0.82),
        face_keep: float = 0.2,
    ) -> None:
        import nvdiffrast.torch as dr

        self.width = int(width)
        self.height = int(height)
        self.fov_y_deg = float(fov_y_deg)
        self.near = float(near)
        self.far = float(far)
        self.device = torch.device(device)
        self.ambient = float(ambient)
        self.diffuse = float(diffuse)
        self.light_dir = tuple(float(x) for x in light_dir)
        self.face_keep = float(face_keep)

        self._glctx = dr.RasterizeCudaContext()
        self._bundles: list[CachedEntityMesh] = []
        self._light_dir_t: torch.Tensor | None = None

    @property
    def n_meshes(self) -> int:
        return len(self._bundles)

    @property
    def has_meshes(self) -> bool:
        return bool(self._bundles)

    def clear(self) -> None:
        self._bundles.clear()

    def register_entity(
        self,
        name: str,
        entity: object,
        meshes: Sequence[trimesh.Trimesh],
        *,
        face_keep: float | None = None,
    ) -> None:
        """Replace any prior registration for ``name``."""
        self._bundles = [b for b in self._bundles if b.name != name]
        bundle = build_cached_entity_mesh(
            name,
            entity,
            list(meshes),
            device=self.device,
            face_keep=self.face_keep if face_keep is None else float(face_keep),
        )
        self._bundles.append(bundle)
        keep = self.face_keep if face_keep is None else float(face_keep)
        print(
            f"[DiffrastCamera] Cached '{name}': "
            f"V={bundle.num_verts:,} F={bundle.num_faces:,} "
            f"L={bundle.num_bodies} face_keep={keep}"
        )

    def set_resolution(
        self,
        width: int,
        height: int,
        *,
        fov_y_deg: float | None = None,
        near: float | None = None,
        far: float | None = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        if fov_y_deg is not None:
            self.fov_y_deg = float(fov_y_deg)
        if near is not None:
            self.near = float(near)
        if far is not None:
            self.far = float(far)

    @torch.no_grad()
    def render(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
        origin_w: torch.Tensor | None = None,
        width: int | None = None,
        height: int | None = None,
        fov_y_deg: float | None = None,
        near: float | None = None,
        far: float | None = None,
        light_dir: torch.Tensor | tuple[float, float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render mesh RGB-D → ``rgb[N,H,W,3]``, ``depth[N,H,W]``, ``mask[N,H,W]``."""
        if not self._bundles:
            raise RuntimeError("DiffrastCamera has no meshes; call register_entity first")

        if width is not None or height is not None or fov_y_deg is not None:
            self.set_resolution(
                width or self.width,
                height or self.height,
                fov_y_deg=fov_y_deg,
                near=near,
                far=far,
            )
        elif near is not None or far is not None:
            self.set_resolution(self.width, self.height, near=near, far=far)

        n = cam_pos_w.shape[0]
        w, h = self.width, self.height
        device = cam_pos_w.device
        dtype = cam_pos_w.dtype

        if light_dir is None:
            light = self.light_dir
        elif isinstance(light_dir, torch.Tensor):
            light = (
                float(light_dir[0]),
                float(light_dir[1]),
                float(light_dir[2]),
            )
        else:
            light = tuple(float(x) for x in light_dir)
        if self._light_dir_t is None or self._light_dir_t.device != device:
            self._light_dir_t = torch.tensor(light, device=device, dtype=dtype)
        else:
            self._light_dir_t = torch.tensor(light, device=device, dtype=dtype)

        w2c_cv = _pos_quat_to_w2c(cam_pos_w, cam_quat_wxyz)
        w2c_gl = opencv_w2c_to_opengl(w2c_cv)
        proj = perspective_opengl(
            self.fov_y_deg,
            w / max(h, 1),
            self.near,
            self.far,
            device=device,
            dtype=dtype,
            batch=n,
        )
        mvp = proj @ w2c_gl

        if env_ids is None:
            env_ids = torch.arange(n, device=device)

        rgb_acc = depth_acc = mask_acc = None
        for bundle in self._bundles:
            poses = bundle.entity.data.body_link_pose_w[env_ids]
            body_T = body_link_pose_to_mat(poses, origin_w=origin_w)
            rgb, depth, mask = render_posed(
                self._glctx,
                bundle,
                body_T,
                mvp,
                w2c_gl,
                width=w,
                height=h,
                light_dir=self._light_dir_t,
                ambient=self.ambient,
                diffuse=self.diffuse,
                far_fill=self.far,
            )
            rgb_acc, depth_acc, mask_acc = merge_rgbd(
                rgb_acc, depth_acc, mask_acc, rgb, depth, mask
            )
        assert rgb_acc is not None and depth_acc is not None and mask_acc is not None
        return rgb_acc, depth_acc, mask_acc
