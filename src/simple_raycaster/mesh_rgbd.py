"""Factory for mesh RGB-D backends used by GS compositors.

* ``diffrast`` → :class:`DiffrastCamera`
* ``raycast`` → one :class:`RaycastCamera` per entity
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence

import torch
import trimesh

from .diffrast_camera import DiffrastCamera
from .mesh_cache import build_simplified_body_trimeshes, merge_rgbd
from .raycast_camera import CameraIntrinsics, RaycastCamera

MeshRendererKind = Literal["diffrast", "raycast"]


class MeshRGBDRenderer(ABC):
    """Cache body-local entity meshes and render OpenCV RGB-D."""

    kind: str

    @property
    @abstractmethod
    def has_meshes(self) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def register_entity(
        self,
        name: str,
        entity: object,
        meshes: Sequence[trimesh.Trimesh],
    ) -> None: ...

    @abstractmethod
    def render(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
        origin_w: torch.Tensor | None = None,
        width: int,
        height: int,
        fov_y_deg: float,
        near: float,
        far: float,
        light_dir: tuple[float, float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``rgb[N,H,W,3]``, ``depth[N,H,W]``, ``mask[N,H,W]``."""


class DiffrastMeshRenderer(MeshRGBDRenderer):
    kind = "diffrast"

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        face_keep: float = 0.2,
        light_dir: tuple[float, float, float] = (0.45, -0.35, 0.82),
    ) -> None:
        self._cam = DiffrastCamera(
            device=device, face_keep=face_keep, light_dir=light_dir
        )
        self.light_dir = self._cam.light_dir

    @property
    def has_meshes(self) -> bool:
        return self._cam.has_meshes

    def clear(self) -> None:
        self._cam.clear()

    def register_entity(
        self,
        name: str,
        entity: object,
        meshes: Sequence[trimesh.Trimesh],
    ) -> None:
        self._cam.register_entity(name, entity, meshes)

    def render(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
        origin_w: torch.Tensor | None = None,
        width: int,
        height: int,
        fov_y_deg: float,
        near: float,
        far: float,
        light_dir: tuple[float, float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._cam.render(
            cam_pos_w,
            cam_quat_wxyz,
            env_ids=env_ids,
            origin_w=origin_w,
            width=width,
            height=height,
            fov_y_deg=fov_y_deg,
            near=near,
            far=far,
            light_dir=light_dir if light_dir is not None else self.light_dir,
        )


class RaycastMeshRenderer(MeshRGBDRenderer):
    kind = "raycast"

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        face_keep: float = 0.2,
        light_dir: tuple[float, float, float] = (0.45, -0.35, 0.82),
    ) -> None:
        self.device = torch.device(device)
        self.face_keep = float(face_keep)
        self.light_dir = tuple(float(x) for x in light_dir)
        self._cams: list[tuple[str, RaycastCamera, object, list[int]]] = []

    @property
    def has_meshes(self) -> bool:
        return bool(self._cams)

    def clear(self) -> None:
        self._cams.clear()

    def register_entity(
        self,
        name: str,
        entity: object,
        meshes: Sequence[trimesh.Trimesh],
    ) -> None:
        self._cams = [(n, c, e, b) for n, c, e, b in self._cams if n != name]
        kept, body_ids, albedos = build_simplified_body_trimeshes(
            list(meshes), face_keep=self.face_keep
        )
        if not kept:
            raise ValueError(f"no non-empty visual meshes for entity {name!r}")
        cam = RaycastCamera(
            64,
            48,
            fov_y_deg=70.0,
            convention="opencv",
            depth_mode="planar",
            device=self.device,
            light_dir=self.light_dir,
        )
        cam.bind_trimeshes(kept, albedos=albedos)
        self._cams.append((name, cam, entity, body_ids))
        n_faces = sum(len(m.faces) for m in kept)
        n_verts = sum(len(m.vertices) for m in kept)
        print(
            f"[RaycastMeshRenderer] Bound '{name}': "
            f"V={n_verts:,} F={n_faces:,} M={len(kept)} face_keep={self.face_keep}"
        )

    def render(
        self,
        cam_pos_w: torch.Tensor,
        cam_quat_wxyz: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
        origin_w: torch.Tensor | None = None,
        width: int,
        height: int,
        fov_y_deg: float,
        near: float,
        far: float,
        light_dir: tuple[float, float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._cams:
            raise RuntimeError("RaycastMeshRenderer has no registered meshes")

        light = self.light_dir if light_dir is None else light_dir
        rgb_acc = depth_acc = mask_acc = None
        for _name, cam, entity, body_ids in self._cams:
            intr = cam.intrinsics
            cam.intrinsics = CameraIntrinsics(
                width=width,
                height=height,
                fov_y_deg=fov_y_deg,
                near=near,
                far=far,
                convention=intr.convention,
                depth_mode=intr.depth_mode,
            )
            poses = entity.data.body_link_pose_w
            cam_pos = cam_pos_w
            cam_quat = cam_quat_wxyz
            if env_ids is not None:
                poses = poses[env_ids]
                cam_pos = cam_pos_w[env_ids]
                cam_quat = cam_quat_wxyz[env_ids]
            mesh_pos = poses[:, body_ids, :3]
            mesh_quat = poses[:, body_ids, 3:7]
            if origin_w is not None:
                mesh_pos = mesh_pos - origin_w.unsqueeze(1)
            rgb, depth, mask = cam.render(
                cam_pos,
                cam_quat,
                mesh_pos_w=mesh_pos.contiguous(),
                mesh_quat_w=mesh_quat.contiguous(),
                light_dir=light,
            )
            rgb_acc, depth_acc, mask_acc = merge_rgbd(
                rgb_acc, depth_acc, mask_acc, rgb, depth, mask
            )
        assert rgb_acc is not None and depth_acc is not None and mask_acc is not None
        return rgb_acc, depth_acc, mask_acc


MESH_RGBD_RENDERERS: dict[str, type[MeshRGBDRenderer]] = {
    "diffrast": DiffrastMeshRenderer,
    "raycast": RaycastMeshRenderer,
}


def make_mesh_rgbd_renderer(
    kind: str | MeshRGBDRenderer,
    *,
    device: str | torch.device = "cuda",
    face_keep: float = 0.2,
    light_dir: tuple[float, float, float] = (0.45, -0.35, 0.82),
) -> MeshRGBDRenderer:
    if isinstance(kind, MeshRGBDRenderer):
        return kind
    key = str(kind).lower()
    if key not in MESH_RGBD_RENDERERS:
        raise ValueError(
            f"Unknown mesh renderer {kind!r}; expected one of "
            f"{sorted(MESH_RGBD_RENDERERS)} or a MeshRGBDRenderer instance"
        )
    return MESH_RGBD_RENDERERS[key](
        device=device, face_keep=face_keep, light_dir=light_dir
    )
