"""Cached body-local meshes for RGB-D mesh cameras (diffrast / raycast).

Cache visual geometry in link frames, optionally quadric-decimate, then each
frame only gathers body transforms::

    v_world[b, v] = T[b, body[v]] @ v_local[v]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import trimesh

from .helpers import matrix_from_quat


@dataclass
class CachedEntityMesh:
    """Visual meshes in body/link frames + topology (built once)."""

    name: str
    entity: object
    verts_local: torch.Tensor  # [V, 3]
    faces: torch.Tensor  # [F, 3] int32
    colors: torch.Tensor  # [V, 3]
    normals_local: torch.Tensor  # [V, 3]
    body_index: torch.Tensor  # [V] long → body slot in [B, L, 4, 4]
    num_bodies: int

    @property
    def num_verts(self) -> int:
        return int(self.verts_local.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])


def simplify_trimesh(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    face_keep: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadric-decimate a single body mesh; ``face_keep`` in (0, 1]."""
    if face_keep >= 1.0 or len(faces) < 8:
        return verts.astype(np.float32), faces.astype(np.int32)
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    target = max(4, int(round(len(tm.faces) * face_keep)))
    if target >= len(tm.faces):
        return verts.astype(np.float32), faces.astype(np.int32)
    simplified = tm.simplify_quadric_decimation(face_count=target)
    return (
        np.asarray(simplified.vertices, dtype=np.float32),
        np.asarray(simplified.faces, dtype=np.int32),
    )


def _vertex_colors(mesh: trimesh.Trimesh, n_verts: int) -> np.ndarray:
    visual = getattr(mesh, "visual", None)
    if visual is not None:
        vc = getattr(visual, "vertex_colors", None)
        if vc is not None:
            vc = np.asarray(vc)
            if vc.ndim == 2 and vc.shape[0] == n_verts:
                rgb = vc[:, :3].astype(np.float32)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                return np.clip(rgb, 0.0, 1.0)
        fc = getattr(visual, "face_colors", None)
        if fc is not None:
            fc = np.asarray(fc)
            if fc.ndim == 2 and len(mesh.faces) == fc.shape[0]:
                rgb = np.full((n_verts, 3), 0.65, dtype=np.float32)
                face_rgb = fc[:, :3].astype(np.float32)
                if face_rgb.max() > 1.0:
                    face_rgb = face_rgb / 255.0
                for fi, face in enumerate(mesh.faces):
                    rgb[face] = face_rgb[fi]
                return np.clip(rgb, 0.0, 1.0)
    return np.full((n_verts, 3), 0.65, dtype=np.float32)


def _recompute_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    normals = np.asarray(tm.vertex_normals, dtype=np.float32)
    nrm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.clip(nrm, 1e-8, None)


def build_simplified_body_trimeshes(
    meshes: list[trimesh.Trimesh],
    *,
    face_keep: float = 1.0,
) -> tuple[list[trimesh.Trimesh], list[int], list[np.ndarray]]:
    """Simplify non-empty body meshes for :class:`RaycastCamera.bind_trimeshes`.

    Returns:
        ``(kept_meshes, body_ids, albedos)`` — parallel lists (albedo is RGB float32).
    """
    kept: list[trimesh.Trimesh] = []
    body_ids: list[int] = []
    albedos: list[np.ndarray] = []
    for body_i, mesh in enumerate(meshes):
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        colors = _vertex_colors(mesh, len(verts))
        verts_s, faces_s = simplify_trimesh(verts, faces, face_keep=face_keep)
        kept.append(trimesh.Trimesh(vertices=verts_s, faces=faces_s, process=False))
        body_ids.append(body_i)
        albedos.append(colors.mean(axis=0).astype(np.float32))
    return kept, body_ids, albedos


def build_cached_entity_mesh(
    name: str,
    entity: object,
    meshes: list[trimesh.Trimesh],
    *,
    device: torch.device,
    face_keep: float = 1.0,
) -> CachedEntityMesh:
    """Pack body trimeshes into one cached GPU mesh (skip empty bodies)."""
    if len(meshes) != entity.num_bodies:
        raise ValueError(
            f"Expected {entity.num_bodies} meshes for '{name}', got {len(meshes)}"
        )

    face_keep = float(face_keep)
    if face_keep <= 0.0:
        raise ValueError(f"face_keep must be > 0, got {face_keep}")

    all_v: list[np.ndarray] = []
    all_f: list[np.ndarray] = []
    all_c: list[np.ndarray] = []
    all_n: list[np.ndarray] = []
    all_body: list[np.ndarray] = []
    v_offset = 0
    faces_in = 0
    faces_out = 0

    for body_i, mesh in enumerate(meshes):
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        faces_in += len(faces)
        colors_full = _vertex_colors(mesh, len(verts))
        verts_s, faces_s = simplify_trimesh(verts, faces, face_keep=face_keep)
        faces_out += len(faces_s)
        if len(verts_s) == len(verts):
            colors = colors_full
        else:
            colors = np.broadcast_to(
                colors_full.mean(axis=0, keepdims=True), (len(verts_s), 3)
            ).copy()
        normals = _recompute_normals(verts_s, faces_s)
        body_ids = np.full((len(verts_s),), body_i, dtype=np.int64)

        all_v.append(verts_s)
        all_f.append(faces_s + v_offset)
        all_c.append(colors.astype(np.float32))
        all_n.append(normals)
        all_body.append(body_ids)
        v_offset += len(verts_s)

    if not all_v:
        raise ValueError(f"no non-empty visual meshes for entity {name!r}")

    if face_keep < 1.0:
        print(
            f"[mesh_cache] '{name}' face_keep={face_keep:.2f}: "
            f"{faces_in:,} → {faces_out:,} faces "
            f"({100.0 * faces_out / max(faces_in, 1):.1f}% kept)"
        )

    return CachedEntityMesh(
        name=name,
        entity=entity,
        verts_local=torch.as_tensor(np.concatenate(all_v, axis=0), device=device),
        faces=torch.as_tensor(np.concatenate(all_f, axis=0), device=device, dtype=torch.int32),
        colors=torch.as_tensor(np.concatenate(all_c, axis=0), device=device),
        normals_local=torch.as_tensor(np.concatenate(all_n, axis=0), device=device),
        body_index=torch.as_tensor(np.concatenate(all_body, axis=0), device=device),
        num_bodies=int(entity.num_bodies),
    )


def body_link_pose_to_mat(
    body_pose_w: torch.Tensor,
    *,
    origin_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """``body_link_pose_w`` ``[B, L, 7]`` → homogeneous ``[B, L, 4, 4]``."""
    pos = body_pose_w[..., :3]
    if origin_w is not None:
        pos = pos - origin_w.unsqueeze(1)
    rot = matrix_from_quat(body_pose_w[..., 3:7])
    b, l = body_pose_w.shape[:2]
    t = torch.eye(4, device=body_pose_w.device, dtype=body_pose_w.dtype)
    t = t.expand(b, l, 4, 4).clone()
    t[..., :3, :3] = rot
    t[..., :3, 3] = pos
    return t


@torch.no_grad()
def pose_verts(
    mesh: CachedEntityMesh,
    body_T: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``body_T[B,L,4,4]`` → world verts/normals ``[B,V,3]``."""
    if body_T.dim() == 3:
        body_T = body_T.unsqueeze(0)
    t_v = body_T[:, mesh.body_index]
    verts = torch.einsum("bvij,vj->bvi", t_v[:, :, :3, :3], mesh.verts_local) + t_v[
        :, :, :3, 3
    ]
    normals = torch.einsum("bvij,vj->bvi", t_v[:, :, :3, :3], mesh.normals_local)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return verts, normals


def merge_rgbd(
    rgb_acc: torch.Tensor | None,
    depth_acc: torch.Tensor | None,
    mask_acc: torch.Tensor | None,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the closer hit across sequential entity renders."""
    if rgb_acc is None:
        return rgb, depth, mask
    closer = mask & ((~mask_acc) | (depth < depth_acc))
    rgb_acc = torch.where(closer.unsqueeze(-1), rgb, rgb_acc)
    depth_acc = torch.where(closer, depth, depth_acc)
    mask_acc = mask_acc | mask
    return rgb_acc, depth_acc, mask_acc
