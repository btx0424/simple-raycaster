"""Per-mesh PBR tables (G1 name heuristics + vertex-color extraction)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import trimesh

from ..mesh_cache import _recompute_normals, _vertex_colors

# Trimesh ColorVisuals default (102,102,102,255) — not an authored color.
_TRIMESH_DEFAULT_FACE = np.array([102.0, 102.0, 102.0], dtype=np.float32)

# rubber / painted-metal / plastic — roughness, metallic
_RUBBER = (0.92, 0.00)
_HAND = (0.62, 0.02)
_PAINTED = (0.32, 0.55)
_HEAD = (0.40, 0.08)
_GROUND = (0.95, 0.00)
_DEFAULT = (0.45, 0.12)


def g1_roughness_metallic(name: str) -> tuple[float, float]:
    """Heuristic roughness/metallic for a Unitree G1 (or similar) body name."""
    n = name.lower()
    if any(k in n for k in ("ground", "floor", "plane", "terrain")):
        return _GROUND
    if any(k in n for k in ("ankle", "foot", "rubber", "sole")):
        return _RUBBER
    if any(
        k in n
        for k in (
            "hand",
            "thumb",
            "index",
            "middle",
            "ring",
            "pinky",
            "finger",
            "wrist",
        )
    ):
        return _HAND
    if any(k in n for k in ("head", "logo")):
        return _HEAD
    if any(
        k in n
        for k in (
            "pelvis",
            "torso",
            "waist",
            "hip",
            "knee",
            "shoulder",
            "elbow",
            "contour",
        )
    ):
        return _PAINTED
    return _DEFAULT


def materials_for_names(
    names: Sequence[str],
    *,
    default_roughness: float = 0.45,
    default_metallic: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-mesh roughness/metallic arrays for ``names`` (empty name → defaults)."""
    r = np.empty(len(names), dtype=np.float32)
    m = np.empty(len(names), dtype=np.float32)
    for i, name in enumerate(names):
        if not name:
            r[i] = default_roughness
            m[i] = default_metallic
        else:
            rr, mm = g1_roughness_metallic(str(name))
            r[i] = rr
            m[i] = mm
    return r, m


def mesh_has_authored_colors(mesh: trimesh.Trimesh) -> bool:
    """True when the trimesh has vertex or non-default face colors."""
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return False
    kind = getattr(visual, "kind", None)
    if kind == "vertex":
        vc = getattr(visual, "vertex_colors", None)
        return vc is not None and np.asarray(vc).ndim == 2
    if kind == "face":
        fc = getattr(visual, "face_colors", None)
        if fc is None:
            return False
        rgb = np.asarray(fc)[:, :3].astype(np.float32)
        if rgb.size == 0:
            return False
        if not np.allclose(rgb, rgb[0], atol=1.5):
            return True
        return not np.allclose(rgb[0], _TRIMESH_DEFAULT_FACE, atol=2.0)
    return False


def vertex_colors_or_albedo(
    mesh: trimesh.Trimesh,
    albedo: np.ndarray,
) -> np.ndarray:
    """Barycentric-ready vertex RGB. Falls back to a constant per-mesh albedo."""
    n_verts = int(len(mesh.vertices))
    alb = np.asarray(albedo, dtype=np.float32).reshape(3)
    if mesh_has_authored_colors(mesh):
        return _vertex_colors(mesh, n_verts)
    return np.broadcast_to(alb, (n_verts, 3)).copy()


def vertex_normals_local(mesh: trimesh.Trimesh) -> np.ndarray:
    """Smooth vertex normals in the mesh-local frame."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return _recompute_normals(verts, faces)


def pack_vertex_attrs(
    meshes: Sequence[trimesh.Trimesh],
    albedos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate per-mesh vertex colors/normals.

    Returns:
        ``(vert_colors [V,3], vert_normals [V,3], vert_offset [M])`` — ``vert_offset[m]``
        is the start index of mesh ``m`` in the packed arrays.
    """
    albedos = np.asarray(albedos, dtype=np.float32).reshape(-1, 3)
    if albedos.shape[0] != len(meshes):
        raise ValueError(f"albedos {albedos.shape[0]} != n_meshes {len(meshes)}")
    colors: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    offsets = np.empty(len(meshes), dtype=np.int32)
    off = 0
    for i, mesh in enumerate(meshes):
        offsets[i] = off
        colors.append(vertex_colors_or_albedo(mesh, albedos[i]))
        normals.append(vertex_normals_local(mesh))
        off += int(len(mesh.vertices))
    if not colors:
        raise ValueError("pack_vertex_attrs: no meshes")
    return (
        np.concatenate(colors, axis=0).astype(np.float32, copy=False),
        np.concatenate(normals, axis=0).astype(np.float32, copy=False),
        offsets,
    )


def pack_vertex_attrs_from_wp(
    meshes_wp: Sequence,
    albedos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same packing from ``wp.Mesh`` (no authored vertex colors — albedo fill)."""
    albedos = np.asarray(albedos, dtype=np.float32).reshape(-1, 3)
    if albedos.shape[0] != len(meshes_wp):
        raise ValueError(f"albedos {albedos.shape[0]} != n_meshes {len(meshes_wp)}")
    colors: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    offsets = np.empty(len(meshes_wp), dtype=np.int32)
    off = 0
    for i, mesh in enumerate(meshes_wp):
        pts = np.asarray(mesh.points.numpy(), dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(mesh.indices.numpy(), dtype=np.int32).reshape(-1, 3)
        offsets[i] = off
        colors.append(np.broadcast_to(albedos[i], (len(pts), 3)).copy())
        normals.append(_recompute_normals(pts, faces))
        off += int(len(pts))
    return (
        np.concatenate(colors, axis=0).astype(np.float32, copy=False),
        np.concatenate(normals, axis=0).astype(np.float32, copy=False),
        offsets,
    )
