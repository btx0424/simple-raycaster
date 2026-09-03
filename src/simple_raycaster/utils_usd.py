"""USD → trimesh extraction for Isaac / Warp consumers.

Supports ``Mesh`` plus common ``UsdGeom`` primitives (``Cube``, ``Sphere``,
``Cylinder``, ``Capsule``, ``Cone``). Primitives are tessellated for raycasting;
callers that want native Viser shapes can use :func:`get_geom_parts_from_prim`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import trimesh
from pxr import Usd, UsdGeom

# Tessellated for Warp / RaycastCamera / MeshRegistry.
GEOM_TYPE_NAMES = ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")

# Viser SceneApi natives (no capsule/cone — tessellate those).
ViserKind = Literal["box", "sphere", "cylinder", "mesh"]
VISER_NATIVE_TYPES = frozenset({"Cube", "Sphere", "Cylinder"})


@dataclass(frozen=True)
class UsdGeomPart:
    """One renderable child under a body visuals / collisions prim."""

    type_name: str
    """USD type name (``Mesh``, ``Cube``, …)."""

    local_pos: np.ndarray
    """Translation in the parent body frame, shape ``(3,)``."""

    local_quat_wxyz: np.ndarray
    """Rotation in the parent body frame (WXYZ), shape ``(4,)``."""

    mesh: trimesh.Trimesh
    """Always-populated tessellation (body-local vertices)."""

    viser_kind: ViserKind
    """``box`` / ``sphere`` / ``cylinder`` when Viser has a native API, else ``mesh``."""

    box_dimensions: tuple[float, float, float] | None = None
    sphere_radius: float | None = None
    cylinder_radius: float | None = None
    cylinder_height: float | None = None


def get_trimesh_from_prim(
    prim: Usd.Prim, predicate: Callable[[Usd.Prim], bool] = lambda _: True
):
    """Extract Mesh + primitive geometry under ``prim`` as one combined trimesh.

    The result is expressed in the frame of ``prim``: each child's local-to-world
    transform is composed with the inverse of ``prim``'s **rigid** (rotation +
    translation) world transform. Scale/shear anywhere in the hierarchy —
    including on ``prim`` itself — is baked into the vertices, since runtime
    poses (e.g. ``body_link_pose_w``) carry no scale.
    """
    parts = get_geom_parts_from_prim(prim, predicate)
    if not parts:
        raise ValueError(f"No mesh primitives found in {prim.GetPath().pathString}")
    meshes = [p.mesh for p in parts]
    combined: trimesh.Trimesh = (
        meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    )
    combined.merge_vertices()
    return combined


def get_geom_parts_from_prim(
    prim: Usd.Prim, predicate: Callable[[Usd.Prim], bool] = lambda _: True
) -> list[UsdGeomPart]:
    """Return per-child geometry parts under ``prim`` (body-local frames)."""
    mesh_prims = get_mesh_prims_subtree(prim, predicate)
    if not mesh_prims:
        return []

    time = Usd.TimeCode.Default()
    parent_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
    parent_rigid_inv = parent_transform.RemoveScaleShear().GetInverse()

    parts: list[UsdGeomPart] = []
    for mesh_prim in mesh_prims:
        transform = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(time)
        if not mesh_prim.IsInPrototype():
            transform = transform * parent_rigid_inv
        # Gf matrix is column-major; NumPy expects row-major 4x4.
        transform_np = np.array(transform, dtype=np.float64).transpose()
        parts.append(_make_geom_part(mesh_prim, transform_np))
    return parts


def get_mesh_prims_subtree(
    prim: Usd.Prim, predicate: Callable[[Usd.Prim], bool] = lambda _: True
):
    """Recursively collect Mesh + primitive geom prims under ``prim``."""
    if prim.IsInstance():
        prim = prim.GetPrototype()
    mesh_prims = []
    all_prims = [prim]
    while len(all_prims) > 0:
        child_prim = all_prims.pop(0)
        type_name = child_prim.GetTypeName()
        if type_name in GEOM_TYPE_NAMES and predicate(child_prim):
            mesh_prims.append(child_prim)
        all_prims += child_prim.GetChildren()
    return mesh_prims


def _get_cube_extents(cube_prim: Usd.Prim) -> np.ndarray:
    """Edge lengths for a ``UsdGeom.Cube`` (size or extent; default 2³)."""
    cube = UsdGeom.Cube(cube_prim)
    time = Usd.TimeCode.Default()
    size_attr = cube.GetSizeAttr()
    if size_attr:
        size_val = size_attr.Get(time)
        if size_val is not None:
            size = np.array(size_val, dtype=np.float64)
            if size.shape == ():
                size = np.array([size, size, size])
            return size
    extent_attr = cube.GetExtentAttr()
    if extent_attr:
        extent = np.array(extent_attr.Get(time), dtype=np.float64)
        if extent is not None and len(extent) >= 6:
            return np.array(
                [
                    extent[3] - extent[0],
                    extent[4] - extent[1],
                    extent[5] - extent[2],
                ]
            )
    return np.array([2.0, 2.0, 2.0])


def _get_radius(prim: Usd.Prim, default: float = 1.0) -> float:
    time = Usd.TimeCode.Default()
    type_name = prim.GetTypeName()
    geom = {
        "Sphere": UsdGeom.Sphere,
        "Cylinder": UsdGeom.Cylinder,
        "Capsule": UsdGeom.Capsule,
        "Cone": UsdGeom.Cone,
    }[type_name](prim)
    val = geom.GetRadiusAttr().Get(time)
    return float(default if val is None else val)


def _get_height(prim: Usd.Prim, default: float = 2.0) -> float:
    time = Usd.TimeCode.Default()
    type_name = prim.GetTypeName()
    geom = {
        "Cylinder": UsdGeom.Cylinder,
        "Capsule": UsdGeom.Capsule,
        "Cone": UsdGeom.Cone,
    }[type_name](prim)
    val = geom.GetHeightAttr().Get(time)
    return float(default if val is None else val)


def _get_axis(prim: Usd.Prim) -> str:
    time = Usd.TimeCode.Default()
    type_name = prim.GetTypeName()
    if type_name not in ("Cylinder", "Capsule", "Cone"):
        return "Z"
    geom = {
        "Cylinder": UsdGeom.Cylinder,
        "Capsule": UsdGeom.Capsule,
        "Cone": UsdGeom.Cone,
    }[type_name](prim)
    attr = geom.GetAxisAttr()
    val = attr.Get(time) if attr else None
    if val is None:
        return "Z"
    return str(val)


def _axis_rotation(axis: str) -> np.ndarray:
    """4x4 that maps trimesh Z-up primitives onto USD ``axis``."""
    axis = axis.upper()
    if axis == "Z":
        return np.eye(4)
    if axis == "X":
        # Z -> X: rotate +90° about Y: (x,y,z) -> (z, y, -x)
        return np.array(
            [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
    if axis == "Y":
        # Z -> Y: rotate -90° about X: (x,y,z) -> (x, z, -y)
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
    return np.eye(4)


def _triangulate(indices: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Fan-triangulate USD face data (mixed polygon sizes) into an [F, 3] array."""
    if np.all(counts == 3):
        return indices.reshape(-1, 3)
    faces = []
    offset = 0
    for c in counts:
        poly = indices[offset : offset + c]
        for i in range(1, c - 1):
            faces.append((poly[0], poly[i], poly[i + 1]))
        offset += c
    return np.asarray(faces, dtype=np.int64)


def usd2trimesh(prim: Usd.Prim) -> trimesh.Trimesh:
    """Convert a USD Mesh or primitive prim to a local-frame trimesh.

    Vertices are in the prim's local frame (xformOps, including scale, are
    applied by the caller via the prim's transform). Cylinder / Capsule / Cone
    are created Z-aligned then rotated to the USD ``axis``.
    """
    type_name = prim.GetTypeName()
    if type_name == "Cube":
        return trimesh.creation.box(extents=_get_cube_extents(prim))
    if type_name == "Sphere":
        return trimesh.creation.icosphere(
            subdivisions=3, radius=_get_radius(prim)
        )
    if type_name == "Cylinder":
        mesh = trimesh.creation.cylinder(
            radius=_get_radius(prim), height=_get_height(prim)
        )
        mesh.apply_transform(_axis_rotation(_get_axis(prim)))
        return mesh
    if type_name == "Capsule":
        mesh = trimesh.creation.capsule(
            radius=_get_radius(prim), height=_get_height(prim)
        )
        mesh.apply_transform(_axis_rotation(_get_axis(prim)))
        return mesh
    if type_name == "Cone":
        mesh = trimesh.creation.cone(
            radius=_get_radius(prim), height=_get_height(prim)
        )
        mesh.apply_transform(_axis_rotation(_get_axis(prim)))
        return mesh
    if type_name != "Mesh":
        raise ValueError(f"Unsupported USD geom type {type_name!r}")
    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    faces = _triangulate(indices, counts)
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        faces = faces[:, ::-1]
    return trimesh.Trimesh(vertices, faces)


def _rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to WXYZ quaternion."""
    # Shepperd's method
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def _decompose_transform(
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose 4x4 into ``(pos, quat_wxyz, scale_xyz)`` (column scales)."""
    pos = T[:3, 3].copy()
    M = T[:3, :3]
    scale = np.linalg.norm(M, axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    R = M / scale
    # Fix improper rotations from negative scales / reflections.
    if np.linalg.det(R) < 0:
        scale[0] *= -1.0
        R[:, 0] *= -1.0
    quat = _rotation_matrix_to_quat_wxyz(R)
    return pos, quat, scale


def _make_geom_part(prim: Usd.Prim, transform_np: np.ndarray) -> UsdGeomPart:
    type_name = prim.GetTypeName()
    # Unoriented primitive (Z-up for cylinders/capsules/cones), then:
    #   tessellation uses transform_np @ axis_T
    #   native Viser pose uses decompose(transform_np) composed with axis quat
    if type_name == "Cube":
        base = trimesh.creation.box(extents=_get_cube_extents(prim))
        axis_T = np.eye(4)
    elif type_name == "Sphere":
        base = trimesh.creation.icosphere(subdivisions=3, radius=_get_radius(prim))
        axis_T = np.eye(4)
    elif type_name == "Cylinder":
        base = trimesh.creation.cylinder(
            radius=_get_radius(prim), height=_get_height(prim)
        )
        axis_T = _axis_rotation(_get_axis(prim))
    elif type_name == "Capsule":
        base = trimesh.creation.capsule(
            radius=_get_radius(prim), height=_get_height(prim)
        )
        axis_T = _axis_rotation(_get_axis(prim))
    elif type_name == "Cone":
        base = trimesh.creation.cone(
            radius=_get_radius(prim), height=_get_height(prim)
        )
        axis_T = _axis_rotation(_get_axis(prim))
    elif type_name == "Mesh":
        base = usd2trimesh(prim)
        axis_T = np.eye(4)
    else:
        raise ValueError(f"Unsupported USD geom type {type_name!r}")

    baked = base.copy()
    baked.apply_transform(transform_np @ axis_T)

    pos, quat_x, scale = _decompose_transform(transform_np)
    axis_quat = _rotation_matrix_to_quat_wxyz(axis_T[:3, :3])
    # quat_mul(a, b): apply b first, then a  (body <- xform <- axis)
    quat = _quat_mul_wxyz(quat_x, axis_quat)

    viser_kind: ViserKind = "mesh"
    box_dimensions = None
    sphere_radius = None
    cylinder_radius = None
    cylinder_height = None

    if type_name == "Cube":
        extents = _get_cube_extents(prim) * scale
        box_dimensions = (float(extents[0]), float(extents[1]), float(extents[2]))
        viser_kind = "box"
    elif type_name == "Sphere" and np.allclose(scale, scale[0], rtol=1e-3, atol=1e-5):
        sphere_radius = float(_get_radius(prim) * scale[0])
        viser_kind = "sphere"
    elif type_name == "Cylinder":
        axis = _get_axis(prim).upper()
        if axis == "Z":
            radial = 0.5 * (scale[0] + scale[1])
            axial = scale[2]
            ok = np.isclose(scale[0], scale[1], rtol=1e-3, atol=1e-5)
        elif axis == "Y":
            radial = 0.5 * (scale[0] + scale[2])
            axial = scale[1]
            ok = np.isclose(scale[0], scale[2], rtol=1e-3, atol=1e-5)
        else:  # X
            radial = 0.5 * (scale[1] + scale[2])
            axial = scale[0]
            ok = np.isclose(scale[1], scale[2], rtol=1e-3, atol=1e-5)
        if ok:
            cylinder_radius = float(_get_radius(prim) * radial)
            cylinder_height = float(_get_height(prim) * axial)
            viser_kind = "cylinder"

    return UsdGeomPart(
        type_name=type_name,
        local_pos=pos.astype(np.float64),
        local_quat_wxyz=quat.astype(np.float64),
        mesh=baked,
        viser_kind=viser_kind,
        box_dimensions=box_dimensions,
        sphere_radius=sphere_radius,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height,
    )


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product ``q1 ⊗ q2`` (WXYZ), apply q2 then q1."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def usd2wp(prim: Usd.Prim, device):
    """Convert a USD Mesh or primitive prim to a ``wp.Mesh``."""
    from .helpers import trimesh2wp

    return trimesh2wp(usd2trimesh(prim), device)


def find_matching_prims(prim_path_regex: str, stage: Usd.Stage):
    if not prim_path_regex.startswith("^"):
        prim_path_regex = "^" + prim_path_regex
    if not prim_path_regex.endswith("$"):
        prim_path_regex = prim_path_regex + "$"
    pattern = re.compile(prim_path_regex)
    results = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if pattern.match(prim_path) is not None:
            results.append(prim)
    return results
