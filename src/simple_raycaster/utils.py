import torch
import trimesh
import numpy as np
import warp as wp
import re

from pxr import Usd, UsdGeom, UsdPhysics


def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor):
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    xyz = quat[..., 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[..., 0:1] * t + xyz.cross(t, dim=-1))


def get_mesh_prims_subtree(prim: Usd.Prim):
    """
    Recursively get all mesh primitives from a USD prim.
    """
    if prim.IsInstance():
        prim = prim.GetPrototype()
    mesh_prims = []
    all_prims = [prim]
    while len(all_prims) > 0:
        child_prim = all_prims.pop(0)
        if child_prim.GetTypeName() == "Mesh":
            mesh_prims.append(child_prim)
        all_prims += child_prim.GetChildren()
    return mesh_prims


def usd2trimesh(prim: Usd.Prim, apply_transform: bool=True):
    """
    Convert a USD prim to a trimesh.Trimesh object.

    Args:
        prim: The USD prim to convert.
        apply_transform: Whether to apply the local transform to the mesh.
    """
    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get())
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    mesh = trimesh.Trimesh(vertices, faces.reshape(-1, 3))
    if apply_transform:
        xform = UsdGeom.Xformable(prim)
        transform = xform.GetLocalTransformation(Usd.TimeCode.Default())
        translation = np.array(transform.ExtractTranslation())
        orientation = transform.ExtractRotationQuat()
        orientation = np.array([orientation.GetReal(), *orientation.GetImaginary()])
        transform = trimesh.transformations.concatenate_matrices(
            trimesh.transformations.translation_matrix(translation),
            trimesh.transformations.quaternion_matrix(orientation),
        )
        mesh.apply_transform(transform)
    return mesh


def usd2wp(prim: Usd.Prim, device):
    """
    Convert a USD prim to a wp.Mesh object.
    """
    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get())
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    return wp.Mesh(
        points=wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(faces.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )


def trimesh2wp(mesh: trimesh.Trimesh, device):
    """
    Convert a trimesh.Trimesh object to a wp.Mesh object.
    """
    return wp.Mesh(
        points=wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(mesh.faces.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )


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

