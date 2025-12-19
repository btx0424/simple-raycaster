import trimesh
import numpy as np
import warp as wp
import re
from typing import Callable

from pxr import Usd, UsdGeom

def get_trimesh_from_prim(prim: Usd.Prim, predicate: Callable[[Usd.Prim], bool] = lambda _: True):
    mesh_prims = get_mesh_prims_subtree(prim, predicate)
    if len(mesh_prims) == 0:
        raise ValueError(f"No mesh primitives found in {prim.GetPath().pathString}")
    
    trimesh_list = []
    time = Usd.TimeCode.Default()
    parent_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
    
    # print(prim, parent_transform.ExtractTranslation(), parent_transform.ExtractRotationQuat())
    for mesh_prim in mesh_prims:
        mesh = usd2trimesh(mesh_prim)
        transform = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(time)
        if mesh_prim.IsInPrototype():
            pass # no global transform for prototype meshes
        else:
            transform = transform * parent_transform.GetInverse()

        transform_np = np.array(transform).transpose()
        mesh.apply_transform(transform_np)
        trimesh_list.append(mesh)
    trimesh_combined: trimesh.Trimesh = trimesh.util.concatenate(trimesh_list)
    trimesh_combined.merge_vertices()
    return trimesh_combined


def get_mesh_prims_subtree(prim: Usd.Prim, predicate: Callable[[Usd.Prim], bool] = lambda _: True):
    """
    Recursively get all mesh primitives from a USD prim.
    """
    if prim.IsInstance():
        prim = prim.GetPrototype()
    mesh_prims = []
    all_prims = [prim]
    while len(all_prims) > 0:
        child_prim = all_prims.pop(0)
        if child_prim.GetTypeName() == "Mesh" and predicate(child_prim):
            mesh_prims.append(child_prim)
        all_prims += child_prim.GetChildren()
    return mesh_prims


def usd2trimesh(prim: Usd.Prim):
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

