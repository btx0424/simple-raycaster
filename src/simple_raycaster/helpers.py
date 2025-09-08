import trimesh
import warp as wp
import numpy as np
import torch


def trimesh2wp(mesh: trimesh.Trimesh, device):
    """
    Convert a trimesh.Trimesh object to a wp.Mesh object.
    """
    return wp.Mesh(
        points=wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(mesh.faces.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )


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

