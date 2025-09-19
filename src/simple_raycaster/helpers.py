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


@wp.kernel(enable_backward=False)
def voxelize_kernel(
    # input
    hit_positions_b: wp.array(dtype=wp.vec3, ndim=2), # in body frame
    resolution: wp.vec3, # [3]
    # output
    voxel_grid: wp.array(dtype=wp.bool, ndim=4),
):
    i, ray_id = wp.tid()
    voxel_x_size = float(voxel_grid.shape[1]) * resolution.x
    voxel_y_size = float(voxel_grid.shape[2]) * resolution.y
    voxel_z_size = float(voxel_grid.shape[3]) * resolution.z
    
    hit_position_b = hit_positions_b[i, ray_id]

    x = int((hit_position_b.x + 0.5 * voxel_x_size) / resolution.x)
    if x < 0 or x >= voxel_grid.shape[1]:
        return
    y = int((hit_position_b.y + 0.5 * voxel_y_size) / resolution.y)
    if y < 0 or y >= voxel_grid.shape[2]:
        return
    z = int((hit_position_b.z + 0.5 * voxel_z_size) / resolution.z)
    if z < 0 or z >= voxel_grid.shape[3]:
        return
    voxel_grid[i, x, y, z] = True


def voxelize_wp(
    voxel_shape: tuple[int, int, int], # [3]
    resolution: tuple[float, float, float], # [3]
    hit_positions_b: torch.Tensor, # [N, n_rays, 3]
):
    """
    Voxelize the hit positions in the body frame.
    This implementation is much faster than `voxelize_torch`.

    Args:
        voxel_shape: The shape of the voxel grid.
        resolution: The resolution of the voxel grid.
        hit_positions_b: The hit positions in the body frame. Shape [N, n_rays, 3].

    Returns:
        The voxel grid. Shape [N, voxel_shape[0], voxel_shape[1], voxel_shape[2]].

    """

    n_rays = hit_positions_b.shape[1]
    N = hit_positions_b.shape[0]
    device = hit_positions_b.device
    device_wp = wp.device_from_torch(device)

    voxel_grid = torch.zeros(N, *voxel_shape, dtype=torch.bool, device=device)
    resolution = wp.vec3(resolution[0], resolution[1], resolution[2])
    wp.launch(
        voxelize_kernel,
        dim=(N, n_rays),
        inputs=[
            wp.from_torch(hit_positions_b, dtype=wp.vec3, return_ctype=True),
            resolution,
        ],
        outputs=[
            wp.from_torch(voxel_grid, dtype=wp.bool, return_ctype=True),
        ],
        device=device_wp,
        record_tape=False,
    )
    return voxel_grid


def voxelize_torch(
    voxel_shape: tuple[int, int, int],
    resolution: tuple[float, float, float],
    hit_positions_b: torch.Tensor, # [N, n_rays, 3]
):
    """
    Voxelize the hit positions in the body frame. This implementation is slow.

    Args:
        voxel_shape: The shape of the voxel grid.
        resolution: The resolution of the voxel grid.
        hit_positions_b: The hit positions in the body frame. Shape [N, n_rays, 3].

    Returns:
        The voxel grid. Shape [N, voxel_shape[0], voxel_shape[1], voxel_shape[2]].
    """

    N = hit_positions_b.shape[0]
    n_rays = hit_positions_b.shape[1]
    device = hit_positions_b.device
    voxel_grid = torch.zeros(N, *voxel_shape, dtype=torch.bool, device=device)
    voxel_shape = torch.tensor(voxel_shape, dtype=torch.long, device=device)
    resolution = torch.tensor(resolution, device=device)
    
    half_voxel_size = voxel_shape / 2 * resolution
    
    valid = (hit_positions_b < half_voxel_size) & (hit_positions_b > -half_voxel_size)
    valid = valid.all(dim=-1) # [N, n_rays]
    hit_positions_b = hit_positions_b[valid]
    grid_idx = (hit_positions_b + half_voxel_size / resolution).long()
    x, y, z = grid_idx.unbind(-1)
    i = torch.arange(N, device=device).unsqueeze(1).expand(N, n_rays)
    i = i[valid]
    voxel_grid[i, x, y, z] = True
    return voxel_grid

