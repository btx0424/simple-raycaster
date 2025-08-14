import trimesh
import numpy as np
import warp as wp
import torch

from pxr import Usd

from .utils import (
    quat_rotate_inverse,
    find_matching_prims,
    get_mesh_prims_subtree,
    usd2trimesh,
    trimesh2wp,
)


@wp.kernel
def multi_mesh_raycast_kernel(
    meshes: wp.array(dtype=wp.uint64),
    ray_starts: wp.array(dtype=wp.vec3, ndim=3),
    ray_dirs: wp.array(dtype=wp.vec3, ndim=3),
    max_dist: float,
    hit_distances: wp.array(dtype=wp.float32, ndim=3),
):
    i, mesh_id, ray_id = wp.tid()
    mesh = meshes[mesh_id]
    ray_start = ray_starts[i, mesh_id, ray_id]
    ray_dir = ray_dirs[i, mesh_id, ray_id]
    result = wp.mesh_query_ray(
        mesh,
        ray_start,
        ray_dir,
        max_dist,
    )
    if result.result:
        t = result.t
    else:
        t = max_dist
    hit_distances[i, mesh_id, ray_id] = t


class MultiMeshRaycaster:
    """
    Raycaster that supports multiple and dynamic meshes.

    Args:
        meshes: List of wp.Mesh objects.
    """
    def __init__(self, meshes: list[wp.Mesh], device: str):
        self.meshes = meshes
        self.n_meshes = len(meshes)
        self.meshes_array = wp.array([mesh.id for mesh in meshes], dtype=wp.uint64)
        self.device = device

        self.n_points = sum(mesh.points.shape[0] for mesh in meshes)
        self.n_faces = sum(mesh.indices.reshape((-1, 3)).shape[0] for mesh in meshes)
    
    def __repr__(self) -> str:
        return f"MultiMeshRaycaster(n_meshes={self.n_meshes}, n_points={self.n_points}, n_faces={self.n_faces})"

    def raycast(
        self,
        mesh_pos_w: torch.Tensor, # [N, n_meshes, 3]
        mesh_quat_w: torch.Tensor, # [N, n_meshes, 4]
        ray_starts_w: torch.Tensor, # [N, n_rays, 3]
        ray_dirs_w: torch.Tensor, # [N, n_rays, 3]
        max_dist: float=100.0,
    ):
        n_rays = ray_dirs_w.shape[1]
        N = mesh_pos_w.shape[0]
        mesh_pos_w = mesh_pos_w.reshape(N, self.n_meshes, 1, 3) # [N, n_meshes, 1, 3]
        mesh_quat_w = mesh_quat_w.reshape(N, self.n_meshes, 1, 4) # [N, n_meshes, 1, 4]
        _ray_starts_w = ray_starts_w.reshape(N, 1, n_rays, 3) # [N, 1, n_rays, 3]
        _ray_dirs_w = ray_dirs_w.reshape(N, 1, n_rays, 3) # [N, 1, n_rays, 3]
        
        # convert to mesh frame
        ray_starts_b = quat_rotate_inverse(mesh_quat_w, _ray_starts_w - mesh_pos_w)
        ray_dirs_b = quat_rotate_inverse(mesh_quat_w, _ray_dirs_w)

        ray_starts_wp = wp.from_torch(ray_starts_b, dtype=wp.vec3, return_ctype=True)
        ray_dirs_wp = wp.from_torch(ray_dirs_b, dtype=wp.vec3, return_ctype=True)
        hit_distances = torch.empty(N, self.n_meshes, n_rays, device=ray_starts_w.device)
        launch: wp.Launch = wp.launch(
            multi_mesh_raycast_kernel,
            dim=(N, self.n_meshes, n_rays),
            inputs=[
                self.meshes_array,
                ray_starts_wp,
                ray_dirs_wp,
                max_dist,
            ],
            outputs=[wp.from_torch(hit_distances, dtype=wp.float32),],
            device=self.device,
        )
        hit_distances = hit_distances.min(dim=1).values
        hit_positions = ray_starts_w + hit_distances.unsqueeze(-1) * ray_dirs_w
        return hit_positions, hit_distances
    
    @classmethod
    def from_prim_paths(cls, paths: list[str], stage: Usd.Stage, device: str, simplify_factor: float=0.0):
        """
        Args:
            paths: List of prim paths (can be regex) to find, e.g. ["World/.*/visuals"].
            stage: The USD stage to search in.
            device: The device to use for the raycaster.
            simplify_factor: The factor to simplify the meshes. 0.0 means no simplification.
        """
        meshes_wp = []
        all_prims = []
        for path in paths:
            if not (prims := find_matching_prims(path, stage)):
                raise ValueError(f"No prims found for path {path}")
            all_prims.extend(prims)
        
        n_faces_before = 0
        n_faces_after = 0
        for prim in all_prims:
            mesh_prims = get_mesh_prims_subtree(prim)
            meshes_trimesh = []
            for mesh_prim in mesh_prims:
                mesh = usd2trimesh(mesh_prim)
                n_faces_before += mesh.faces.shape[0]
                if simplify_factor > 0.0:
                    mesh = mesh.simplify_quadric_decimation(simplify_factor)
                n_faces_after += mesh.faces.shape[0]
                meshes_trimesh.append(mesh)
            mesh_combined: trimesh.Trimesh = trimesh.util.concatenate(meshes_trimesh)
            mesh_combined.merge_vertices()
            meshes_wp.append(trimesh2wp(mesh_combined, device))
        
        if n_faces_before != n_faces_after:
            print(f"Simplified {n_faces_before} to {n_faces_after} faces")
        
        return cls(meshes_wp, device)

