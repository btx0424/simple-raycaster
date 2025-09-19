import numpy as np
import torch
import trimesh
import warp as wp
import mujoco

from .helpers import quat_rotate_inverse, trimesh2wp

@wp.kernel(enable_backward=False)
def raycast_kernel(
    meshes: wp.array(dtype=wp.uint64),
    ray_starts: wp.array(dtype=wp.vec3, ndim=3),
    ray_dirs: wp.array(dtype=wp.vec3, ndim=3),
    enabled: wp.array(dtype=wp.bool, ndim=1),
    min_dist: float,
    max_dist: float,
    hit_distances: wp.array(dtype=wp.float32, ndim=3),
):
    i, mesh_id, ray_id = wp.tid()
    if not enabled[i]:
        hit_distances[i, mesh_id, ray_id] = wp.INF
        return
    mesh = meshes[mesh_id]
    ray_start = ray_starts[i, mesh_id, ray_id]
    ray_dir = ray_dirs[i, mesh_id, ray_id]
    result = wp.mesh_query_ray(
        mesh,
        ray_start,
        ray_dir,
        max_dist,
    )
    t = max_dist
    if result.result and result.t >= min_dist:
        t = result.t
    hit_distances[i, mesh_id, ray_id] = t


@wp.kernel(enable_backward=False)
def transform_and_raycast_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),
    ray_starts_w: wp.array(dtype=wp.vec3, ndim=2),
    ray_dirs_w: wp.array(dtype=wp.vec3, ndim=2),
    enabled: wp.array(dtype=wp.bool, ndim=1),
    min_dist: float,
    max_dist: float,
    hit_distances: wp.array(dtype=wp.float32, ndim=3),
):
    i, mesh_id, ray_id = wp.tid()
    if not enabled[i]:
        hit_distances[i, mesh_id, ray_id] = wp.INF
        return
    
    # transform ray starts and dirs to mesh frame
    quat_wxyz = mesh_quat_w[i, mesh_id]
    quat_xyzw = wp.quat(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0])
    ray_start_b = wp.quat_rotate_inv(
        quat_xyzw,
        ray_starts_w[i, ray_id] - mesh_pos_w[i, mesh_id],
    )
    ray_dir_b = wp.quat_rotate_inv(
        quat_xyzw,
        ray_dirs_w[i, ray_id],
    )
    
    result = wp.mesh_query_ray(
        meshes[mesh_id],
        ray_start_b,
        ray_dir_b,
        max_dist,
    )
    t = max_dist
    if result.result and result.t >= min_dist:
        t = result.t
    hit_distances[i, mesh_id, ray_id] = t


class MultiMeshRaycaster:
    """
    Raycaster that supports multiple and dynamic meshes.

    Args:
        meshes: List of wp.Mesh objects.
    """

    def __init__(self, meshes: list[wp.Mesh], device: str):
        self.meshes = meshes
        self.device = device
        self.meshes_array = wp.array([mesh.id for mesh in self.meshes], device=device, dtype=wp.uint64)
        self.mesh_names = None
    
    def add_mesh(self, mesh):
        if isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh2wp(mesh, self.device)
        self.meshes.append(mesh)
        self.meshes_array = wp.array([mesh.id for mesh in self.meshes], device=self.device, dtype=wp.uint64)

    @property
    def n_points(self):
        return sum(mesh.points.shape[0] for mesh in self.meshes)

    @property
    def n_faces(self):
        return sum(mesh.indices.reshape((-1, 3)).shape[0] for mesh in self.meshes)

    @property
    def n_meshes(self):
        return len(self.meshes)

    def __repr__(self) -> str:
        return f"MultiMeshRaycaster(n_meshes={self.n_meshes}, n_points={self.n_points}, n_faces={self.n_faces})"

    def raycast(
        self,
        mesh_pos_w: torch.Tensor,  # [N, n_meshes, 3]
        mesh_quat_w: torch.Tensor,  # [N, n_meshes, 4]
        ray_starts_w: torch.Tensor,  # [N, n_rays, 3]
        ray_dirs_w: torch.Tensor,  # [N, n_rays, 3]
        enabled: torch.Tensor=None,  # [N]
        min_dist: float = 0.0,
        max_dist: float = 100.0,
    ):
        """
        Args:
            mesh_pos_w: The position of the meshes in the world frame. Shape [N, n_meshes, 3].
            mesh_quat_w: The orientation of the meshes in the world frame. Shape [N, n_meshes, 4].
            ray_starts_w: The starting points of the rays in the world frame. Shape [N, n_rays, 3].
            ray_dirs_w: The directions of the rays in the world frame. Shape [N, n_rays, 3].

            min_dist: The minimum distance to the mesh. Defaults to 0.0.
            max_dist: The maximum distance to the mesh. Defaults to 100.0.

        Returns:
            hit_positions: The positions of the hits in the world frame. Shape [N, n_rays, 3].
            hit_distances: The distances to the hits. Shape [N, n_rays].
        """
        n_rays = ray_dirs_w.shape[1]
        N = mesh_pos_w.shape[0]
        result_shape = (N, self.n_meshes, n_rays)

        mesh_pos_w = mesh_pos_w.reshape(N, self.n_meshes, 1, 3)  # [N, n_meshes, 1, 3]
        mesh_quat_w = mesh_quat_w.reshape(N, self.n_meshes, 1, 4)  # [N, n_meshes, 1, 4]

        if enabled is None:
            enabled = torch.ones(N, dtype=torch.bool, device=ray_starts_w.device)
        else:
            enabled = enabled.reshape(N,)

        # convert to mesh frame
        ray_starts_b = quat_rotate_inverse(mesh_quat_w, ray_starts_w.unsqueeze(1) - mesh_pos_w)
        ray_dirs_b = quat_rotate_inverse(mesh_quat_w, ray_dirs_w.unsqueeze(1))

        ray_starts_wp = wp.from_torch(ray_starts_b, dtype=wp.vec3, return_ctype=True)
        ray_dirs_wp = wp.from_torch(ray_dirs_b, dtype=wp.vec3, return_ctype=True)
        enabled_wp = wp.from_torch(enabled, dtype=wp.bool, return_ctype=True)

        hit_distances = torch.empty(result_shape, device=ray_starts_w.device)
        wp.launch(
            raycast_kernel,
            dim=(N, self.n_meshes, n_rays),
            inputs=[
                self.meshes_array,
                ray_starts_wp,
                ray_dirs_wp,
                enabled_wp,
                min_dist,
                max_dist,
            ],
            outputs=[
                wp.from_torch(hit_distances, dtype=wp.float32),
            ],
            device=self.device,
            record_tape=False,
        )
        
        hit_distances = hit_distances.min(dim=1).values # [N, n_rays]
        hit_positions = ray_starts_w + hit_distances.unsqueeze(-1) * ray_dirs_w # [N, n_rays, 3]
        return hit_positions, hit_distances
    
    def raycast_fused(
        self,
        mesh_pos_w: torch.Tensor,  # [N, n_meshes, 3]
        mesh_quat_w: torch.Tensor,  # [N, n_meshes, 4]
        ray_starts_w: torch.Tensor,  # [N, n_rays, 3]
        ray_dirs_w: torch.Tensor,  # [N, n_rays, 3]
        enabled: torch.Tensor=None,  # [N]
        min_dist: float = 0.0,
        max_dist: float = 100.0,
    ):
        """
        Args:
            mesh_pos_w: The position of the meshes in the world frame. Shape [N, n_meshes, 3].
            mesh_quat_w: The orientation of the meshes in the world frame. Shape [N, n_meshes, 4].
            ray_starts_w: The starting points of the rays in the world frame. Shape [N, n_rays, 3].
            ray_dirs_w: The directions of the rays in the world frame. Shape [N, n_rays, 3].
            enabled: The enabled flag for the rays. Shape [N].
            min_dist: The minimum distance to the mesh. Defaults to 0.0.
            max_dist: The maximum distance to the mesh. Defaults to 100.0.

        Returns:
            hit_positions: The positions of the hits in the world frame. Shape [N, n_rays, 3].
            hit_distances: The distances to the hits. Shape [N, n_rays].
        """
        n_rays = ray_dirs_w.shape[1]
        N = mesh_pos_w.shape[0]
        result_shape = (N, self.n_meshes, n_rays)

        if enabled is None:
            enabled = torch.ones(N, dtype=torch.bool, device=ray_starts_w.device)
        else:
            enabled = enabled.reshape(N,)

        hit_distances = torch.empty(result_shape, device=ray_starts_w.device)
        wp.launch(
            transform_and_raycast_kernel,
            dim=(N, self.n_meshes, n_rays),
            inputs=[
                self.meshes_array,
                wp.from_torch(mesh_pos_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(mesh_quat_w, dtype=wp.vec4, return_ctype=True),
                wp.from_torch(ray_starts_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(ray_dirs_w, dtype=wp.vec3, return_ctype=True),
                wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                min_dist,
                max_dist,
            ],
            outputs=[
                wp.from_torch(hit_distances, dtype=wp.float32),
            ],
            device=self.device,
            record_tape=False,
        )
        
        hit_distances = hit_distances.min(dim=1).values # [N, n_rays]
        hit_positions = ray_starts_w + hit_distances.unsqueeze(-1) * ray_dirs_w # [N, n_rays, 3]
        return hit_positions, hit_distances

    @classmethod
    def from_prim_paths(
        cls,
        paths: list[str],
        stage: "Usd.Stage",
        device: str,
        simplify_factor: float = 0.0,
    ):
        """
        Args:
            paths: List of prim paths (can be regex) to find, e.g. ["World/.*/visuals"].
            stage: The USD stage to search in.
            device: The device to use for the raycaster.
            simplify_factor: The factor to simplify the meshes. 0.0 means no simplification.
                If a single float is provided, it will be used for all meshes.
        """
        if isinstance(simplify_factor, float):
            simplify_factor = [simplify_factor] * len(paths)
        if not len(paths) == len(simplify_factor):
            raise ValueError(
                "`simplify_factor` must be a single float or a list of floats with the same length as `paths`"
            )
        
        from .utils_usd import find_matching_prims, get_trimesh_from_prim

        meshes_wp = []
        n_verts_before = 0
        n_verts_after = 0
        n_faces_before = 0
        n_faces_after = 0
        for path, factor in zip(paths, simplify_factor):
            if not (prims := find_matching_prims(path, stage)):
                raise ValueError(f"No prims found for path {path}")

            for prim in prims:
                mesh_combined = get_trimesh_from_prim(prim)
                
                n_verts_before += mesh_combined.vertices.shape[0]
                n_faces_before += mesh_combined.faces.shape[0]
                if factor > 0.0:
                    mesh_combined = mesh_combined.simplify_quadric_decimation(factor)
                n_verts_after += mesh_combined.vertices.shape[0]
                n_faces_after += mesh_combined.faces.shape[0]

                meshes_wp.append(trimesh2wp(mesh_combined, device))

        if n_faces_before != n_faces_after:
            print(f"Simplified from ({n_verts_before}, {n_faces_before}) to ({n_verts_after}, {n_faces_after})")

        return cls(meshes_wp, device)
    

    @classmethod
    def from_MjModel(
        cls,
        body_names: list[str],
        model: mujoco.MjModel,
        device: str,
        simplify_factor: float = 0.0,
    ):
        """
        Args:
            model: The Mujoco model to use for the raycaster.
            device: The device to use for the raycaster.
            simplify_factor: The factor to simplify the meshes. 0.0 means no simplification.
                If a single float is provided, it will be used for all meshes.
        """
        from .utils_mjc import get_trimesh_from_body
        
        mesh_names = []
        meshes_wp = []

        n_verts_before = 0
        n_verts_after = 0
        n_faces_before = 0
        n_faces_after = 0

        for body_name in body_names:
            body = model.body(body_name)
            if body.geomnum.item() > 0:
                mesh = get_trimesh_from_body(body, model)
                n_verts_before += mesh.vertices.shape[0]
                n_faces_before += mesh.faces.shape[0]
                if simplify_factor > 0.0:
                    mesh = mesh.simplify_quadric_decimation(simplify_factor)
                n_verts_after += mesh.vertices.shape[0]
                n_faces_after += mesh.faces.shape[0]

                mesh_names.append(body.name)
                meshes_wp.append(trimesh2wp(mesh, device))

        if n_faces_before != n_faces_after:
            print(f"Simplified from ({n_verts_before}, {n_faces_before}) to ({n_verts_after}, {n_faces_after})")

        return cls(meshes_wp, device)

