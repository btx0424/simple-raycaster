from __future__ import annotations

import torch
import trimesh
import warp as wp

from typing import Optional, List, Union, TYPE_CHECKING
from jaxtyping import Float, Bool, Int
from .helpers import quat_rotate, quat_rotate_inverse, trimesh2wp, workspace_tensor
from .kernels import (
    raycast_kernel,
    raycast_against_meshes_kernel,
    transform_and_raycast_kernel,
    transform_and_raycast_against_meshes_kernel,
    transform_and_raycast_closest_kernel,
    transform_and_raycast_closest_against_meshes_kernel,
)


MeshType = Union[wp.Mesh, trimesh.Trimesh]


if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    IsaacEntity = Union[Articulation, RigidObject]


class MultiMeshRaycasterV2:
    """
    Raycaster that supports multiple and dynamic meshes.
    Different from MultiMeshRaycaster, this version no longer needs providing `mesh_pos_w` and `mesh_quat_w` as inputs for `raycast` and `raycast_fused`.
    Instead, it automatically updates the mesh positions and orientations.

    TODO: add support for backends other than isaac.

    """
    def __init__(
        self,
        device: str | torch.device,
        *,
        bvh_constructor: str | None = None,
        bvh_leaf_size: int | None = None,
    ):
        self.entities: List[Optional[IsaacEntity]] = []
        self.meshes_wp = []
        if not isinstance(device, str):
            device = wp.get_device(str(device))
        self.device = device
        self.bvh_constructor = bvh_constructor
        self.bvh_leaf_size = bvh_leaf_size
        self.initialized = False
        self._work: dict = {}
    
    def initialize(self):
        if self.initialized:
            return
        self._validate_registration()
        self.initialized = True
        self.meshes_array = wp.array([mesh_wp.id for mesh_wp in self.meshes_wp], device=self.device, dtype=wp.uint64)

    def _cached(self, key: str, shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
        return workspace_tensor(self._work, key, shape, dtype, device)

    def _default_enabled(self, n: int, device) -> torch.Tensor:
        prev = self._work.get("enabled_true")
        enabled = self._cached("enabled_true", (n,), torch.bool, device)
        if prev is None or prev is not enabled:
            enabled.fill_(True)
        return enabled

    def _add_mesh(self, mesh: MeshType):
        """Add a mesh to the raycaster (internal)."""
        if isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh2wp(
                mesh,
                self.device,
                bvh_constructor=self.bvh_constructor,
                bvh_leaf_size=self.bvh_leaf_size,
            )
        self.meshes_wp.append(mesh)
        self.initialized = False

    def _validate_registration(self):
        """Validate that registered entities match the number of loaded meshes."""
        if not self.entities:
            return

        expected_meshes = sum(1 if entity is None else entity.num_bodies for entity in self.entities)
        if expected_meshes != self.n_meshes:
            raise ValueError(
                f"Registered entities account for {expected_meshes} meshes, "
                f"but the raycaster has {self.n_meshes} meshes"
            )

    def _get_mesh_poses_w(
        self,
        N: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather world-frame mesh poses from registered Isaac entities."""
        if self.n_meshes == 0:
            raise ValueError("No meshes registered with the raycaster")

        zero_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

        if not self.entities:
            mesh_pos_w = torch.zeros(N, self.n_meshes, 3, device=device)
            mesh_quat_w = zero_quat.expand(N, self.n_meshes, 4)
            return mesh_pos_w, mesh_quat_w

        mesh_pos_w = []
        mesh_quat_w = []
        for entity in self.entities:
            if entity is None:
                mesh_pos_w.append(torch.zeros(N, 1, 3, device=device))
                mesh_quat_w.append(zero_quat.expand(N, 1, 4))
            else:
                body_link_pose_w = entity.data.body_link_pose_w
                if body_link_pose_w.shape[0] != N:
                    raise ValueError(
                        f"Batch size {N} does not match entity instance count {body_link_pose_w.shape[0]}"
                    )
                mesh_pos_w.append(body_link_pose_w[:, :, :3])
                mesh_quat_w.append(body_link_pose_w[:, :, 3:7])

        mesh_pos_w = torch.cat(mesh_pos_w, dim=1)
        mesh_quat_w = torch.cat(mesh_quat_w, dim=1)
        return mesh_pos_w, mesh_quat_w
    
    def _add_from_path(
        self,
        path: str | list[str],
        stage: "Usd.Stage",
        combine: bool = True,
    ) -> List[str]:
        """
        Add meshes from USD prim paths to the raycaster (internal).

        Args:
            path: Prim path(s) to search for (supports regex). Can be a single string or list of strings.
            stage: The USD stage to search in.
            combine: If True, combine all matching meshes into one. If False, add each mesh separately.
        returns:
            List of prim paths that were found.
        """
        
        if isinstance(path, str):
            path = [path]
        path_expr = "(" + "|".join(path) + ")"
        from .utils_usd import find_matching_prims, get_trimesh_from_prim
        prims = find_matching_prims(path_expr, stage)
        trimesh_list = []
        
        for prim in prims:
            mesh = get_trimesh_from_prim(prim)
            trimesh_list.append(mesh)
        
        if combine:
            mesh = trimesh.util.concatenate(trimesh_list)
            self._add_mesh(mesh)
        else:
            for mesh in trimesh_list:
                self._add_mesh(mesh)
        self.initialized = False
        return [prim.GetPath().pathString for prim in prims]

    def add_isaac_static(self, prim_path: str) -> List[str]:
        """Register a static mesh from the current USD stage.

        Matching visual meshes under ``prim_path`` (regex supported) are combined
        into a single Warp mesh baked in world frame. At raycast time the mesh pose
        is fixed at the origin with identity orientation — geometry must already
        include any static world transform from USD.

        Typical use: ground / terrain that does not move relative to the world.

        Args:
            prim_path: USD prim path or regex, e.g. ``"/World/ground"``.

        Returns:
            List of matched prim path strings.

        Raises:
            ValueError: If entity/mesh registration becomes inconsistent.
        """
        from isaacsim.core.utils.stage import get_current_stage
        prim_paths = self._add_from_path(prim_path, get_current_stage(), combine=True)
        self.entities.append(None)
        self._validate_registration()
        return prim_paths

    def add_isaac_entity(self, entity: IsaacEntity) -> List[str]:
        """Register a dynamic Isaac Lab articulation or rigid object.

        Loads one visual mesh per body from the current USD stage. Body prim paths
        are derived from ``entity.root_physx_view.prim_paths[0]`` by substituting
        each name in ``entity.body_names``. At raycast time, poses are read from
        ``entity.data.body_link_pose_w`` (WXYZ quaternions).

        Args:
            entity: An Isaac Lab ``Articulation`` or ``RigidObject`` instance.
                The number of matched visual prims must equal ``entity.num_bodies``.

        Returns:
            List of matched prim path strings (one entry per body mesh).

        Raises:
            ValueError: If the number of matched prims does not equal
                ``entity.num_bodies``, or if entity/mesh registration is inconsistent.

        Note:
            Primarily tested with ``Articulation``; ``RigidObject`` support is
            untested but follows the same body/visual layout.
        """
        from isaacsim.core.utils.stage import get_current_stage
        stage = get_current_stage()
        # template path is, for example, /World/envs/env_0/robot/base_link for an Articulation
        # we will need to find the paths for other bodies in the articulation
        template_path = entity.root_physx_view.prim_paths[0]
        root_body_name = template_path.split("/")[-1]
        if root_body_name == entity.body_names[0]:
            prim_paths = [
                template_path.replace(root_body_name, body_name) + "/visuals"
                for body_name in entity.body_names
            ]
        prim_paths = self._add_from_path(prim_paths, stage, combine=False)
        if len(prim_paths) != entity.num_bodies:
            raise ValueError(f"Number of prim paths {len(prim_paths)} does not match number of bodies {entity.num_bodies}")
        self.entities.append(entity)
        self._validate_registration()
        return prim_paths
    
    def raycast(
        self,
        ray_starts_w: Float[torch.Tensor, "N n_rays 3"],  # [N, n_rays, 3]
        ray_dirs_w: Float[torch.Tensor, "N n_rays 3"],  # [N, n_rays, 3]
        min_dist: float = 0.0,
        max_dist: float = 100.0,
        *,
        enabled: Optional[Bool[torch.Tensor, "N"]]=None,  # [N]
        mesh_indices: Optional[Int[torch.Tensor, "N n_meshes"]]=None,  # [N, n_meshes]
    ) -> tuple[Float[torch.Tensor, "N n_rays 3"], Float[torch.Tensor, "N n_rays"], Float[torch.Tensor, "N n_rays 3"]]:
        if not self.initialized:
            self.initialize()

        N, n_rays = ray_dirs_w.shape[:2]
        mesh_pos_w, mesh_quat_w = self._get_mesh_poses_w(N, ray_starts_w.device)

        if enabled is None:
            enabled = torch.ones(N, dtype=torch.bool, device=ray_starts_w.device)
        else:
            enabled = enabled.reshape(N,)

        if mesh_indices is None:
            result_shape = (N, self.n_meshes, n_rays)
            mesh_pos_w = mesh_pos_w.reshape(N, self.n_meshes, 1, 3)
            mesh_quat_w = mesh_quat_w.reshape(N, self.n_meshes, 1, 4)

            ray_starts_b = quat_rotate_inverse(mesh_quat_w, ray_starts_w.unsqueeze(1) - mesh_pos_w)
            ray_dirs_b = quat_rotate_inverse(mesh_quat_w, ray_dirs_w.unsqueeze(1))

            hit_distances = torch.empty(result_shape, device=ray_starts_w.device)
            hit_normals = torch.empty(*result_shape, 3, device=ray_starts_w.device)
            wp.launch(
                raycast_kernel,
                dim=(N, self.n_meshes, n_rays),
                inputs=[
                    self.meshes_array,
                    wp.from_torch(ray_starts_b, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(ray_dirs_b, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                    min_dist,
                    max_dist,
                ],
                outputs=[
                    wp.from_torch(hit_distances, dtype=wp.float32, return_ctype=True),
                    wp.from_torch(hit_normals, dtype=wp.vec3, return_ctype=True),
                ],
                device=self.device,
                record_tape=False,
            )
        else:
            assert mesh_indices.shape == mesh_pos_w.shape[:2] == mesh_quat_w.shape[:2], (
                "`mesh_indices` must have the same number of meshes as the registered mesh poses"
            )
            n_meshes = mesh_indices.shape[1]
            result_shape = (N, n_meshes, n_rays)
            mesh_pos_w = mesh_pos_w.reshape(N, n_meshes, 1, 3)
            mesh_quat_w = mesh_quat_w.reshape(N, n_meshes, 1, 4)

            ray_starts_b = quat_rotate_inverse(mesh_quat_w, ray_starts_w.unsqueeze(1) - mesh_pos_w)
            ray_dirs_b = quat_rotate_inverse(mesh_quat_w, ray_dirs_w.unsqueeze(1))

            hit_distances = torch.empty(result_shape, device=ray_starts_w.device)
            hit_normals = torch.empty(*result_shape, 3, device=ray_starts_w.device)
            wp.launch(
                raycast_against_meshes_kernel,
                dim=(N, n_meshes, n_rays),
                inputs=[
                    self.meshes_array,
                    wp.from_torch(mesh_indices, dtype=wp.int64, return_ctype=True),
                    wp.from_torch(ray_starts_b, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(ray_dirs_b, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                    min_dist,
                    max_dist,
                ],
                outputs=[
                    wp.from_torch(hit_distances, dtype=wp.float32, return_ctype=True),
                    wp.from_torch(hit_normals, dtype=wp.vec3, return_ctype=True),
                ],
                device=self.device,
                record_tape=False,
            )

        # kernel returns normals in mesh-local frame; rotate back to world
        hit_normals = quat_rotate(mesh_quat_w, hit_normals)

        min_result = hit_distances.min(dim=1)
        hit_distances = min_result.values  # [N, n_rays]
        # select the normal of the closest-hit mesh for each ray
        gather_idx = min_result.indices.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, 3)
        hit_normals = hit_normals.gather(1, gather_idx).squeeze(1)  # [N, n_rays, 3]
        hit_positions = ray_starts_w + hit_distances.unsqueeze(-1) * ray_dirs_w
        return hit_positions, hit_distances, hit_normals
    
    def raycast_fused(
        self,
        ray_starts_w: Float[torch.Tensor, "N n_rays 3"],  # [N, n_rays, 3]
        ray_dirs_w: Float[torch.Tensor, "N n_rays 3"],  # [N, n_rays, 3]
        min_dist: float = 0.0,
        max_dist: float = 100.0,
        *,
        enabled: Optional[Bool[torch.Tensor, "N"]]=None,  # [N]
        mesh_indices: Optional[Int[torch.Tensor, "N n_meshes"]]=None,  # [N, n_meshes]
        closest_hit: bool = True,
    ) -> tuple[Float[torch.Tensor, "N n_rays 3"], Float[torch.Tensor, "N n_rays"], Float[torch.Tensor, "N n_rays 3"]]:
        """
        Perform raycasting against multiple meshes using a fused GPU kernel.
        
        This method is an optimized version of `raycast()` that fuses the coordinate
        transformation and raycasting operations into a single GPU kernel, reducing
        memory transfers and improving performance. The transformation from world
        coordinates to mesh-local coordinates is performed on the GPU within the
        kernel, eliminating the need for intermediate tensor allocations.
        
        For each batch element, rays are cast against all meshes (or a subset specified
        by `mesh_indices`), and the closest hit point across all meshes is returned for
        each ray. Rays that don't hit any mesh within the distance range will have
        a hit distance equal to `max_dist`.
        
        Args:
            ray_starts_w: The starting points of the rays in the world frame. Shape [N, n_rays, 3],
                where n_rays is the number of rays per batch element. Each row represents
                the 3D starting position (x, y, z) of a ray.
            ray_dirs_w: The direction vectors of the rays in the world frame. Shape [N, n_rays, 3].
                Direction vectors should be normalized (unit vectors). Each row represents
                the 3D direction vector of a ray.
            min_dist: The minimum distance along the ray to consider for hits. Rays that hit
                closer than this distance will be ignored. Defaults to 0.0. Useful for avoiding
                self-intersections or ignoring hits at the ray origin.
            max_dist: The maximum distance along the ray to search for hits. Rays that don't
                hit within this distance will return `max_dist` as the hit distance. Defaults
                to 100.0. Acts as the maximum ray length.
            enabled: Optional boolean tensor indicating which batch elements should be processed.
                Shape [N]. If None (default), all batch elements are enabled. Disabled elements
                will have their hit distances set to infinity. This is a keyword-only argument.
            mesh_indices: Optional tensor specifying which meshes to raycast against for each
                batch element. Shape [N, n_meshes], where each element is an index into the
                mesh array. If None (default), all meshes are tested. When provided, allows
                different batch elements to test against different subsets of meshes. This is
                a keyword-only argument.
        
        Returns:
            tuple: A tuple containing:
                - hit_positions (torch.Tensor): The 3D positions of the closest hit points
                  in the world frame. Shape [N, n_rays, 3]. Computed as `ray_starts_w + 
                  hit_distances * ray_dirs_w`. For rays that don't hit, positions correspond
                  to the point at `max_dist` along the ray direction.
                - hit_distances (torch.Tensor): The distances along each ray to the closest
                  hit point. Shape [N, n_rays]. Values are in the range [min_dist, max_dist],
                  or `max_dist` if no hit occurred. The minimum is taken across all meshes
                  for each ray.
                - hit_normals (torch.Tensor): The world-frame face normals at the closest
                  hit points. Shape [N, n_rays, 3]. Zero vectors for rays that miss or for
                  disabled batch elements.
        
        Note:
            - This method automatically initializes the raycaster if not already initialized.
            - The coordinate transformation and raycast are fused into a single Warp kernel,
              reducing intermediate allocations compared to `raycast()`.
            - When `mesh_indices` is provided, its second dimension should match the number
              of registered mesh poses for each batch element.
            - Mesh poses are fetched automatically from registered Isaac entities.
            - Quaternions are converted from WXYZ to XYZW format internally for Warp compatibility.
            - All input tensors must be on the same device as the raycaster.
        
        Example:
            >>> raycaster = MultiMeshRaycasterV2(device="cuda")
            >>> ray_starts = torch.randn(10, 100, 3, device="cuda")  # 100 rays per batch
            >>> ray_dirs = torch.randn(10, 100, 3, device="cuda")
            >>> ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)  # normalize
            >>> hit_pos, hit_dist, hit_normal = raycaster.raycast_fused(
            ...     ray_starts, ray_dirs,
            ...     min_dist=0.1, max_dist=50.0
            ... )
        """
        if not self.initialized:
            self.initialize()
        
        N, n_rays = ray_dirs_w.shape[:2]
        mesh_pos_w, mesh_quat_w = self._get_mesh_poses_w(N, ray_starts_w.device)

        if enabled is None:
            enabled = self._default_enabled(N, ray_starts_w.device)
        else:
            enabled = enabled.reshape(N,)

        if closest_hit:
            hit_distances = self._cached(
                "closest_dist", (N, n_rays), torch.float32, ray_starts_w.device
            )
            hit_normals = self._cached(
                "closest_nrm", (N, n_rays, 3), torch.float32, ray_starts_w.device
            )
            hit_positions = self._cached(
                "closest_pos", (N, n_rays, 3), torch.float32, ray_starts_w.device
            )
            if mesh_indices is None:
                wp.launch(
                    transform_and_raycast_closest_kernel,
                    dim=(N, n_rays),
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
                        wp.from_torch(hit_distances, dtype=wp.float32, return_ctype=True),
                        wp.from_torch(hit_normals, dtype=wp.vec3, return_ctype=True),
                        wp.from_torch(hit_positions, dtype=wp.vec3, return_ctype=True),
                    ],
                    device=self.device,
                    record_tape=False,
                )
            else:
                assert mesh_indices.shape == mesh_pos_w.shape[:2] == mesh_quat_w.shape[:2], (
                    "`mesh_indices` must have the same number of meshes as `mesh_pos_w` and `mesh_quat_w`"
                )
                wp.launch(
                    transform_and_raycast_closest_against_meshes_kernel,
                    dim=(N, n_rays),
                    inputs=[
                        self.meshes_array,
                        wp.from_torch(mesh_indices, dtype=wp.int64, return_ctype=True),
                        wp.from_torch(mesh_pos_w, dtype=wp.vec3, return_ctype=True),
                        wp.from_torch(mesh_quat_w, dtype=wp.vec4, return_ctype=True),
                        wp.from_torch(ray_starts_w, dtype=wp.vec3, return_ctype=True),
                        wp.from_torch(ray_dirs_w, dtype=wp.vec3, return_ctype=True),
                        wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                        min_dist,
                        max_dist,
                    ],
                    outputs=[
                        wp.from_torch(hit_distances, dtype=wp.float32, return_ctype=True),
                        wp.from_torch(hit_normals, dtype=wp.vec3, return_ctype=True),
                        wp.from_torch(hit_positions, dtype=wp.vec3, return_ctype=True),
                    ],
                    device=self.device,
                    record_tape=False,
                )
            return hit_positions, hit_distances, hit_normals

        if mesh_indices is None:
            result_shape = (N, self.n_meshes, n_rays)
            hit_distances = torch.empty(result_shape, device=ray_starts_w.device)
            hit_normals = torch.empty(*result_shape, 3, device=ray_starts_w.device)
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
                    wp.from_torch(hit_distances, dtype=wp.float32, return_ctype=True),
                    wp.from_torch(hit_normals, dtype=wp.vec3, return_ctype=True),
                ],
                device=self.device,
                record_tape=False,
            )
        else:
            assert mesh_indices.shape == mesh_pos_w.shape[:2] == mesh_quat_w.shape[:2], (
                "`mesh_indices` must have the same number of meshes as `mesh_pos_w` and `mesh_quat_w`"
            )
            n_meshes = mesh_indices.shape[1]
            result_shape = (N, n_meshes, n_rays)
            hit_distances = torch.empty(result_shape, device=ray_starts_w.device)
            hit_normals = torch.empty(*result_shape, 3, device=ray_starts_w.device)
            wp.launch(
                transform_and_raycast_against_meshes_kernel,
                dim=(N, mesh_indices.shape[1], n_rays),
                inputs=[
                    self.meshes_array,
                    wp.from_torch(mesh_indices, dtype=wp.int64, return_ctype=True),
                    wp.from_torch(mesh_pos_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(mesh_quat_w, dtype=wp.vec4, return_ctype=True),
                    wp.from_torch(ray_starts_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(ray_dirs_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                    min_dist,
                    max_dist,
                ],
                outputs=[
                    wp.from_torch(hit_distances, dtype=wp.float32, return_ctype=True),
                    wp.from_torch(hit_normals, dtype=wp.vec3, return_ctype=True),
                ],
                device=self.device,
                record_tape=False,
            )
        
        min_result = hit_distances.min(dim=1)
        hit_distances = min_result.values # [N, n_rays]
        # select the normal of the closest-hit mesh for each ray (already world-frame)
        gather_idx = min_result.indices.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, 3)
        hit_normals = hit_normals.gather(1, gather_idx).squeeze(1) # [N, n_rays, 3]
        hit_positions = ray_starts_w + hit_distances.unsqueeze(-1) * ray_dirs_w # [N, n_rays, 3]
        return hit_positions, hit_distances, hit_normals

    @property
    def n_points(self):
        return sum(mesh_wp.points.shape[0] for mesh_wp in self.meshes_wp)

    @property
    def n_faces(self):
        return sum(mesh_wp.indices.reshape((-1, 3)).shape[0] for mesh_wp in self.meshes_wp)

    @property
    def n_meshes(self):
        return len(self.meshes_wp)

    def __repr__(self) -> str:
        return f"MultiMeshRaycasterV2(n_meshes={self.n_meshes}, n_points={self.n_points}, n_faces={self.n_faces})"

