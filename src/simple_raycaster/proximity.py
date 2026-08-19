from __future__ import annotations

import torch
import warp as wp

from typing import Optional, TYPE_CHECKING, Union
from jaxtyping import Bool, Float, Int

from .kernels import (
    transform_and_query_point_against_meshes_kernel,
    transform_and_query_point_closest_against_meshes_kernel,
    transform_and_query_point_closest_kernel,
    transform_and_query_point_kernel,
)
from .raycaster_v2 import MultiMeshRaycasterV2

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

    IsaacEntity = Union[Articulation, RigidObject]


class MeshProximitySensor:
    """Closest-point sensor over Isaac-registered meshes via Warp ``mesh_query_point``.

    Registration mirrors :class:`MultiMeshRaycasterV2` — either share an existing
    raycaster instance or own one for static / dynamic meshes:

    .. code-block:: python

        sensor = MeshProximitySensor(device="cuda")
        sensor.add_isaac_static("/World/ground")
        sensor.add_isaac_entity(env.scene.articulations["robot"])
        closest_w, dist = sensor.query(points_w, max_dist=2.0)

    Intended for observation terms that sample body-frame probe points, transform
    them to world, and feed ``query``. Quaternions are **WXYZ**.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        *,
        meshes: MultiMeshRaycasterV2 | None = None,
    ):
        if meshes is not None:
            self._meshes = meshes
        else:
            if device is None:
                raise ValueError("Provide `device` or an existing `meshes` MultiMeshRaycasterV2")
            self._meshes = MultiMeshRaycasterV2(device)

    @property
    def device(self):
        return self._meshes.device

    @property
    def n_meshes(self) -> int:
        return self._meshes.n_meshes

    @property
    def meshes(self) -> MultiMeshRaycasterV2:
        """Underlying mesh registry (same object as a shared raycaster, if any)."""
        return self._meshes

    def add_isaac_static(self, prim_path: str) -> list[str]:
        """Register a static world-frame mesh (see :meth:`MultiMeshRaycasterV2.add_isaac_static`)."""
        return self._meshes.add_isaac_static(prim_path)

    def add_isaac_entity(self, entity: IsaacEntity) -> list[str]:
        """Register body visuals of an Isaac articulation / rigid object."""
        return self._meshes.add_isaac_entity(entity)

    def query(
        self,
        positions_w: Float[torch.Tensor, "N n_points 3"],
        max_dist: float = 10.0,
        *,
        enabled: Optional[Bool[torch.Tensor, "N"]] = None,
        mesh_indices: Optional[Int[torch.Tensor, "N n_meshes"]] = None,
        signed: bool = False,
        closest_hit: bool = True,
    ) -> tuple[
        Float[torch.Tensor, "N n_points 3"],
        Float[torch.Tensor, "N n_points"],
    ]:
        """Closest points on registered meshes to each query position.

        For each batch element and query point, all meshes (or the subset in
        ``mesh_indices``) are tested and the nearest surface point is kept.
        With ``signed=True``, distance uses angle-weighted normals (negative
        when the query is inside a watertight mesh); the nearest mesh is still
        chosen by absolute distance.

        Args:
            positions_w: Query positions in the world frame. Shape ``[N, n_points, 3]``.
            max_dist: Faces beyond this distance are ignored. Misses return
                ``max_dist`` and ``closest_pos_w == positions_w``.
            enabled: Optional per-batch mask. Disabled rows get ``inf`` distance.
            mesh_indices: Optional per-batch mesh subset. Shape ``[N, n_subset]``.
            signed: If True, use ``mesh_query_point_sign_normal``; else unsigned
                ``mesh_query_point_no_sign``.
            closest_hit: If True (default), loop meshes in-kernel and write
                ``[N, n_points]``. Output buffers are reused across calls with
                the same shape. If False, write a per-mesh buffer and reduce
                with PyTorch ``argmin`` (parity / baseline).

        Returns:
            ``(closest_pos_w, distances)`` with shapes ``[N, n_points, 3]`` and
            ``[N, n_points]``.
        """
        if not self._meshes.initialized:
            self._meshes.initialize()

        N, n_points = positions_w.shape[:2]
        mesh_pos_w, mesh_quat_w = self._meshes._get_mesh_poses_w(N, positions_w.device)

        if enabled is None:
            enabled = self._meshes._default_enabled(N, positions_w.device)
        else:
            enabled = enabled.reshape(N)

        sign_flag = 1 if signed else 0

        if closest_hit:
            closest_pos_w = self._meshes._cached(
                "prox_pos", (N, n_points, 3), torch.float32, positions_w.device
            )
            distances = self._meshes._cached(
                "prox_dist", (N, n_points), torch.float32, positions_w.device
            )
            if mesh_indices is None:
                wp.launch(
                    transform_and_query_point_closest_kernel,
                    dim=(N, n_points),
                    inputs=[
                        self._meshes.meshes_array,
                        wp.from_torch(mesh_pos_w, dtype=wp.vec3, return_ctype=True),
                        wp.from_torch(mesh_quat_w, dtype=wp.vec4, return_ctype=True),
                        wp.from_torch(positions_w, dtype=wp.vec3, return_ctype=True),
                        wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                        max_dist,
                        sign_flag,
                    ],
                    outputs=[
                        wp.from_torch(closest_pos_w, dtype=wp.vec3, return_ctype=True),
                        wp.from_torch(distances, dtype=wp.float32, return_ctype=True),
                    ],
                    device=self.device,
                    record_tape=False,
                )
            else:
                assert mesh_indices.shape == mesh_pos_w.shape[:2] == mesh_quat_w.shape[:2], (
                    "`mesh_indices` must have the same number of meshes as the registered mesh poses"
                )
                wp.launch(
                    transform_and_query_point_closest_against_meshes_kernel,
                    dim=(N, n_points),
                    inputs=[
                        self._meshes.meshes_array,
                        wp.from_torch(mesh_indices, dtype=wp.int64, return_ctype=True),
                        wp.from_torch(mesh_pos_w, dtype=wp.vec3, return_ctype=True),
                        wp.from_torch(mesh_quat_w, dtype=wp.vec4, return_ctype=True),
                        wp.from_torch(positions_w, dtype=wp.vec3, return_ctype=True),
                        wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                        max_dist,
                        sign_flag,
                    ],
                    outputs=[
                        wp.from_torch(closest_pos_w, dtype=wp.vec3, return_ctype=True),
                        wp.from_torch(distances, dtype=wp.float32, return_ctype=True),
                    ],
                    device=self.device,
                    record_tape=False,
                )
            return closest_pos_w, distances

        if mesh_indices is None:
            result_shape = (N, self.n_meshes, n_points)
            closest_pos_w = torch.empty(result_shape + (3,), device=positions_w.device)
            distances = torch.empty(result_shape, device=positions_w.device)
            wp.launch(
                transform_and_query_point_kernel,
                dim=(N, self.n_meshes, n_points),
                inputs=[
                    self._meshes.meshes_array,
                    wp.from_torch(mesh_pos_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(mesh_quat_w, dtype=wp.vec4, return_ctype=True),
                    wp.from_torch(positions_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                    max_dist,
                    sign_flag,
                ],
                outputs=[
                    wp.from_torch(closest_pos_w, dtype=wp.vec3),
                    wp.from_torch(distances, dtype=wp.float32),
                ],
                device=self.device,
                record_tape=False,
            )
        else:
            assert mesh_indices.shape == mesh_pos_w.shape[:2] == mesh_quat_w.shape[:2], (
                "`mesh_indices` must have the same number of meshes as the registered mesh poses"
            )
            n_subset = mesh_indices.shape[1]
            result_shape = (N, n_subset, n_points)
            closest_pos_w = torch.empty(result_shape + (3,), device=positions_w.device)
            distances = torch.empty(result_shape, device=positions_w.device)
            wp.launch(
                transform_and_query_point_against_meshes_kernel,
                dim=(N, n_subset, n_points),
                inputs=[
                    self._meshes.meshes_array,
                    wp.from_torch(mesh_indices, dtype=wp.int64, return_ctype=True),
                    wp.from_torch(mesh_pos_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(mesh_quat_w, dtype=wp.vec4, return_ctype=True),
                    wp.from_torch(positions_w, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                    max_dist,
                    sign_flag,
                ],
                outputs=[
                    wp.from_torch(closest_pos_w, dtype=wp.vec3),
                    wp.from_torch(distances, dtype=wp.float32),
                ],
                device=self.device,
                record_tape=False,
            )

        # Nearest mesh per query point (signed: compare absolute distance).
        if signed:
            mesh_idx = distances.abs().argmin(dim=1)  # [N, n_points]
        else:
            mesh_idx = distances.argmin(dim=1)

        gather_idx = mesh_idx.unsqueeze(1)  # [N, 1, n_points]
        distances = distances.gather(1, gather_idx).squeeze(1)
        closest_pos_w = closest_pos_w.gather(
            1, gather_idx.unsqueeze(-1).expand(-1, 1, -1, 3)
        ).squeeze(1)
        return closest_pos_w, distances

    def __repr__(self) -> str:
        return (
            f"MeshProximitySensor(n_meshes={self.n_meshes}, "
            f"n_points={self._meshes.n_points}, n_faces={self._meshes.n_faces})"
        )
