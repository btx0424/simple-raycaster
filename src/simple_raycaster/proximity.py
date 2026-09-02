from __future__ import annotations

MESH_PROXIMITY_UPDATE_REQUIRED = (
    "MeshProximitySensor is pending rewrite for explicit mesh poses. "
    "Register geometry with active_adaptation.envs.mesh_registry.MeshRegistry "
    "(ensure_targets / update_poses / poses_for_indices) and pass mesh_pos_w / "
    "mesh_quat_w into proximity kernels. MultiMeshRaycasterV2 was removed."
)


class MeshProximitySensor:
    """Closest-point sensor over scene meshes (rewrite in progress).

    Update required: depended on ``MultiMeshRaycasterV2``. Migrate to
    ``MeshRegistry`` + explicit ``mesh_pos_w`` / ``mesh_quat_w`` (see
    ``MESH_PROXIMITY_UPDATE_REQUIRED``).
    """

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        raise NotImplementedError(MESH_PROXIMITY_UPDATE_REQUIRED)
