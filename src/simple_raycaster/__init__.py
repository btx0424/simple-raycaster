from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .diffrast_camera import DiffrastCamera
from .mesh_cache import CachedEntityMesh, build_cached_entity_mesh, build_simplified_body_trimeshes
from .mesh_rgbd import (
    DiffrastMeshRenderer,
    MeshRGBDRenderer,
    RaycastMeshRenderer,
    make_mesh_rgbd_renderer,
)
from .pbr import (
    EnvironmentHDRI,
    RaycastPBRCamera,
    RaycastSSGICamera,
    default_hdri_path,
    resolve_hdri_path,
)
from .proximity import MESH_PROXIMITY_UPDATE_REQUIRED, MeshProximitySensor
from .raycast_camera import CameraIntrinsics, RaycastCamera
from .raycaster import MultiMeshRaycaster

try:
    __version__ = _pkg_version("simple-raycaster")
except PackageNotFoundError:
    # Editable / source tree without an installed distribution metadata.
    __version__ = "0.4.2"

__all__ = [
    "__version__",
    "MESH_PROXIMITY_UPDATE_REQUIRED",
    "MultiMeshRaycaster",
    "MeshProximitySensor",
    "RaycastCamera",
    "RaycastPBRCamera",
    "RaycastSSGICamera",
    "EnvironmentHDRI",
    "default_hdri_path",
    "resolve_hdri_path",
    "CameraIntrinsics",
    "DiffrastCamera",
    "CachedEntityMesh",
    "build_cached_entity_mesh",
    "build_simplified_body_trimeshes",
    "MeshRGBDRenderer",
    "DiffrastMeshRenderer",
    "RaycastMeshRenderer",
    "make_mesh_rgbd_renderer",
]
