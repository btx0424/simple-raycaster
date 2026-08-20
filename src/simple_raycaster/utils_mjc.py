from typing import Optional

import mujoco
import numpy as np
import trimesh


def _geom_rgba_rgb(geom) -> np.ndarray:
    rgba = np.asarray(geom.rgba, dtype=np.float32).reshape(-1)
    return np.clip(rgba[:3], 0.0, 1.0).astype(np.float32)


def body_mesh_albedo(
    body,
    model: mujoco.MjModel,
    *,
    default: tuple[float, float, float] = (0.65, 0.65, 0.65),
) -> np.ndarray:
    """Mean RGB of mesh geoms on ``body`` (prefer visual ``group==1``).

    MuJoCo stores appearance on ``geom_rgba``, not in STL vertex colors. Visual
    Unitree assets typically use ``group="1"`` for the display mesh.
    """
    geomadr = int(body.geomadr.item())
    geomnum = int(body.geomnum.item())
    preferred: list[np.ndarray] = []
    fallback: list[np.ndarray] = []
    for j in range(geomadr, geomadr + geomnum):
        geom = model.geom(j)
        if int(geom.type.item()) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        rgb = _geom_rgba_rgb(geom)
        if int(geom.group.item()) == 1:
            preferred.append(rgb)
        else:
            fallback.append(rgb)
    chosen = preferred or fallback
    if not chosen:
        return np.asarray(default, dtype=np.float32)
    return np.mean(np.stack(chosen, axis=0), axis=0).astype(np.float32)


def get_trimesh_from_body(body, model: mujoco.MjModel) -> Optional[trimesh.Trimesh]:
    """Combine mesh geoms on a body. Face colors come from each geom's ``rgba``."""
    geomadr = body.geomadr.item()
    geomnum = body.geomnum.item()

    # mesh_id -> (pos, quat, rgb). Later geoms overwrite earlier ones (same as before).
    meshes: dict[int, tuple] = {}
    for j in range(geomadr, geomadr + geomnum):
        geom = model.geom(j)
        is_mesh = geom.type.item() == mujoco.mjtGeom.mjGEOM_MESH
        if is_mesh:
            meshes[geom.dataid.item()] = (geom.pos, geom.quat, _geom_rgba_rgb(geom))
    trimesh_list = []
    for mesh_id, (pos, quat, rgb) in meshes.items():
        mesh = model.mesh(mesh_id)
        faceadr = mesh.faceadr.item()
        facenum = mesh.facenum.item()
        vertadr = mesh.vertadr.item()
        vertnum = mesh.vertnum.item()
        mesh = trimesh.Trimesh(
            vertices=model.mesh_vert[vertadr : vertadr + vertnum],
            faces=model.mesh_face[faceadr : faceadr + facenum],
            process=False,
        )
        # uint8 face colors so authored-color detection + _vertex_colors work.
        face_rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        rgba = np.concatenate([face_rgb, np.array([255], dtype=np.uint8)])
        mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))
        transform = trimesh.transformations.concatenate_matrices(
            trimesh.transformations.translation_matrix(pos),
            trimesh.transformations.quaternion_matrix(quat),
        )
        mesh.apply_transform(transform)
        trimesh_list.append(mesh)
    if not trimesh_list:
        return None
    trimesh_combined: trimesh.Trimesh = trimesh.util.concatenate(trimesh_list)
    trimesh_combined.merge_vertices()
    return trimesh_combined
