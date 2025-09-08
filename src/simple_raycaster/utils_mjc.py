import mujoco
import trimesh


def get_trimesh_from_body(body, model: mujoco.MjModel):
    geomadr = body.geomadr.item()
    geomnum = body.geomnum.item()

    meshes = {}
    for j in range(geomadr, geomadr + geomnum):
        geom = model.geom(j)
        is_mesh = geom.type.item() == mujoco.mjtGeom.mjGEOM_MESH
        if is_mesh:
            meshes[geom.dataid.item()] = (geom.pos, geom.quat)
    trimesh_list = []
    for mesh_id, (pos, quat) in meshes.items():
        mesh = model.mesh(mesh_id)
        faceadr = mesh.faceadr.item()
        facenum = mesh.facenum.item()
        vertadr = mesh.vertadr.item()
        vertnum = mesh.vertnum.item()
        mesh = trimesh.Trimesh(
            vertices=model.mesh_vert[vertadr:vertadr + vertnum],
            faces=model.mesh_face[faceadr:faceadr + facenum]
        )
        transform = trimesh.transformations.concatenate_matrices(
            trimesh.transformations.translation_matrix(pos),
            trimesh.transformations.quaternion_matrix(quat)
        )
        mesh.apply_transform(transform)
        trimesh_list.append(mesh)
    trimesh_combined: trimesh.Trimesh = trimesh.util.concatenate(trimesh_list)
    trimesh_combined.merge_vertices()
    return trimesh_combined

