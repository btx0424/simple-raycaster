import torch
import numpy as np
import trimesh
import warp as wp
import argparse

from pxr import Usd, UsdGeom, UsdPhysics

from simple_raycaster.raycaster import MultiMeshRaycaster
from simple_raycaster.utils import get_mesh_prims_subtree, usd2trimesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=str, required=True)
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.usd)
    path_regex = f"{stage.GetDefaultPrim().GetPath()}/.*/visuals"
    
    trimesh_list = []
    translations = []
    quats = []

    default_prim = stage.GetDefaultPrim()

    for child in default_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            visuals_prim = stage.GetPrimAtPath(str(child.GetPath()) + "/visuals")
            if visuals_prim.IsValid():
                xform = UsdGeom.Xformable(visuals_prim)
                transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                translation = np.array(transform.ExtractTranslation())
                orientation = transform.ExtractRotationQuat()
                orientation = np.array([orientation.GetReal(), *orientation.GetImaginary()])
                transform = trimesh.transformations.concatenate_matrices(
                    trimesh.transformations.translation_matrix(translation),
                    trimesh.transformations.quaternion_matrix(orientation),
                )
                mesh_prims = get_mesh_prims_subtree(visuals_prim)
                meshes = []
                for mesh_prim in mesh_prims:
                    mesh = usd2trimesh(mesh_prim)
                    faces_before = mesh.faces.shape[0]
                    mesh = mesh.simplify_quadric_decimation(0.5)
                    faces_after = mesh.faces.shape[0]
                    meshes.append(mesh)
                mesh: trimesh.Trimesh = trimesh.util.concatenate(meshes)
                mesh.merge_vertices()
                mesh.apply_transform(transform)
                trimesh_list.append(mesh)

                translations.append(translation)
                quats.append(orientation)

    device = "cuda"
    wp.init()

    raycaster = MultiMeshRaycaster.from_prim_paths(
        [path_regex],
        stage=stage,
        device=device,
    )
    print(raycaster)

    horizontal_angles = torch.linspace(-torch.pi / 4, torch.pi / 4, 20)
    vertical_angles = torch.linspace(-torch.pi / 6, torch.pi / 6, 10) - torch.pi / 2
    hh, vv = torch.meshgrid(horizontal_angles, vertical_angles)

    ray_dirs = torch.stack([
        torch.cos(hh) * torch.cos(vv),
        torch.sin(hh) * torch.cos(vv),
        torch.sin(vv),
    ], dim=2)

    ray_dirs = ray_dirs.reshape(-1, 3).to(device)
    ray_starts = torch.zeros(ray_dirs.shape[0], 3, device=device)
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=1, keepdim=True)

    ray_starts[:, 2] = 1.0
    # ray_dirs[:, 0] = - ray_dirs[:, 0].abs()

    translations = torch.from_numpy(np.stack(translations, axis=0)).float().to(device)
    quats = torch.from_numpy(np.stack(quats, axis=0)).float().to(device)
    print(translations.shape, quats.shape)

    hit_positions, hit_distances = raycaster.raycast(
        translations.unsqueeze(0),
        quats.unsqueeze(0),
        ray_starts.unsqueeze(0),
        ray_dirs.unsqueeze(0),
        5.0,
    )
    hit_positions = hit_positions.squeeze(0)

    scene = trimesh.Scene([mesh for mesh in trimesh_list])
    segments = torch.stack([ray_starts, hit_positions], dim=1).cpu().numpy()
    lines = trimesh.load_path(segments)
    frame = trimesh.creation.axis()
    scene.add_geometry(lines)
    scene.add_geometry(frame)
    scene.show()

