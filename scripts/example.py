import torch
import numpy as np
import trimesh
import warp as wp
import argparse
import mujoco
import time

from simple_raycaster.raycaster import MultiMeshRaycaster


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=str)
    parser.add_argument("--mjcf", type=str)
    args = parser.parse_args()

    if args.usd is None and args.mjcf is None:
        raise ValueError("Either --usd or --mjcf must be provided")
    if args.usd is not None and args.mjcf is not None:
        raise ValueError("Only one of --usd or --mjcf must be provided")

    trimesh_list = []
    translations = []
    quats = []
    device = "cuda"
    wp.init()

    if args.usd is not None:
        from pxr import Usd, UsdGeom, UsdPhysics
        from simple_raycaster.utils_usd import get_trimesh_from_prim

        stage = Usd.Stage.Open(args.usd)
        path_regex = f"{stage.GetDefaultPrim().GetPath()}/.*/visuals"
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
                    mesh = get_trimesh_from_prim(visuals_prim)
                    mesh.apply_transform(transform)
                    trimesh_list.append(mesh)

                    translations.append(translation)
                    quats.append(orientation)
        
        raycaster = MultiMeshRaycaster.from_prim_paths(
            [path_regex],
            stage=stage,
            device=device,
            simplify_factor=0.5,
        )
    else:
        from simple_raycaster.utils_mjc import get_trimesh_from_body

        model = mujoco.MjModel.from_xml_path(args.mjcf)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)

        body_names = []
        for i in range(model.nbody):
            body = model.body(i)
            geomadr = body.geomadr.item()
            geomnum = body.geomnum.item()
            if body.geomnum > 0:
                body_names.append(body.name)
                mesh = get_trimesh_from_body(body, model)
                transform = trimesh.transformations.concatenate_matrices(
                    trimesh.transformations.translation_matrix(data.xpos[body.id]),
                    trimesh.transformations.quaternion_matrix(data.xquat[body.id])
                )
                mesh.apply_transform(transform)
                trimesh_list.append(mesh)
                translations.append(data.xpos[body.id])
                quats.append(data.xquat[body.id])
        
        raycaster = MultiMeshRaycaster.from_MjModel(
            body_names=body_names,
            model=model,
            device=device,
            simplify_factor=0.5,
        )

    print(raycaster)

    horizontal_angles = torch.linspace(-torch.pi / 4, torch.pi / 4, 20)
    vertical_angles = torch.linspace(-torch.pi / 6, torch.pi / 6, 10)
    hh, vv = torch.meshgrid(horizontal_angles, vertical_angles)

    ray_dirs = torch.stack([
        torch.cos(hh) * torch.cos(vv),
        torch.sin(hh) * torch.cos(vv),
        torch.sin(vv),
    ], dim=2)

    ray_dirs = ray_dirs.reshape(-1, 3).to(device)
    ray_starts = torch.zeros(ray_dirs.shape[0], 3, device=device)
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=1, keepdim=True)

    ray_starts[:, 0] = - 1.0
    ray_dirs[:, 0] = ray_dirs[:, 0].abs()

    translations = torch.from_numpy(np.stack(translations, axis=0)).float().to(device)
    quats = torch.from_numpy(np.stack(quats, axis=0)).float().to(device)
    print(translations.shape, quats.shape)

    N = 4096
    T = 100
    start_time = time.perf_counter()
    for i in range(T):
        hit_positions, hit_distances = raycaster.raycast(
            translations.expand(N, *translations.shape),
            quats.expand(N, *quats.shape),
            ray_starts.expand(N, *ray_starts.shape),
            ray_dirs.expand(N, *ray_dirs.shape),
            min_dist=0.0,
            max_dist=5.0,
        )
    end_time = time.perf_counter()
    print(f"Average time: {(end_time - start_time) / T} s")
    hit_positions = hit_positions[0]

    scene = trimesh.Scene([mesh for mesh in trimesh_list])
    segments = torch.stack([ray_starts, hit_positions], dim=1).cpu().numpy()
    lines = trimesh.load_path(segments)
    frame = trimesh.creation.axis()
    scene.add_geometry(lines)
    scene.add_geometry(frame)
    scene.show()

