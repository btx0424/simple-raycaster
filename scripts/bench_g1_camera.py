"""G1 camera-scene benchmark for MultiMeshRaycaster / RaycastCamera.

Scene: ground plane + cube + Unitree G1 (Inspire hands). Rays are OpenCV
pinhole cameras with randomized look-at poses.

    .venv/bin/python scripts/bench_g1_camera.py
    .venv/bin/python scripts/bench_g1_camera.py --n 16 --width 64 --height 48
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh
import warp as wp
from scipy.spatial.transform import Rotation

from simple_raycaster import MultiMeshRaycaster, RaycastCamera
from simple_raycaster.helpers import quat_rotate
from simple_raycaster.kernels import (
    transform_and_query_point_closest_kernel,
    transform_and_query_point_kernel,
)

DEFAULT_G1_XML = (
    "/mnt/workspace/btx0424/aa-projects/object_hoi/src/assets/unitree_g1/"
    "g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
)


def format_memory(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def look_at_opencv_wxyz(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """OpenCV camera quat WXYZ: +Z forward, +Y down, +X right."""
    z_cam = target - eye
    z_norm = np.linalg.norm(z_cam)
    if z_norm < 1e-8:
        raise ValueError("eye and target are coincident")
    z_cam = z_cam / z_norm
    # right = forward × up (not up × forward — that yields left / Y-up).
    x_cam = np.cross(z_cam, world_up)
    x_norm = np.linalg.norm(x_cam)
    if x_norm < 1e-6:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_cam = np.cross(z_cam, fallback)
        x_norm = np.linalg.norm(x_cam)
    x_cam = x_cam / x_norm
    y_cam = np.cross(z_cam, x_cam)  # down
    rot = np.stack([x_cam, y_cam, z_cam], axis=1)
    xyzw = Rotation.from_matrix(rot).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def sample_cameras(
    n: int,
    *,
    seed: int,
    radius_min: float,
    radius_max: float,
    elev_min_deg: float,
    elev_max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    az = rng.uniform(0.0, 2.0 * math.pi, size=n)
    el = rng.uniform(math.radians(elev_min_deg), math.radians(elev_max_deg), size=n)
    radius = rng.uniform(radius_min, radius_max, size=n)
    eye = np.stack(
        [
            radius * np.cos(el) * np.cos(az),
            radius * np.cos(el) * np.sin(az),
            radius * np.sin(el),
        ],
        axis=-1,
    ).astype(np.float32)
    target = np.zeros(3, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    quats = np.stack([look_at_opencv_wxyz(eye[i].astype(np.float64), target, up) for i in range(n)])
    return eye, quats.astype(np.float32)


def pinhole_rays(
    cam_pos: torch.Tensor,
    cam_quat_wxyz: torch.Tensor,
    width: int,
    height: int,
    fov_y_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OpenCV pinhole rays matching ``RaycastCamera`` (convention=opencv)."""
    n = cam_pos.shape[0]
    device = cam_pos.device
    fy = height / (2.0 * math.tan(math.radians(fov_y_deg) * 0.5))
    fx = fy
    cx = 0.5 * width
    cy = 0.5 * height
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    u = (xs + 0.5 - cx) / fx
    v = (ys + 0.5 - cy) / fy
    dir_c = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(1, -1, 3)
    dir_c = dir_c / dir_c.norm(dim=-1, keepdim=True)
    dir_c = dir_c.expand(n, -1, -1)
    quat = cam_quat_wxyz.unsqueeze(1).expand(-1, dir_c.shape[1], -1)
    ray_dirs = quat_rotate(quat, dir_c)
    ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
    ray_starts = cam_pos.unsqueeze(1).expand(-1, dir_c.shape[1], -1).contiguous()
    return ray_starts.contiguous(), ray_dirs.contiguous()


def load_scene(
    xml_path: str,
    device: str,
    *,
    bvh_constructor: str | None = None,
    merge_static: bool = False,
    robot_first: bool = False,
    cube_first: bool = False,
    simplify_factor: float = 0.0,
) -> tuple[MultiMeshRaycaster, torch.Tensor, torch.Tensor, dict]:
    xml_path = str(Path(xml_path).resolve())
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    # Inspire-hand DFQ parks the free joint at the origin; the stock G1 XML
    # places the pelvis at z≈0.793 so the feet rest on z=0.
    if model.nq >= 7 and abs(float(data.qpos[2])) < 1e-6:
        data.qpos[2] = 0.793
    mujoco.mj_forward(model, data)

    body_names = [model.body(i).name for i in range(model.nbody)]
    g1 = MultiMeshRaycaster.from_MjModel(
        body_names,
        model,
        device=device,
        simplify_factor=simplify_factor,
        bvh_constructor=bvh_constructor,
    )
    mesh_names = list(g1.mesh_names or [])
    name_to_id = {model.body(i).name: i for i in range(model.nbody)}

    g1_pos = torch.tensor(
        np.stack([data.xpos[name_to_id[name]] for name in mesh_names], axis=0),
        device=device,
        dtype=torch.float32,
    )
    g1_quat = torch.tensor(
        np.stack([data.xquat[name_to_id[name]] for name in mesh_names], axis=0),
        device=device,
        dtype=torch.float32,
    )

    ground = trimesh.creation.box(extents=(10.0, 10.0, 0.05))
    ground.apply_translation((0.0, 0.0, -0.025))
    cube = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    cube.apply_translation((0.45, 0.0, 0.25))
    static_meshes: list = [ground, cube]
    if cube_first:
        static_meshes = [cube, ground]
    if merge_static:
        static_meshes = [trimesh.util.concatenate(static_meshes)]
    n_static = len(static_meshes)

    if robot_first:
        meshes = [*g1.meshes_wp, *static_meshes]
    else:
        meshes = [*static_meshes, *g1.meshes_wp]
    raycaster = MultiMeshRaycaster(
        meshes,
        device=device,
        bvh_constructor=bvh_constructor,
    )
    stats = {
        "g1_xml": xml_path,
        "n_g1_bodies": len(mesh_names),
        "n_meshes": raycaster.n_meshes,
        "n_points": raycaster.n_points,
        "n_faces": raycaster.n_faces,
        "g1_names": mesh_names,
        "bvh_constructor": bvh_constructor or "default(lbvh)",
        "merge_static": merge_static,
        "robot_first": robot_first,
        "cube_first": cube_first,
        "simplify_factor": simplify_factor,
    }
    return raycaster, g1_pos, g1_quat, stats | {"n_static": n_static}


def expand_poses(
    n: int,
    g1_pos: torch.Tensor,
    g1_quat: torch.Tensor,
    n_static: int,
    *,
    robot_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = g1_pos.device
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    static_pos = torch.zeros(n, n_static, 3, device=device, dtype=torch.float32)
    static_quat = ident.view(1, 1, 4).expand(n, n_static, 4).contiguous()
    g1_pos_b = g1_pos.unsqueeze(0).expand(n, -1, -1)
    g1_quat_b = g1_quat.unsqueeze(0).expand(n, -1, -1)
    if robot_first:
        mesh_pos = torch.cat([g1_pos_b, static_pos], dim=1)
        mesh_quat = torch.cat([g1_quat_b, static_quat], dim=1)
    else:
        mesh_pos = torch.cat([static_pos, g1_pos_b], dim=1)
        mesh_quat = torch.cat([static_quat, g1_quat_b], dim=1)
    return mesh_pos.contiguous(), mesh_quat.contiguous()


def check_parity(ref: torch.Tensor, other: torch.Tensor, name: str, atol: float) -> dict:
    diff = (ref - other).abs()
    ok = bool(torch.allclose(ref, other, atol=atol, rtol=0.0))
    return {
        "name": name,
        "ok": ok,
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "atol": atol,
    }


@torch.no_grad()
def benchmark_fn(fn, warmup: int, iters: int, device: torch.device) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    wp.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
        wp.synchronize()
    end.record()
    torch.cuda.synchronize(device)

    mem_after = torch.cuda.memory_allocated(device)
    mem_peak = torch.cuda.max_memory_allocated(device)
    ms = start.elapsed_time(end) / max(iters, 1)
    return {
        "ms_per_iter": float(ms),
        "mem_before": int(mem_before),
        "mem_after": int(mem_after),
        "mem_peak": int(mem_peak),
        "mem_delta": int(mem_after - mem_before),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_G1_XML)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--max-dist", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip-unfused", action="store_true")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument(
        "--bvh",
        type=str,
        default=None,
        help="wp.Mesh bvh_constructor: sah, median, lbvh (default: Warp CUDA lbvh)",
    )
    parser.add_argument(
        "--sweep-bvh",
        action="store_true",
        help="Rebuild the scene for sah/median/lbvh and time closest-hit only",
    )
    parser.add_argument("--merge-static", action="store_true", help="Concatenate ground+cube into one mesh")
    parser.add_argument("--robot-first", action="store_true", help="Query G1 body meshes before ground/cube")
    parser.add_argument("--cube-first", action="store_true", help="Query the nearby cube before the ground plane")
    parser.add_argument("--block-dim", type=int, default=128, help="CUDA threads per block for closest-hit kernel")
    parser.add_argument(
        "--sweep-block-dim",
        action="store_true",
        help="Time closest-hit at block_dim in {32,64,128,256,512}",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.0,
        help="Quadric decimation factor for G1 bodies (0 = none)",
    )
    parser.add_argument("--graph", action="store_true", help="Also bench CUDA graph replay of fused closest-hit")
    parser.add_argument("--proximity", action="store_true", help="Also bench fused closest-point vs per-mesh argmin")
    parser.add_argument("--n-probes", type=int, default=256, help="Probe points per env for --proximity")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    torch_device = torch.device(device)

    wp.init()
    raycaster, g1_pos, g1_quat, stats = load_scene(
        args.xml,
        device,
        bvh_constructor=args.bvh,
        merge_static=args.merge_static,
        robot_first=args.robot_first,
        cube_first=args.cube_first,
        simplify_factor=args.simplify,
    )
    n_static = stats["n_static"]
    mesh_pos, mesh_quat = expand_poses(
        args.n, g1_pos, g1_quat, n_static, robot_first=args.robot_first
    )
    raycaster.block_dim = int(args.block_dim)

    cam_pos_np, cam_quat_np = sample_cameras(
        args.n,
        seed=args.seed,
        radius_min=1.5,
        radius_max=3.5,
        elev_min_deg=15.0,
        elev_max_deg=60.0,
    )
    cam_pos = torch.from_numpy(cam_pos_np).to(device=torch_device)
    cam_quat = torch.from_numpy(cam_quat_np).to(device=torch_device)
    ray_starts, ray_dirs = pinhole_rays(cam_pos, cam_quat, args.width, args.height, args.fov)
    n_rays = ray_dirs.shape[1]

    print("=" * 70)
    print("G1 CAMERA SCENE")
    print("=" * 70)
    print(f"warp {wp.__version__}  torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}")
    print(f"xml  {stats['g1_xml']}")
    print(
        f"meshes {stats['n_meshes']} (static {n_static} + g1 {stats['n_g1_bodies']})  "
        f"verts {stats['n_points']}  faces {stats['n_faces']}"
    )
    print(f"N={args.n}  {args.width}x{args.height} ({n_rays} rays)  fov={args.fov}  max_dist={args.max_dist}")
    print(f"bvh={stats['bvh_constructor']}  merge_static={stats['merge_static']}  "
          f"robot_first={stats['robot_first']}  cube_first={stats['cube_first']}  "
          f"simplify={stats['simplify_factor']}  block_dim={raycaster.block_dim}")
    print(raycaster)

    kwargs = dict(
        mesh_pos_w=mesh_pos,
        mesh_quat_w=mesh_quat,
        ray_starts_w=ray_starts,
        ray_dirs_w=ray_dirs,
        min_dist=args.min_dist,
        max_dist=args.max_dist,
    )

    print("\nParity (one shot)...")
    pos_new, dist_new, nrm_new = raycaster.raycast_fused(**kwargs, closest_hit=True)
    pos_new, dist_new, nrm_new = pos_new.clone(), dist_new.clone(), nrm_new.clone()
    hit_frac = float((dist_new < args.max_dist - 1e-4).float().mean())
    print(f"  closest-hit hit_frac={hit_frac:.3f}")
    if hit_frac < 0.3:
        raise SystemExit(f"sanity failed: hit_frac={hit_frac:.3f} < 0.3 (cameras likely misoriented)")

    parity_rows = []
    if not args.skip_legacy:
        pos_old, dist_old, nrm_old = raycaster.raycast_fused(**kwargs, closest_hit=False)
        parity_rows.extend(
            [
                check_parity(dist_old, dist_new, "dist legacy vs closest", 1e-5),
                check_parity(pos_old, pos_new, "pos legacy vs closest", 1e-5),
                check_parity(nrm_old, nrm_new, "normal legacy vs closest", 1e-4),
            ]
        )
        mesh_idx = torch.arange(raycaster.n_meshes, device=torch_device).expand(args.n, -1)
        pos_sub, dist_sub, nrm_sub = raycaster.raycast_fused(
            **kwargs, closest_hit=True, mesh_indices=mesh_idx
        )
        parity_rows.extend(
            [
                check_parity(dist_new, dist_sub, "dist closest vs mesh_indices", 1e-5),
                check_parity(pos_new, pos_sub, "pos closest vs mesh_indices", 1e-5),
                check_parity(nrm_new, nrm_sub, "normal closest vs mesh_indices", 1e-4),
            ]
        )
        del pos_sub, dist_sub, nrm_sub, mesh_idx
    if not args.skip_unfused:
        pos_u, dist_u, nrm_u = raycaster.raycast(**kwargs)
        parity_rows.extend(
            [
                check_parity(dist_u, dist_new, "dist unfused vs closest", 1e-5),
                check_parity(pos_u, pos_new, "pos unfused vs closest", 1e-5),
                check_parity(nrm_u, nrm_new, "normal unfused vs closest", 1e-4),
            ]
        )

    cam = None
    depth_cam = None
    if not args.skip_camera:
        cam = RaycastCamera(
            args.width,
            args.height,
            fov_y_deg=args.fov,
            near=args.min_dist,
            far=args.max_dist,
            convention="opencv",
            depth_mode="ray",
            device=device,
        )
        cam.bind_meshes(raycaster)
        rgb, depth_cam, mask = cam.render(
            cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat
        )
        dist_img = dist_new.reshape(args.n, args.height, args.width)
        parity_rows.append(check_parity(dist_img, depth_cam, "dist lidar vs camera depth", 1e-4))
        print(f"  camera mask_frac={float(mask.float().mean()):.3f}  rgb_mean={float(rgb.mean()):.3f}")

    for row in parity_rows:
        mark = "ok" if row["ok"] else "FAIL"
        print(f"  [{mark}] {row['name']}: max={row['max_abs']:.3e} mean={row['mean_abs']:.3e}")

    # Drop parity tensors so timed peak-memory is not polluted by 3D leftovers.
    del pos_new, dist_new, nrm_new
    if not args.skip_legacy:
        del pos_old, dist_old, nrm_old
    if not args.skip_unfused:
        del pos_u, dist_u, nrm_u
    if depth_cam is not None:
        del rgb, depth_cam, mask, dist_img
    if cam is not None:
        cam._work.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(torch_device)

    results = []

    def add_result(name: str, fn, *, n_queries: int | None = None) -> None:
        print(f"\nBenchmarking {name}...")
        timing = benchmark_fn(fn, args.warmup, args.iters, torch_device)
        q = n_rays if n_queries is None else n_queries
        rays_per_s = (args.n * q) / (timing["ms_per_iter"] / 1000.0)
        row = {"name": name, "rays_per_s": float(rays_per_s), "n_queries": int(args.n * q), **timing}
        results.append(row)
        print(
            f"  {timing['ms_per_iter']:.3f} ms/iter  {rays_per_s/1e6:.2f} M queries/s  "
            f"peak {format_memory(timing['mem_peak'])}"
        )

    add_result("lidar_fused_closest", lambda: raycaster.raycast_fused(**kwargs, closest_hit=True))
    if not args.skip_legacy:
        add_result("lidar_fused_legacy", lambda: raycaster.raycast_fused(**kwargs, closest_hit=False))
    if not args.skip_unfused:
        add_result("lidar_unfused", lambda: raycaster.raycast(**kwargs))
    if cam is not None:
        add_result(
            "camera_rgbd",
            lambda: cam.render(cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat),
        )
    if args.graph:
        print("\nCapturing CUDA graph...")
        try:
            raycaster.capture_fused(**kwargs)
            add_result(
                "lidar_fused_graph",
                lambda: raycaster.raycast_fused(**kwargs, closest_hit=True, use_graph=True),
            )
        except Exception as exc:
            print(f"  graph capture failed: {type(exc).__name__}: {exc}")
    if args.sweep_bvh:
        already = args.bvh or "lbvh"
        for ctor in ("lbvh", "sah", "median"):
            if ctor == already:
                continue
            print(f"\nRebuilding scene bvh_constructor={ctor}...")
            rc, gp, gq, st = load_scene(
                args.xml,
                device,
                bvh_constructor=ctor,
                merge_static=args.merge_static,
                robot_first=args.robot_first,
                cube_first=args.cube_first,
                simplify_factor=args.simplify,
            )
            print(f"  {rc}")
            mp, mq = expand_poses(
                args.n, gp, gq, st["n_static"], robot_first=args.robot_first
            )
            kw = dict(
                mesh_pos_w=mp,
                mesh_quat_w=mq,
                ray_starts_w=ray_starts,
                ray_dirs_w=ray_dirs,
                min_dist=args.min_dist,
                max_dist=args.max_dist,
            )
            add_result(
                f"lidar_fused_bvh_{ctor}",
                lambda rc=rc, kw=kw: rc.raycast_fused(**kw, closest_hit=True),
            )

    if args.sweep_block_dim:
        saved = raycaster.block_dim
        for bd in (32, 64, 128, 256, 512):
            if bd == saved:
                continue
            raycaster.block_dim = bd
            add_result(
                f"lidar_fused_block_{bd}",
                lambda: raycaster.raycast_fused(**kwargs, closest_hit=True),
            )
        raycaster.block_dim = saved

    if args.proximity:
        if not raycaster.initialized:
            raycaster.initialize()
        rng = torch.Generator(device=torch_device)
        rng.manual_seed(args.seed)
        probes = torch.randn(args.n, args.n_probes, 3, device=torch_device, generator=rng)
        probes = probes * 0.6 + torch.tensor([0.0, 0.0, 0.4], device=torch_device)
        enabled = torch.ones(args.n, dtype=torch.bool, device=torch_device)
        prox_max = 2.0

        def launch_prox_legacy():
            closest = torch.empty(
                args.n, raycaster.n_meshes, args.n_probes, 3,
                device=torch_device, dtype=torch.float32,
            )
            dist = torch.empty(
                args.n, raycaster.n_meshes, args.n_probes,
                device=torch_device, dtype=torch.float32,
            )
            wp.launch(
                transform_and_query_point_kernel,
                dim=(args.n, raycaster.n_meshes, args.n_probes),
                inputs=[
                    raycaster.meshes_array,
                    wp.from_torch(mesh_pos, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(mesh_quat, dtype=wp.vec4, return_ctype=True),
                    wp.from_torch(probes, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                    prox_max,
                    0,
                ],
                outputs=[
                    wp.from_torch(closest, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(dist, dtype=wp.float32, return_ctype=True),
                ],
                device=raycaster.device,
                record_tape=False,
            )
            idx = dist.argmin(dim=1)
            gather = idx.unsqueeze(1)
            return (
                closest.gather(1, gather.unsqueeze(-1).expand(-1, 1, -1, 3)).squeeze(1),
                dist.gather(1, gather).squeeze(1),
            )

        prox_pos_cache = torch.empty(args.n, args.n_probes, 3, device=torch_device)
        prox_dist_cache = torch.empty(args.n, args.n_probes, device=torch_device)

        def launch_prox_closest():
            wp.launch(
                transform_and_query_point_closest_kernel,
                dim=(args.n, args.n_probes),
                inputs=[
                    raycaster.meshes_array,
                    wp.from_torch(mesh_pos, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(mesh_quat, dtype=wp.vec4, return_ctype=True),
                    wp.from_torch(probes, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(enabled, dtype=wp.bool, return_ctype=True),
                    prox_max,
                    0,
                ],
                outputs=[
                    wp.from_torch(prox_pos_cache, dtype=wp.vec3, return_ctype=True),
                    wp.from_torch(prox_dist_cache, dtype=wp.float32, return_ctype=True),
                ],
                device=raycaster.device,
                record_tape=False,
                block_dim=raycaster.block_dim,
            )
            return prox_pos_cache, prox_dist_cache

        print("\nProximity parity (one shot)...")
        pos_pl, dist_pl = launch_prox_legacy()
        pos_pc, dist_pc = launch_prox_closest()
        for row in (
            check_parity(dist_pl, dist_pc, "prox dist legacy vs closest", 1e-5),
            check_parity(pos_pl, pos_pc, "prox pos legacy vs closest", 1e-5),
        ):
            mark = "ok" if row["ok"] else "FAIL"
            print(f"  [{mark}] {row['name']}: max={row['max_abs']:.3e} mean={row['mean_abs']:.3e}")
            parity_rows.append(row)
        del pos_pl, dist_pl, pos_pc, dist_pc
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch_device)
        add_result("proximity_closest", launch_prox_closest, n_queries=args.n_probes)
        add_result("proximity_legacy", launch_prox_legacy, n_queries=args.n_probes)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'name':<24} {'ms/iter':>10} {'M q/s':>10} {'peak mem':>12}")
    for row in results:
        print(
            f"{row['name']:<24} {row['ms_per_iter']:10.3f} {row['rays_per_s']/1e6:10.2f} "
            f"{format_memory(row['mem_peak']):>12}"
        )

    blob = {
        "warp": wp.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "n": args.n,
        "width": args.width,
        "height": args.height,
        "fov": args.fov,
        "n_rays": n_rays,
        "hit_frac": hit_frac,
        "scene": {
            k: stats[k]
            for k in (
                "n_meshes",
                "n_points",
                "n_faces",
                "n_g1_bodies",
                "bvh_constructor",
                "merge_static",
                "robot_first",
                "cube_first",
                "simplify_factor",
            )
        },
        "parity": parity_rows,
        "results": results,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(blob, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
