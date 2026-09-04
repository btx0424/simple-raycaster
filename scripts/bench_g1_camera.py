"""G1 camera-scene benchmark for MultiMeshRaycaster / RaycastCamera.

Scene: ground plane + cube + Unitree G1 (Inspire hands). Rays are OpenCV
pinhole cameras with randomized look-at poses.

    python scripts/bench_g1_camera.py
    python scripts/bench_g1_camera.py --n 16 --width 64 --height 48

BVH tuning (constructor + leaf size) for lidar fused closest-hit and camera:

    python scripts/bench_g1_camera.py --sweep-bvh --skip-legacy --skip-unfused
    python scripts/bench_g1_camera.py --sweep-bvh --sweep-bvh-mode full
    python scripts/bench_g1_camera.py --bvh lbvh --bvh-leaf 4
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
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

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from g1_assets import find_g1_xml, resolve_g1_xml

__all__ = ["find_g1_xml", "resolve_g1_xml", "DEFAULT_G1_XML"]

# Back-compat for ``from bench_g1_camera import DEFAULT_G1_XML``.
DEFAULT_G1_XML = str(find_g1_xml() or "")


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


def _ctor_label(ctor: str | None) -> str:
    return "default" if ctor is None else str(ctor)


def _leaf_label(leaf: int | None) -> str:
    return "default" if leaf is None else str(int(leaf))


def probe_bvh_support(device: str) -> dict:
    """Discover which ``wp.Mesh`` BVH knobs this Warp build accepts."""
    params = inspect.signature(wp.Mesh.__init__).parameters
    has_ctor = "bvh_constructor" in params
    has_leaf = "bvh_leaf_size" in params
    constructors: list[str | None] = [None]
    if has_ctor:
        constructors = [None, "lbvh", "sah", "median"]
        # cuBQL is Warp ≥ 1.13 and may be compiled out; probe with a tiny mesh.
        try:
            pts = wp.array(
                np.zeros((3, 3), dtype=np.float32), dtype=wp.vec3, device=device
            )
            idx = wp.array(
                np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device
            )
            mesh = wp.Mesh(points=pts, indices=idx, bvh_constructor="cubql")
            del mesh
            constructors.append("cubql")
        except Exception:
            pass
    leaf_sizes: list[int | None] = [None]
    if has_leaf:
        # Warp docs: 1 favors ray/AABB intersection; 4–8 often helps closest-point.
        leaf_sizes = [None, 1, 2, 4, 8, 16]
    return {
        "has_constructor": has_ctor,
        "has_leaf_size": has_leaf,
        "constructors": constructors,
        "leaf_sizes": leaf_sizes,
        "warp": wp.__version__,
    }


def parse_ctor_list(raw: str | None, available: list[str | None]) -> list[str | None]:
    if not raw:
        return list(available)
    out: list[str | None] = []
    for part in raw.split(","):
        token = part.strip().lower()
        if token in ("", "default", "none"):
            out.append(None)
        else:
            out.append(token)
    return out


def parse_leaf_list(raw: str | None, available: list[int | None]) -> list[int | None]:
    if not raw:
        return list(available)
    out: list[int | None] = []
    for part in raw.split(","):
        token = part.strip().lower()
        if token in ("", "default", "none"):
            out.append(None)
        else:
            out.append(int(token))
    return out


def load_scene(
    xml_path: str,
    device: str,
    *,
    bvh_constructor: str | None = None,
    bvh_leaf_size: int | None = None,
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
    # Rebuild G1 wp.Meshes with the requested BVH knobs so leaf_size/ctor
    # apply to every body (not only static trimeshes converted later).
    g1 = MultiMeshRaycaster.from_MjModel(
        body_names,
        model,
        device=device,
        simplify_factor=simplify_factor,
        bvh_constructor=bvh_constructor,
        bvh_leaf_size=bvh_leaf_size,
    )
    mesh_names = list(g1.mesh_names or [])
    name_to_id = {model.body(i).name: i for i in range(model.nbody)}

    # Keep soles clearly above the ground top (z=0). Dark feet on a light
    # plane look "sunk" when coplanar even if the BVH hits are correct.
    foot_clearance = 0.03
    z_min = np.inf
    for i, name in enumerate(mesh_names):
        bid = name_to_id[name]
        pos = np.asarray(data.xpos[bid], dtype=np.float64)
        quat = np.asarray(data.xquat[bid], dtype=np.float64)  # wxyz
        pts = g1.meshes_wp[i].points.numpy().astype(np.float64)
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        z_min = min(z_min, float((pts @ rot.T + pos)[:, 2].min()))
    if np.isfinite(z_min) and z_min < foot_clearance and model.nq >= 7:
        data.qpos[2] = float(data.qpos[2]) + (foot_clearance - z_min)
        mujoco.mj_forward(model, data)

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
    # Bottom of the 0.5 m cube sits at the same clearance as the feet.
    cube = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    cube.apply_translation((0.45, 0.0, 0.25 + foot_clearance))
    static_meshes: list = [ground, cube]
    # Authored-looking static albedos (not from MuJoCo).
    static_albedos = np.asarray(
        [[0.38, 0.38, 0.36], [0.78, 0.28, 0.16]], dtype=np.float32
    )
    if cube_first:
        static_meshes = [cube, ground]
        static_albedos = static_albedos[[1, 0]]
    if merge_static:
        static_meshes = [trimesh.util.concatenate(static_meshes)]
        static_albedos = np.mean(static_albedos, axis=0, keepdims=True).astype(np.float32)
    n_static = len(static_meshes)
    g1_albedos = np.asarray(getattr(g1, "mesh_albedos"), dtype=np.float32).reshape(-1, 3)
    if g1_albedos.shape[0] != len(mesh_names):
        raise RuntimeError(
            f"G1 mesh_albedos {g1_albedos.shape[0]} != mesh_names {len(mesh_names)}"
        )

    if robot_first:
        meshes = [*g1.meshes_wp, *static_meshes]
        mesh_albedos = np.concatenate([g1_albedos, static_albedos], axis=0)
    else:
        meshes = [*static_meshes, *g1.meshes_wp]
        mesh_albedos = np.concatenate([static_albedos, g1_albedos], axis=0)
    raycaster = MultiMeshRaycaster(
        meshes,
        device=device,
        bvh_constructor=bvh_constructor,
        bvh_leaf_size=bvh_leaf_size,
    )
    raycaster.mesh_albedos = mesh_albedos
    raycaster.mesh_names = (
        [*mesh_names, *(["ground", "cube"][:n_static] if not merge_static else ["static"])]
        if robot_first
        else [
            *(["ground", "cube"][:n_static] if not merge_static else ["static"]),
            *mesh_names,
        ]
    )
    stats = {
        "g1_xml": xml_path,
        "n_g1_bodies": len(mesh_names),
        "n_meshes": raycaster.n_meshes,
        "n_points": raycaster.n_points,
        "n_faces": raycaster.n_faces,
        "g1_names": mesh_names,
        "albedos": mesh_albedos,
        "bvh_constructor": _ctor_label(bvh_constructor),
        "bvh_leaf_size": _leaf_label(bvh_leaf_size),
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


def _bvh_config_pairs(
    mode: str,
    constructors: list[str | None],
    leaf_sizes: list[int | None],
) -> list[tuple[str | None, int | None]]:
    """Build (constructor, leaf_size) pairs for a sweep mode.

    ``staged`` means: all constructors at default leaf, then all leaves at the
    constructor named ``__best__`` placeholder (resolved after phase 1).
    Callers should use ``staged_phase`` helpers instead for staged runs.
    """
    if mode == "constructors":
        leaf = leaf_sizes[0] if leaf_sizes else None
        return [(c, leaf) for c in constructors]
    if mode == "leaf":
        ctor = constructors[0] if constructors else None
        return [(ctor, leaf) for leaf in leaf_sizes]
    if mode == "full":
        return [(c, leaf) for c in constructors for leaf in leaf_sizes]
    raise ValueError(f"unknown sweep mode {mode!r}")


def recommend_bvh(rows: list[dict]) -> dict:
    """Pick best BVH configs by lidar, camera, and combined ms/iter."""
    by_key: dict[tuple, dict] = {}
    for row in rows:
        key = (row["bvh_constructor"], row["bvh_leaf_size"])
        slot = by_key.setdefault(
            key,
            {
                "bvh_constructor": row["bvh_constructor"],
                "bvh_leaf_size": row["bvh_leaf_size"],
                "build_s": row.get("build_s"),
            },
        )
        if row["path"] == "lidar":
            slot["lidar_ms"] = row["ms_per_iter"]
            slot["lidar_peak"] = row.get("mem_peak")
        elif row["path"] == "camera":
            slot["camera_ms"] = row["ms_per_iter"]
            slot["camera_peak"] = row.get("mem_peak")
    configs = list(by_key.values())
    for cfg in configs:
        lidar = cfg.get("lidar_ms")
        cam = cfg.get("camera_ms")
        cfg["combined_ms"] = (
            (lidar or 0.0) + (cam or 0.0) if lidar is not None or cam is not None else None
        )

    def best(metric: str) -> dict | None:
        scored = [c for c in configs if c.get(metric) is not None]
        if not scored:
            return None
        return min(scored, key=lambda c: c[metric])

    return {
        "lidar": best("lidar_ms"),
        "camera": best("camera_ms"),
        "combined": best("combined_ms"),
        "configs": configs,
    }


@torch.no_grad()
def run_bvh_sweep(
    *,
    xml_path: str,
    device: str,
    torch_device: torch.device,
    n: int,
    width: int,
    height: int,
    fov: float,
    min_dist: float,
    max_dist: float,
    warmup: int,
    iters: int,
    ray_starts: torch.Tensor,
    ray_dirs: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_quat: torch.Tensor,
    merge_static: bool,
    robot_first: bool,
    cube_first: bool,
    simplify_factor: float,
    block_dim: int,
    mode: str,
    constructors: list[str | None],
    leaf_sizes: list[int | None],
    skip_camera: bool,
    n_rays: int,
) -> tuple[list[dict], dict]:
    """Rebuild the scene for each BVH config; time lidar + camera query paths."""

    def configs_for_staged_phase1() -> list[tuple[str | None, int | None]]:
        # Default leaf (Warp's per-constructor default) while comparing builders.
        return [(c, None) for c in constructors]

    def configs_for_staged_phase2(best_ctor: str | None) -> list[tuple[str | None, int | None]]:
        # Skip leaf=None (already measured in phase 1 for this ctor).
        return [(best_ctor, leaf) for leaf in leaf_sizes if leaf is not None]

    if mode == "staged":
        phases: list[tuple[str, list[tuple[str | None, int | None]]]] = [
            ("constructors@leaf=default", configs_for_staged_phase1()),
        ]
    else:
        phases = [(mode, _bvh_config_pairs(mode, constructors, leaf_sizes))]

    sweep_rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def time_config(ctor: str | None, leaf: int | None, phase: str) -> None:
        key = (_ctor_label(ctor), _leaf_label(leaf))
        if key in seen:
            return
        seen.add(key)
        label = f"ctor={key[0]} leaf={key[1]}"
        print(f"\nBVH rebuild [{phase}] {label}...")
        t0 = time.perf_counter()
        try:
            rc, gp, gq, st = load_scene(
                xml_path,
                device,
                bvh_constructor=ctor,
                bvh_leaf_size=leaf,
                merge_static=merge_static,
                robot_first=robot_first,
                cube_first=cube_first,
                simplify_factor=simplify_factor,
            )
        except Exception as exc:
            print(f"  BUILD FAIL: {type(exc).__name__}: {exc}")
            sweep_rows.append(
                {
                    "path": "build",
                    "bvh_constructor": key[0],
                    "bvh_leaf_size": key[1],
                    "phase": phase,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return
        build_s = time.perf_counter() - t0
        rc.block_dim = int(block_dim)
        print(f"  built in {build_s:.2f}s  {rc}")
        mp, mq = expand_poses(n, gp, gq, st["n_static"], robot_first=robot_first)
        kw = dict(
            mesh_pos_w=mp,
            mesh_quat_w=mq,
            ray_starts_w=ray_starts,
            ray_dirs_w=ray_dirs,
            min_dist=min_dist,
            max_dist=max_dist,
        )

        timing = benchmark_fn(
            lambda: rc.raycast_fused(**kw, closest_hit=True),
            warmup,
            iters,
            torch_device,
        )
        rays_per_s = (n * n_rays) / (timing["ms_per_iter"] / 1000.0)
        lidar_row = {
            "path": "lidar",
            "name": f"lidar_bvh_{key[0]}_L{key[1]}",
            "bvh_constructor": key[0],
            "bvh_leaf_size": key[1],
            "phase": phase,
            "build_s": build_s,
            "ok": True,
            "rays_per_s": float(rays_per_s),
            "n_queries": int(n * n_rays),
            **timing,
        }
        sweep_rows.append(lidar_row)
        print(
            f"  lidar  {timing['ms_per_iter']:.3f} ms/iter  "
            f"{rays_per_s/1e6:.2f} M q/s  peak {format_memory(timing['mem_peak'])}"
        )

        if not skip_camera:
            cam = RaycastCamera(
                width,
                height,
                fov_y_deg=fov,
                near=min_dist,
                far=max_dist,
                convention="opencv",
                depth_mode="ray",
                device=device,
            )
            cam.bvh_constructor = ctor
            cam.bvh_leaf_size = leaf
            cam.bind_meshes(rc)
            timing_c = benchmark_fn(
                lambda: cam.render(cam_pos, cam_quat, mesh_pos_w=mp, mesh_quat_w=mq),
                warmup,
                iters,
                torch_device,
            )
            cam_qps = (n * n_rays) / (timing_c["ms_per_iter"] / 1000.0)
            cam_row = {
                "path": "camera",
                "name": f"camera_bvh_{key[0]}_L{key[1]}",
                "bvh_constructor": key[0],
                "bvh_leaf_size": key[1],
                "phase": phase,
                "build_s": build_s,
                "ok": True,
                "rays_per_s": float(cam_qps),
                "n_queries": int(n * n_rays),
                **timing_c,
            }
            sweep_rows.append(cam_row)
            print(
                f"  camera {timing_c['ms_per_iter']:.3f} ms/iter  "
                f"{cam_qps/1e6:.2f} M q/s  peak {format_memory(timing_c['mem_peak'])}"
            )
            cam._work.clear()
            del cam

        del rc, gp, gq, st, mp, mq
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch_device)

    for phase_name, pairs in phases:
        for ctor, leaf in pairs:
            time_config(ctor, leaf, phase_name)

    if mode == "staged":
        # Prefer lidar winner for leaf sweep; fall back to combined / first ctor.
        rec_partial = recommend_bvh(sweep_rows)
        best = rec_partial["lidar"] or rec_partial["combined"]
        if best is not None:
            best_ctor_label = best["bvh_constructor"]
            best_ctor = None if best_ctor_label == "default" else best_ctor_label
            print(
                f"\nStaged leaf sweep using best constructor so far: "
                f"{best_ctor_label} (lidar {best.get('lidar_ms', float('nan')):.3f} ms)"
            )
            for ctor, leaf in configs_for_staged_phase2(best_ctor):
                time_config(ctor, leaf, "leaf@best_ctor")
            # Also sweep leaves on explicit lbvh if the winner was Warp default —
            # default == lbvh on CUDA, but labeling differs for clarity.
            if best_ctor is None and "lbvh" in {_ctor_label(c) for c in constructors}:
                for ctor, leaf in configs_for_staged_phase2("lbvh"):
                    time_config(ctor, leaf, "leaf@lbvh")

    recommendation = recommend_bvh([r for r in sweep_rows if r.get("ok")])
    return sweep_rows, recommendation


def _print_bvh_recommendation(rec: dict) -> None:
    print("\n" + "=" * 70)
    print("BVH RECOMMENDATION (lower ms/iter is better)")
    print("=" * 70)

    def fmt(slot: str, cfg: dict | None) -> None:
        if cfg is None:
            print(f"  {slot:<10} (no data)")
            return
        lidar = cfg.get("lidar_ms")
        cam = cfg.get("camera_ms")
        comb = cfg.get("combined_ms")
        lidar_s = f"{lidar:.3f}" if lidar is not None else "n/a"
        cam_s = f"{cam:.3f}" if cam is not None else "n/a"
        comb_s = f"{comb:.3f}" if comb is not None else "n/a"
        print(
            f"  {slot:<10} ctor={cfg['bvh_constructor']:<8} leaf={str(cfg['bvh_leaf_size']):<8}  "
            f"lidar={lidar_s} ms  camera={cam_s} ms  sum={comb_s} ms"
        )

    fmt("lidar", rec.get("lidar"))
    fmt("camera", rec.get("camera"))
    fmt("combined", rec.get("combined"))
    print(
        "  Tip: lidar-only sensing → lidar winner; RGB-D obs → camera or combined winner."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="G1 MJCF path (default: SIMPLE_RAYCASTER_G1_XML or assets/unitree_g1/...)",
    )
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
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
        help="wp.Mesh bvh_constructor: sah, median, lbvh, cubql (default: Warp device default)",
    )
    parser.add_argument(
        "--bvh-leaf",
        type=int,
        default=None,
        help="wp.Mesh bvh_leaf_size (default: Warp per-constructor default; typically 1)",
    )
    parser.add_argument(
        "--sweep-bvh",
        action="store_true",
        help=(
            "Sweep BVH constructor/leaf_size; time MultiMeshRaycaster closest-hit "
            "and RaycastCamera; print a recommendation"
        ),
    )
    parser.add_argument(
        "--sweep-bvh-mode",
        type=str,
        default="staged",
        choices=("staged", "constructors", "leaf", "full"),
        help=(
            "staged=ctors@default leaf then leaves@best ctor (default); "
            "constructors|leaf|full=explicit grid"
        ),
    )
    parser.add_argument(
        "--bvh-constructors",
        type=str,
        default=None,
        help="Comma list for sweeps (e.g. default,lbvh,sah,median,cubql)",
    )
    parser.add_argument(
        "--bvh-leaf-sizes",
        type=str,
        default=None,
        help="Comma list for sweeps (e.g. default,1,2,4,8,16)",
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

    xml_path = str(resolve_g1_xml(args.xml))
    wp.init()
    bvh_support = probe_bvh_support(device)
    raycaster, g1_pos, g1_quat, stats = load_scene(
        xml_path,
        device,
        bvh_constructor=args.bvh,
        bvh_leaf_size=args.bvh_leaf,
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
    print(
        f"bvh={stats['bvh_constructor']}  leaf={stats['bvh_leaf_size']}  "
        f"merge_static={stats['merge_static']}  "
        f"robot_first={stats['robot_first']}  cube_first={stats['cube_first']}  "
        f"simplify={stats['simplify_factor']}  block_dim={raycaster.block_dim}"
    )
    print(
        f"Warp BVH support: constructors={[ _ctor_label(c) for c in bvh_support['constructors'] ]}  "
        f"leaf_sizes={[ _leaf_label(x) for x in bvh_support['leaf_sizes'] ]}"
    )
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
        cam.bvh_constructor = args.bvh
        cam.bvh_leaf_size = args.bvh_leaf
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

    bvh_sweep_rows: list[dict] = []
    bvh_recommendation: dict | None = None
    if args.sweep_bvh:
        ctors = parse_ctor_list(args.bvh_constructors, bvh_support["constructors"])
        leaves = parse_leaf_list(args.bvh_leaf_sizes, bvh_support["leaf_sizes"])
        if args.sweep_bvh_mode == "leaf" and args.bvh is not None:
            ctors = [args.bvh]
        if args.sweep_bvh_mode == "constructors" and args.bvh_leaf is not None:
            leaves = [args.bvh_leaf]
        print("\n" + "=" * 70)
        print(
            f"BVH SWEEP mode={args.sweep_bvh_mode}  "
            f"ctors={[ _ctor_label(c) for c in ctors ]}  "
            f"leaves={[ _leaf_label(x) for x in leaves ]}"
        )
        print("=" * 70)
        bvh_sweep_rows, bvh_recommendation = run_bvh_sweep(
            xml_path=xml_path,
            device=device,
            torch_device=torch_device,
            n=args.n,
            width=args.width,
            height=args.height,
            fov=args.fov,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            warmup=args.warmup,
            iters=args.iters,
            ray_starts=ray_starts,
            ray_dirs=ray_dirs,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            merge_static=args.merge_static,
            robot_first=args.robot_first,
            cube_first=args.cube_first,
            simplify_factor=args.simplify,
            block_dim=args.block_dim,
            mode=args.sweep_bvh_mode,
            constructors=ctors,
            leaf_sizes=leaves,
            skip_camera=args.skip_camera,
            n_rays=n_rays,
        )
        for row in bvh_sweep_rows:
            if row.get("path") in ("lidar", "camera") and row.get("ok"):
                results.append(
                    {
                        "name": row["name"],
                        "ms_per_iter": row["ms_per_iter"],
                        "rays_per_s": row["rays_per_s"],
                        "mem_peak": row["mem_peak"],
                        "n_queries": row["n_queries"],
                        "bvh_constructor": row["bvh_constructor"],
                        "bvh_leaf_size": row["bvh_leaf_size"],
                    }
                )
        _print_bvh_recommendation(bvh_recommendation)

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
        "bvh_support": {
            "constructors": [_ctor_label(c) for c in bvh_support["constructors"]],
            "leaf_sizes": [_leaf_label(x) for x in bvh_support["leaf_sizes"]],
            "has_constructor": bvh_support["has_constructor"],
            "has_leaf_size": bvh_support["has_leaf_size"],
        },
        "scene": {
            k: stats[k]
            for k in (
                "n_meshes",
                "n_points",
                "n_faces",
                "n_g1_bodies",
                "bvh_constructor",
                "bvh_leaf_size",
                "merge_static",
                "robot_first",
                "cube_first",
                "simplify_factor",
            )
        },
        "parity": parity_rows,
        "results": results,
        "bvh_sweep": bvh_sweep_rows,
        "bvh_recommendation": bvh_recommendation,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(blob, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
