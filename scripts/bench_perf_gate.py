"""Routine PBR performance gate (synthetic scene — no G1 MJCF required).

Canonical long-term check: fixed 192×144, N=256, ``quality="fast"`` (FXAA +
sun shadow rays). Emits a versioned JSON blob and can fail on regression vs a
baseline from the same GPU class.

    # Measure + write results
    .venv/bin/python scripts/bench_perf_gate.py --json-out bench/results/latest.json

    # Compare against a saved baseline (exit 1 on >15% slowdown)
    .venv/bin/python scripts/bench_perf_gate.py \\
        --json-out bench/results/latest.json \\
        --compare bench/baselines/rtx4090.json

    # Faster agent loop
    .venv/bin/python scripts/bench_perf_gate.py --quick

Full G1 / pretty / filter probes stay in ``bench_pbr_camera.py`` and
``bench_tiled_filters.py`` — use those for deep dives, not daily gates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp

from simple_raycaster import MultiMeshRaycaster, RaycastCamera, RaycastPBRCamera
from simple_raycaster.pbr.hdri import EnvironmentHDRI, default_hdri_path

SCHEMA_VERSION = 1
GATE_NAME = "pbr_perf_gate"
DEFAULT_N = 256
DEFAULT_W, DEFAULT_H = 192, 144


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _look_at_opencv(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    z_cam = target - eye
    z_cam = z_cam / np.linalg.norm(z_cam)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_cam = np.cross(z_cam, world_up)
    if np.linalg.norm(x_cam) < 1e-6:
        x_cam = np.cross(z_cam, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    rot = np.stack([x_cam, y_cam, z_cam], axis=1)
    xyzw = Rotation.from_matrix(rot).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def _build_scene(device: str):
    ground = trimesh.creation.box(extents=[4.0, 4.0, 0.05])
    ground.apply_translation([0.0, 0.0, -0.025])
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.35)
    sphere.apply_translation([0.0, 0.0, 0.35])
    box = trimesh.creation.box(extents=[0.45, 0.45, 0.45])
    box.apply_translation([0.7, -0.2, 0.225])
    meshes = [ground, sphere, box]
    names = ["ground", "sphere", "box"]
    albedos = np.array(
        [[0.45, 0.45, 0.42], [0.75, 0.25, 0.18], [0.22, 0.45, 0.75]],
        dtype=np.float32,
    )
    raycaster = MultiMeshRaycaster(meshes, device=device)
    return raycaster, names, albedos


def _time_ms(fn, *, warmup: int, iters: int, device: torch.device) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    wp.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
        wp.synchronize()
    end.record()
    torch.cuda.synchronize(device)
    return {
        "ms_per_iter": float(start.elapsed_time(end) / max(iters, 1)),
        "mem_peak": int(torch.cuda.max_memory_allocated(device)),
    }


def _compare(current: dict, baseline: dict, *, fail_pct: float) -> list[str]:
    """Return human-readable failure lines (empty = ok)."""
    failures: list[str] = []
    if baseline.get("schema_version") != SCHEMA_VERSION:
        failures.append(
            f"baseline schema {baseline.get('schema_version')!r} != {SCHEMA_VERSION}"
        )
        return failures
    if baseline.get("gate") != GATE_NAME:
        failures.append(f"baseline gate {baseline.get('gate')!r} != {GATE_NAME!r}")
        return failures
    # Soft warnings for env drift (do not fail).
    for key in ("gpu", "width", "height", "n"):
        if baseline.get(key) != current.get(key):
            print(f"warning: {key} baseline={baseline.get(key)!r} current={current.get(key)!r}")

    base_paths = {p["name"]: p for p in baseline.get("paths", [])}
    for row in current.get("paths", []):
        name = row["name"]
        ref = base_paths.get(name)
        if ref is None:
            failures.append(f"missing baseline path {name!r}")
            continue
        cur_ms = float(row["ms_per_iter"])
        ref_ms = float(ref["ms_per_iter"])
        if ref_ms <= 0:
            failures.append(f"baseline {name} has non-positive ms_per_iter")
            continue
        ratio = cur_ms / ref_ms
        limit = 1.0 + fail_pct / 100.0
        if ratio > limit:
            failures.append(
                f"{name}: {cur_ms:.3f} ms is {ratio:.2f}× baseline {ref_ms:.3f} ms "
                f"(limit {limit:.2f}× / +{fail_pct:.0f}%)"
            )
        # Portable signal: vs-Lambert should not explode even if clocks differ.
        if name == "pbr_fast" and "vs_lambert" in row and "vs_lambert" in ref:
            cur_v = float(row["vs_lambert"])
            ref_v = float(ref["vs_lambert"])
            if ref_v > 0 and cur_v / ref_v > limit:
                failures.append(
                    f"pbr_fast vs_lambert: {cur_v:.2f}× is {cur_v / ref_v:.2f}× "
                    f"baseline {ref_v:.2f}× (limit {limit:.2f}×)"
                )
    return failures


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--height", type=int, default=DEFAULT_H)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer warmup/iters for agent loops (not for baselines)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument(
        "--compare",
        type=str,
        default="",
        help="Baseline JSON from a prior gate run on the same GPU class",
    )
    parser.add_argument(
        "--fail-pct",
        type=float,
        default=15.0,
        help="Fail if a path is slower than baseline by more than this percent",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune-no-cudagraphs",
        help="Pass empty string to disable torch.compile",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_perf_gate requires CUDA")

    if args.quick:
        args.warmup = min(args.warmup, 3)
        args.iters = min(args.iters, 10)

    compile_mode = args.compile_mode.strip() or None
    device = args.device
    td = torch.device(device)
    n, w, h = int(args.n), int(args.width), int(args.height)
    n_pix = w * h

    wp.init()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    raycaster, names, albedos = _build_scene(device)
    env = EnvironmentHDRI.from_path(default_hdri_path(), device=td)

    lambert = RaycastCamera(
        w, h, fov_y_deg=50.0, near=0.05, far=20.0, convention="opencv", device=device
    )
    lambert.bind_meshes(raycaster)

    pbr_gbuf = RaycastPBRCamera(
        w,
        h,
        fov_y_deg=50.0,
        near=0.05,
        far=20.0,
        convention="opencv",
        device=device,
        hdri=env,
        quality="fast",
        fxaa_enabled=False,
        shadow_rays=False,
        compile_mode=None,
    )
    pbr_gbuf.bind_meshes(raycaster, names=names, albedos=albedos)

    pbr_fast = RaycastPBRCamera(
        w,
        h,
        fov_y_deg=50.0,
        near=0.05,
        far=20.0,
        convention="opencv",
        device=device,
        hdri=env,
        quality="fast",
        compile_mode=compile_mode,
        fxaa_impl="torch",
    )
    pbr_fast.bind_meshes(raycaster, names=names, albedos=albedos)

    eye = np.array([1.8, -1.6, 1.1], dtype=np.float64)
    quat = _look_at_opencv(eye, np.array([0.2, 0.0, 0.3], dtype=np.float64))
    cam_pos = torch.as_tensor(eye, device=td, dtype=torch.float32).view(1, 3).expand(n, 3).contiguous()
    cam_quat = torch.as_tensor(quat, device=td, dtype=torch.float32).view(1, 4).expand(n, 4).contiguous()
    m = raycaster.n_meshes
    mesh_pos = torch.zeros(n, m, 3, device=td, dtype=torch.float32)
    mesh_quat = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=td)
        .view(1, 1, 4)
        .expand(n, m, 4)
        .contiguous()
    )
    kw = dict(mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat)

    # Sanity + compile warmup (excluded from timed region via _time_ms warmup).
    rgb_l, depth_l, mask_l = lambert.render(cam_pos, cam_quat, **kw)
    rgb_p, depth_p, mask_p = pbr_fast.render(cam_pos, cam_quat, **kw)
    depth_err = float((depth_l - depth_p).abs().max())
    mask_frac = float(mask_p.float().mean())
    if mask_frac < 0.2:
        raise SystemExit(f"sanity failed: mask_frac={mask_frac:.3f} < 0.2")
    if depth_err > 1e-3:
        print(f"warning: Lambert vs PBR depth max|Δ|={depth_err:.3e}")

    print("=" * 72)
    print(f"{GATE_NAME}  N={n}  {w}x{h}  compile={compile_mode!r}")
    print(f"gpu {torch.cuda.get_device_name(0)}  warp {wp.__version__}  torch {torch.__version__}")
    print("=" * 72)

    paths: list[dict] = []

    def record(name: str, cam, fn) -> None:
        # Drop workspaces between paths for fair-ish peak mem.
        work = getattr(cam, "_work", None)
        if work is not None:
            work.clear()
        geom = getattr(cam, "geom", None)
        if geom is not None and getattr(geom, "_work", None) is not None:
            geom._work.clear()
        torch.cuda.empty_cache()
        stats = _time_ms(fn, warmup=args.warmup, iters=args.iters, device=td)
        row = {
            "name": name,
            "ms_per_iter": stats["ms_per_iter"],
            "ms_per_cam": stats["ms_per_iter"] / n,
            "mpix_per_s": (n * n_pix) / (stats["ms_per_iter"] * 1e-3) / 1e6,
            "mem_peak": stats["mem_peak"],
        }
        paths.append(row)
        print(
            f"  {name:<12} {row['ms_per_iter']:8.3f} ms/iter  "
            f"{row['ms_per_cam']:6.3f} ms/cam  peak={row['mem_peak'] / 1e6:.1f} MB"
        )

    t_wall0 = time.perf_counter()
    record("lambert", lambert, lambda: lambert.render(cam_pos, cam_quat, **kw))
    record(
        "pbr_gbuffer",
        pbr_gbuf,
        lambda: pbr_gbuf.render_gbuffer(cam_pos, cam_quat, **kw),
    )
    record("pbr_fast", pbr_fast, lambda: pbr_fast.render(cam_pos, cam_quat, **kw))
    wall_s = time.perf_counter() - t_wall0

    lambert_ms = next(p["ms_per_iter"] for p in paths if p["name"] == "lambert")
    for p in paths:
        p["vs_lambert"] = float(p["ms_per_iter"] / lambert_ms)

    fast = next(p for p in paths if p["name"] == "pbr_fast")
    print(
        f"\nHEADLINE N={n} pbr_fast: {fast['ms_per_iter']:.3f} ms/iter  "
        f"({fast['vs_lambert']:.2f}× Lambert)  wall={wall_s:.1f}s"
    )

    blob = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "git_sha": _git_sha(),
        "warp": wp.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "device": device,
        "n": n,
        "width": w,
        "height": h,
        "n_pix": n_pix,
        "warmup": args.warmup,
        "iters": args.iters,
        "compile_mode": compile_mode,
        "scene": "synthetic_ground_sphere_box",
        "n_meshes": int(m),
        "mask_frac": mask_frac,
        "depth_max_abs": depth_err,
        "paths": paths,
        "headline": {
            "name": "pbr_fast",
            "ms_per_iter": fast["ms_per_iter"],
            "vs_lambert": fast["vs_lambert"],
        },
    }

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"wrote {out}")

    if args.compare:
        base = json.loads(Path(args.compare).read_text())
        failures = _compare(blob, base, fail_pct=float(args.fail_pct))
        if failures:
            print("\nREGRESSION vs", args.compare)
            for line in failures:
                print("  FAIL:", line)
            raise SystemExit(1)
        print(f"\ncompare ok vs {args.compare} (fail_pct={args.fail_pct})")

    print("ok")


if __name__ == "__main__":
    main()
