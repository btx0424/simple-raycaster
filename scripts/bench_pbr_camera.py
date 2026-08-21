"""Sweep ``RaycastPBRCamera`` vs Lambert ``RaycastCamera`` over camera count N.

G1 scene matches ``scripts/bench_g1_camera.py`` (ground + cube + Inspire G1).
Headline batch for RL is **N=256**.

    .venv/bin/python scripts/bench_pbr_camera.py
    .venv/bin/python scripts/bench_pbr_camera.py --n 64,256,512
    .venv/bin/python scripts/bench_pbr_camera.py --pretty --shadow-map
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

from simple_raycaster import RaycastCamera, RaycastPBRCamera
from simple_raycaster.pbr.hdri import EnvironmentHDRI, default_hdri_path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from bench_g1_camera import (  # noqa: E402
    resolve_g1_xml,
    benchmark_fn,
    expand_poses,
    format_memory,
    load_scene,
    sample_cameras,
)


def _parse_n_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(n < 1 for n in values):
        raise argparse.ArgumentTypeError(f"expected positive ints, got {raw!r}")
    return values


def _scene_names(stats: dict) -> list[str]:
    g1 = list(stats["g1_names"])
    n_static = int(stats["n_static"])
    if stats.get("merge_static"):
        static = ["static"] * n_static
    else:
        static = ["ground", "cube"][:n_static]
    if stats.get("robot_first"):
        return g1 + static
    return static + g1


def _make_lambert(args, device: str) -> RaycastCamera:
    return RaycastCamera(
        args.width,
        args.height,
        fov_y_deg=args.fov,
        near=args.min_dist,
        far=args.max_dist,
        convention="opencv",
        depth_mode="planar",
        device=device,
    )


def _make_pbr(args, device: str, env: EnvironmentHDRI, **overrides) -> RaycastPBRCamera:
    kw = dict(
        width=args.width,
        height=args.height,
        fov_y_deg=args.fov,
        near=args.min_dist,
        far=args.max_dist,
        convention="opencv",
        depth_mode="planar",
        device=device,
        hdri=env,
        quality="fast",
    )
    kw.update(overrides)
    return RaycastPBRCamera(**kw)


def _clear_work(cam) -> None:
    work = getattr(cam, "_work", None)
    if work is not None:
        work.clear()
    geom = getattr(cam, "geom", None)
    if geom is not None and getattr(geom, "_work", None) is not None:
        geom._work.clear()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=None, help="G1 MJCF (or SIMPLE_RAYCASTER_G1_XML)")
    parser.add_argument(
        "--n",
        type=_parse_n_list,
        default=_parse_n_list("256"),
        help="Camera counts to time (default: 256, the RL headline batch)",
    )
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--max-dist", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Also time quality=pretty (SSAO + 128² shadow map)",
    )
    parser.add_argument(
        "--shadow-map",
        action="store_true",
        help="Also time pretty with shadow_map_size=256 (pretty already uses 128²)",
    )
    parser.add_argument(
        "--no-shadow-rays",
        action="store_true",
        help="Also time fast with shadow_rays=False (default fast now includes sray)",
    )
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    torch_device = torch.device(device)
    n_pix = args.width * args.height

    wp.init()
    raycaster, g1_pos, g1_quat, stats = load_scene(str(resolve_g1_xml(args.xml)), device)
    n_static = stats["n_static"]
    names = _scene_names(stats)

    print("=" * 78)
    print("PBR CAMERA N-SWEEP  (headline N=256 for parallel RL)")
    print("=" * 78)
    print(f"warp {wp.__version__}  torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}")
    print(f"xml  {stats['g1_xml']}")
    print(
        f"meshes {stats['n_meshes']} (static {n_static} + g1 {stats['n_g1_bodies']})  "
        f"verts {stats['n_points']}  faces {stats['n_faces']}"
    )
    print(
        f"res {args.width}x{args.height} ({n_pix} pix)  fov={args.fov}  "
        f"max_dist={args.max_dist}  N={args.n}"
    )
    print(f"warmup={args.warmup}  iters={args.iters}")
    print(raycaster)

    print("\nLoading HDRI + binding cameras...")
    env = EnvironmentHDRI.from_path(default_hdri_path(), device=torch_device)
    lambert = _make_lambert(args, device)
    lambert.bind_meshes(raycaster)
    pbr_fast = _make_pbr(args, device, env, quality="fast")
    pbr_fast.bind_meshes(raycaster, names=names)
    pbr_gbuf = _make_pbr(
        args, device, env, quality="fast", fxaa_enabled=False, shadow_rays=False
    )
    pbr_gbuf.bind_meshes(raycaster, names=names)
    pbr_nofxaa = _make_pbr(args, device, env, quality="fast", fxaa_enabled=False)
    pbr_nofxaa.bind_meshes(raycaster, names=names)
    pbr_pretty = None
    if args.pretty or args.shadow_map:
        pbr_pretty = _make_pbr(args, device, env, quality="pretty")
        pbr_pretty.bind_meshes(raycaster, names=names)
    pbr_smap = None
    pbr_nosray = None
    if args.shadow_map:
        pbr_smap = _make_pbr(
            args, device, env, quality="pretty", shadow_map_enabled=True, shadow_map_size=256
        )
        pbr_smap.bind_meshes(raycaster, names=names)
    if args.no_shadow_rays:
        pbr_nosray = _make_pbr(
            args, device, env, quality="fast", fxaa_enabled=False, shadow_rays=False
        )
        pbr_nosray.bind_meshes(raycaster, names=names)

    n0 = min(args.n)
    mesh_pos0, mesh_quat0 = expand_poses(n0, g1_pos, g1_quat, n_static)
    cam_pos0_np, cam_quat0_np = sample_cameras(
        n0, seed=args.seed, radius_min=1.5, radius_max=3.5, elev_min_deg=15.0, elev_max_deg=60.0
    )
    cam_pos0 = torch.from_numpy(cam_pos0_np).to(device=torch_device)
    cam_quat0 = torch.from_numpy(cam_quat0_np).to(device=torch_device)
    rgb_l, depth_l, mask_l = lambert.render(
        cam_pos0, cam_quat0, mesh_pos_w=mesh_pos0, mesh_quat_w=mesh_quat0
    )
    rgb_p, depth_p, mask_p = pbr_fast.render(
        cam_pos0, cam_quat0, mesh_pos_w=mesh_pos0, mesh_quat_w=mesh_quat0
    )
    depth_err = float((depth_l - depth_p).abs().max())
    mask_frac = float(mask_p.float().mean())
    print(
        f"  warmup N={n0}: lambert rgb_mean={float(rgb_l.mean()):.3f}  "
        f"pbr_fast rgb_mean={float(rgb_p.mean()):.3f}  mask_frac={mask_frac:.3f}  "
        f"depth max|Δ|={depth_err:.3e}"
    )
    if mask_frac < 0.3:
        raise SystemExit(f"sanity failed: mask_frac={mask_frac:.3f} < 0.3")
    if depth_err > 1e-4:
        print(f"  WARNING: Lambert vs PBR depth max|Δ|={depth_err:.3e} > 1e-4")
    del rgb_l, depth_l, mask_l, rgb_p, depth_p, mask_p, mesh_pos0, mesh_quat0, cam_pos0, cam_quat0
    for cam in (lambert, pbr_fast, pbr_nofxaa, pbr_gbuf, pbr_pretty, pbr_smap, pbr_nosray):
        if cam is not None:
            _clear_work(cam)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(torch_device)

    results: list[dict] = []

    def time_path(n: int, name: str, cam, fn) -> None:
        _clear_work(cam)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch_device)
        print(f"\n  {name} N={n}...")
        timing = benchmark_fn(fn, args.warmup, args.iters, torch_device)
        ms = timing["ms_per_iter"]
        pix_per_s = (n * n_pix) / (ms / 1000.0)
        cams_per_s = n / (ms / 1000.0)
        row = {
            "n": n,
            "name": name,
            "ms_per_iter": float(ms),
            "ms_per_cam": float(ms / n),
            "mpix_per_s": float(pix_per_s / 1e6),
            "cams_per_s": float(cams_per_s),
            "n_pixels": int(n * n_pix),
            **{k: timing[k] for k in ("mem_before", "mem_after", "mem_peak", "mem_delta")},
        }
        results.append(row)
        print(
            f"    {ms:.3f} ms/iter  {ms / n:.3f} ms/cam  {pix_per_s / 1e6:.2f} Mpix/s  "
            f"{cams_per_s:.1f} cam/s  peak {format_memory(timing['mem_peak'])}"
        )

    for n in args.n:
        print("\n" + "-" * 78)
        print(f"N={n}  ({n * n_pix} pixels)")
        mesh_pos, mesh_quat = expand_poses(n, g1_pos, g1_quat, n_static)
        cam_pos_np, cam_quat_np = sample_cameras(
            n, seed=args.seed, radius_min=1.5, radius_max=3.5, elev_min_deg=15.0, elev_max_deg=60.0
        )
        cam_pos = torch.from_numpy(cam_pos_np).to(device=torch_device)
        cam_quat = torch.from_numpy(cam_quat_np).to(device=torch_device)
        kw = dict(mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat)

        time_path(n, "lambert", lambert, lambda: lambert.render(cam_pos, cam_quat, **kw))
        time_path(
            n, "pbr_gbuffer", pbr_gbuf, lambda: pbr_gbuf.render_gbuffer(cam_pos, cam_quat, **kw)
        )
        time_path(n, "pbr_nofxaa", pbr_nofxaa, lambda: pbr_nofxaa.render(cam_pos, cam_quat, **kw))
        time_path(n, "pbr_fast", pbr_fast, lambda: pbr_fast.render(cam_pos, cam_quat, **kw))
        if pbr_pretty is not None:
            time_path(
                n, "pbr_pretty", pbr_pretty, lambda: pbr_pretty.render(cam_pos, cam_quat, **kw)
            )
        if pbr_smap is not None:
            time_path(n, "pbr_smap", pbr_smap, lambda: pbr_smap.render(cam_pos, cam_quat, **kw))
        if pbr_nosray is not None:
            time_path(
                n, "pbr_nosray", pbr_nosray, lambda: pbr_nosray.render(cam_pos, cam_quat, **kw)
            )

        del mesh_pos, mesh_quat, cam_pos, cam_quat
        torch.cuda.empty_cache()

    lambert_ms = {row["n"]: row["ms_per_iter"] for row in results if row["name"] == "lambert"}

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    header = (
        f"{'N':>5} {'path':<14} {'ms/iter':>10} {'ms/cam':>8} "
        f"{'Mpix/s':>8} {'cam/s':>8} {'vs L':>7} {'peak':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        vs = row["ms_per_iter"] / lambert_ms[row["n"]]
        print(
            f"{row['n']:5d} {row['name']:<14} {row['ms_per_iter']:10.3f} {row['ms_per_cam']:8.3f} "
            f"{row['mpix_per_s']:8.2f} {row['cams_per_s']:8.1f} {vs:6.2f}x "
            f"{format_memory(row['mem_peak']):>12}"
        )

    if 256 in lambert_ms:
        fast_rows = [r for r in results if r["n"] == 256 and r["name"] == "pbr_fast"]
        if fast_rows:
            r = fast_rows[0]
            print(
                f"\nHEADLINE N=256 pbr_fast: {r['ms_per_iter']:.3f} ms/iter  "
                f"{r['ms_per_cam']:.3f} ms/cam  {r['cams_per_s']:.0f} cam/s  "
                f"({r['ms_per_iter'] / lambert_ms[256]:.2f}x Lambert)"
            )

    blob = {
        "warp": wp.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "width": args.width,
        "height": args.height,
        "fov": args.fov,
        "n_list": args.n,
        "n_pix": n_pix,
        "warmup": args.warmup,
        "iters": args.iters,
        "depth_max_abs": depth_err,
        "mask_frac": mask_frac,
        "scene": {
            k: stats[k]
            for k in ("n_meshes", "n_points", "n_faces", "n_g1_bodies", "bvh_constructor")
        },
        "results": results,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(blob, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
