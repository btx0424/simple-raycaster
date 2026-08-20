"""Probe ``torch.compile`` modes on the PBR Torch shade path (N=256 headline).

Compiles shade+FXAA on a frozen G-buffer (Warp stays out of the graph). Modes:
default / reduce-overhead / max-autotune / max-autotune-no-cudagraphs.

    .venv/bin/python scripts/probe_pbr_compile.py
    .venv/bin/python scripts/probe_pbr_compile.py --modes default,reduce-overhead
    .venv/bin/python scripts/probe_pbr_compile.py --json-out /tmp/probe_compile.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import warp as wp

from simple_raycaster import RaycastPBRCamera
from simple_raycaster.pbr.hdri import EnvironmentHDRI, default_hdri_path
from simple_raycaster.pbr.shade import aces_tonemap, fxaa, shade_pbr

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from bench_g1_camera import (  # noqa: E402
    DEFAULT_G1_XML,
    expand_poses,
    load_scene,
    sample_cameras,
)


DEFAULT_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


def _scene_names(stats: dict) -> list[str]:
    g1 = list(stats["g1_names"])
    n_static = int(stats["n_static"])
    static = ["ground", "cube"][:n_static]
    return static + g1


def _time_fn(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / iters * 1000.0


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=str, default=DEFAULT_G1_XML)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--max-dist", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help="Comma-separated torch.compile modes",
    )
    parser.add_argument(
        "--full-render",
        action="store_true",
        help="Also try compile on cam.render (likely fails / no win; Warp+Torch)",
    )
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch_device = torch.device(args.device)
    wp.init()

    raycaster, g1_pos, g1_quat, stats = load_scene(args.xml, args.device)
    names = _scene_names(stats)
    env = EnvironmentHDRI.from_path(default_hdri_path(), device=torch_device)

    mesh_pos, mesh_quat = expand_poses(args.n, g1_pos, g1_quat, stats["n_static"])
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
    kw = dict(mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat)

    cam = RaycastPBRCamera(
        args.width,
        args.height,
        fov_y_deg=args.fov,
        near=args.min_dist,
        far=args.max_dist,
        convention="opencv",
        depth_mode="planar",
        device=args.device,
        hdri=env,
        quality="fast",
        fxaa_enabled=True,
        compile_mode=None,
    )
    cam.bind_meshes(raycaster, names=names)

    print("=" * 72)
    print(
        f"torch.compile probe  N={args.n} {args.width}x{args.height}  "
        f"torch {torch.__version__}  gpu={torch.cuda.get_device_name(0)}"
    )
    print("=" * 72)

    gb = cam.render_gbuffer(cam_pos, cam_quat, **kw)
    albedo = gb["albedo"].contiguous()
    normal = gb["normal"].contiguous()
    view = gb["view"].contiguous()
    rough = gb["roughness"].contiguous()
    metal = gb["metallic"].contiguous()
    mask = gb["mask"].contiguous()
    sun = cam._sun_dir_t
    vis = torch.ones_like(gb["depth"])
    sun_color = cam._sun_color_t
    exposure = float(cam.exposure)
    sun_intensity = float(cam.sun_intensity)

    def shade_once() -> torch.Tensor:
        hdr = shade_pbr(
            albedo,
            normal,
            view,
            rough,
            metal,
            cam.env,
            sun_dir_w=sun,
            sun_intensity=sun_intensity,
            sun_color=sun_color,
            sun_visibility=vis,
        )
        bg = cam.env.sample(-view, roughness=0.0)
        hdr = torch.where(mask.unsqueeze(-1), hdr, bg)
        rgb = aces_tonemap(hdr, exposure=exposure)
        return fxaa(rgb)

    def shade_nofxaa() -> torch.Tensor:
        hdr = shade_pbr(
            albedo,
            normal,
            view,
            rough,
            metal,
            cam.env,
            sun_dir_w=sun,
            sun_intensity=sun_intensity,
            sun_color=sun_color,
            sun_visibility=vis,
        )
        bg = cam.env.sample(-view, roughness=0.0)
        hdr = torch.where(mask.unsqueeze(-1), hdr, bg)
        return aces_tonemap(hdr, exposure=exposure)

    # Eager baselines.
    eager_full = _time_fn(
        lambda: cam.render(cam_pos, cam_quat, **kw),
        warmup=args.warmup,
        iters=args.iters,
        device=torch_device,
    )
    # sync Warp for fair full-path; shade-only is pure Torch.
    wp.synchronize()
    eager_shade = _time_fn(shade_once, warmup=args.warmup, iters=args.iters, device=torch_device)
    eager_nofxaa = _time_fn(
        shade_nofxaa, warmup=args.warmup, iters=args.iters, device=torch_device
    )
    print(f"  eager full pbr_fast:     {eager_full:.3f} ms/iter")
    print(f"  eager shade+fxaa:        {eager_shade:.3f} ms/iter")
    print(f"  eager shade (no fxaa):   {eager_nofxaa:.3f} ms/iter")

    results: dict = {
        "n": args.n,
        "width": args.width,
        "height": args.height,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "eager_full_ms": eager_full,
        "eager_shade_ms": eager_shade,
        "eager_nofxaa_ms": eager_nofxaa,
        "modes": {},
    }

    ref = shade_once()
    for mode in modes:
        print(f"\n--- mode={mode!r} ---")
        entry: dict = {"ok": False, "error": None}
        try:
            compiled = torch.compile(shade_once, mode=mode)
            # First calls trigger compile / CUDA graph capture.
            t_compile0 = time.perf_counter()
            for _ in range(max(3, args.warmup)):
                out = compiled()
            torch.cuda.synchronize(torch_device)
            compile_s = time.perf_counter() - t_compile0
            mae = float((out - ref).abs().mean())
            ms = _time_fn(compiled, warmup=args.warmup, iters=args.iters, device=torch_device)
            entry.update(
                {
                    "ok": True,
                    "ms": ms,
                    "speedup_vs_eager": eager_shade / ms if ms > 1e-9 else None,
                    "compile_warmup_s": compile_s,
                    "mae_vs_eager": mae,
                }
            )
            print(
                f"  shade+fxaa: {ms:.3f} ms/iter  "
                f"({eager_shade / ms:.2f}x)  compile+warmup {compile_s:.1f}s  mae={mae:.3e}"
            )

            # Also compile no-FXAA path (fewer dynamo issues sometimes).
            compiled_nf = torch.compile(shade_nofxaa, mode=mode)
            t0 = time.perf_counter()
            for _ in range(max(3, args.warmup)):
                compiled_nf()
            torch.cuda.synchronize(torch_device)
            compile_nf_s = time.perf_counter() - t0
            ms_nf = _time_fn(
                compiled_nf, warmup=args.warmup, iters=args.iters, device=torch_device
            )
            entry["nofxaa_ms"] = ms_nf
            entry["nofxaa_speedup"] = eager_nofxaa / ms_nf if ms_nf > 1e-9 else None
            entry["nofxaa_compile_warmup_s"] = compile_nf_s
            print(
                f"  shade nofxaa: {ms_nf:.3f} ms/iter  "
                f"({eager_nofxaa / ms_nf:.2f}x)  compile+warmup {compile_nf_s:.1f}s"
            )
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {entry['error']}")
            traceback.print_exc()
        results["modes"][mode] = entry
        # Drop compiled modules between modes to limit memory growth.
        torch.compiler.reset()
        torch.cuda.empty_cache()

    if args.full_render:
        print("\n--- compile full cam.render (experimental) ---")
        try:
            # Likely unsupported (Warp launches inside).
            compiled_render = torch.compile(
                lambda: cam.render(cam_pos, cam_quat, **kw), mode="default"
            )
            for _ in range(3):
                compiled_render()
            torch.cuda.synchronize(torch_device)
            ms = _time_fn(
                compiled_render, warmup=args.warmup, iters=args.iters, device=torch_device
            )
            results["full_render_compile"] = {
                "ok": True,
                "ms": ms,
                "speedup": eager_full / ms,
            }
            print(f"  full render compiled: {ms:.3f} ms ({eager_full / ms:.2f}x)")
        except Exception as exc:
            results["full_render_compile"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  full render FAILED: {type(exc).__name__}: {exc}")
        torch.compiler.reset()

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  eager shade+fxaa: {eager_shade:.3f} ms")
    best = None
    for mode, entry in results["modes"].items():
        if entry.get("ok") and entry.get("ms") is not None:
            print(
                f"  {mode:<28} {entry['ms']:7.3f} ms  "
                f"{entry['speedup_vs_eager']:.2f}x"
            )
            if best is None or entry["ms"] < best[1]:
                best = (mode, entry["ms"], entry["speedup_vs_eager"])
    if best:
        print(f"  best: {best[0]}  {best[1]:.3f} ms  ({best[2]:.2f}x)")
    else:
        print("  no successful compile modes")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
