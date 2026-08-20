"""Benchmark Warp tiled FXAA/SSAO vs Torch reference @ N=256.

    .venv/bin/python scripts/bench_tiled_filters.py
    .venv/bin/python scripts/bench_tiled_filters.py --json-out /tmp/bench_tiled.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from simple_raycaster import RaycastPBRCamera
from simple_raycaster.pbr.hdri import EnvironmentHDRI, default_hdri_path
from simple_raycaster.pbr.shade import fxaa, ssao
from simple_raycaster.pbr.tiled_filters import fxaa_tiled, ssao_tiled

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from bench_g1_camera import (  # noqa: E402
    DEFAULT_G1_XML,
    expand_poses,
    load_scene,
    sample_cameras,
)


def _scene_names(stats: dict) -> list[str]:
    return ["ground", "cube"][: int(stats["n_static"])] + list(stats["g1_names"])


def _time(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    wp.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=str, default=DEFAULT_G1_XML)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    td = torch.device(device)
    wp.init()

    raycaster, g1_pos, g1_quat, stats = load_scene(args.xml, device)
    names = _scene_names(stats)
    albedos = stats["albedos"]
    env = EnvironmentHDRI.from_path(default_hdri_path(), device=td)
    mesh_pos, mesh_quat = expand_poses(args.n, g1_pos, g1_quat, stats["n_static"])
    cam_pos_np, cam_quat_np = sample_cameras(
        args.n,
        seed=args.seed,
        radius_min=1.5,
        radius_max=3.5,
        elev_min_deg=15.0,
        elev_max_deg=60.0,
    )
    cam_pos = torch.from_numpy(cam_pos_np).to(td)
    cam_quat = torch.from_numpy(cam_quat_np).to(td)
    kw = dict(mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat)

    print("=" * 72)
    print(
        f"tiled filters bench  N={args.n} {args.width}x{args.height}  "
        f"warp {wp.__version__}  torch {torch.__version__}"
    )
    print("=" * 72)

    # --- Isolated FXAA / SSAO on frozen buffers ---
    cam0 = RaycastPBRCamera(
        args.width,
        args.height,
        device=device,
        hdri=env,
        quality="fast",
        fxaa_enabled=False,
        fxaa_impl="torch",
    )
    cam0.bind_meshes(raycaster, names=names, albedos=albedos)
    rgb0, depth0, mask0 = cam0.render(cam_pos, cam_quat, **kw)
    # Fake LDR for FXAA
    ldr = rgb0.clamp(0, 1).contiguous()
    planar = depth0  # already planar for opencv cam
    fx, fy, cx, cy = cam0.intrinsics.resolved_k()

    # Warmup compile tiled kernels
    _ = fxaa_tiled(ldr[:1])
    _ = ssao_tiled(planar[:1], mask0[:1], fx=fx, fy=fy)

    ms_fxaa_t = _time(lambda: fxaa(ldr), warmup=args.warmup, iters=args.iters, device=td)
    ms_fxaa_w = _time(lambda: fxaa_tiled(ldr), warmup=args.warmup, iters=args.iters, device=td)
    ms_ssao_t = _time(
        lambda: ssao(planar, mask0, fx=fx, fy=fy, cx=cx, cy=cy),
        warmup=args.warmup,
        iters=args.iters,
        device=td,
    )
    ms_ssao_w = _time(
        lambda: ssao_tiled(planar, mask0, fx=fx, fy=fy, cx=cx, cy=cy),
        warmup=args.warmup,
        iters=args.iters,
        device=td,
    )
    mae_fxaa = float((fxaa_tiled(ldr) - fxaa(ldr)).abs().mean())
    mae_ssao = float(
        (ssao_tiled(planar, mask0, fx=fx, fy=fy) - ssao(planar, mask0, fx=fx, fy=fy, cx=cx, cy=cy))
        .abs()
        .mean()
    )
    print(f"  fxaa  torch {ms_fxaa_t:.3f} ms   tiled {ms_fxaa_w:.3f} ms  "
          f"({ms_fxaa_t/ms_fxaa_w:.2f}x)  mae={mae_fxaa:.3e}")
    print(f"  ssao  torch {ms_ssao_t:.3f} ms   tiled {ms_ssao_w:.3f} ms  "
          f"({ms_ssao_t/ms_ssao_w:.2f}x)  mae={mae_ssao:.3e}")

    # --- Full camera paths ---
    results = {
        "warp": wp.__version__,
        "torch": torch.__version__,
        "n": args.n,
        "fxaa_torch_ms": ms_fxaa_t,
        "fxaa_tiled_ms": ms_fxaa_w,
        "fxaa_speedup": ms_fxaa_t / ms_fxaa_w,
        "fxaa_mae": mae_fxaa,
        "ssao_torch_ms": ms_ssao_t,
        "ssao_tiled_ms": ms_ssao_w,
        "ssao_speedup": ms_ssao_t / ms_ssao_w,
        "ssao_mae": mae_ssao,
        "full": {},
    }

    variants = {
        "fast_torch_fxaa": dict(quality="fast", fxaa_enabled=True, fxaa_impl="torch"),
        "fast_tiled_fxaa": dict(quality="fast", fxaa_enabled=True, fxaa_impl="tiled"),
        "fast_nofxaa": dict(quality="fast", fxaa_enabled=False),
        "pretty_torch": dict(
            quality="pretty", fxaa_enabled=True, fxaa_impl="torch", ssao_impl="torch"
        ),
        "pretty_tiled": dict(
            quality="pretty", fxaa_enabled=True, fxaa_impl="tiled", ssao_impl="tiled"
        ),
    }
    for name, overrides in variants.items():
        cam = RaycastPBRCamera(
            args.width,
            args.height,
            device=device,
            hdri=env,
            **overrides,
        )
        cam.bind_meshes(raycaster, names=names, albedos=albedos)
        ms = _time(
            lambda c=cam: c.render(cam_pos, cam_quat, **kw),
            warmup=args.warmup,
            iters=args.iters,
            device=td,
        )
        results["full"][name] = ms
        print(f"  full {name:<18} {ms:.3f} ms/iter")

    print("\nVERDICT")
    print(
        f"  FXAA tiled {results['fxaa_speedup']:.2f}x vs torch; "
        f"SSAO tiled {results['ssao_speedup']:.2f}x vs torch"
    )
    print(
        f"  pbr_fast: torch_fxaa {results['full']['fast_torch_fxaa']:.2f} → "
        f"tiled {results['full']['fast_tiled_fxaa']:.2f} ms"
    )
    print(
        f"  pbr_pretty: torch {results['full']['pretty_torch']:.2f} → "
        f"tiled {results['full']['pretty_tiled']:.2f} ms"
    )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
