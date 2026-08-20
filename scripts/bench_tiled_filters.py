"""Benchmark Torch (eager + compile) vs Warp tiled FXAA/SSAO @ N=256.

Default resolution is 192×144 (script default across PBR benches).

    .venv/bin/python scripts/bench_tiled_filters.py
    .venv/bin/python scripts/bench_tiled_filters.py --compile-mode max-autotune-no-cudagraphs
    .venv/bin/python scripts/bench_tiled_filters.py --json-out /tmp/bench_tiled.json
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
from simple_raycaster.pbr.shade import aces_tonemap, fxaa, shade_pbr, ssao
from simple_raycaster.pbr.tiled_filters import fxaa_tiled, ssao_tiled

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from bench_g1_camera import (  # noqa: E402
    resolve_g1_xml,
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
    parser.add_argument("--xml", type=str, default=None, help="G1 MJCF (or SIMPLE_RAYCASTER_G1_XML)")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode for Torch FXAA/SSAO / shade+FXAA",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    td = torch.device(device)
    wp.init()

    raycaster, g1_pos, g1_quat, stats = load_scene(str(resolve_g1_xml(args.xml)), device)
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
        compile_mode=None,
    )
    cam0.bind_meshes(raycaster, names=names, albedos=albedos)
    rgb0, depth0, mask0 = cam0.render(cam_pos, cam_quat, **kw)
    ldr = rgb0.clamp(0, 1).contiguous()
    planar = depth0
    fx, fy, cx, cy = cam0.intrinsics.resolved_k()

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
    print(
        f"  fxaa  torch {ms_fxaa_t:.3f} ms   tiled {ms_fxaa_w:.3f} ms  "
        f"({ms_fxaa_t / ms_fxaa_w:.2f}x)  mae={mae_fxaa:.3e}"
    )
    print(
        f"  ssao  torch {ms_ssao_t:.3f} ms   tiled {ms_ssao_w:.3f} ms  "
        f"({ms_ssao_t / ms_ssao_w:.2f}x)  mae={mae_ssao:.3e}"
    )

    results: dict = {
        "warp": wp.__version__,
        "torch": torch.__version__,
        "n": args.n,
        "width": args.width,
        "height": args.height,
        "compile_mode": args.compile_mode,
        "fxaa_torch_ms": ms_fxaa_t,
        "fxaa_tiled_ms": ms_fxaa_w,
        "fxaa_speedup_tiled_vs_torch": ms_fxaa_t / ms_fxaa_w,
        "fxaa_mae": mae_fxaa,
        "ssao_torch_ms": ms_ssao_t,
        "ssao_tiled_ms": ms_ssao_w,
        "ssao_speedup_tiled_vs_torch": ms_ssao_t / ms_ssao_w,
        "ssao_mae": mae_ssao,
        "compile": {},
        "full": {},
        "shade": {},
    }

    # --- torch.compile on Torch FXAA / SSAO ---
    if not args.skip_compile:
        print(f"\n--- torch.compile mode={args.compile_mode!r} (filters) ---")
        try:

            def fxaa_fn() -> torch.Tensor:
                return fxaa(ldr)

            def ssao_fn() -> torch.Tensor:
                return ssao(planar, mask0, fx=fx, fy=fy, cx=cx, cy=cy)

            fxaa_c = torch.compile(fxaa_fn, mode=args.compile_mode)
            ssao_c = torch.compile(ssao_fn, mode=args.compile_mode)
            t0 = time.perf_counter()
            for _ in range(max(3, args.warmup)):
                fxaa_c()
                ssao_c()
            torch.cuda.synchronize(td)
            compile_s = time.perf_counter() - t0
            ms_fxaa_c = _time(fxaa_c, warmup=args.warmup, iters=args.iters, device=td)
            ms_ssao_c = _time(ssao_c, warmup=args.warmup, iters=args.iters, device=td)
            mae_fxaa_c = float((fxaa_c() - fxaa(ldr)).abs().mean())
            mae_ssao_c = float((ssao_c() - ssao_fn()).abs().mean())
            results["compile"].update(
                {
                    "ok": True,
                    "warmup_s": compile_s,
                    "fxaa_ms": ms_fxaa_c,
                    "ssao_ms": ms_ssao_c,
                    "fxaa_mae": mae_fxaa_c,
                    "ssao_mae": mae_ssao_c,
                    "fxaa_speedup_vs_eager": ms_fxaa_t / ms_fxaa_c,
                    "ssao_speedup_vs_eager": ms_ssao_t / ms_ssao_c,
                    "fxaa_speedup_vs_tiled": ms_fxaa_w / ms_fxaa_c,
                    "ssao_speedup_vs_tiled": ms_ssao_w / ms_ssao_c,
                }
            )
            print(
                f"  fxaa  compile {ms_fxaa_c:.3f} ms  "
                f"({ms_fxaa_t / ms_fxaa_c:.2f}x eager, {ms_fxaa_w / ms_fxaa_c:.2f}x vs tiled)  "
                f"mae={mae_fxaa_c:.3e}"
            )
            print(
                f"  ssao  compile {ms_ssao_c:.3f} ms  "
                f"({ms_ssao_t / ms_ssao_c:.2f}x eager, {ms_ssao_w / ms_ssao_c:.2f}x vs tiled)  "
                f"mae={mae_ssao_c:.3e}"
            )
            print(f"  compile+warmup {compile_s:.1f}s")
        except Exception as exc:
            results["compile"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  filter compile FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        torch.compiler.reset()
        torch.cuda.empty_cache()

    # --- Full camera paths ---
    print("\n--- full camera ---")
    variants = {
        "fast_torch_fxaa": dict(
            quality="fast", fxaa_enabled=True, fxaa_impl="torch", compile_mode=None
        ),
        "fast_tiled_fxaa": dict(
            quality="fast", fxaa_enabled=True, fxaa_impl="tiled", compile_mode=None
        ),
        "fast_nofxaa": dict(quality="fast", fxaa_enabled=False, compile_mode=None),
        "pretty_torch": dict(
            quality="pretty",
            fxaa_enabled=True,
            fxaa_impl="torch",
            ssao_impl="torch",
            compile_mode=None,
        ),
        "pretty_tiled": dict(
            quality="pretty",
            fxaa_enabled=True,
            fxaa_impl="tiled",
            ssao_impl="tiled",
            compile_mode=None,
        ),
        "pretty_tiled_fxaa_torch_ssao": dict(
            quality="pretty",
            fxaa_enabled=True,
            fxaa_impl="tiled",
            ssao_impl="torch",
            compile_mode=None,
        ),
        "pretty_compile_torch": dict(
            quality="pretty",
            fxaa_enabled=True,
            fxaa_impl="torch",
            ssao_impl="torch",
            compile_mode=args.compile_mode,
        ),
        "fast_compile_torch": dict(
            quality="fast",
            fxaa_enabled=True,
            fxaa_impl="torch",
            compile_mode=args.compile_mode,
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
        print(f"  full {name:<28} {ms:.3f} ms/iter")

    # --- Frozen G-buffer: shade + FXAA (eager / compile / tiled FXAA) ---
    print(f"\n--- shade+FXAA on frozen G-buffer (compile={args.compile_mode!r}) ---")
    cam_gb = RaycastPBRCamera(
        args.width,
        args.height,
        device=device,
        hdri=env,
        quality="fast",
        fxaa_enabled=False,
        compile_mode=None,
    )
    cam_gb.bind_meshes(raycaster, names=names, albedos=albedos)
    gb = cam_gb.render_gbuffer(cam_pos, cam_quat, **kw)
    albedo = gb["albedo"].contiguous()
    normal = gb["normal"].contiguous()
    view = gb["view"].contiguous()
    rough = gb["roughness"].contiguous()
    metal = gb["metallic"].contiguous()
    mask = gb["mask"].contiguous()
    sun = cam_gb._sun_dir_t
    vis = torch.ones_like(gb["depth"])
    sun_color = cam_gb._sun_color_t
    exposure = float(cam_gb.exposure)
    sun_intensity = float(cam_gb.sun_intensity)

    def _shade_hdr() -> torch.Tensor:
        hdr = shade_pbr(
            albedo,
            normal,
            view,
            rough,
            metal,
            cam_gb.env,
            sun_dir_w=sun,
            sun_intensity=sun_intensity,
            sun_color=sun_color,
            sun_visibility=vis,
        )
        bg = cam_gb.env.sample(-view, roughness=0.0)
        return torch.where(mask.unsqueeze(-1), hdr, bg)

    def shade_torch_fxaa() -> torch.Tensor:
        return fxaa(aces_tonemap(_shade_hdr(), exposure=exposure))

    def shade_tiled_fxaa() -> torch.Tensor:
        return fxaa_tiled(aces_tonemap(_shade_hdr(), exposure=exposure))

    def shade_nofxaa() -> torch.Tensor:
        return aces_tonemap(_shade_hdr(), exposure=exposure)

    ms_sh_t = _time(shade_torch_fxaa, warmup=args.warmup, iters=args.iters, device=td)
    ms_sh_w = _time(shade_tiled_fxaa, warmup=args.warmup, iters=args.iters, device=td)
    ms_sh_n = _time(shade_nofxaa, warmup=args.warmup, iters=args.iters, device=td)
    results["shade"]["eager_torch_fxaa_ms"] = ms_sh_t
    results["shade"]["eager_tiled_fxaa_ms"] = ms_sh_w
    results["shade"]["eager_nofxaa_ms"] = ms_sh_n
    print(f"  eager shade+torch_fxaa  {ms_sh_t:.3f} ms")
    print(f"  eager shade+tiled_fxaa  {ms_sh_w:.3f} ms")
    print(f"  eager shade nofxaa      {ms_sh_n:.3f} ms")

    if not args.skip_compile:
        try:
            compiled = torch.compile(shade_torch_fxaa, mode=args.compile_mode)
            t0 = time.perf_counter()
            for _ in range(max(3, args.warmup)):
                out = compiled()
            torch.cuda.synchronize(td)
            compile_s = time.perf_counter() - t0
            ms_sh_c = _time(compiled, warmup=args.warmup, iters=args.iters, device=td)
            mae = float((out - shade_torch_fxaa()).abs().mean())
            results["shade"]["compile_torch_fxaa_ms"] = ms_sh_c
            results["shade"]["compile_warmup_s"] = compile_s
            results["shade"]["compile_mae"] = mae
            results["shade"]["compile_vs_eager"] = ms_sh_t / ms_sh_c
            results["shade"]["compile_vs_tiled_fxaa"] = ms_sh_w / ms_sh_c
            print(
                f"  compile shade+torch_fxaa {ms_sh_c:.3f} ms  "
                f"({ms_sh_t / ms_sh_c:.2f}x eager, {ms_sh_w / ms_sh_c:.2f}x vs tiled_fxaa)  "
                f"warmup {compile_s:.1f}s  mae={mae:.3e}"
            )
        except Exception as exc:
            results["shade"]["compile_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  shade compile FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        torch.compiler.reset()
        torch.cuda.empty_cache()

    print("\nVERDICT")
    print(
        f"  FXAA: tiled {results['fxaa_speedup_tiled_vs_torch']:.2f}x vs torch eager"
        + (
            f"; compile {results['compile']['fxaa_speedup_vs_eager']:.2f}x eager / "
            f"{results['compile']['fxaa_ms'] / results['fxaa_tiled_ms']:.2f}x vs tiled"
            if results.get("compile", {}).get("ok")
            else ""
        )
    )
    print(
        f"  SSAO: tiled {results['ssao_speedup_tiled_vs_torch']:.2f}x vs torch eager"
        + (
            f"; compile {results['compile']['ssao_speedup_vs_eager']:.2f}x eager / "
            f"{results['compile']['ssao_ms'] / results['ssao_tiled_ms']:.2f}x vs tiled"
            if results.get("compile", {}).get("ok")
            else ""
        )
    )
    print(
        f"  pbr_fast: torch_fxaa {results['full']['fast_torch_fxaa']:.2f} → "
        f"tiled {results['full']['fast_tiled_fxaa']:.2f} → "
        f"compile {results['full']['fast_compile_torch']:.2f} ms"
    )
    print(
        f"  pbr_pretty: torch {results['full']['pretty_torch']:.2f} → "
        f"tiled_fxaa+torch_ssao {results['full']['pretty_tiled_fxaa_torch_ssao']:.2f} → "
        f"all_tiled {results['full']['pretty_tiled']:.2f} → "
        f"compile {results['full']['pretty_compile_torch']:.2f} ms"
    )
    if "compile_torch_fxaa_ms" in results["shade"]:
        print(
            f"  shade+FXAA: eager_torch {results['shade']['eager_torch_fxaa_ms']:.2f} / "
            f"compile {results['shade']['compile_torch_fxaa_ms']:.2f} / "
            f"tiled_fxaa {results['shade']['eager_tiled_fxaa_ms']:.2f} ms"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
