"""Probe PBR shadows (cost vs fidelity) and CUDA graph capture at N=256.

Shadow rays are treated as the visibility reference. Graphs: Warp capture of
the G-buffer launch, and a Torch CUDAGraph of shade+FXAA on a frozen G-buffer.

    .venv/bin/python scripts/probe_pbr_opt.py
    .venv/bin/python scripts/probe_pbr_opt.py --n 256 --width 192 --height 144
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

from simple_raycaster import RaycastCamera, RaycastPBRCamera
from simple_raycaster.pbr.hdri import default_hdri_path, EnvironmentHDRI
from simple_raycaster.pbr.shade import aces_tonemap, fxaa, shade_pbr

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from bench_g1_camera import (  # noqa: E402
    DEFAULT_G1_XML,
    benchmark_fn,
    expand_poses,
    format_memory,
    load_scene,
    sample_cameras,
)


def _scene_names(stats: dict) -> list[str]:
    g1 = list(stats["g1_names"])
    n_static = int(stats["n_static"])
    static = ["ground", "cube"][:n_static]
    return static + g1


def _vis_stats(vis: torch.Tensor, mask: torch.Tensor, name: str) -> dict:
    m = mask
    v = vis[m]
    dark = v < 0.5
    soft = (v < 0.95) & (v >= 0.5)
    return {
        "name": name,
        "dark_frac": float(dark.float().mean()) if v.numel() else 0.0,
        "soft_frac": float(soft.float().mean()) if v.numel() else 0.0,
        "vis_mean": float(v.mean()) if v.numel() else 1.0,
        "vis_min": float(v.min()) if v.numel() else 1.0,
    }


def _agree(ref: torch.Tensor, other: torch.Tensor, mask: torch.Tensor) -> dict:
    """Compare vis maps on hits; ref is usually shadow-ray visibility."""
    m = mask
    a = ref[m]
    b = other[m]
    mae = float((a - b).abs().mean())
    # Binary dark agreement (vis < 0.5).
    ra = a < 0.5
    rb = b < 0.5
    inter = (ra & rb).float().sum()
    union = (ra | rb).float().sum().clamp_min(1.0)
    iou = float(inter / union)
    # Correlation on continuous vis.
    if a.numel() > 8 and float(a.std()) > 1e-6 and float(b.std()) > 1e-6:
        corr = float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])
    else:
        corr = 0.0
    return {"mae": mae, "dark_iou": iou, "corr": corr}


@torch.no_grad()
def probe_shadows(
    args,
    raycaster,
    g1_pos,
    g1_quat,
    n_static,
    names,
    env,
    torch_device,
) -> dict:
    print("\n" + "=" * 72)
    print("SHADOW METHODS @ N=%d" % args.n)
    print("=" * 72)

    mesh_pos, mesh_quat = expand_poses(args.n, g1_pos, g1_quat, n_static)
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

    variants = {
        "none": dict(quality="fast", fxaa_enabled=False),
        "contact": dict(
            quality="fast",
            fxaa_enabled=False,
            contact_shadows_enabled=True,
            ssao_enabled=False,
        ),
        "smap128": dict(
            quality="fast",
            fxaa_enabled=False,
            contact_shadows_enabled=False,
            ssao_enabled=False,
            shadow_map_enabled=True,
            shadow_map_size=128,
        ),
        "smap256": dict(
            quality="fast",
            fxaa_enabled=False,
            contact_shadows_enabled=False,
            ssao_enabled=False,
            shadow_map_enabled=True,
            shadow_map_size=256,
        ),
        "sray": dict(
            quality="fast",
            fxaa_enabled=False,
            contact_shadows_enabled=False,
            ssao_enabled=False,
            shadow_rays=True,
        ),
        "contact+smap256": dict(
            quality="fast",
            fxaa_enabled=False,
            contact_shadows_enabled=True,
            ssao_enabled=False,
            shadow_map_enabled=True,
            shadow_map_size=256,
        ),
    }

    cams = {}
    for name, overrides in variants.items():
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
            compile_mode=None,
            **overrides,
        )
        cam.bind_meshes(raycaster, names=names)
        cams[name] = cam

    # Warmup + fidelity vs shadow rays.
    ref_rgb, _, ref_mask, ref_gb = cams["sray"].render(
        cam_pos, cam_quat, return_gbuffer=True, **kw
    )
    ref_vis = ref_gb["sun_visibility"]
    fidelity = {}
    for name, cam in cams.items():
        rgb, _, mask, gb = cam.render(cam_pos, cam_quat, return_gbuffer=True, **kw)
        assert torch.equal(mask, ref_mask)
        fidelity[name] = {
            **_vis_stats(gb["sun_visibility"], mask, name),
            **{f"vs_sray_{k}": v for k, v in _agree(ref_vis, gb["sun_visibility"], mask).items()},
        }
        print(
            f"  {name:<16} dark={fidelity[name]['dark_frac']:.3f}  "
            f"vs_sray mae={fidelity[name]['vs_sray_mae']:.3f}  "
            f"iou={fidelity[name]['vs_sray_dark_iou']:.3f}  "
            f"corr={fidelity[name]['vs_sray_corr']:.3f}"
        )
        del rgb, mask, gb

    # Timing (include full render so shade cost is in the total).
    timings = {}
    for name, cam in cams.items():
        cam._work.clear()
        torch.cuda.empty_cache()
        timing = benchmark_fn(
            lambda c=cam: c.render(cam_pos, cam_quat, **kw),
            args.warmup,
            args.iters,
            torch_device,
        )
        timings[name] = timing
        print(
            f"  time {name:<16} {timing['ms_per_iter']:.3f} ms/iter  "
            f"peak {format_memory(timing['mem_peak'])}"
        )

    # Dump one-env comparison for camera 0 (PPM/PNG via smoke helpers).
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import importlib.util

    smoke_path = Path(__file__).resolve().parent / "smoke_pbr_camera.py"
    spec = importlib.util.spec_from_file_location("smoke_pbr_camera", smoke_path)
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    smoke._write_preview(out / "shadow_ref_sray", ref_rgb[0])
    for name in ("none", "contact", "smap256", "contact+smap256"):
        rgb, _, _ = cams[name].render(
            cam_pos[:1], cam_quat[:1], mesh_pos_w=mesh_pos[:1], mesh_quat_w=mesh_quat[:1]
        )
        smoke._write_preview(out / f"shadow_{name}", rgb[0])

    return {"fidelity": fidelity, "timings": timings}


@torch.no_grad()
def probe_graphs(
    args,
    raycaster,
    g1_pos,
    g1_quat,
    n_static,
    names,
    env,
    torch_device,
) -> dict:
    print("\n" + "=" * 72)
    print("CUDA GRAPH PROBE @ N=%d" % args.n)
    print("=" * 72)

    mesh_pos, mesh_quat = expand_poses(args.n, g1_pos, g1_quat, n_static)
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
    )
    cam.bind_meshes(raycaster, names=names)

    # --- A: full pbr_fast eager ---
    for _ in range(args.warmup):
        cam.render(cam_pos, cam_quat, **kw)
    torch.cuda.synchronize(torch_device)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        cam.render(cam_pos, cam_quat, **kw)
        wp.synchronize()
    torch.cuda.synchronize(torch_device)
    eager_ms = (time.perf_counter() - t0) / args.iters * 1000.0
    print(f"  full pbr_fast eager:     {eager_ms:.3f} ms/iter")

    # --- B: Warp graph on G-buffer only ---
    gb = cam.render_gbuffer(cam_pos, cam_quat, **kw)
    torch.cuda.synchronize(torch_device)
    wp.synchronize()

    def _launch_gbuffer():
        # Re-enter the same path; workspace addresses stay stable after first call.
        return cam.render_gbuffer(cam_pos, cam_quat, **kw)

    # Warm then capture the *kernel* via ScopedCapture by monkey-patching? Simpler:
    # capture inside a thin wrapper that only does the wp.launch of the last render_gbuffer.
    # We capture by calling render_gbuffer once to fill workspaces, then ScopedCapture
    # around a second identical call.
    _launch_gbuffer()
    torch.cuda.synchronize(torch_device)
    wp.synchronize()
    graph_ok = True
    graph_ms = None
    try:
        with wp.ScopedCapture(device=cam.geom.device, force_module_load=False) as cap:
            _launch_gbuffer()
        gbuf_graph = cap.graph
        for _ in range(args.warmup):
            wp.capture_launch(gbuf_graph)
            wp.synchronize()
        torch.cuda.synchronize(torch_device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            wp.capture_launch(gbuf_graph)
            wp.synchronize()
        torch.cuda.synchronize(torch_device)
        graph_ms = (time.perf_counter() - t0) / args.iters * 1000.0
        # Eager gbuffer alone for comparison.
        for _ in range(args.warmup):
            _launch_gbuffer()
            wp.synchronize()
        torch.cuda.synchronize(torch_device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _launch_gbuffer()
            wp.synchronize()
        torch.cuda.synchronize(torch_device)
        gbuf_eager_ms = (time.perf_counter() - t0) / args.iters * 1000.0
        print(f"  gbuffer eager:           {gbuf_eager_ms:.3f} ms/iter")
        print(
            f"  gbuffer Warp graph:      {graph_ms:.3f} ms/iter  "
            f"({gbuf_eager_ms / graph_ms:.2f}x)" if graph_ms > 1e-6 else ""
        )
    except Exception as exc:
        graph_ok = False
        gbuf_eager_ms = None
        print(f"  gbuffer Warp graph FAILED: {type(exc).__name__}: {exc}")

    # --- C: Torch CUDAGraph on shade+FXAA with frozen G-buffer ---
    gb = cam.render_gbuffer(cam_pos, cam_quat, **kw)
    # Detach / clone so shade reads fixed tensors.
    albedo = gb["albedo"].contiguous()
    normal = gb["normal"].contiguous()
    view = gb["view"].contiguous()
    rough = gb["roughness"].contiguous()
    metal = gb["metallic"].contiguous()
    mask = gb["mask"].contiguous()
    sun = torch.tensor(cam.sun_dir, device=torch_device, dtype=torch.float32)
    vis = torch.ones_like(gb["depth"])

    static_input = (albedo, normal, view, rough, metal, mask, sun, vis)

    def shade_once():
        hdr = shade_pbr(
            albedo,
            normal,
            view,
            rough,
            metal,
            cam.env,
            sun_dir_w=sun,
            sun_intensity=cam.sun_intensity,
            sun_color=cam._sun_color_t,
            sun_visibility=vis,
        )
        bg = cam.env.sample(-view, roughness=0.0)
        hdr = torch.where(mask.unsqueeze(-1), hdr, bg)
        rgb = aces_tonemap(hdr, exposure=cam.exposure)
        return fxaa(rgb)

    # Eager shade
    for _ in range(args.warmup):
        shade_once()
    torch.cuda.synchronize(torch_device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        shade_once()
    torch.cuda.synchronize(torch_device)
    shade_eager_ms = (time.perf_counter() - t0) / args.iters * 1000.0
    print(f"  shade+fxaa eager:        {shade_eager_ms:.3f} ms/iter")

    torch_graph_ok = True
    shade_graph_ms = None
    try:
        # CUDAGraph requires no CPU↔GPU sync and stable allocations.
        g = torch.cuda.CUDAGraph()
        # Warmup on side stream (PyTorch recommendation).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                shade_once()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            rgb_out = shade_once()
        for _ in range(args.warmup):
            g.replay()
        torch.cuda.synchronize(torch_device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            g.replay()
        torch.cuda.synchronize(torch_device)
        shade_graph_ms = (time.perf_counter() - t0) / args.iters * 1000.0
        print(
            f"  shade+fxaa CUDAGraph:    {shade_graph_ms:.3f} ms/iter  "
            f"({shade_eager_ms / shade_graph_ms:.2f}x)"
        )
        del rgb_out, static_input
    except Exception as exc:
        torch_graph_ok = False
        print(f"  shade+fxaa CUDAGraph FAILED: {type(exc).__name__}: {exc}")

    # --- D: FXAA-only CUDAGraph (known graph-safe) ---
    ldr = torch.rand(args.n, args.height, args.width, 3, device=torch_device).clamp(0, 1)
    for _ in range(args.warmup):
        fxaa(ldr)
    torch.cuda.synchronize(torch_device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        fxaa(ldr)
    torch.cuda.synchronize(torch_device)
    fxaa_eager_ms = (time.perf_counter() - t0) / args.iters * 1000.0
    fxaa_graph_ok = True
    fxaa_graph_ms = None
    try:
        gf = torch.cuda.CUDAGraph()
        s2 = torch.cuda.Stream()
        s2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s2):
            for _ in range(3):
                fxaa(ldr)
        torch.cuda.current_stream().wait_stream(s2)
        with torch.cuda.graph(gf):
            _ = fxaa(ldr)
        for _ in range(args.warmup):
            gf.replay()
        torch.cuda.synchronize(torch_device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            gf.replay()
        torch.cuda.synchronize(torch_device)
        fxaa_graph_ms = (time.perf_counter() - t0) / args.iters * 1000.0
        print(
            f"  fxaa CUDAGraph:          {fxaa_graph_ms:.3f} ms/iter  "
            f"({fxaa_eager_ms / fxaa_graph_ms:.2f}x vs eager {fxaa_eager_ms:.3f})"
        )
    except Exception as exc:
        fxaa_graph_ok = False
        print(f"  fxaa CUDAGraph FAILED: {type(exc).__name__}: {exc}")

    return {
        "eager_full_ms": eager_ms,
        "gbuffer_eager_ms": gbuf_eager_ms,
        "gbuffer_graph_ms": graph_ms,
        "gbuffer_graph_ok": graph_ok,
        "shade_eager_ms": shade_eager_ms,
        "shade_graph_ms": shade_graph_ms,
        "shade_graph_ok": torch_graph_ok,
        "fxaa_eager_ms": fxaa_eager_ms,
        "fxaa_graph_ms": fxaa_graph_ms,
        "fxaa_graph_ok": fxaa_graph_ok,
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_G1_XML)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--max-dist", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=str, default="scripts/_pbr_smoke")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--skip-shadows", action="store_true")
    parser.add_argument("--skip-graphs", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch_device = torch.device(args.device)
    wp.init()

    raycaster, g1_pos, g1_quat, stats = load_scene(args.xml, args.device)
    names = _scene_names(stats)
    env = EnvironmentHDRI.from_path(default_hdri_path(), device=torch_device)

    print(
        f"probe N={args.n} {args.width}x{args.height}  "
        f"meshes={raycaster.n_meshes}  gpu={torch.cuda.get_device_name(0)}"
    )

    blob: dict = {
        "n": args.n,
        "width": args.width,
        "height": args.height,
        "gpu": torch.cuda.get_device_name(0),
        "warp": wp.__version__,
        "torch": torch.__version__,
    }
    if not args.skip_shadows:
        blob["shadows"] = probe_shadows(
            args, raycaster, g1_pos, g1_quat, stats["n_static"], names, env, torch_device
        )
    if not args.skip_graphs:
        blob["graphs"] = probe_graphs(
            args, raycaster, g1_pos, g1_quat, stats["n_static"], names, env, torch_device
        )

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if "shadows" in blob:
        t = blob["shadows"]["timings"]
        f = blob["shadows"]["fidelity"]
        base = t["none"]["ms_per_iter"]
        print("Shadows (extra ms over none, fidelity vs shadow-ray):")
        for name in ("contact", "smap128", "smap256", "sray", "contact+smap256"):
            extra = t[name]["ms_per_iter"] - base
            print(
                f"  {name:<16} +{extra:6.2f} ms   "
                f"iou={f[name]['vs_sray_dark_iou']:.3f}  "
                f"corr={f[name]['vs_sray_corr']:.3f}  "
                f"mae={f[name]['vs_sray_mae']:.3f}"
            )
    if "graphs" in blob:
        g = blob["graphs"]
        print("Graphs:")
        print(f"  full eager pbr_fast:     {g['eager_full_ms']:.3f} ms")
        if g.get("gbuffer_graph_ok") and g.get("gbuffer_graph_ms") and g.get("gbuffer_eager_ms"):
            print(
                f"  gbuffer graph speedup:   "
                f"{g['gbuffer_eager_ms'] / g['gbuffer_graph_ms']:.2f}x "
                f"({g['gbuffer_eager_ms']:.3f} → {g['gbuffer_graph_ms']:.3f} ms)"
            )
        if g.get("shade_graph_ok") and g.get("shade_graph_ms") and g.get("shade_eager_ms"):
            print(
                f"  shade graph speedup:     "
                f"{g['shade_eager_ms'] / g['shade_graph_ms']:.2f}x "
                f"({g['shade_eager_ms']:.3f} → {g['shade_graph_ms']:.3f} ms)"
            )
        elif not g.get("shade_graph_ok"):
            print("  shade graph:             FAILED (see log)")
        if g.get("fxaa_graph_ok") and g.get("fxaa_graph_ms") and g.get("fxaa_eager_ms"):
            print(
                f"  fxaa graph speedup:      "
                f"{g['fxaa_eager_ms'] / g['fxaa_graph_ms']:.2f}x "
                f"({g['fxaa_eager_ms']:.3f} → {g['fxaa_graph_ms']:.3f} ms)"
            )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(blob, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
