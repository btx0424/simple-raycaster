"""Benchmark dense ``render`` vs tile-sparse ``render_sparse`` (1/4 tiles).

Synthetic scene, N=256, 192×144, tile_size=16 → 12×9=108 tiles; sparse uses 27.

    .venv/bin/python scripts/bench_render_sparse.py
    .venv/bin/python scripts/bench_render_sparse.py --json-out bench/results/sparse.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp

from simple_raycaster import MultiMeshRaycaster, RaycastPBRCamera
from simple_raycaster.pbr.hdri import EnvironmentHDRI, default_hdri_path


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    xyzw = Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def _time_ms(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    wp.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
        wp.synchronize()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / max(iters, 1))


def _strided_quarter_tiles(n_tx: int, n_ty: int, device: torch.device) -> torch.Tensor:
    """Deterministic ~1/4 of the tile grid (every other tile in x and y)."""
    xs = torch.arange(0, n_tx, 2, device=device)
    ys = torch.arange(0, n_ty, 2, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).to(torch.int32)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="",
        help="Empty = None (fair sparse vs dense shade; dense FXAA still on)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    wp.init()
    device = args.device
    td = torch.device(device)
    n, w, h, s = args.n, args.width, args.height, args.tile_size
    assert w % s == 0 and h % s == 0
    n_tx, n_ty = w // s, h // s
    n_tiles_full = n_tx * n_ty
    compile_mode = args.compile_mode.strip() or None

    ground = trimesh.creation.box(extents=[4.0, 4.0, 0.05])
    ground.apply_translation([0.0, 0.0, -0.025])
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.35)
    sphere.apply_translation([0.0, 0.0, 0.35])
    box = trimesh.creation.box(extents=[0.45, 0.45, 0.45])
    box.apply_translation([0.7, -0.2, 0.225])
    meshes = [ground, sphere, box]
    names = ["ground", "sphere", "box"]
    albedos = np.array(
        [[0.45, 0.45, 0.42], [0.75, 0.25, 0.18], [0.22, 0.45, 0.75]], dtype=np.float32
    )
    raycaster = MultiMeshRaycaster(meshes, device=device)
    env = EnvironmentHDRI.from_path(default_hdri_path(), device=td)

    cam_kw = dict(
        width=w,
        height=h,
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
    cam_dense = RaycastPBRCamera(**cam_kw)
    cam_dense.bind_meshes(raycaster, names=names, albedos=albedos)
    cam_sparse = RaycastPBRCamera(**cam_kw)
    cam_sparse.bind_meshes(raycaster, names=names, albedos=albedos)
    # Dense no-FXAA matches sparse shade path for an apples-to-apples pixel cost.
    cam_dense_nf = RaycastPBRCamera(**{**cam_kw, "fxaa_enabled": False})
    cam_dense_nf.bind_meshes(raycaster, names=names, albedos=albedos)

    eye = np.array([1.8, -1.6, 1.1], dtype=np.float64)
    quat = _look_at(eye, np.array([0.2, 0.0, 0.3], dtype=np.float64))
    cam_pos = torch.as_tensor(eye, device=td, dtype=torch.float32).view(1, 3).expand(n, 3).contiguous()
    cam_quat = torch.as_tensor(quat, device=td, dtype=torch.float32).view(1, 4).expand(n, 4).contiguous()
    m = raycaster.n_meshes
    mesh_pos = torch.zeros(n, m, 3, device=td)
    mesh_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=td).view(1, 1, 4).expand(n, m, 4).contiguous()
    pose_kw = dict(mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat)

    tiles_1 = _strided_quarter_tiles(n_tx, n_ty, td)
    t_count = int(tiles_1.shape[0])
    tiles = tiles_1.view(1, t_count, 2).expand(n, t_count, 2).contiguous()
    frac = t_count / float(n_tiles_full)

    # Correctness: sparse tiles match dense crop.
    rgb_d, depth_d, mask_d = cam_dense_nf.render(cam_pos[:1], cam_quat[:1], **{
        k: v[:1] for k, v in pose_kw.items()
    })
    rgb_s, depth_s, mask_s = cam_sparse.render_sparse(
        cam_pos[:1],
        cam_quat[:1],
        tiles=tiles[:1],
        tile_size=s,
        **{k: v[:1] for k, v in pose_kw.items()},
    )
    assert rgb_s.shape == (1, t_count, s, s, 3), rgb_s.shape
    # Compare first tile against dense crop.
    tx0, ty0 = int(tiles[0, 0, 0]), int(tiles[0, 0, 1])
    y0, x0 = ty0 * s, tx0 * s
    crop = rgb_d[0, y0 : y0 + s, x0 : x0 + s]
    err = float((crop - rgb_s[0, 0]).abs().max())
    if err > 1e-3:
        print(f"warning: dense vs sparse tile0 max|Δ|={err:.3e}")
    else:
        print(f"parity tile0 max|Δ|={err:.3e}")

    print("=" * 72)
    print(
        f"render_sparse bench  N={n} {w}x{h}  S={s}  "
        f"tiles={t_count}/{n_tiles_full} ({frac:.1%})  compile={compile_mode!r}"
    )
    print(f"gpu {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    ms_dense = _time_ms(
        lambda: cam_dense.render(cam_pos, cam_quat, **pose_kw),
        warmup=args.warmup,
        iters=args.iters,
        device=td,
    )
    ms_dense_nf = _time_ms(
        lambda: cam_dense_nf.render(cam_pos, cam_quat, **pose_kw),
        warmup=args.warmup,
        iters=args.iters,
        device=td,
    )
    ms_sparse = _time_ms(
        lambda: cam_sparse.render_sparse(
            cam_pos, cam_quat, tiles=tiles, tile_size=s, **pose_kw
        ),
        warmup=args.warmup,
        iters=args.iters,
        device=td,
    )
    # G-buffer only.
    ms_gb = _time_ms(
        lambda: cam_dense_nf.render_gbuffer(cam_pos, cam_quat, **pose_kw),
        warmup=args.warmup,
        iters=args.iters,
        device=td,
    )
    ms_gb_sp = _time_ms(
        lambda: cam_sparse.render_gbuffer_sparse(
            cam_pos, cam_quat, tiles=tiles, tile_size=s, **pose_kw
        ),
        warmup=args.warmup,
        iters=args.iters,
        device=td,
    )

    print(f"  dense pbr_fast (FXAA):     {ms_dense:8.3f} ms/iter")
    print(f"  dense no-FXAA:             {ms_dense_nf:8.3f} ms/iter")
    print(f"  sparse 1/4 tiles:          {ms_sparse:8.3f} ms/iter")
    print(f"  gbuffer dense:             {ms_gb:8.3f} ms/iter")
    print(f"  gbuffer sparse:            {ms_gb_sp:8.3f} ms/iter")
    print(
        f"\n  vs dense FXAA:   {ms_dense / ms_sparse:.2f}×  "
        f"({(1 - ms_sparse / ms_dense) * 100:.0f}% less time)"
    )
    print(
        f"  vs dense noFXAA: {ms_dense_nf / ms_sparse:.2f}×  "
        f"({(1 - ms_sparse / ms_dense_nf) * 100:.0f}% less time)"
    )
    print(
        f"  gbuffer speedup: {ms_gb / ms_gb_sp:.2f}×  "
        f"(pixel frac {frac:.1%}, ideal ~{1 / frac:.1f}×)"
    )
    print(
        f"  RGB storage: full {n * h * w * 3 * 4 / 1e6:.1f} MB fp32  vs  "
        f"sparse {n * t_count * s * s * 3 * 4 / 1e6:.1f} MB "
        f"({frac:.1%})"
    )

    blob = {
        "git_sha": _git_sha(),
        "gpu": torch.cuda.get_device_name(0),
        "n": n,
        "width": w,
        "height": h,
        "tile_size": s,
        "n_tiles_full": n_tiles_full,
        "n_tiles_sparse": t_count,
        "tile_frac": frac,
        "compile_mode": compile_mode,
        "ms": {
            "dense_fxaa": ms_dense,
            "dense_nofxaa": ms_dense_nf,
            "sparse": ms_sparse,
            "gbuffer_dense": ms_gb,
            "gbuffer_sparse": ms_gb_sp,
        },
        "speedup_vs_dense_fxaa": ms_dense / ms_sparse,
        "speedup_vs_dense_nofxaa": ms_dense_nf / ms_sparse,
        "parity_tile0_max_abs": err,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"wrote {out}")
    print("ok")


if __name__ == "__main__":
    main()
