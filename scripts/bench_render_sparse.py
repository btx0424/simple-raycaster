"""Benchmark dense ``render`` vs tile-sparse ``render_sparse`` (1/2 and 1/4 tiles).

Synthetic scene, N=256, 192×144, tile_size=16 → 12×9=108 tiles.

    .venv/bin/python scripts/bench_render_sparse.py
    .venv/bin/python scripts/bench_render_sparse.py --json-out bench/results/sparse.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
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


def _all_tiles(n_tx: int, n_ty: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(n_ty, device=device)
    xs = torch.arange(n_tx, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).to(torch.int32)


def _half_tiles(n_tx: int, n_ty: int, device: torch.device) -> torch.Tensor:
    """~1/2 of the grid: every other column (checker columns)."""
    all_t = _all_tiles(n_tx, n_ty, device)
    return all_t[(all_t[:, 0] % 2) == 0].contiguous()


def _quarter_tiles(n_tx: int, n_ty: int, device: torch.device) -> torch.Tensor:
    """~1/4 of the grid: every other tile in x and y."""
    all_t = _all_tiles(n_tx, n_ty, device)
    return all_t[((all_t[:, 0] % 2) == 0) & ((all_t[:, 1] % 2) == 0)].contiguous()


def _expand_tiles(tiles_1: torch.Tensor, n: int) -> torch.Tensor:
    t = int(tiles_1.shape[0])
    return tiles_1.view(1, t, 2).expand(n, t, 2).contiguous()


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

    variants = {
        "half": _expand_tiles(_half_tiles(n_tx, n_ty, td), n),
        "quarter": _expand_tiles(_quarter_tiles(n_tx, n_ty, td), n),
    }

    # Correctness on quarter (first tile).
    tiles_q = variants["quarter"]
    rgb_d, _, _ = cam_dense_nf.render(
        cam_pos[:1], cam_quat[:1], **{k: v[:1] for k, v in pose_kw.items()}
    )
    rgb_s, _, _ = cam_sparse.render_sparse(
        cam_pos[:1],
        cam_quat[:1],
        tiles=tiles_q[:1],
        tile_size=s,
        **{k: v[:1] for k, v in pose_kw.items()},
    )
    tx0, ty0 = int(tiles_q[0, 0, 0]), int(tiles_q[0, 0, 1])
    crop = rgb_d[0, ty0 * s : ty0 * s + s, tx0 * s : tx0 * s + s]
    err = float((crop - rgb_s[0, 0]).abs().max())
    print(f"parity tile0 max|Δ|={err:.3e}" + ("" if err <= 1e-3 else "  WARNING"))

    print("=" * 72)
    print(
        f"render_sparse bench  N={n} {w}x{h}  S={s}  "
        f"full_tiles={n_tiles_full}  compile={compile_mode!r}"
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
    ms_gb = _time_ms(
        lambda: cam_dense_nf.render_gbuffer(cam_pos, cam_quat, **pose_kw),
        warmup=args.warmup,
        iters=args.iters,
        device=td,
    )

    print(f"  dense pbr_fast (FXAA):     {ms_dense:8.3f} ms/iter")
    print(f"  dense no-FXAA:             {ms_dense_nf:8.3f} ms/iter")
    print(f"  gbuffer dense:             {ms_gb:8.3f} ms/iter")

    sparse_rows: dict[str, dict] = {}
    for label, tiles in variants.items():
        t_count = int(tiles.shape[1])
        frac = t_count / float(n_tiles_full)
        ms_sp = _time_ms(
            lambda t=tiles: cam_sparse.render_sparse(
                cam_pos, cam_quat, tiles=t, tile_size=s, **pose_kw
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=td,
        )
        ms_gb_sp = _time_ms(
            lambda t=tiles: cam_sparse.render_gbuffer_sparse(
                cam_pos, cam_quat, tiles=t, tile_size=s, **pose_kw
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=td,
        )
        row = {
            "n_tiles": t_count,
            "tile_frac": frac,
            "ms": ms_sp,
            "ms_gbuffer": ms_gb_sp,
            "speedup_vs_dense_fxaa": ms_dense / ms_sp,
            "speedup_vs_dense_nofxaa": ms_dense_nf / ms_sp,
            "gbuffer_speedup": ms_gb / ms_gb_sp,
            "rgb_mb_fp32": n * t_count * s * s * 3 * 4 / 1e6,
        }
        sparse_rows[label] = row
        print(
            f"  sparse {label:7s} ({t_count}/{n_tiles_full}={frac:.1%}): "
            f"{ms_sp:8.3f} ms  "
            f"({row['speedup_vs_dense_nofxaa']:.2f}× vs noFXAA, "
            f"{row['speedup_vs_dense_fxaa']:.2f}× vs FXAA)  "
            f"gbuf {ms_gb_sp:.3f} ms ({row['gbuffer_speedup']:.2f}×)"
        )

    full_mb = n * h * w * 3 * 4 / 1e6
    print(f"\n  RGB storage full: {full_mb:.1f} MB fp32")
    for label, row in sparse_rows.items():
        print(
            f"  RGB storage {label:7s}: {row['rgb_mb_fp32']:.1f} MB  "
            f"({row['tile_frac']:.1%})"
        )

    blob = {
        "git_sha": _git_sha(),
        "gpu": torch.cuda.get_device_name(0),
        "n": n,
        "width": w,
        "height": h,
        "tile_size": s,
        "n_tiles_full": n_tiles_full,
        "compile_mode": compile_mode,
        "ms_dense_fxaa": ms_dense,
        "ms_dense_nofxaa": ms_dense_nf,
        "ms_gbuffer_dense": ms_gb,
        "sparse": sparse_rows,
        "parity_tile0_max_abs": err,
        "rgb_mb_full_fp32": full_mb,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        print(f"wrote {out}")
    print("ok")


if __name__ == "__main__":
    main()
