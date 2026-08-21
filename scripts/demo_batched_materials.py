"""Batched PBR material / light overrides in one ``render`` call.

Builds a small scene (ground + sphere + box), then renders N camera rows that
share geometry/poses but differ in albedo, roughness, metallic, sun, and
exposure — all via ``set_materials`` / ``set_lights`` mesh tables (no rebind).

    .venv/bin/python scripts/demo_batched_materials.py
    .venv/bin/python scripts/demo_batched_materials.py --out scripts/_pbr_batch_demo
"""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp

from simple_raycaster import MultiMeshRaycaster, RaycastPBRCamera


def _write_png(path: Path, rgb_u8: np.ndarray) -> None:
    h, w, _ = rgb_u8.shape
    raw = b"".join(b"\x00" + rgb_u8[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


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


def _grid_image(rgbs: torch.Tensor, ncols: int = 3) -> np.ndarray:
    """Stack ``[N,H,W,3]`` LDR into a single uint8 montage."""
    n, h, w, _ = rgbs.shape
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / ncols))
    canvas = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, ncols)
        img = (rgbs[i].detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    return canvas


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=str, default="scripts/_pbr_batch_demo")
    args = parser.parse_args()

    wp.init()
    device = args.device
    td = torch.device(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ground = trimesh.creation.box(extents=[4.0, 4.0, 0.05])
    ground.apply_translation([0.0, 0.0, -0.025])
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.35)
    sphere.apply_translation([0.0, 0.0, 0.35])
    box = trimesh.creation.box(extents=[0.45, 0.45, 0.45])
    box.apply_translation([0.7, -0.2, 0.225])
    meshes = [ground, sphere, box]
    names = ["ground", "sphere", "box"]
    base_alb = np.array(
        [
            [0.45, 0.45, 0.42],
            [0.75, 0.25, 0.18],
            [0.22, 0.45, 0.75],
        ],
        dtype=np.float32,
    )

    raycaster = MultiMeshRaycaster(meshes, device=device)
    cam = RaycastPBRCamera(
        args.width,
        args.height,
        fov_y_deg=50.0,
        near=0.05,
        far=20.0,
        convention="opencv",
        depth_mode="planar",
        device=device,
        quality="fast",
        compile_mode=None,
        fxaa_impl="tiled",
    )
    cam.bind_meshes(raycaster, names=names, albedos=base_alb)

    # --- One batch, six parameter variants ---
    labels = [
        "baseline",
        "red_sphere",
        "chrome_sphere",
        "rubber_box",
        "warm_sun",
        "bright_exp",
    ]
    n = len(labels)
    albedo = torch.as_tensor(base_alb, device=td, dtype=torch.float32).unsqueeze(0).expand(n, -1, -1).contiguous().clone()
    rough = torch.tensor([[0.95, 0.35, 0.40]] * n, device=td, dtype=torch.float32)
    metal = torch.tensor([[0.00, 0.05, 0.10]] * n, device=td, dtype=torch.float32)

    # row 1: red sphere
    albedo[1, 1] = torch.tensor([0.95, 0.08, 0.06], device=td)
    # row 2: chrome-ish sphere
    rough[2, 1] = 0.12
    metal[2, 1] = 0.95
    albedo[2, 1] = torch.tensor([0.9, 0.9, 0.92], device=td)
    # row 3: rubber box
    rough[3, 2] = 0.95
    metal[3, 2] = 0.0
    albedo[3, 2] = torch.tensor([0.1, 0.55, 0.2], device=td)

    sun_dir = torch.tensor(
        [
            [0.45, -0.35, 0.82],
            [0.45, -0.35, 0.82],
            [0.45, -0.35, 0.82],
            [0.45, -0.35, 0.82],
            [0.15, -0.55, 0.55],  # warmer / lower
            [0.45, -0.35, 0.82],
        ],
        device=td,
        dtype=torch.float32,
    )
    sun_color = torch.tensor(
        [
            [1.0, 0.98, 0.92],
            [1.0, 0.98, 0.92],
            [1.0, 0.98, 0.92],
            [1.0, 0.98, 0.92],
            [1.0, 0.72, 0.45],
            [1.0, 0.98, 0.92],
        ],
        device=td,
        dtype=torch.float32,
    )
    sun_intensity = torch.tensor([2.5, 2.5, 2.5, 2.5, 3.2, 2.5], device=td)
    exposure = torch.tensor([0.85, 0.85, 0.85, 0.85, 0.85, 1.35], device=td)

    eye = np.array([1.8, -1.6, 1.1], dtype=np.float64)
    target = np.array([0.2, 0.0, 0.3], dtype=np.float64)
    quat = _look_at_opencv(eye, target)
    cam_pos = torch.as_tensor(eye, device=td, dtype=torch.float32).view(1, 3).expand(n, 3).contiguous()
    cam_quat = torch.as_tensor(quat, device=td, dtype=torch.float32).view(1, 4).expand(n, 4).contiguous()
    mesh_pos = torch.zeros(n, 3, 3, device=td, dtype=torch.float32)
    mesh_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=td).view(1, 1, 4).expand(n, 3, 4).contiguous()

    # One render call carries the full ticket (poses + DR tables).
    rgb, depth, mask = cam.render(
        cam_pos,
        cam_quat,
        mesh_pos_w=mesh_pos,
        mesh_quat_w=mesh_quat,
        albedo=albedo,
        roughness=rough,
        metallic=metal,
        sun_dir=sun_dir,
        sun_color=sun_color,
        sun_intensity=sun_intensity,
        exposure=exposure,
    )
    assert rgb.shape[0] == n

    grid = _grid_image(rgb, ncols=3)
    _write_png(out / "batch_grid.png", grid)
    for i, label in enumerate(labels):
        img = (rgb[i].detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        _write_png(out / f"{i:02d}_{label}.png", img)

    print(
        f"batched materials demo  N={n} {args.width}x{args.height}  "
        f"hit_frac={float(mask.float().mean()):.3f}  wrote {out / 'batch_grid.png'}"
    )
    for i, label in enumerate(labels):
        print(f"  [{i}] {label}  rgb_mean={float(rgb[i][mask[i]].mean()) if bool(mask[i].any()) else 0.0:.3f}")
    print("ok")


if __name__ == "__main__":
    main()
