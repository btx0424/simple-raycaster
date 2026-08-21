"""UV + albedo ground textures (Poly Haven CC0) on a small PBR scene.

    .venv/bin/python scripts/demo_ground_uv_albedo.py
    .venv/bin/python scripts/demo_ground_uv_albedo.py --out scripts/_pbr_ground_uv
"""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp

from simple_raycaster import MultiMeshRaycaster, RaycastPBRCamera
from simple_raycaster.pbr.textures import BUNDLED_ALBEDOS


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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=str, default="scripts/_pbr_ground_uv")
    parser.add_argument("--tile-m", type=float, default=0.75)
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
            [1.0, 1.0, 1.0],
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

    eye = np.array([1.8, -1.6, 1.1], dtype=np.float64)
    target = np.array([0.2, 0.0, 0.3], dtype=np.float64)
    quat = _look_at_opencv(eye, target)
    cam_pos = torch.as_tensor(eye, device=td, dtype=torch.float32).view(1, 3)
    cam_quat = torch.as_tensor(quat, device=td, dtype=torch.float32).view(1, 4)
    mesh_pos = torch.zeros(1, 3, 3, device=td, dtype=torch.float32)
    mesh_quat = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=td)
        .view(1, 1, 4)
        .expand(1, 3, 4)
        .contiguous()
    )

    # Flat baseline (no UV).
    rgb0, _, mask0 = cam.render(
        cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat
    )
    _write_png(
        out / "00_flat_ground.png",
        (rgb0[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8),
    )

    aliases = list(BUNDLED_ALBEDOS.keys())
    for i, alias in enumerate(aliases):
        cam.set_ground_albedo(alias, mesh_name="ground", tile_m=args.tile_m)
        rgb, depth, mask = cam.render(
            cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat
        )
        gb = cam.render_gbuffer(
            cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat
        )
        alb = gb["albedo"][0]
        hit = mask[0]
        # Ground pixels should vary spatially (texture), not be a single flat RGB.
        groundish = hit & (alb[..., 1] > 0.05)
        if bool(groundish.any()):
            spread = float(alb[groundish].std())
        else:
            spread = 0.0
        img = (rgb[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        _write_png(out / f"{i + 1:02d}_{alias}.png", img)
        print(
            f"  [{alias}] hit_frac={float(mask.float().mean()):.3f}  "
            f"albedo_std={spread:.4f}  rgb_mean={float(rgb[0][mask[0]].mean()) if bool(mask[0].any()) else 0.0:.3f}"
        )
        if spread < 1e-3:
            raise RuntimeError(f"expected textured albedo variation for {alias}, std={spread}")

    # Batched DR: same geometry, different ground textures via mesh_tex_id + stack.
    from simple_raycaster.pbr.textures import load_albedo_stack, pack_vertex_uvs_from_wp

    stack = load_albedo_stack(aliases)
    n = len(aliases)
    tex_ids = torch.full((n, 3), -1, device=td, dtype=torch.int32)
    for i in range(n):
        tex_ids[i, 0] = i
    cam.set_albedo_textures(stack, mesh_tex_id=tex_ids)
    uvs = pack_vertex_uvs_from_wp(
        cam.geom.meshes_wp, textured_mesh_ids=[0], scale=float(args.tile_m)
    )
    cam.set_vertex_uvs(uvs)
    alb = torch.ones(n, 3, 3, device=td, dtype=torch.float32)
    alb[:, 1] = torch.tensor([0.75, 0.25, 0.18], device=td)
    alb[:, 2] = torch.tensor([0.22, 0.45, 0.75], device=td)
    cam.set_materials(albedo=alb)

    cam_pos_b = cam_pos.expand(n, 3).contiguous()
    cam_quat_b = cam_quat.expand(n, 4).contiguous()
    mesh_pos_b = mesh_pos.expand(n, 3, 3).contiguous()
    mesh_quat_b = mesh_quat.expand(n, 3, 4).contiguous()
    rgb_b, _, mask_b = cam.render(
        cam_pos_b, cam_quat_b, mesh_pos_w=mesh_pos_b, mesh_quat_w=mesh_quat_b
    )
    # Montage
    h, w = args.height, args.width
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    canvas = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, ncols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = (
            rgb_b[i].clamp(0, 1).cpu().numpy() * 255.0
        ).astype(np.uint8)
    _write_png(out / "batch_ground_textures.png", canvas)

    print(
        f"ground UV albedo demo  {args.width}x{args.height}  "
        f"flat_hit={float(mask0.float().mean()):.3f}  "
        f"batch_hit={float(mask_b.float().mean()):.3f}  wrote {out}"
    )
    print("ok")


if __name__ == "__main__":
    main()
