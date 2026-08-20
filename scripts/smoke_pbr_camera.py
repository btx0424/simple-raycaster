"""Smoke: Lambert ``RaycastCamera`` vs new ``RaycastPBRCamera`` (untouched Lambert path).

    .venv/bin/python scripts/smoke_pbr_camera.py
    .venv/bin/python scripts/smoke_pbr_camera.py --res 512x384
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

from simple_raycaster import RaycastCamera, RaycastPBRCamera, default_hdri_path
from simple_raycaster.pbr.hdri import load_radiance_hdr
from simple_raycaster.pbr.materials import g1_roughness_metallic, materials_for_names

G1_XML = (
    "/mnt/workspace/btx0424/aa-projects/object_hoi/src/assets/unitree_g1/"
    "g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
)


def _parse_res(raw: str) -> tuple[int, int]:
    parts = raw.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected WxH, got {raw!r}")
    width, height = int(parts[0]), int(parts[1])
    if width < 16 or height < 16:
        raise argparse.ArgumentTypeError(f"resolution {width}x{height} is below 16x16")
    return width, height


def _size(
    width: int | None, height: int | None, default_w: int, default_h: int
) -> tuple[int, int]:
    if width is None and height is None:
        return default_w, default_h
    if width is None or height is None:
        raise ValueError("width and height must be set together")
    return width, height


def _as_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    img = (rgb.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {tuple(img.shape)}")
    return np.ascontiguousarray(img)


def _write_ppm(path: Path, img: np.ndarray) -> None:
    """Binary Netpbm P6: ASCII header, then packed 8-bit RGB with no padding."""
    h, w = img.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(img.tobytes())


def _write_png(path: Path, img: np.ndarray) -> None:
    """RGB8 PNG via stdlib zlib (no Pillow). Filter 0, non-interlaced."""
    h, w = img.shape[:2]

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _write_preview(path: Path, rgb: torch.Tensor) -> None:
    """Write ``stem.ppm`` and ``stem.png`` (PPM is the original zero-dep dump)."""
    img = _as_u8_rgb(rgb)
    stem = path.with_suffix("")
    _write_ppm(stem.with_suffix(".ppm"), img)
    _write_png(stem.with_suffix(".png"), img)


def _look_at_opencv(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    z_cam = target - eye
    z_cam = z_cam / np.linalg.norm(z_cam)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # OpenCV: +X right, +Y down, +Z forward → right = forward × up.
    x_cam = np.cross(z_cam, world_up)
    if np.linalg.norm(x_cam) < 1e-6:
        x_cam = np.cross(z_cam, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)  # down
    rot = np.stack([x_cam, y_cam, z_cam], axis=1)
    xyzw = Rotation.from_matrix(rot).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def _assert_lambert_parity(
    device: str, out: Path, *, width: int | None = None, height: int | None = None
) -> None:
    w, h = _size(width, height, 64, 48)
    cy, cx = h // 2, w // 2
    box = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    lambert = RaycastCamera(w, h, fov_y_deg=70.0, convention="opencv", device=device)
    lambert.bind_trimeshes([box], albedos=[(0.9, 0.2, 0.15)])
    pbr = RaycastPBRCamera(
        w,
        h,
        fov_y_deg=70.0,
        convention="opencv",
        device=device,
        default_roughness=0.35,
        default_metallic=0.15,
        contact_shadows_enabled=False,
        ssao_enabled=True,
    )
    pbr.bind_trimeshes(
        [box],
        albedos=[(0.9, 0.2, 0.15)],
        roughness=0.35,
        metallic=0.15,
    )

    n = 1
    cam_pos = torch.zeros(n, 3, device=device)
    cam_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4).contiguous()
    mesh_pos = torch.tensor([[[0.0, 0.0, 2.0]]], device=device)
    mesh_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device)

    rgb_l, depth_l, mask_l = lambert.render(
        cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat
    )
    rgb_p, depth_p, mask_p, gb = pbr.render(
        cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat, return_gbuffer=True
    )

    if rgb_p.shape != rgb_l.shape:
        raise SystemExit(f"shape mismatch {tuple(rgb_p.shape)} vs {tuple(rgb_l.shape)}")
    if not torch.equal(mask_l, mask_p):
        raise SystemExit("mask mismatch vs Lambert camera")
    ddiff = (depth_l - depth_p).abs().max().item()
    if ddiff > 1e-4:
        raise SystemExit(f"depth mismatch vs Lambert max={ddiff}")
    if not bool(mask_p[0, cy, cx]):
        raise SystemExit("expected a hit at the image center")
    alb = gb["albedo"][0, cy, cx]
    if (alb - torch.tensor([0.9, 0.2, 0.15], device=device)).abs().max().item() > 1e-4:
        raise SystemExit(f"constant-albedo lerp failed: {alb.tolist()}")

    _write_preview(out / "lambert", rgb_l[0])
    _write_preview(out / "pbr", rgb_p[0])
    print(
        f"parity {w}x{h} hit_frac={float(mask_p.float().mean()):.3f} "
        f"depth_center={float(depth_p[0, cy, cx]):.3f} depth_max_abs={ddiff:.2e} "
        f"lambert_rgb_mean={float(rgb_l[mask_l].mean()):.3f} "
        f"pbr_rgb_mean={float(rgb_p[mask_p].mean()):.3f}"
    )


def _assert_vertex_colors(
    device: str, out: Path, *, width: int | None = None, height: int | None = None
) -> None:
    """Three-vertex triangle: red / green / blue — barycentric lerp at center."""
    verts = np.array(
        [[-0.6, 0.5, 1.5], [0.6, 0.5, 1.5], [0.0, -0.5, 1.5]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual.vertex_colors = np.array(
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
        dtype=np.uint8,
    )
    w, h = _size(width, height, 64, 48)
    pbr = RaycastPBRCamera(
        w,
        h,
        fov_y_deg=70.0,
        convention="opencv",
        device=device,
        ssao_enabled=False,
        contact_shadows_enabled=False,
    )
    pbr.bind_trimeshes([mesh], albedos=[(0.5, 0.5, 0.5)])
    cam_pos = torch.zeros(1, 3, device=device)
    cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    mesh_pos = torch.zeros(1, 1, 3, device=device)
    mesh_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device)
    rgb, depth, mask, gb = pbr.render(
        cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat, return_gbuffer=True
    )
    del depth
    if float(mask.float().mean()) < 0.02:
        raise SystemExit(f"vertex-color triangle missed, hit_frac={float(mask.float().mean())}")
    alb = gb["albedo"][0]
    m = mask[0]
    # Mid row of the image should clip the triangle; leftmost hit is nearer the
    # red vertex, rightmost nearer green, and a lower row nearer blue.
    row = h // 2
    cols = torch.where(m[row])[0]
    if cols.numel() < 4:
        raise SystemExit(f"vertex-color triangle too thin on row {row}: {int(cols.numel())} hits")
    left = alb[row, int(cols[0])]
    right = alb[row, int(cols[-1])]
    if not (float(left[0]) > float(right[0]) + 0.08 and float(right[1]) > float(left[1]) + 0.08):
        raise SystemExit(f"barycentric colors not split L/R: left={left.tolist()} right={right.tolist()}")
    rows = torch.where(m.any(dim=1))[0]
    top_cols = torch.where(m[int(rows[0])])[0]
    bot_cols = torch.where(m[int(rows[-1])])[0]
    top = alb[int(rows[0]), int(top_cols[top_cols.numel() // 2])]
    bot = alb[int(rows[-1]), int(bot_cols[bot_cols.numel() // 2])]
    # OpenCV +Y is down: v2 (blue) is at negative Y = top of the image.
    if float(top[2]) < float(bot[2]) + 0.2:
        raise SystemExit(f"expected blue at top vertex, top={top.tolist()} bot={bot.tolist()}")
    _write_preview(out / "vertex_colors", rgb[0])
    print(
        f"vertex_colors {w}x{h} hit_frac={float(mask.float().mean()):.3f} "
        f"left_r={float(left[0]):.2f} right_g={float(right[1]):.2f} top_b={float(top[2]):.2f}"
    )


def _assert_smooth_normals(
    device: str, *, width: int | None = None, height: int | None = None
) -> None:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    w, h = _size(width, height, 48, 48)
    kwargs = dict(
        width=w,
        height=h,
        fov_y_deg=50.0,
        convention="opencv",
        device=device,
        ssao_enabled=False,
        contact_shadows_enabled=False,
        default_roughness=0.4,
        default_metallic=0.0,
    )
    smooth = RaycastPBRCamera(**kwargs, smooth_normals=True)
    faceted = RaycastPBRCamera(**kwargs, smooth_normals=False)
    for cam in (smooth, faceted):
        cam.bind_trimeshes([sphere], albedos=[(0.7, 0.7, 0.7)])
    cam_pos = torch.zeros(1, 3, device=device)
    cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    mesh_pos = torch.tensor([[[0.0, 0.0, 1.4]]], device=device)
    mesh_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device)
    _, _, mask_s, gb_s = smooth.render(
        cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat, return_gbuffer=True
    )
    _, _, mask_f, gb_f = faceted.render(
        cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat, return_gbuffer=True
    )
    if not torch.equal(mask_s, mask_f):
        raise SystemExit("smooth vs face-normal mask mismatch")
    n_s = gb_s["normal"][0]
    n_f = gb_f["normal"][0]
    m = mask_s[0]
    center = torch.tensor([0.0, 0.0, 1.4], device=device)
    fx, fy, cx, cy = smooth.intrinsics.resolved_k()
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    z = gb_s["depth"][0]
    hit = torch.stack(
        [(xs + 0.5 - cx) / fx * z, (ys + 0.5 - cy) / fy * z, z], dim=-1
    )
    expected = hit - center
    expected = expected / expected.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # Kernel flips the normal toward the camera (−dir, here −Z).
    expected = torch.where((expected * torch.tensor([0.0, 0.0, 1.0], device=device)).sum(-1, keepdim=True) > 0, -expected, expected)
    err_s = (n_s - expected).norm(dim=-1)[m].mean()
    err_f = (n_f - expected).norm(dim=-1)[m].mean()
    if float(err_s) >= float(err_f) * 0.9:
        raise SystemExit(
            f"smooth normals not closer to sphere: {float(err_s):.4f} vs face {float(err_f):.4f}"
        )
    print(f"smooth_normals {w}x{h} err={float(err_s):.4f} face_err={float(err_f):.4f}")


def _assert_g1_materials() -> None:
    r_ank, m_ank = g1_roughness_metallic("left_ankle_roll_link")
    r_hip, m_hip = g1_roughness_metallic("left_hip_pitch_link")
    r_hand, m_hand = g1_roughness_metallic("L_index_proximal")
    if not (r_ank > 0.8 and m_ank < 0.05):
        raise SystemExit(f"ankle should be rubber, got r={r_ank} m={m_ank}")
    if not (m_hip > 0.3 and r_hip < r_ank):
        raise SystemExit(f"hip should be painted metal, got r={r_hip} m={m_hip}")
    if not (m_hand < 0.1 and r_hand > 0.5):
        raise SystemExit(f"hand should be plastic, got r={r_hand} m={m_hand}")
    names = ["left_ankle_roll_link", "torso_link", "ground"]
    r, m = materials_for_names(names)
    if r[0] <= r[1] or m[1] <= m[0]:
        raise SystemExit(f"materials_for_names order wrong: r={r} m={m}")
    print(f"g1_materials ankle=({r_ank:.2f},{m_ank:.2f}) hip=({r_hip:.2f},{m_hip:.2f}) hand=({r_hand:.2f},{m_hand:.2f})")


def _assert_contact_and_shadows(
    device: str, out: Path, *, width: int | None = None, height: int | None = None
) -> None:
    ground = trimesh.creation.box(extents=(4.0, 4.0, 0.04))
    cube = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    eye = np.array([1.15, -1.25, 0.85], dtype=np.float32)
    quat = _look_at_opencv(eye, np.array([0.2, 0.0, 0.15], dtype=np.float32))
    cam_pos = torch.tensor(eye, device=device).view(1, 3)
    cam_quat = torch.tensor(quat, device=device).view(1, 4)
    mesh_pos = torch.tensor(
        [[[0.0, 0.0, -0.02], [0.15, 0.0, 0.21]]],
        device=device,
    )
    mesh_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], device=device)
    sun = (0.85, 0.15, 0.50)
    w, h = _size(width, height, 96, 64)

    def _cam(**extra):
        c = RaycastPBRCamera(
            w,
            h,
            fov_y_deg=55.0,
            convention="opencv",
            device=device,
            ssao_enabled=False,
            sun_dir=sun,
            default_roughness=0.5,
            default_metallic=0.0,
            **extra,
        )
        c.bind_trimeshes(
            [ground, cube],
            albedos=[(0.55, 0.55, 0.52), (0.75, 0.25, 0.18)],
            names=["ground", "cube"],
        )
        return c

    off = _cam(contact_shadows_enabled=False, shadow_map_enabled=False, shadow_rays=False)
    contact = _cam(contact_shadows_enabled=True, shadow_map_enabled=False, shadow_rays=False)
    smap = _cam(
        contact_shadows_enabled=False,
        shadow_map_enabled=True,
        shadow_map_size=128,
        shadow_map_extent=2.5,
        shadow_rays=False,
    )
    sray = _cam(contact_shadows_enabled=False, shadow_map_enabled=False, shadow_rays=True)

    kwargs = dict(mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat)
    _, d0, m0, gb0 = off.render(cam_pos, cam_quat, return_gbuffer=True, **kwargs)
    rgb_c, _, _, gb_c = contact.render(cam_pos, cam_quat, return_gbuffer=True, **kwargs)
    rgb_s, _, _, gb_s = smap.render(cam_pos, cam_quat, return_gbuffer=True, **kwargs)
    rgb_r, _, _, gb_r = sray.render(cam_pos, cam_quat, return_gbuffer=True, **kwargs)

    if not torch.equal(m0, gb_c["mask"]):
        raise SystemExit("contact-shadow mask drifted")
    vis_c = gb_c["sun_visibility"]
    if float((vis_c[m0] < 0.95).float().mean()) < 0.005:
        raise SystemExit(
            f"contact shadows never darkened (frac={(vis_c[m0] < 0.95).float().mean().item():.4f})"
        )
    vis_s = gb_s["sun_visibility"]
    vis_r = gb_r["sun_visibility"]
    if float((vis_s[m0] < 0.5).float().mean()) < 0.01:
        raise SystemExit("shadow map produced almost no occlusion")
    if float((vis_r[m0] < 0.5).float().mean()) < 0.01:
        raise SystemExit("shadow rays produced almost no occlusion")

    _write_preview(out / "contact", rgb_c[0])
    _write_preview(out / "shadow_map", rgb_s[0])
    _write_preview(out / "shadow_ray", rgb_r[0])
    print(
        f"visibility {w}x{h} contact_dark={float((vis_c[m0] < 0.95).float().mean()):.3f} "
        f"smap_dark={float((vis_s[m0] < 0.5).float().mean()):.3f} "
        f"sray_dark={float((vis_r[m0] < 0.5).float().mean()):.3f} "
        f"depth_max={float(d0[m0].max()):.3f}"
    )


def _maybe_g1_preview(
    device: str, out: Path, *, width: int | None = None, height: int | None = None
) -> None:
    xml = Path(G1_XML)
    if not xml.is_file():
        print(f"skip g1 preview (missing {xml})")
        return
    import importlib.util

    bench_path = Path(__file__).resolve().parent / "bench_g1_camera.py"
    spec = importlib.util.spec_from_file_location("bench_g1_camera", bench_path)
    if spec is None or spec.loader is None:
        print("skip g1 preview (could not load bench_g1_camera.py)")
        return
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    raycaster, g1_pos, g1_quat, stats = bench.load_scene(str(xml), device)
    n_static = int(stats["n_static"])
    names = ["ground", "cube", *list(stats["g1_names"])]
    if len(names) != raycaster.n_meshes:
        names = None
    eye, quats = bench.sample_cameras(
        1, seed=0, radius_min=2.0, radius_max=2.0, elev_min_deg=12.0, elev_max_deg=12.0
    )
    cam_pos = torch.as_tensor(eye, device=device, dtype=torch.float32)
    # Look at mid-torso once the free joint is raised to standing height.
    target = np.array([0.0, 0.0, 0.65], dtype=np.float64)
    cam_quat = torch.as_tensor(
        np.stack(
            [
                bench.look_at_opencv_wxyz(
                    eye[i].astype(np.float64), target, np.array([0.0, 0.0, 1.0])
                )
                for i in range(len(eye))
            ]
        ),
        device=device,
        dtype=torch.float32,
    )
    mesh_pos, mesh_quat = bench.expand_poses(1, g1_pos, g1_quat, n_static, robot_first=False)
    w, h = _size(width, height, 128, 96)
    pbr = RaycastPBRCamera(
        w,
        h,
        fov_y_deg=55.0,
        convention="opencv",
        device=device,
        ssao_enabled=True,
        ssao_radius=0.08,
        contact_shadows_enabled=True,
        shadow_map_enabled=False,
        shadow_map_size=256,
        shadow_map_extent=2.2,
    )
    alb = None
    if names is not None and raycaster.n_meshes >= 2:
        alb = np.tile(np.array([0.65, 0.65, 0.65], dtype=np.float32), (raycaster.n_meshes, 1))
        alb[0] = (0.38, 0.38, 0.36)  # darker ground — white feet read on top
        alb[1] = (0.78, 0.28, 0.16)
    pbr.bind_meshes(raycaster, names=names, albedos=alb)
    rgb, depth, mask = pbr.render(cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat)
    _write_preview(out / "g1_pbr", rgb[0])
    print(
        f"g1 preview {w}x{h} meshes={raycaster.n_meshes} hit_frac={float(mask.float().mean()):.3f} "
        f"rgb_mean={float(rgb[mask].mean()) if bool(mask.any()) else 0.0:.3f} "
        f"wrote {out / 'g1_pbr.png'}"
    )
    del depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=None, help="Override all camera widths")
    parser.add_argument("--height", type=int, default=None, help="Override all camera heights")
    parser.add_argument(
        "--res",
        type=_parse_res,
        default=None,
        help="WxH shortcut (overrides --width/--height), e.g. 512x384",
    )
    parser.add_argument("--out", type=str, default="scripts/_pbr_smoke")
    args = parser.parse_args()

    if args.res is not None:
        width, height = args.res
    elif args.width is not None or args.height is not None:
        width = args.width
        height = args.height
        if width is None:
            width = max(16, round(height * 4 / 3))
        if height is None:
            height = max(16, round(width * 3 / 4))
        if width < 16 or height < 16:
            raise SystemExit(f"resolution {width}x{height} is below 16x16")
    else:
        width = height = None

    wp.init()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("warning: CPU Warp mesh raycast is slow; prefer CUDA")

    hdr_path = default_hdri_path()
    hdri = load_radiance_hdr(hdr_path)
    print(f"hdri {hdr_path.name} shape={hdri.shape} max={float(hdri.max()):.1f} mean={float(hdri.mean()):.3f}")
    if hdri.ndim != 3 or hdri.shape[-1] != 3 or hdri.shape[1] < 256:
        raise SystemExit(f"unexpected HDRI shape {hdri.shape}")
    if float(hdri.max()) < 2.0:
        raise SystemExit("HDRI looks clipped (max < 2); need unclipped .hdr not a JPG")
    if width is not None:
        print(f"resolution override {width}x{height} (all cameras)")
    else:
        print("resolution defaults: box 64x48, sphere 48x48, shadows 96x64, g1 128x96")

    out = Path(args.out)
    size_kw = dict(width=width, height=height)
    _assert_g1_materials()
    _assert_lambert_parity(device, out, **size_kw)
    _assert_vertex_colors(device, out, **size_kw)
    _assert_smooth_normals(device, **size_kw)
    _assert_contact_and_shadows(device, out, **size_kw)
    _maybe_g1_preview(device, out, **size_kw)
    print(f"wrote under {out}")
    print("ok")


if __name__ == "__main__":
    main()
