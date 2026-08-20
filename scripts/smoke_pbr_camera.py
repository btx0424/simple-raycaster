"""Smoke: Lambert ``RaycastCamera`` vs new ``RaycastPBRCamera`` (untouched Lambert path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp

from simple_raycaster import RaycastCamera, RaycastPBRCamera, default_hdri_path
from simple_raycaster.pbr.hdri import load_radiance_hdr


def _write_ppm(path: Path, rgb: torch.Tensor) -> None:
    img = (rgb.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    h, w = img.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.ascontiguousarray(img).tobytes())


def main() -> None:
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

    box = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    lambert = RaycastCamera(64, 48, fov_y_deg=70.0, convention="opencv", device=device)
    lambert.bind_trimeshes([box], albedos=[(0.9, 0.2, 0.15)])
    pbr = RaycastPBRCamera(
        64,
        48,
        fov_y_deg=70.0,
        convention="opencv",
        device=device,
        default_roughness=0.35,
        default_metallic=0.15,
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
    rgb_p, depth_p, mask_p = pbr.render(
        cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat
    )

    if rgb_p.shape != rgb_l.shape:
        raise SystemExit(f"shape mismatch {tuple(rgb_p.shape)} vs {tuple(rgb_l.shape)}")
    if not torch.equal(mask_l, mask_p):
        raise SystemExit("mask mismatch vs Lambert camera")
    ddiff = (depth_l - depth_p).abs().max().item()
    if ddiff > 1e-4:
        raise SystemExit(f"depth mismatch vs Lambert max={ddiff}")
    if not bool(mask_p[0, 24, 32]):
        raise SystemExit("expected a hit at the image center")

    out = Path("scripts/_pbr_smoke")
    _write_ppm(out / "lambert.ppm", rgb_l[0])
    _write_ppm(out / "pbr.ppm", rgb_p[0])
    print(
        f"device={device} hit_frac={float(mask_p.float().mean()):.3f} "
        f"depth_center={float(depth_p[0, 24, 32]):.3f} depth_max_abs={ddiff:.2e}"
    )
    print(f"lambert_rgb_mean={float(rgb_l[mask_l].mean()):.3f} pbr_rgb_mean={float(rgb_p[mask_p].mean()):.3f}")
    print(f"wrote {out / 'lambert.ppm'} and {out / 'pbr.ppm'}")
    print("ok")


if __name__ == "__main__":
    main()
