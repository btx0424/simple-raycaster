"""Smoke test RaycastCamera: unit box in front of an OpenCV pinhole camera."""

from __future__ import annotations

import torch
import trimesh
import warp as wp

from simple_raycaster import RaycastCamera


def main() -> None:
    wp.init()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("warning: CPU Warp mesh raycast is slow; prefer CUDA")

    box = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    cam = RaycastCamera(64, 48, fov_y_deg=70.0, convention="opencv", device=device)
    cam.bind_trimeshes([box], albedos=[(0.9, 0.2, 0.15)])

    # Camera at origin looking +Z; box centered at z=2
    n = 2
    cam_pos = torch.zeros(n, 3, device=device)
    cam_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4).contiguous()
    mesh_pos = torch.tensor([[[0.0, 0.0, 2.0]]], device=device).expand(n, 1, 3).contiguous()
    mesh_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device).expand(n, 1, 4).contiguous()

    rgb, depth, mask = cam.render(
        cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat
    )
    assert rgb.shape == (n, 48, 64, 3)
    assert depth.shape == (n, 48, 64)
    assert mask.shape == (n, 48, 64)
    hit_frac = float(mask.float().mean())
    center_d = float(depth[0, 24, 32])
    print(f"device={device} hit_frac={hit_frac:.3f} center_depth={center_d:.3f}")
    print(f"rgb_mean={float(rgb[mask].mean()) if mask.any() else 0:.3f}")
    if not bool(mask[0, 24, 32]):
        raise SystemExit("smoke failed: expected a hit at the image center")
    # ~0.5 m box at z≈2 with 70° FOV covers only a few percent of pixels.
    if hit_frac < 0.01:
        raise SystemExit("smoke failed: expected box to cover some FOV pixels")
    if not (1.6 < center_d < 1.9):
        raise SystemExit(f"smoke failed: unexpected center depth {center_d} (want ~1.75)")
    print("ok")


if __name__ == "__main__":
    main()
