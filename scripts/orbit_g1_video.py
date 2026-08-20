"""Orbit the G1 scene and write depth / Lambert / PBR videos.

    .venv/bin/python scripts/orbit_g1_video.py
    .venv/bin/python scripts/orbit_g1_video.py --res 512x384 --frames 90 --hdri venice_sunset
    .venv/bin/python scripts/orbit_g1_video.py --width 256 --height 192 --hdri studio_small

Outputs under ``scripts/_orbit_g1/`` (gitignored):
  orbit_depth.mp4, orbit_lambert.mp4, orbit_pbr.mp4, orbit_side_by_side.mp4
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

from simple_raycaster import RaycastCamera, RaycastPBRCamera
from simple_raycaster.pbr.hdri import BUNDLED_HDRIS, EnvironmentHDRI, resolve_hdri_path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from bench_g1_camera import (  # noqa: E402
    DEFAULT_G1_XML,
    expand_poses,
    load_scene,
    look_at_opencv_wxyz,
)


def _parse_res(raw: str) -> tuple[int, int]:
    parts = raw.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected WxH, got {raw!r}")
    width, height = int(parts[0]), int(parts[1])
    if width < 16 or height < 16:
        raise argparse.ArgumentTypeError(f"resolution {width}x{height} is below 16x16")
    return width, height


def _even(n: int) -> int:
    """libx264 wants even frame dimensions."""
    return n if n % 2 == 0 else n + 1


def _as_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    img = (rgb.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    return np.ascontiguousarray(img)


def _depth_to_rgb(
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    near: float,
    far: float,
) -> torch.Tensor:
    d = depth.detach().float()
    m = mask.detach().bool()
    out = torch.zeros(*d.shape, 3, device=d.device, dtype=torch.float32)
    if not bool(m.any()):
        return out
    lo, hi = float(near), float(far)
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    norm = ((d - lo) / (hi - lo)).clamp(0.0, 1.0)
    gray = torch.where(m, 1.0 - norm, torch.zeros_like(norm))
    return gray.unsqueeze(-1).expand_as(out).contiguous()


def _orbit_eyes(
    n_frames: int,
    *,
    radius: float,
    elev_deg: float,
    target: np.ndarray,
    azim0_deg: float = 0.0,
) -> np.ndarray:
    """Eyes on a sphere around ``target`` (Z-up). ``elev_deg`` from horizon."""
    elev = math.radians(float(elev_deg))
    az0 = math.radians(float(azim0_deg))
    azims = az0 + np.linspace(0.0, 2.0 * math.pi, n_frames, endpoint=False)
    ce, se = math.cos(elev), math.sin(elev)
    eyes = np.empty((n_frames, 3), dtype=np.float64)
    tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
    for i, a in enumerate(azims):
        eyes[i, 0] = tx + radius * ce * math.cos(a)
        eyes[i, 1] = ty + radius * ce * math.sin(a)
        eyes[i, 2] = tz + radius * se
    return eyes


class _FFmpegWriter:
    """Pipe RGB24 frames into ffmpeg → H.264 mp4."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.width = int(width)
        self.height = int(height)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_u8: np.ndarray) -> None:
        if frame_u8.shape != (self.height, self.width, 3):
            raise ValueError(
                f"frame shape {frame_u8.shape} != ({self.height}, {self.width}, 3)"
            )
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame_u8).tobytes())

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        rc = self._proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed ({rc}) writing {self.path}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=str, default=DEFAULT_G1_XML)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--res",
        type=_parse_res,
        default=None,
        help="WxH shortcut (overrides --width/--height), e.g. 512x384",
    )
    parser.add_argument("--fov", type=float, default=55.0)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--radius", type=float, default=2.2)
    parser.add_argument("--elev", type=float, default=18.0, help="Orbit elevation degrees")
    parser.add_argument("--azim0", type=float, default=-40.0, help="Start azimuth degrees")
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.65),
        metavar=("X", "Y", "Z"),
        help="Look-at point (world)",
    )
    parser.add_argument(
        "--hdri",
        type=str,
        default="studio",
        help=f"Bundled alias or path. Aliases: {sorted(BUNDLED_HDRIS)}",
    )
    parser.add_argument(
        "--quality",
        choices=("fast", "pretty"),
        default="pretty",
        help="PBR quality preset (pretty = SSAO + 128² shadow map)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=str, default="scripts/_orbit_g1")
    parser.add_argument(
        "--no-side-by-side",
        action="store_true",
        help="Skip the concatenated depth|lambert|pbr video",
    )
    args = parser.parse_args()

    if args.frames < 2:
        raise SystemExit("--frames must be >= 2")
    if args.res is not None:
        width, height = args.res
    else:
        width = 256 if args.width is None else int(args.width)
        height = 192 if args.height is None else int(args.height)
    width, height = _even(width), _even(height)
    if width < 16 or height < 16:
        raise SystemExit(f"resolution {width}x{height} is below 16x16")

    hdri_path = resolve_hdri_path(args.hdri)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wp.init()
    device = args.device
    torch_device = torch.device(device)
    raycaster, g1_pos, g1_quat, stats = load_scene(args.xml, device)
    n_static = int(stats["n_static"])
    names = list(getattr(raycaster, "mesh_names", []) or [])
    if len(names) != raycaster.n_meshes:
        names = ["ground", "cube", *list(stats["g1_names"])][: raycaster.n_meshes]
    albedos = np.asarray(stats["albedos"], dtype=np.float32)

    mesh_pos, mesh_quat = expand_poses(1, g1_pos, g1_quat, n_static, robot_first=False)
    target = np.asarray(args.target, dtype=np.float64)
    eyes = _orbit_eyes(
        args.frames,
        radius=args.radius,
        elev_deg=args.elev,
        target=target,
        azim0_deg=args.azim0,
    )
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    quats = np.stack(
        [look_at_opencv_wxyz(eyes[i], target, up) for i in range(args.frames)],
        axis=0,
    ).astype(np.float32)

    env = EnvironmentHDRI.from_path(hdri_path, device=torch_device)
    cam_kw = dict(
        width=width,
        height=height,
        fov_y_deg=args.fov,
        near=args.near,
        far=args.far,
        convention="opencv",
        depth_mode="planar",
        device=device,
    )
    lambert = RaycastCamera(**cam_kw)
    lambert.bind_meshes(raycaster)
    pbr = RaycastPBRCamera(**cam_kw, hdri=env, quality=args.quality, shadow_map_extent=2.2)
    pbr.bind_meshes(raycaster, names=names, albedos=albedos)

    w_depth = _FFmpegWriter(out / "orbit_depth.mp4", width, height, args.fps)
    w_lambert = _FFmpegWriter(out / "orbit_lambert.mp4", width, height, args.fps)
    w_pbr = _FFmpegWriter(out / "orbit_pbr.mp4", width, height, args.fps)
    w_sbs = None
    if not args.no_side_by_side:
        w_sbs = _FFmpegWriter(
            out / "orbit_side_by_side.mp4", width * 3, height, args.fps
        )

    print(
        f"orbit {args.frames} frames @ {width}x{height}  fps={args.fps}  "
        f"hdri={hdri_path.name}  quality={args.quality}  out={out}"
    )
    pose_kw = dict(mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat)
    for i in range(args.frames):
        cam_pos = torch.as_tensor(eyes[i : i + 1], device=torch_device, dtype=torch.float32)
        cam_quat = torch.as_tensor(quats[i : i + 1], device=torch_device, dtype=torch.float32)
        rgb_l, depth_l, mask_l = lambert.render(cam_pos, cam_quat, **pose_kw)
        rgb_p, depth_p, mask_p = pbr.render(cam_pos, cam_quat, **pose_kw)
        if not torch.equal(mask_l, mask_p):
            raise SystemExit(f"frame {i}: Lambert/PBR mask mismatch")
        depth_rgb = _depth_to_rgb(depth_l[0], mask_l[0], near=args.near, far=args.far)
        f_d = _as_u8_rgb(depth_rgb)
        f_l = _as_u8_rgb(rgb_l[0])
        f_p = _as_u8_rgb(rgb_p[0])
        w_depth.write(f_d)
        w_lambert.write(f_l)
        w_pbr.write(f_p)
        if w_sbs is not None:
            w_sbs.write(np.concatenate([f_d, f_l, f_p], axis=1))
        if (i + 1) % max(1, args.frames // 8) == 0 or i + 1 == args.frames:
            print(f"  frame {i + 1}/{args.frames}")

    w_depth.close()
    w_lambert.close()
    w_pbr.close()
    if w_sbs is not None:
        w_sbs.close()
    print(
        f"wrote {out / 'orbit_depth.mp4'} {out / 'orbit_lambert.mp4'} "
        f"{out / 'orbit_pbr.mp4'}"
        + (f" {out / 'orbit_side_by_side.mp4'}" if w_sbs is not None else "")
    )
    print("ok")


if __name__ == "__main__":
    main()
