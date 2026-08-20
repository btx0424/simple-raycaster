"""Radiance HDRI load, SH irradiance, and lat-long sampling (Y-up maps, Z-up world)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT_HDRI = _ASSETS / "poly_haven_studio_1k.hdr"

# Ramamoorthi & Hanrahan cosine-lobe factors for irradiance SH.
_A0 = np.pi
_A1 = 2.0 * np.pi / 3.0
_A2 = np.pi / 4.0


def default_hdri_path() -> Path:
    if not DEFAULT_HDRI.is_file():
        raise FileNotFoundError(f"bundled HDRI missing: {DEFAULT_HDRI}")
    return DEFAULT_HDRI


def _rgbe_to_float(rgbe: np.ndarray) -> np.ndarray:
    rgb = rgbe[..., :3].astype(np.float32)
    exp = rgbe[..., 3].astype(np.int32)
    out = np.zeros_like(rgb)
    nz = exp > 0
    out[nz] = rgb[nz] * np.ldexp(1.0, exp[nz] - 128).astype(np.float32)[:, None] / 256.0
    return out


def load_radiance_hdr(path: str | Path) -> np.ndarray:
    """Load a Radiance RGBE ``.hdr`` to float32 RGB, shape ``[H, W, 3]``, linear."""
    path = Path(path)
    with path.open("rb") as f:
        header = b""
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"truncated HDR header: {path}")
            if line in (b"\n", b"\r\n"):
                break
            header += line
        res = f.readline().decode("ascii", errors="replace").strip()
        parts = res.split()
        if len(parts) != 4:
            raise ValueError(f"bad HDR resolution line {res!r} in {path}")
        height = int(parts[1])
        width = int(parts[3])
        y_sign = 1 if parts[0].startswith("+") else -1
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    n = width * height
    if raw.size == n * 4:
        rgbe = raw.reshape(height, width, 4)
    else:
        rgbe = _decode_rle(raw, width, height)
    rgb = _rgbe_to_float(rgbe)
    if y_sign > 0:
        rgb = np.flipud(rgb)
    return np.ascontiguousarray(rgb)


def _decode_rle(data: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.empty((height, width, 4), dtype=np.uint8)
    i = 0
    for y in range(height):
        if i + 4 > data.size:
            raise ValueError("truncated HDR RLE scanline")
        if data[i] != 2 or data[i + 1] != 2:
            rest = data[i:]
            need = (height - y) * width * 4
            if rest.size < need:
                raise ValueError("truncated uncompressed HDR")
            chunk = rest[:need].reshape((-1, width, 4))
            out[y:] = chunk
            return out
        scan_w = (int(data[i + 2]) << 8) | int(data[i + 3])
        if scan_w != width:
            raise ValueError(f"HDR scanline width {scan_w} != {width}")
        i += 4
        for ch in range(4):
            x = 0
            while x < width:
                if i >= data.size:
                    raise ValueError("truncated HDR RLE channel")
                count = int(data[i])
                i += 1
                if count > 128:
                    count -= 128
                    if i >= data.size:
                        raise ValueError("truncated HDR RLE run")
                    val = data[i]
                    i += 1
                    out[y, x : x + count, ch] = val
                    x += count
                else:
                    out[y, x : x + count, ch] = data[i : i + count]
                    i += count
                    x += count
    return out


def direction_to_latlong_uv(dir_yup: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, z = dir_yup.unbind(-1)
    phi = torch.atan2(x, z)
    theta = torch.acos(y.clamp(-1.0, 1.0))
    u = phi * (0.5 / torch.pi) + 0.5
    v = theta / torch.pi
    return u, v


def z_up_to_y_up(vec: torch.Tensor) -> torch.Tensor:
    """Robotics Z-up ``(x,y,z)`` → IBL Y-up ``(x,z,y)``."""
    x, y, z = vec.unbind(-1)
    return torch.stack([x, z, y], dim=-1)


def sample_latlong(hdri: torch.Tensor, dir_yup: torch.Tensor) -> torch.Tensor:
    """Bilinear sample ``hdri[H,W,3]`` with Y-up directions ``[..., 3]``."""
    d = dir_yup / dir_yup.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    orig = d.shape
    d = d.reshape(-1, 3)
    u, v = direction_to_latlong_uv(d)
    grid = torch.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], dim=-1).view(1, -1, 1, 2)
    img = hdri.permute(2, 0, 1).unsqueeze(0)
    samp = torch.nn.functional.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    rgb = samp[0, :, :, 0].permute(1, 0)
    return rgb.reshape(*orig[:-1], 3)


def project_sh9(hdri: np.ndarray) -> np.ndarray:
    """Project lat-long HDRI (Y-up) onto 9 SH RGB coefficients ``[9, 3]``."""
    h, w = hdri.shape[:2]
    ys = (np.arange(h, dtype=np.float64) + 0.5) / h
    xs = (np.arange(w, dtype=np.float64) + 0.5) / w
    theta = ys * np.pi
    phi = xs * 2.0 * np.pi - np.pi
    sin_t = np.sin(theta)
    # dω = sin(θ) dθ dφ ; texel solid angle
    dtheta = np.pi / h
    dphi = 2.0 * np.pi / w
    weight = (sin_t * dtheta * dphi)[:, None]

    ct, st = np.cos(theta), sin_t
    cp, sp = np.cos(phi), np.sin(phi)
    # Y-up: x=st*sp, y=ct, z=st*cp
    x = st[:, None] * sp[None, :]
    y = np.broadcast_to(ct[:, None], (h, w))
    z = st[:, None] * cp[None, :]

    y00 = 0.282095
    y1m1 = 0.488603 * y
    y10 = 0.488603 * z
    y11 = 0.488603 * x
    y2m2 = 1.092548 * x * y
    y2m1 = 1.092548 * y * z
    y20 = 0.315392 * (3.0 * z * z - 1.0)
    y21 = 1.092548 * x * z
    y22 = 0.546274 * (x * x - y * y)
    bands = [y00 * np.ones_like(x), y1m1, y10, y11, y2m2, y2m1, y20, y21, y22]

    rgb = hdri.astype(np.float64)
    sh = np.zeros((9, 3), dtype=np.float64)
    wgt = weight
    for i, ylm in enumerate(bands):
        sh[i] = (rgb * (ylm * wgt)[..., None]).sum(axis=(0, 1))
    return sh.astype(np.float32)


def eval_sh_irradiance(sh9: torch.Tensor, n_yup: torch.Tensor) -> torch.Tensor:
    """Diffuse irradiance from 9 SH coeffs. ``n_yup[...,3]``, ``sh9[9,3]``."""
    n = n_yup / n_yup.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = n.unbind(-1)
    c = sh9
    e = (
        _A0 * c[0] * 0.282095
        + _A1 * c[1] * (0.488603 * y).unsqueeze(-1)
        + _A1 * c[2] * (0.488603 * z).unsqueeze(-1)
        + _A1 * c[3] * (0.488603 * x).unsqueeze(-1)
        + _A2 * c[4] * (1.092548 * x * y).unsqueeze(-1)
        + _A2 * c[5] * (1.092548 * y * z).unsqueeze(-1)
        + _A2 * c[6] * (0.315392 * (3.0 * z * z - 1.0)).unsqueeze(-1)
        + _A2 * c[7] * (1.092548 * x * z).unsqueeze(-1)
        + _A2 * c[8] * (0.546274 * (x * x - y * y)).unsqueeze(-1)
    )
    return e.clamp_min(0.0)


def make_specular_mips(hdri: torch.Tensor, n_mips: int = 5) -> list[torch.Tensor]:
    """Successive 2× box downsamples as roughness lods (mip 0 = sharp)."""
    mips = [hdri.contiguous()]
    img = hdri.permute(2, 0, 1).unsqueeze(0)
    for _ in range(n_mips - 1):
        img = torch.nn.functional.avg_pool2d(img, kernel_size=2, stride=2, ceil_mode=False)
        if img.shape[-2] < 4 or img.shape[-1] < 8:
            break
        mips.append(img[0].permute(1, 2, 0).contiguous())
    return mips


class EnvironmentHDRI:
    """GPU HDRI ready for IBL (Z-up world normals/dirs)."""

    def __init__(self, hdri_hw3: torch.Tensor, *, n_mips: int = 5):
        if hdri_hw3.ndim != 3 or hdri_hw3.shape[-1] != 3:
            raise ValueError(f"expected [H,W,3] HDRI, got {tuple(hdri_hw3.shape)}")
        self.hdri = hdri_hw3.contiguous()
        sh = project_sh9(hdri_hw3.detach().float().cpu().numpy())
        self.sh9 = torch.as_tensor(sh, device=hdri_hw3.device, dtype=hdri_hw3.dtype)
        self.mips = make_specular_mips(self.hdri, n_mips=n_mips)

    @classmethod
    def from_path(cls, path: str | Path | None = None, *, device: str | torch.device = "cuda") -> "EnvironmentHDRI":
        path = default_hdri_path() if path is None else Path(path)
        rgb = load_radiance_hdr(path)
        t = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
        return cls(t)

    def sample(self, dir_zup: torch.Tensor, *, roughness: torch.Tensor | float = 0.0) -> torch.Tensor:
        d = z_up_to_y_up(dir_zup)
        if not torch.is_tensor(roughness):
            roughness = torch.full(
                dir_zup.shape[:-1], float(roughness), device=dir_zup.device, dtype=dir_zup.dtype
            )
        sharp = sample_latlong(self.mips[0], d)
        blur = sample_latlong(self.mips[-1], d)
        r = roughness.clamp(0.0, 1.0).unsqueeze(-1)
        return sharp * (1.0 - r) + blur * r

    def irradiance(self, n_zup: torch.Tensor) -> torch.Tensor:
        return eval_sh_irradiance(self.sh9, z_up_to_y_up(n_zup))
