"""Warp tiled FXAA + SSAO (shared-memory tiles over planar float maps)."""

from __future__ import annotations

import torch
import warp as wp

_TILE = 8
_FXAA_PAD = 8
_FXAA_HALO = _TILE + 2 * _FXAA_PAD  # 24
_SSAO_PAD = 16
_SSAO_HALO = _TILE + 2 * _SSAO_PAD  # 40

TILE = wp.constant(_TILE)
FXAA_PAD = wp.constant(_FXAA_PAD)
FXAA_HALO = wp.constant(_FXAA_HALO)
SSAO_PAD = wp.constant(_SSAO_PAD)
SSAO_HALO = wp.constant(_SSAO_HALO)


@wp.func
def _luma3(r: float, g: float, b: float) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@wp.func
def _bilerp(
    img: wp.tile(dtype=float, shape=(_FXAA_HALO, _FXAA_HALO)),
    y: float,
    x: float,
) -> float:
    y0 = int(wp.floor(y))
    x0 = int(wp.floor(x))
    y0c = wp.clamp(y0, 0, FXAA_HALO - 1)
    x0c = wp.clamp(x0, 0, FXAA_HALO - 1)
    y1c = wp.clamp(y0 + 1, 0, FXAA_HALO - 1)
    x1c = wp.clamp(x0 + 1, 0, FXAA_HALO - 1)
    fy = y - float(y0)
    fx = x - float(x0)
    c00 = img[y0c, x0c]
    c01 = img[y0c, x1c]
    c10 = img[y1c, x0c]
    c11 = img[y1c, x1c]
    return (c00 * (1.0 - fx) + c01 * fx) * (1.0 - fy) + (c10 * (1.0 - fx) + c11 * fx) * fy


@wp.kernel(enable_backward=False)
def fxaa_tiled_kernel(
    r_in: wp.array3d(dtype=float),
    g_in: wp.array3d(dtype=float),
    b_in: wp.array3d(dtype=float),
    r_out: wp.array3d(dtype=float),
    g_out: wp.array3d(dtype=float),
    b_out: wp.array3d(dtype=float),
    height: int,
    width: int,
    edge_threshold: float,
    edge_threshold_min: float,
    subpix: float,
):
    n, ty, tx = wp.tid()
    y_base = ty * TILE
    x_base = tx * TILE
    hr = wp.tile_load(r_in[n], shape=(FXAA_HALO, FXAA_HALO), offset=(y_base, x_base), storage="shared")
    hg = wp.tile_load(g_in[n], shape=(FXAA_HALO, FXAA_HALO), offset=(y_base, x_base), storage="shared")
    hb = wp.tile_load(b_in[n], shape=(FXAA_HALO, FXAA_HALO), offset=(y_base, x_base), storage="shared")
    or_ = wp.tile_zeros(shape=(TILE, TILE), dtype=float, storage="shared")
    og = wp.tile_zeros(shape=(TILE, TILE), dtype=float, storage="shared")
    ob = wp.tile_zeros(shape=(TILE, TILE), dtype=float, storage="shared")

    for ly in range(wp.static(_TILE)):
        for lx in range(wp.static(_TILE)):
            gy = y_base + ly
            gx = x_base + lx
            cy = ly + FXAA_PAD
            cx = lx + FXAA_PAD
            if gy >= height or gx >= width:
                or_[ly, lx] = 0.0
                og[ly, lx] = 0.0
                ob[ly, lx] = 0.0
                continue

            rm = hr[cy, cx]
            gm = hg[cy, cx]
            bm = hb[cy, cx]
            luma_nw = _luma3(hr[cy - 1, cx - 1], hg[cy - 1, cx - 1], hb[cy - 1, cx - 1])
            luma_n = _luma3(hr[cy - 1, cx], hg[cy - 1, cx], hb[cy - 1, cx])
            luma_ne = _luma3(hr[cy - 1, cx + 1], hg[cy - 1, cx + 1], hb[cy - 1, cx + 1])
            luma_w = _luma3(hr[cy, cx - 1], hg[cy, cx - 1], hb[cy, cx - 1])
            luma_m = _luma3(rm, gm, bm)
            luma_e = _luma3(hr[cy, cx + 1], hg[cy, cx + 1], hb[cy, cx + 1])
            luma_sw = _luma3(hr[cy + 1, cx - 1], hg[cy + 1, cx - 1], hb[cy + 1, cx - 1])
            luma_s = _luma3(hr[cy + 1, cx], hg[cy + 1, cx], hb[cy + 1, cx])
            luma_se = _luma3(hr[cy + 1, cx + 1], hg[cy + 1, cx + 1], hb[cy + 1, cx + 1])

            luma_min = wp.min(wp.min(wp.min(wp.min(luma_m, luma_n), luma_s), luma_w), luma_e)
            luma_max = wp.max(wp.max(wp.max(wp.max(luma_m, luma_n), luma_s), luma_w), luma_e)
            luma_range = luma_max - luma_min
            if luma_range < wp.max(edge_threshold_min, luma_max * edge_threshold):
                or_[ly, lx] = rm
                og[ly, lx] = gm
                ob[ly, lx] = bm
                continue

            dir_x = -((luma_nw + luma_ne) - (luma_sw + luma_se))
            dir_y = (luma_nw + luma_sw) - (luma_ne + luma_se)
            dir_reduce = wp.max((luma_nw + luma_ne + luma_sw + luma_se) * 0.0625, 1.0 / 128.0)
            dir_min = 1.0 / (wp.min(wp.abs(dir_x), wp.abs(dir_y)) + dir_reduce)
            dir_x = wp.clamp(dir_x * dir_min, -8.0, 8.0)
            dir_y = wp.clamp(dir_y * dir_min, -8.0, 8.0)

            fx = float(cx)
            fy = float(cy)
            ra = 0.5 * (_bilerp(hr, fy + dir_y * 0.5, fx + dir_x * 0.5) + _bilerp(hr, fy - dir_y * 0.5, fx - dir_x * 0.5))
            ga = 0.5 * (_bilerp(hg, fy + dir_y * 0.5, fx + dir_x * 0.5) + _bilerp(hg, fy - dir_y * 0.5, fx - dir_x * 0.5))
            ba = 0.5 * (_bilerp(hb, fy + dir_y * 0.5, fx + dir_x * 0.5) + _bilerp(hb, fy - dir_y * 0.5, fx - dir_x * 0.5))
            rb = ra * 0.5 + 0.25 * (_bilerp(hr, fy + dir_y, fx + dir_x) + _bilerp(hr, fy - dir_y, fx - dir_x))
            gb_ = ga * 0.5 + 0.25 * (_bilerp(hg, fy + dir_y, fx + dir_x) + _bilerp(hg, fy - dir_y, fx - dir_x))
            bb = ba * 0.5 + 0.25 * (_bilerp(hb, fy + dir_y, fx + dir_x) + _bilerp(hb, fy - dir_y, fx - dir_x))
            luma_b = _luma3(rb, gb_, bb)
            fr = ra
            fg = ga
            fb = ba
            if luma_b >= luma_min and luma_b <= luma_max:
                fr = rb
                fg = gb_
                fb = bb
            or_[ly, lx] = rm * (1.0 - subpix) + fr * subpix
            og[ly, lx] = gm * (1.0 - subpix) + fg * subpix
            ob[ly, lx] = bm * (1.0 - subpix) + fb * subpix

    wp.tile_store(r_out[n], or_, offset=(y_base, x_base))
    wp.tile_store(g_out[n], og, offset=(y_base, x_base))
    wp.tile_store(b_out[n], ob, offset=(y_base, x_base))


@wp.kernel(enable_backward=False)
def ssao_tiled_kernel(
    depth_in: wp.array3d(dtype=float),
    ao_out: wp.array3d(dtype=float),
    height: int,
    width: int,
    fx: float,
    fy: float,
    radius: float,
    bias: float,
    n_samples: int,
):
    n, ty, tx = wp.tid()
    y_base = ty * TILE
    x_base = tx * TILE
    depth_halo = wp.tile_load(
        depth_in[n],
        shape=(SSAO_HALO, SSAO_HALO),
        offset=(y_base, x_base),
        storage="shared",
    )
    out = wp.tile_zeros(shape=(TILE, TILE), dtype=float, storage="shared")

    for ly in range(wp.static(_TILE)):
        for lx in range(wp.static(_TILE)):
            gy = y_base + ly
            gx = x_base + lx
            if gy >= height or gx >= width:
                out[ly, lx] = 1.0
                continue
            cy = ly + SSAO_PAD
            cx = lx + SSAO_PAD
            z = depth_halo[cy, cx]
            if z > 500.0:
                # miss sentinel from wrapper
                out[ly, lx] = 1.0
                continue
            ao = float(0.0)
            denom = float(wp.max(n_samples - 1, 1))
            for s in range(n_samples):
                ang = 6.28318530718 * float(s) / float(n_samples)
                rad = 0.35 + 0.65 * float(s) / denom
                px = (radius * rad * fx) / wp.max(z, 1.0e-3) * wp.cos(ang)
                py = (radius * rad * fy) / wp.max(z, 1.0e-3) * wp.sin(ang)
                xi = wp.clamp(int(wp.round(float(cx) + px)), 0, SSAO_HALO - 1)
                yi = wp.clamp(int(wp.round(float(cy) + py)), 0, SSAO_HALO - 1)
                z2 = depth_halo[yi, xi]
                closer = float(0.0)
                if (z - z2) > bias:
                    closer = 1.0
                fade = 1.0 - wp.clamp(wp.abs(z - z2) / radius, 0.0, 1.0)
                ao += closer * fade
            out[ly, lx] = wp.clamp(1.0 - ao / float(n_samples), 0.15, 1.0)

    wp.tile_store(ao_out[n], out, offset=(y_base, x_base))


def _pad_nhw(t: torch.Tensor, pad: int) -> torch.Tensor:
    return torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="replicate")


def fxaa_tiled(
    rgb: torch.Tensor,
    *,
    edge_threshold: float = 0.166,
    edge_threshold_min: float = 0.0833,
    subpix: float = 0.75,
) -> torch.Tensor:
    """Warp tiled FXAA on LDR ``[N,H,W,3]``."""
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"fxaa_tiled expects [N,H,W,3], got {tuple(rgb.shape)}")
    n, h, w, _ = rgb.shape
    device = rgb.device
    pad = _FXAA_PAD
    rgb_c = rgb.detach().float().contiguous()
    r = _pad_nhw(rgb_c[..., 0], pad).contiguous()
    g = _pad_nhw(rgb_c[..., 1], pad).contiguous()
    b = _pad_nhw(rgb_c[..., 2], pad).contiguous()
    out = torch.empty(n, h, w, 3, device=device, dtype=torch.float32)
    r_o = torch.empty(n, h, w, device=device, dtype=torch.float32)
    g_o = torch.empty(n, h, w, device=device, dtype=torch.float32)
    b_o = torch.empty(n, h, w, device=device, dtype=torch.float32)

    wp.launch_tiled(
        fxaa_tiled_kernel,
        dim=(n, (h + _TILE - 1) // _TILE, (w + _TILE - 1) // _TILE),
        inputs=[
            wp.from_torch(r, dtype=wp.float32),
            wp.from_torch(g, dtype=wp.float32),
            wp.from_torch(b, dtype=wp.float32),
            wp.from_torch(r_o, dtype=wp.float32),
            wp.from_torch(g_o, dtype=wp.float32),
            wp.from_torch(b_o, dtype=wp.float32),
            h,
            w,
            float(edge_threshold),
            float(edge_threshold_min),
            float(subpix),
        ],
        block_dim=64,
        device=str(wp.device_from_torch(device)),
    )
    wp.synchronize()
    return torch.stack([r_o, g_o, b_o], dim=-1)


def ssao_tiled(
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    fx: float,
    fy: float,
    cx: float = 0.0,
    cy: float = 0.0,
    n_samples: int = 4,
    radius: float = 0.08,
    bias: float = 0.004,
) -> torch.Tensor:
    """Warp tiled SSAO. ``depth/mask`` are ``[N,H,W]``."""
    del cx, cy
    n, h, w = depth.shape
    device = depth.device
    pad = _SSAO_PAD
    depth_work = torch.where(mask, depth.float(), torch.full_like(depth, 1.0e3, dtype=torch.float32))
    depth_p = _pad_nhw(depth_work.contiguous(), pad).contiguous()
    ao = torch.empty(n, h, w, device=device, dtype=torch.float32)
    wp.launch_tiled(
        ssao_tiled_kernel,
        dim=(n, (h + _TILE - 1) // _TILE, (w + _TILE - 1) // _TILE),
        inputs=[
            wp.from_torch(depth_p, dtype=wp.float32),
            wp.from_torch(ao, dtype=wp.float32),
            h,
            w,
            float(fx),
            float(fy),
            float(radius),
            float(bias),
            int(n_samples),
        ],
        block_dim=64,
        device=str(wp.device_from_torch(device)),
    )
    wp.synchronize()
    return torch.where(mask, ao, torch.ones_like(ao))
