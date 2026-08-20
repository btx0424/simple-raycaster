"""Warp tiled screen-space global illumination (short-range color bleeding + AO)."""

from __future__ import annotations

import torch
import warp as wp

_TILE = 8
_PAD = 24
_HALO = _TILE + 2 * _PAD  # 56
_RAYS = 4
_STEPS = 8

TILE = wp.constant(_TILE)
PAD = wp.constant(_PAD)
HALO = wp.constant(_HALO)
RAYS = wp.constant(_RAYS)
STEPS = wp.constant(_STEPS)


@wp.kernel(enable_backward=False)
def ssgi_tiled_kernel(
    depth_in: wp.array3d(dtype=float),
    rad_r: wp.array3d(dtype=float),
    rad_g: wp.array3d(dtype=float),
    rad_b: wp.array3d(dtype=float),
    out_r: wp.array3d(dtype=float),
    out_g: wp.array3d(dtype=float),
    out_b: wp.array3d(dtype=float),
    out_ao: wp.array3d(dtype=float),
    height: int,
    width: int,
    fx: float,
    fy: float,
    radius_m: float,
    thickness: float,
    bias: float,
):
    """Per-tile SSGI: march screen rays in a shared depth/radiance halo."""
    n, ty, tx = wp.tid()
    y_base = ty * TILE
    x_base = tx * TILE
    d_h = wp.tile_load(depth_in[n], shape=(HALO, HALO), offset=(y_base, x_base), storage="shared")
    rr = wp.tile_load(rad_r[n], shape=(HALO, HALO), offset=(y_base, x_base), storage="shared")
    rg = wp.tile_load(rad_g[n], shape=(HALO, HALO), offset=(y_base, x_base), storage="shared")
    rb = wp.tile_load(rad_b[n], shape=(HALO, HALO), offset=(y_base, x_base), storage="shared")
    or_ = wp.tile_zeros(shape=(TILE, TILE), dtype=float, storage="shared")
    og = wp.tile_zeros(shape=(TILE, TILE), dtype=float, storage="shared")
    ob = wp.tile_zeros(shape=(TILE, TILE), dtype=float, storage="shared")
    oa = wp.tile_zeros(shape=(TILE, TILE), dtype=float, storage="shared")

    for ly in range(wp.static(_TILE)):
        for lx in range(wp.static(_TILE)):
            gy = y_base + ly
            gx = x_base + lx
            if gy >= height or gx >= width:
                or_[ly, lx] = 0.0
                og[ly, lx] = 0.0
                ob[ly, lx] = 0.0
                oa[ly, lx] = 1.0
                continue
            cy = ly + PAD
            cx = lx + PAD
            z0 = d_h[cy, cx]
            if z0 > 500.0 or z0 < 1.0e-4:
                or_[ly, lx] = 0.0
                og[ly, lx] = 0.0
                ob[ly, lx] = 0.0
                oa[ly, lx] = 1.0
                continue

            # Screen-space step scale (metres → pixels), clamped to halo budget.
            step_px = wp.clamp((radius_m * fx) / wp.max(z0, 1.0e-3) / float(STEPS), 1.0, float(PAD - 1))
            acc_r = float(0.0)
            acc_g = float(0.0)
            acc_b = float(0.0)
            hits = float(0.0)

            for ray_i in range(wp.static(_RAYS)):
                ang = 6.28318530718 * (float(ray_i) + 0.5) / float(RAYS)
                # Slight spiral so steps aren't collinear across rays.
                ang = ang + 0.35 * float(ly + lx)
                dir_x = wp.cos(ang)
                dir_y = wp.sin(ang)
                hit_r = float(0.0)
                hit_g = float(0.0)
                hit_b = float(0.0)
                hit = float(0.0)
                for s in range(1, wp.static(_STEPS) + 1):
                    t = step_px * float(s)
                    sx = float(cx) + dir_x * t
                    sy = float(cy) + dir_y * t
                    xi = wp.clamp(int(wp.round(sx)), 0, HALO - 1)
                    yi = wp.clamp(int(wp.round(sy)), 0, HALO - 1)
                    z_s = d_h[yi, xi]
                    if z_s > 500.0:
                        continue
                    # Expected depth along a roughly fronto-parallel march.
                    z_exp = z0
                    delta = z_exp - z_s
                    if delta > bias and delta < thickness:
                        # Occluder / bounce surface in front → gather its radiance.
                        fade = 1.0 - float(s) / float(STEPS)
                        hit_r = rr[yi, xi] * fade
                        hit_g = rg[yi, xi] * fade
                        hit_b = rb[yi, xi] * fade
                        hit = 1.0
                        break
                acc_r += hit_r
                acc_g += hit_g
                acc_b += hit_b
                hits += hit

            inv = 1.0 / float(RAYS)
            or_[ly, lx] = acc_r * inv
            og[ly, lx] = acc_g * inv
            ob[ly, lx] = acc_b * inv
            # Free AO-like term: more hits → darker (SSGI subsumes SSAO).
            oa[ly, lx] = wp.clamp(1.0 - 0.65 * hits * inv, 0.2, 1.0)

    wp.tile_store(out_r[n], or_, offset=(y_base, x_base))
    wp.tile_store(out_g[n], og, offset=(y_base, x_base))
    wp.tile_store(out_b[n], ob, offset=(y_base, x_base))
    wp.tile_store(out_ao[n], oa, offset=(y_base, x_base))


def _pad_nhw(t: torch.Tensor, pad: int) -> torch.Tensor:
    return torch.nn.functional.pad(t, (pad, pad, pad, pad), mode="replicate")


def ssgi_tiled(
    depth: torch.Tensor,
    mask: torch.Tensor,
    radiance: torch.Tensor,
    *,
    fx: float,
    fy: float,
    radius_m: float = 0.25,
    thickness: float = 0.08,
    bias: float = 0.004,
    intensity: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiled SSGI.

    Args:
        depth: planar depth ``[N,H,W]``
        mask: hit mask ``[N,H,W]``
        radiance: direct HDR RGB ``[N,H,W,3]`` (used as bounce source)

    Returns:
        ``(indirect_rgb [N,H,W,3], ao [N,H,W])`` — apply as
        ``hdr * ao + intensity * indirect`` (albedo already in radiance path
        via caller, or multiply indirect by albedo outside).
    """
    if radiance.ndim != 4 or radiance.shape[-1] != 3:
        raise ValueError(f"radiance expects [N,H,W,3], got {tuple(radiance.shape)}")
    n, h, w = depth.shape
    device = depth.device
    pad = _PAD
    depth_work = torch.where(
        mask, depth.float(), torch.full_like(depth, 1.0e3, dtype=torch.float32)
    )
    depth_p = _pad_nhw(depth_work.contiguous(), pad).contiguous()
    # Misses contribute black radiance.
    rad = torch.where(mask.unsqueeze(-1), radiance.float(), torch.zeros_like(radiance))
    r_p = _pad_nhw(rad[..., 0].contiguous(), pad).contiguous()
    g_p = _pad_nhw(rad[..., 1].contiguous(), pad).contiguous()
    b_p = _pad_nhw(rad[..., 2].contiguous(), pad).contiguous()

    ir = torch.empty(n, h, w, device=device, dtype=torch.float32)
    ig = torch.empty(n, h, w, device=device, dtype=torch.float32)
    ib = torch.empty(n, h, w, device=device, dtype=torch.float32)
    ao = torch.empty(n, h, w, device=device, dtype=torch.float32)

    wp.launch_tiled(
        ssgi_tiled_kernel,
        dim=(n, (h + _TILE - 1) // _TILE, (w + _TILE - 1) // _TILE),
        inputs=[
            wp.from_torch(depth_p, dtype=wp.float32),
            wp.from_torch(r_p, dtype=wp.float32),
            wp.from_torch(g_p, dtype=wp.float32),
            wp.from_torch(b_p, dtype=wp.float32),
            wp.from_torch(ir, dtype=wp.float32),
            wp.from_torch(ig, dtype=wp.float32),
            wp.from_torch(ib, dtype=wp.float32),
            wp.from_torch(ao, dtype=wp.float32),
            h,
            w,
            float(fx),
            float(fy),
            float(radius_m),
            float(thickness),
            float(bias),
        ],
        block_dim=64,
        device=str(wp.device_from_torch(device)),
    )
    wp.synchronize()
    indirect = torch.stack([ir, ig, ib], dim=-1) * float(intensity)
    ao = torch.where(mask, ao, torch.ones_like(ao))
    indirect = torch.where(mask.unsqueeze(-1), indirect, torch.zeros_like(indirect))
    return indirect, ao
