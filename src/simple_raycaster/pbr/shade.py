"""PyTorch PBR + SSAO + FXAA + ACES. Differentiable w.r.t. G-buffer/materials if they require grad."""

from __future__ import annotations

import math

import torch

from .hdri import EnvironmentHDRI

_PI = math.pi


def _fresnel_schlick(cos_theta: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    t = (1.0 - cos_theta.clamp(0.0, 1.0)).unsqueeze(-1)
    return f0 + (1.0 - f0) * t.pow(5.0)


def _d_ggx(n_dot_h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    a2 = a * a
    d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 / (_PI * d * d).clamp_min(1e-8)


def _g_smith(n_dot_v: torch.Tensor, n_dot_l: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    r = a + 1.0
    k = (r * r) / 8.0

    def g1(nd: torch.Tensor) -> torch.Tensor:
        return nd / (nd * (1.0 - k) + k).clamp_min(1e-8)

    return g1(n_dot_v) * g1(n_dot_l)


def shade_pbr(
    albedo: torch.Tensor,
    normal_w: torch.Tensor,
    view_w: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    env: EnvironmentHDRI,
    *,
    sun_dir_w: torch.Tensor,
    sun_intensity: float = 2.5,
    sun_color: tuple[float, float, float] = (1.0, 0.98, 0.92),
    sun_visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear HDR RGB. ``albedo/normal/view [... 3]``, ``roughness/metallic [...]``.

    ``sun_visibility`` (optional, ``[...]`` in ``[0,1]``) scales only the sun
    terms — IBL is unchanged. Contact shadows / shadow maps / shadow rays.
    """
    n = normal_w / normal_w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    v = view_w / view_w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    l = sun_dir_w / sun_dir_w.norm().clamp_min(1e-8)
    h = l + v
    h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    n_dot_l = (n * l).sum(-1).clamp(0.0, 1.0)
    if sun_visibility is not None:
        n_dot_l = n_dot_l * sun_visibility.clamp(0.0, 1.0)
    n_dot_v = (n * v).sum(-1).clamp(1e-4, 1.0)
    n_dot_h = (n * h).sum(-1).clamp(0.0, 1.0)
    v_dot_h = (v * h).sum(-1).clamp(0.0, 1.0)

    a = roughness.clamp(0.04, 1.0).pow(2)
    f0 = 0.04 * (1.0 - metallic.unsqueeze(-1)) + albedo * metallic.unsqueeze(-1)
    f = _fresnel_schlick(v_dot_h, f0)
    d = _d_ggx(n_dot_h, a)
    g = _g_smith(n_dot_v, n_dot_l, a)
    spec_sun = (d * g).unsqueeze(-1) * f / (4.0 * n_dot_v * n_dot_l).unsqueeze(-1).clamp_min(1e-4)
    spec_sun = spec_sun * n_dot_l.unsqueeze(-1)

    kd = (1.0 - f) * (1.0 - metallic.unsqueeze(-1))
    sun = sun_intensity * torch.tensor(sun_color, device=albedo.device, dtype=albedo.dtype)
    diff_sun = kd * albedo * (n_dot_l.unsqueeze(-1) / _PI)

    irr = env.irradiance(n)
    diff_ibl = kd * albedo * irr

    r = (2.0 * n_dot_v.unsqueeze(-1) * n - v)
    spec_env = env.sample(r, roughness=roughness)
    fresnel_v = _fresnel_schlick(n_dot_v, f0)
    spec_ibl = spec_env * (fresnel_v * (1.0 - roughness.unsqueeze(-1)) + 0.04)

    return diff_sun * sun + spec_sun * sun + diff_ibl + spec_ibl


def aces_tonemap(hdr: torch.Tensor, exposure: float = 1.0) -> torch.Tensor:
    x = hdr * float(exposure)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)


def fxaa(
    rgb: torch.Tensor,
    *,
    edge_threshold: float = 0.166,
    edge_threshold_min: float = 0.0833,
    subpix: float = 0.75,
) -> torch.Tensor:
    """FXAA on LDR RGB ``[N,H,W,3]`` (Lottes console-quality).

    Softens silhouette stairs after tonemap. Flat regions early-out. Uses
    padded slices + ``grid_sample`` (no extra mesh rays).
    """
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"fxaa expects [N,H,W,3], got {tuple(rgb.shape)}")

    # NCHW for pad / grid_sample.
    x = rgb.permute(0, 3, 1, 2).contiguous()
    n, _, h, w = x.shape
    luma = (0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]).contiguous()

    def neigh(img: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """``out[y,x] = img[y+dy, x+dx]`` with replicate borders."""
        pt, pb = max(-dy, 0), max(dy, 0)
        pl, pr = max(-dx, 0), max(dx, 0)
        padded = torch.nn.functional.pad(img, (pl, pr, pt, pb), mode="replicate")
        return padded[:, :, pt + dy : pt + dy + h, pl + dx : pl + dx + w]

    luma_nw = neigh(luma, -1, -1)
    luma_n = neigh(luma, -1, 0)
    luma_ne = neigh(luma, -1, 1)
    luma_w = neigh(luma, 0, -1)
    luma_m = luma
    luma_e = neigh(luma, 0, 1)
    luma_sw = neigh(luma, 1, -1)
    luma_s = neigh(luma, 1, 0)
    luma_se = neigh(luma, 1, 1)

    luma_min = luma_m
    for t in (luma_n, luma_s, luma_w, luma_e):
        luma_min = torch.minimum(luma_min, t)
    luma_max = luma_m
    for t in (luma_n, luma_s, luma_w, luma_e):
        luma_max = torch.maximum(luma_max, t)
    luma_range = luma_max - luma_min
    early = luma_range < torch.maximum(
        torch.full_like(luma_range, float(edge_threshold_min)),
        luma_max * float(edge_threshold),
    )

    dir_x = -((luma_nw + luma_ne) - (luma_sw + luma_se))
    dir_y = (luma_nw + luma_sw) - (luma_ne + luma_se)
    dir_reduce = torch.maximum(
        (luma_nw + luma_ne + luma_sw + luma_se) * (0.25 * 0.25),
        torch.full_like(luma_m, 1.0 / 128.0),
    )
    dir_min = 1.0 / (torch.minimum(dir_x.abs(), dir_y.abs()) + dir_reduce)
    dir_x = (dir_x * dir_min).clamp(-8.0, 8.0)
    dir_y = (dir_y * dir_min).clamp(-8.0, 8.0)

    ys = torch.arange(h, device=x.device, dtype=x.dtype)
    xs = torch.arange(w, device=x.device, dtype=x.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_x = (grid_x + 0.5) / float(w) * 2.0 - 1.0
    base_y = (grid_y + 0.5) / float(h) * 2.0 - 1.0
    base = torch.stack([base_x, base_y], dim=-1).unsqueeze(0).expand(n, -1, -1, -1)

    def sample_offset(ox: torch.Tensor, oy: torch.Tensor) -> torch.Tensor:
        gx = base[..., 0] + ox.squeeze(1) / float(w) * 2.0
        gy = base[..., 1] + oy.squeeze(1) / float(h) * 2.0
        grid = torch.stack([gx, gy], dim=-1)
        return torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

    rgb_a = 0.5 * (
        sample_offset(dir_x * 0.5, dir_y * 0.5) + sample_offset(dir_x * -0.5, dir_y * -0.5)
    )
    rgb_b = rgb_a * 0.5 + 0.25 * (
        sample_offset(dir_x, dir_y) + sample_offset(dir_x * -1.0, dir_y * -1.0)
    )

    luma_b = 0.2126 * rgb_b[:, 0:1] + 0.7152 * rgb_b[:, 1:2] + 0.0722 * rgb_b[:, 2:3]
    use_b = (luma_b >= luma_min) & (luma_b <= luma_max)
    filtered = torch.where(use_b, rgb_b, rgb_a)
    amount = (~early).float() * float(subpix)
    out = x * (1.0 - amount) + filtered * amount
    return out.permute(0, 2, 3, 1).contiguous()


def ssao(
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    n_samples: int = 4,
    radius: float = 0.08,
    bias: float = 0.004,
) -> torch.Tensor:
    """Screen-space AO from OpenCV *planar* depth. Returns ``[N,H,W]`` in ``[0,1]``.

    ``radius`` is metres in world/camera space. ``0.08`` fits G1 ankles and a
    cube sitting on the ground; ``0.12`` is a looser living-room scale.
    Default ``n_samples=4`` (was 8) — enough for pretty dumps; the RL path
    leaves SSAO off.
    """
    del cx, cy
    n, h, w = depth.shape
    device = depth.device
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth
    ang = torch.linspace(0.0, 2.0 * math.pi, n_samples + 1, device=device)[:-1]
    rad = torch.linspace(0.35, 1.0, n_samples, device=device)
    ao = torch.zeros(n, h, w, device=device, dtype=depth.dtype)
    bidx = torch.arange(n, device=device)[:, None, None]
    for s in range(n_samples):
        px = (radius * rad[s] * fx) / z.clamp_min(1e-3) * torch.cos(ang[s])
        py = (radius * rad[s] * fy) / z.clamp_min(1e-3) * torch.sin(ang[s])
        xi = (xs + px).round().long().clamp(0, w - 1).expand(n, -1, -1)
        yi = (ys + py).round().long().clamp(0, h - 1).expand(n, -1, -1)
        z2 = depth[bidx, yi, xi]
        closer = (z - z2) > bias
        fade = (1.0 - (z - z2).abs() / radius).clamp(0.0, 1.0)
        ao = ao + closer.float() * fade
    ao = (1.0 - ao / float(n_samples)).clamp(0.15, 1.0)
    return torch.where(mask, ao, torch.ones_like(ao))


def contact_shadows(
    depth: torch.Tensor,
    mask: torch.Tensor,
    sun_dir_cam: torch.Tensor,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    n_steps: int = 8,
    length: float = 0.08,
    thickness: float = 0.024,
    bias: float = 0.002,
    intensity: float = 0.85,
) -> torch.Tensor:
    """March along the projected sun in the depth image (no BVH).

    ``sun_dir_cam`` is the direction *toward* the sun in OpenCV camera space
    (``+Z`` forward, ``+Y`` down), shape ``[3]`` or ``[N,3]``. ``depth`` is
    planar camera-Z. Returns sun visibility ``[N,H,W]`` in ``[0,1]``.
    """
    n, h, w = depth.shape
    device = depth.device
    if sun_dir_cam.ndim == 1:
        l = sun_dir_cam.reshape(1, 1, 1, 3)
    else:
        l = sun_dir_cam.reshape(n, 1, 1, 3)
    l = l / l.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32) + 0.5,
        torch.arange(w, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    z = depth.clamp_min(1e-4)
    pos_x = (xs - cx) / fx * z
    pos_y = (ys - cy) / fy * z
    pos_z = z
    bidx = torch.arange(n, device=device)[:, None, None]
    shadow = torch.zeros(n, h, w, device=device, dtype=depth.dtype)
    step = float(length) / float(n_steps)
    for i in range(1, n_steps + 1):
        t = step * float(i)
        px = pos_x + l[..., 0] * t
        py = pos_y + l[..., 1] * t
        pz = pos_z + l[..., 2] * t
        sx = fx * px / pz.clamp_min(1e-4) + cx
        sy = fy * py / pz.clamp_min(1e-4) + cy
        inb = (sx >= 0.0) & (sx <= float(w - 1)) & (sy >= 0.0) & (sy <= float(h - 1)) & (pz > 1e-4)
        xi = sx.round().long().clamp(0, w - 1)
        yi = sy.round().long().clamp(0, h - 1)
        z_s = depth[bidx, yi, xi]
        delta = pz - z_s
        occ = (delta > bias) & (delta < thickness) & inb
        fade = 1.0 - (float(i) / float(n_steps))
        shadow = torch.maximum(shadow, occ.float() * fade)

    vis = (1.0 - float(intensity) * shadow).clamp(0.0, 1.0)
    return torch.where(mask, vis, torch.ones_like(vis))


def sample_shadow_map(
    hit_w: torch.Tensor,
    mask: torch.Tensor,
    sm_depth: torch.Tensor,
    *,
    center_w: torch.Tensor,
    right_w: torch.Tensor,
    up_w: torch.Tensor,
    forward_w: torch.Tensor,
    half_extent: float,
    catch_dist: float,
    bias: float = 0.012,
) -> torch.Tensor:
    """Compare reconstructed hits to an ortho sun depth map. ``sm_depth [N,S,S]``."""
    n, h, w = mask.shape
    del h, w
    size = sm_depth.shape[-1]
    rel = hit_w - center_w.view(n, 1, 1, 3)
    u = (rel * right_w.view(1, 1, 1, 3)).sum(-1) / float(half_extent)
    v = (rel * up_w.view(1, 1, 1, 3)).sum(-1) / float(half_extent)
    t = (rel * forward_w.view(1, 1, 1, 3)).sum(-1) + float(catch_dist)
    xi = ((u * 0.5 + 0.5) * float(size)).long()
    yi = ((v * 0.5 + 0.5) * float(size)).long()
    inb = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    xi = xi.clamp(0, size - 1)
    yi = yi.clamp(0, size - 1)
    bidx = torch.arange(n, device=mask.device)[:, None, None]
    sampled = sm_depth[bidx, yi, xi]
    vis = (t <= sampled + bias).float()
    vis = torch.where(inb, vis, torch.ones_like(vis))
    return torch.where(mask, vis, torch.ones_like(vis))
