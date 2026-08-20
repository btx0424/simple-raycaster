# Standalone PBR camera (separate from Lambert `RaycastCamera`)

Working plan for a **second** RGB-D path: closest-hit G-buffer + PyTorch
PBR / HDRI / SSAO. The existing `RaycastCamera` (Lambert megakernel) is
**not** modified; keep it for GS compositing and A/B comparison.

## Why a new class

| Path | Class | Role |
| --- | --- | --- |
| Composite / GS overlay | `RaycastCamera` | Depth + mask (+ cheap Lambert RGB). Unchanged. |
| Standalone (no GS) | `RaycastPBRCamera` | Same hits, shaded with PBR + HDRI + SSAO. |

Shared geometry: both can `bind_meshes(raycaster)` / `bind_trimeshes(...)`.
Hits should match Lambert depth at `atol=1e-4` (same BVH, same tmax shrink).

## Phase 0 (this change) — shippable viewer

1. G-buffer Warp kernel (`pbr/kernels.py`): per pixel `depth`, `mask`,
   world **face** normal, per-mesh **albedo**, roughness, metallic.
   No RGB in the kernel.
2. `pbr/hdri.py`: Radiance `.hdr` loader, 3-band SH irradiance, lat-long
   sample, a few roughness mips (downsample).
3. `pbr/shade.py` (PyTorch, differentiable w.r.t. materials/lights/SH if
   the G-buffer is detached):
   - GGX Cook–Torrance + 1 directional sun
   - Diffuse IBL from SH, specular IBL from mips
   - SSAO from planar depth (no extra mesh ray)
   - ACES tonemap + exposure
   - Misses sample the HDRI along the camera ray
4. Bundled test HDRI: `pbr/assets/poly_haven_studio_1k.hdr`
   (Poly Haven, CC0, 1K — enough for SH + IBL).
5. Smoke: `scripts/smoke_pbr_camera.py` vs Lambert on the same box.

**Out of Phase 0:** vertex-color lerp, smooth normals, shadow rays,
shadow maps, contact-shadow march, CUDA graphs on the shade pass.

## Phase 1 — look (still no extra rays)

- Barycentric vertex colors + vertex normals (`result.face/u/v`).
- Per-mesh roughness/metallic tables for G1 (metal links vs rubber feet).
- Optional `return_gbuffer=True` for real2sim (detach hits, optimize
  albedo/rough/metal + SH).

## Phase 2 — cheap local occlusion

- Screen-space **contact shadows** along the sun in the depth image.
- Tune SSAO radius in metres for G1 (ankles / cube on ground).

## Phase 3 — optional visibility (pretty mode only)

- Sun **shadow map** (depth-only pass, static meshes, ~256²), not a
  second full-resolution camera ray.
- True shadow ray remains a debug knob.

## Real2sim (later)

Do **not** enable Warp `enable_backward` on closest-hit. Freeze G-buffer,
optimize shade parameters in PyTorch. See discussion in-thread:
materials/SH/exposure yes; poses/mesh assignment no.

## Files

```
src/simple_raycaster/pbr/
  __init__.py          # RaycastPBRCamera, default_hdri_path
  camera.py
  kernels.py           # raycast_gbuffer_kernel only
  hdri.py
  shade.py
  assets/poly_haven_studio_1k.hdr
  assets/README.md
scripts/smoke_pbr_camera.py
_pbr_camera.md         # this file
```

`RaycastCamera`, `kernels.raycast_camera_kernel`, and
`RaycastMeshRenderer` stay Lambert.
