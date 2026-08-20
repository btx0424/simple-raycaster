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

## Phase 0 — shippable viewer

1. G-buffer Warp kernel (`pbr/kernels.py`): per pixel `depth`, `mask`,
   world **face** normal (now optional), per-mesh **albedo**, roughness, metallic.
   No RGB in the kernel.
2. `pbr/hdri.py`: Radiance `.hdr` loader, 3-band SH irradiance, lat-long
   sample, a few roughness mips (downsample).
3. `pbr/shade.py` (PyTorch, differentiable w.r.t. materials/lights/SH if
   the G-buffer is detached):
   - GGX Cook–Torrance + 1 directional sun
   - Diffuse IBL from SH, specular IBL from mips
   - SSAO from planar depth (no extra mesh ray)
   - ACES tonemap + exposure
   - FXAA on LDR RGB (default on; no extra rays)
   - Misses sample the HDRI along the camera ray
4. Bundled test HDRI: `pbr/assets/poly_haven_studio_1k.hdr`
   (Poly Haven, CC0, 1K — enough for SH + IBL).
5. Smoke: `scripts/smoke_pbr_camera.py` vs Lambert on the same box.

## Phase 1 — look (still no extra rays)

- [x] Barycentric vertex colors + vertex normals (`result.face/u/v`).
      Packed `vert_colors` / `vert_normals` + `vert_offset[M]`.
      Fallback: constant per-mesh albedo when the trimesh has no authored colors.
- [x] Per-mesh roughness/metallic tables for G1 (`pbr/materials.py`):
      rubber ankles/feet, painted-metal links, plastic hands.
      Applied when `bind_*` is given `names=` (or `raycaster.mesh_names`).
- [x] `return_gbuffer=True` for real2sim (detach hits, optimize
      albedo/rough/metal + SH). Hits stay `enable_backward=False`.

## Phase 2 — cheap local occlusion

- [x] Screen-space **contact shadows** along the sun in the depth image
      (`contact_shadows_enabled=True` by default; sun term only).
- [x] SSAO radius default **0.08 m** (G1 ankles / cube on ground).
      `0.12` is still fine for a larger room.

## Phase 3 — optional visibility (pretty mode only)

- [x] Sun **shadow map**: directional ortho, default **256²**, all bound
      meshes. Off by default (`shadow_map_enabled=False`).
- [x] True shadow ray toward the sun: debug knob, default **off**
      (`shadow_rays=False`).

## Real2sim (later)

Do **not** enable Warp `enable_backward` on closest-hit. Freeze G-buffer,
optimize shade parameters in PyTorch. See discussion in-thread:
materials/SH/exposure yes; poses/mesh assignment no.

## Files

```
src/simple_raycaster/pbr/
  __init__.py          # RaycastPBRCamera, default_hdri_path, G1 materials
  camera.py
  kernels.py           # g-buffer + ortho shadow map + shadow-ray debug
  hdri.py
  shade.py             # PBR, SSAO, contact shadows, shadow-map compare
  materials.py         # G1 roughness/metallic + vertex pack
  assets/poly_haven_studio_1k.hdr
  assets/README.md
scripts/smoke_pbr_camera.py
scripts/bench_pbr_camera.py
_pbr_camera.md         # this file
```

`RaycastCamera`, `kernels.raycast_camera_kernel`, and
`RaycastMeshRenderer` stay Lambert.

## G1 camera-count bench (RTX 4090)

Scene matches `scripts/bench_g1_camera.py`: ground + cube + G1 Inspire
(61 meshes, 271k verts, 542k faces), OpenCV 128×96, fov 70°, `far=10`.
Warp 1.6.0, torch 2.8.0+cu128. Depth max|Δ| vs Lambert = 0.

```
.venv/bin/python scripts/bench_pbr_camera.py --n 1,8,16,64,256
.venv/bin/python scripts/bench_pbr_camera.py --shadow-map --shadow-rays
```

| N | lambert | gbuffer | shade | **default** | +smap | +sray |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.47 | 0.45 | 1.94 | **6.06** | 6.02 | 2.60 |
| 8 | 0.45 | 0.43 | 1.93 | **5.29** | 6.28 | 2.41 |
| 16 | 0.44 | 0.43 | 1.96 | **5.37** | 6.51 | 2.72 |
| 64 | 0.93 | 0.94 | 2.57 | **5.88** | 9.19 | 3.74 |
| 128 | 1.66 | 1.74 | 4.97 | **8.42** | — | — |
| 256 | 2.72 | 2.82 | 11.88 | **19.61** | 31.42 | 14.75 |
| 512 | 5.32 | 5.36 | 29.56 | **51.17** | — | — |
| 1024 | 10.28 | 10.43 | 63.68 | **120.93** | — | — |

Times are ms/iter. **default** = GGX + HDRI + SSAO + contact shadows.
**shade** = GGX + HDRI only. **+smap** is default plus the 256² sun
shadow map. **+sray** is shade plus a closest-hit sun ray (no SSAO /
contact, so the extra ray is visible). Peak torch mem for default is
~367 MB at N=64 and ~4.8 GB at N=1024.

Takeaways:

- Closest-hit is not the extra cost. `render_gbuffer` tracks Lambert
  within a few percent at every N (both ~1.2 Gpix/s once the GPU is full).
- Small batches (N≤16) sit on a ~0.45 ms trace floor and a ~5–6 ms
  PyTorch shade/SSAO floor. Default PBR is ~12× Lambert there because
  SSAO + contact are many small tensor ops, not extra mesh rays.
- At a typical train batch (N=64, 128×96) default PBR is **5.9 ms**
  (~6× Lambert 0.93 ms) and still ~11k cameras/s. Drop SSAO/contact
  (`pbr_shade`) and that falls to **2.6 ms**.
- Past N=256, SSAO/contact scale worse than GGX (104 Mpix/s vs 198 at
  N=1024). The 256² shadow map is cheap at N=1 and +12 ms at N=256.
  A true shadow ray is about one extra closest-hit (~+3 ms at N=256).
