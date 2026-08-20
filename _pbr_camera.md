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

## Phase 2 — cheap local occlusion (pretty mode)

- [x] Screen-space **contact shadows** along the sun in the depth image
      (sun term only). **Opt-in only** — not a good shadow proxy (see Round 2).
- [x] SSAO radius default **0.08 m**, **4 taps**. Pretty-only.

## Phase 3 — optional visibility

- [x] Sun **shadow map**: directional ortho, default **128²** (was 256).
      On in ``quality="pretty"``; off in ``fast``.
- [x] True shadow ray toward the sun: debug / GT knob
      (`shadow_rays=False` by default).

## Quality presets

| `quality` | SSAO | shadow map | contact | FXAA | Use |
| --- | --- | --- | --- | --- | --- |
| `"fast"` (default) | off | off | off | on | parallel RL / train RGB |
| `"pretty"` | on (4-tap) | **128²** | off | on | dumps / viewer |

Explicit `ssao_enabled=` / `shadow_map_enabled=` / `contact_shadows_enabled=`
still override the preset. Prefer **shadow rays** over stacking contact+map
when you need GT-quality visibility at similar cost (see Round 2).

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
scripts/probe_pbr_opt.py   # shadow fidelity + CUDA graph probes
_pbr_camera.md         # this file
```

`RaycastCamera`, `kernels.raycast_camera_kernel`, and
`RaycastMeshRenderer` stay Lambert.

## G1 camera-count bench (RTX 4090)

Scene matches `scripts/bench_g1_camera.py`: ground + cube + G1 Inspire
(61 meshes, standing free-joint, 3 cm foot clearance), OpenCV 128×96,
fov 70°, `far=10`. Warp 1.6.0, torch 2.8.0+cu128. Depth max|Δ| vs Lambert = 0.

**Headline batch is N=256** (parallel RL). Re-run:

```
.venv/bin/python scripts/bench_pbr_camera.py              # N=256 fast
.venv/bin/python scripts/bench_pbr_camera.py --pretty
.venv/bin/python scripts/probe_pbr_opt.py --json-out /tmp/probe_pbr_opt.json
```

### Round 1 (shade_fast default) — N=256

| path | ms/iter | ms/cam | cam/s | vs Lambert | peak |
| --- | ---: | ---: | ---: | ---: | --- |
| lambert | 5.54 | 0.022 | 46.2k | 1.00× | 85 MB |
| pbr_gbuffer | 5.56 | 0.022 | 46.0k | 1.00× | 232 MB |
| pbr_nofxaa | 14.65 | 0.057 | 17.5k | 2.65× | 1.07 GB |
| **pbr_fast** (default) | **17.40** | **0.068** | **14.7k** | **3.14×** | 1.22 GB |
| pbr_pretty (SSAO+contact) | 23.51 | 0.092 | 10.9k | 4.25× | 1.36 GB |

`pbr_fast` = GGX + HDRI + ACES + FXAA. Round-1 `pretty` was SSAO + contact.
Gating SSAO/contact saved **~6 ms** at N=256. FXAA alone is ~2.7 ms.

### Round 2 — shadows + graphs (N=256)

Probe: `scripts/probe_pbr_opt.py`. Shadow-ray visibility is the reference.

| method | +ms vs none | dark IoU vs sray | corr |
| --- | ---: | ---: | ---: |
| contact | +4.4 | 0.08 | 0.11 |
| **smap128** | **+5.5** | **0.78** | **0.87** |
| smap256 | +15.1 | 0.87 | 0.93 |
| **sray** | **+4.9** | **1.00** | **1.00** |
| contact+smap256 | +19.4 | 0.27 | 0.60 |

**Shadows verdict:** use **128² shadow map** for pretty / viewer (default now),
or **shadow rays** when you want GT visibility at similar cost to smap128 at
this res/N. Skip contact-as-shadow; do not stack contact+map.

**Graphs verdict** (after fixing host `torch.tensor` inside `shade_pbr`):

| capture | speedup @ N=256 |
| --- | ---: |
| Warp graph on G-buffer | ~1.01× |
| Torch CUDAGraph shade+FXAA | ~1.02× |
| Torch CUDAGraph FXAA only | ~1.03× |

Not worth productizing at the RL headline batch — launch overhead is already
amortized. (Earlier lidar notes: graphs help small N ~1.1×, ~0% at large N.)

Round-2 `pretty` = SSAO + **smap128** (refreshed bench):

| path | ms/iter | vs Lambert | peak |
| --- | ---: | ---: | --- |
| pbr_fast | 17.65 | 3.23× | 1.22 GB |
| **pbr_pretty** (SSAO+smap128) | **24.42** | **4.47×** | 1.41 GB |

Similar cost to old SSAO+contact pretty (~23.5), with real shadows.

### Next iteration candidates

- FXAA opt-in / pretty-only if RL does not need silhouette AA (~2.7 ms @256).
- Workspace / mem for G-buffer at N≥256 (fast peaks ~1.2 GB).
- Shadow rays as optional pretty mode when light moves every frame (no map rebuild).
