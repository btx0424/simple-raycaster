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
4. Bundled test HDRIs (Poly Haven CC0, 1K):
   `poly_haven_studio_1k.hdr` (default), `studio_small_09_1k.hdr`,
   `venice_sunset_1k.hdr`. Aliases: ``resolve_hdri_path("studio"|...)``.
5. Smoke: `scripts/smoke_pbr_camera.py` vs Lambert on the same box.
   Orbit video: `scripts/orbit_g1_video.py` (depth / Lambert / PBR).

## Phase 1 — look (still no extra rays)

- [x] Barycentric vertex colors + vertex normals (`result.face/u/v`).
      Packed `vert_colors` / `vert_normals` + `vert_offset[M]`.
      Fallback: constant per-mesh albedo when the trimesh has no authored colors.
- [x] Per-mesh roughness/metallic tables for G1 (`pbr/materials.py`):
      rubber ankles/feet, painted-metal links, plastic hands.
      Applied when `bind_*` is given `names=` (or `raycaster.mesh_names`).
- [x] **MuJoCo `geom_rgba` → albedo** (`utils_mjc.body_mesh_albedo`,
      `MultiMeshRaycaster.mesh_albedos`). STLs have no vertex colors; appearance
      lives on the geom. `bind_meshes` picks this up when `albedos=` is omitted.
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
fov 70°, `far=10`. **Warp 1.16.0**, torch 2.8.0+cu128. Depth max|Δ| vs
Lambert = 0. (Rounds 1–3 below were recorded on Warp **1.6.0**; Round 4
re-benches the same scripts after upgrading.)

**Headline batch is N=256** (parallel RL). Re-run:

```
.venv/bin/python scripts/bench_pbr_camera.py              # N=256 fast
.venv/bin/python scripts/bench_pbr_camera.py --pretty
.venv/bin/python scripts/probe_pbr_opt.py --json-out /tmp/probe_pbr_opt.json
.venv/bin/python scripts/probe_pbr_compile.py --json-out /tmp/probe_compile.json
```

### Round 4 — Warp 1.16.0 re-bench (N=256)

Same scene / scripts as Round 1–3. Warp raycast got clearly faster; Torch
shade is unchanged (as expected).

| path | ms/iter (1.16) | was (1.6) | Δ |
| --- | ---: | ---: | ---: |
| lambert | **3.56** | 5.54 | −36% |
| pbr_gbuffer | **3.59** | 5.56 | −35% |
| pbr_nofxaa | 12.63 | 14.65 | −14% |
| **pbr_fast** | **15.48** | 17.40 | −11% |
| pbr_pretty (SSAO+smap128) | **21.79** | 24.42 | −11% |

vs Lambert rose to **4.35×** for pbr_fast because Lambert sped up more than
shade. Absolute pbr_fast is still better (~15.5 vs ~17.4 ms).

Shadows (+ms over none, fidelity unchanged):

| method | +ms (1.16) | was (1.6) | IoU vs sray |
| --- | ---: | ---: | ---: |
| contact | +4.2 | +4.4 | 0.08 |
| smap128 | +4.5 | +5.5 | 0.78 |
| smap256 | +12.8 | +15.1 | 0.87 |
| **sray** | **+3.0** | +4.9 | 1.00 |

Graphs still ~1.02×. ``torch.compile`` shade numbers match Round 3
(~1.56× shade+FXAA; ~6× shade no-FXAA). Rough full-frame with compile:
G-buffer ~3.6 + compiled shade+FXAA ~7.6 ≈ **~11 ms**.

### Round 1 (shade_fast default, Warp 1.6) — N=256

| path | ms/iter | ms/cam | cam/s | vs Lambert | peak |
| --- | ---: | ---: | ---: | ---: | --- |
| lambert | 5.54 | 0.022 | 46.2k | 1.00× | 85 MB |
| pbr_gbuffer | 5.56 | 0.022 | 46.0k | 1.00× | 232 MB |
| pbr_nofxaa | 14.65 | 0.057 | 17.5k | 2.65× | 1.07 GB |
| **pbr_fast** (default) | **17.40** | **0.068** | **14.7k** | **3.14×** | 1.22 GB |
| pbr_pretty (SSAO+contact) | 23.51 | 0.092 | 10.9k | 4.25× | 1.36 GB |

`pbr_fast` = GGX + HDRI + ACES + FXAA. Round-1 `pretty` was SSAO + contact.
Gating SSAO/contact saved **~6 ms** at N=256. FXAA alone is ~2.7 ms.

### Round 2 — shadows + graphs (Warp 1.6, N=256)

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

Round-2 `pretty` = SSAO + **smap128** (refreshed bench on Warp 1.6):

| path | ms/iter | vs Lambert | peak |
| --- | ---: | ---: | --- |
| pbr_fast | 17.65 | 3.23× | 1.22 GB |
| **pbr_pretty** (SSAO+smap128) | **24.42** | **4.47×** | 1.41 GB |

Similar cost to old SSAO+contact pretty (~23.5), with real shadows.

### Round 3 — ``torch.compile`` on shade (Warp 1.6 / still valid on 1.16)

Probe: `scripts/probe_pbr_compile.py` (frozen G-buffer; Warp out of the
compiled region). Eager shade+FXAA **11.78 ms**; shade without FXAA **8.97 ms**.

| mode | shade+FXAA | vs eager | shade no-FXAA | vs eager |
| --- | ---: | ---: | ---: | ---: |
| default | 7.85 | 1.50× | 1.85 | 4.86× |
| reduce-overhead | 8.20 | 1.44× | 2.20 | 4.08× |
| max-autotune | 7.90 | 1.49× | 1.85 | 4.85× |
| **max-autotune-no-cudagraphs** | **7.55** | **1.56×** | **1.47** | **6.12×** |

MAE vs eager ~2e-7. Compile+warmup ~9–52 s (max-autotune longest).

**Compile verdict:** worth it for the Torch shade path. Best mode here is
``max-autotune-no-cudagraphs``. FXAA still dominates after compile (~6 ms of
the 7.6 ms stack); shade+IBL alone collapses to ~1.5 ms. On Warp 1.16,
rough full-frame with compile: G-buffer ~3.6 + compiled shade+FXAA ~7.6 ≈
**~11 ms**, or **~5 ms** if FXAA is off (~1.4× Lambert).

Not productized in `RaycastPBRCamera` yet — wire as an optional
``compile_shade=True`` / mode knob when integrating.

### Round 5 — Warp tiled FXAA / SSAO (Warp 1.16, N=256)

Kernels in `pbr/tiled_filters.py` (`wp.launch_tiled` + shared halo tiles).
Probe: `scripts/bench_tiled_filters.py`.

| filter | torch ms | tiled ms | speedup | mae |
| --- | ---: | ---: | ---: | ---: |
| **FXAA** | 2.89 | **1.72** | **1.68×** | ~2e-8 |
| SSAO (4-tap) | **1.70** | 3.90 | 0.44× | ~1e-10 |

Full frame:

| path | ms/iter |
| --- | ---: |
| pbr_fast + torch FXAA | 15.30 |
| **pbr_fast + tiled FXAA** | **14.17** |
| pbr_fast no FXAA | 12.51 |
| pretty torch FXAA+SSAO | **21.61** |
| pretty tiled FXAA+SSAO | 22.67 |

**Verdict:** default ``fxaa_impl="tiled"``. Keep ``ssao_impl="torch"`` —
tiled SSAO matches numerically but the TILE×TILE serial loops inside each
block lose to Torch’s vectorized 4-tap path at this resolution. Use
``ssao_impl="tiled"`` only to exercise / extend the tile path.

### Round 6 — Tiled SSGI camera (replaces SSAO)

``RaycastSSGICamera`` subclasses ``RaycastPBRCamera`` and overrides
``_compose_indirect`` with Warp tiled SSGI (`pbr/ssgi.py`: shared halo,
4 rays × 8 steps → short-range color bleed + AO-like term). SSAO knobs stay
on the base camera; SSGI knobs (`ssgi_radius` / `thickness` / `intensity` /
`bias`) live only on the subclass.

Smoke writes ``g1_ssgi.png`` next to ``g1_pbr.png`` (same pose / meshes /
shadow map). This is short-range screen-space GI, not full SSR + denoise.

Quick N=1 @ 128×96 (cached kernels): pretty+SSAO ~3.5 ms, pretty+SSGI ~3.0 ms
(SSGI is in the same ballpark; tile halo fits the short march well).

### Round 7 — Default res 192×144; torch.compile vs Warp tiled

Script defaults (`bench_*`, `probe_*`, `smoke_pbr_camera`, `orbit_g1_video`)
are now **192×144**. Probe: `scripts/bench_tiled_filters.py` (N=256, Warp 1.16,
torch 2.8, `max-autotune-no-cudagraphs`).

Isolated filters:

| filter | torch eager | torch.compile | tiled | best |
| --- | ---: | ---: | ---: | --- |
| **FXAA** | 8.71 | **3.13** (2.78×) | 3.92 (2.22×) | compile ≳ tiled |
| **SSAO** | 5.78 | **0.09** (~60×) | 8.75 (0.66×) | compile ≫ torch; tiled loses |

Full frame (ms/iter):

| path | ms |
| --- | ---: |
| pbr_fast + torch FXAA | 43.85 |
| **pbr_fast + tiled FXAA** | **39.03** |
| pbr_fast no FXAA | 35.12 |
| pretty torch FXAA+SSAO | 57.49 |
| pretty all tiled | 55.59 |
| **pretty tiled FXAA + torch SSAO** | **52.66** |

Frozen G-buffer shade+FXAA:

| path | ms |
| --- | ---: |
| eager torch FXAA | 36.37 |
| eager tiled FXAA | 31.52 |
| **compile shade+torch FXAA** | **17.73** (2.05× eager) |

**Verdict @ 192×144:** default camera config is now Round-7 best —
``compile_mode="max-autotune-no-cudagraphs"`` with ``fxaa_impl="torch"`` /
``ssao_impl="torch"`` (compiled shade+SSAO+FXAA graph). Pass
``compile_mode=None, fxaa_impl="tiled"`` for the no-compile full-frame path
(tiled FXAA still beats eager Torch FXAA; tiled SSAO still loses).

### Next iteration candidates

- SIMT Warp SSAO (not tiled) if pretty AO must beat Torch eager without compile.
- FXAA opt-in / pretty-only if RL does not need silhouette AA.
- Warp tile-friendly shade only after a Warp shade megakernel.
- Workspace / mem for G-buffer at N≥256 (fast peaks ~1.2 GB).
- Longer SSGI / better normal-aware thickness / denoise if bleed looks noisy.
- Compile shade alone for ``RaycastSSGICamera`` (SSGI stays outside the graph).
