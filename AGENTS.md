# Repository Guidelines

Agent-oriented guide for working in `simple-raycaster`. Also check the [Warp changelog](https://nvidia.github.io/warp/latest/user_guide/changelog.html) when upgrading `warp-lang` or adding kernel features.

## Project Structure

```
src/simple_raycaster/
  raycaster.py       # MultiMeshRaycaster (manual poses)
  raycaster_v2.py    # MultiMeshRaycasterV2 (Isaac entity poses)
  proximity.py       # MeshProximitySensor (closest-point / SDF-style queries)
  raycast_camera.py  # RaycastCamera — Lambert RGB-D megakernel (do not break for GS A/B)
  diffrast_camera.py # DiffrastCamera — nvdiffrast RGB-D (optional dep)
  mesh_cache.py      # CachedEntityMesh + simplify / pose helpers
  mesh_rgbd.py       # MeshRGBDRenderer factory (diffrast | raycast)
  kernels.py         # Warp raycast + fused transform + proximity + camera kernels
  helpers.py         # trimesh→wp conversion, quat math, voxelize_* utilities
  utils_usd.py       # USD prim search + trimesh extraction
  utils_mjc.py       # MuJoCo body → trimesh extraction
  pbr/               # Standalone PBR path (does not modify RaycastCamera)
    camera.py        # RaycastPBRCamera — G-buffer + Torch shade / compile
    ssgi_camera.py   # RaycastSSGICamera — tiled SSGI instead of SSAO
    ssgi.py          # Warp tiled SSGI kernel
    tiled_filters.py # Warp tiled FXAA / SSAO
    shade.py         # GGX, SSAO, FXAA, ACES, shadows (Torch)
    hdri.py          # EnvironmentHDRI + bundled maps under pbr/assets/
    kernels.py       # G-buffer / shadow-map / shadow-ray Warp kernels
    materials.py     # Per-mesh roughness/metallic + vertex packs
scripts/
  example.py / advanced.py
  g1_assets.py       # Resolve G1 MJCF (env / --xml / relative) — no machine paths
  smoke_pbr_camera.py, bench_pbr_camera.py, bench_tiled_filters.py, orbit_g1_video.py, …
_pbr_camera.md       # PBR design notes + bench rounds (not user-facing README)
_optimization.md     # Lidar / camera optimization plan
```

Public exports include raycasters, `RaycastCamera`, `RaycastPBRCamera`,
`RaycastSSGICamera`, `EnvironmentHDRI`, `DiffrastCamera`,
`make_mesh_rgbd_renderer`, and mesh-cache helpers. Optional: `pip install .[diffrast]`.

### Mesh RGB-D cameras

* Pinhole **RGB-D** mesh renderers for 3DGS composite in active-adaptation
  (`task.visual.mesh_renderer: diffrast|raycast`). AA only depth-composites.
* **Not** a replacement for AA’s MDP obs `extero.raycast_camera` — that stays on
  `MultiMeshRaycasterV2` for range sensing.
* Shared API shape: register / bind body-local meshes, then
  `render(cam_pos, cam_quat, …) → (rgb, depth, mask)` (OpenCV convention).
* Prefer `make_mesh_rgbd_renderer("diffrast"|"raycast")` for multi-entity use.
* **Hard rule:** do not change Lambert `RaycastCamera` / `raycast_camera_kernel`
  behavior when extending PBR — keep it for GS compositing A/B.

### PBR camera (`pbr/`)

* `quality="fast"` — GGX + HDRI + ACES + FXAA + **sun shadow rays**
  (second closest-hit cast; no SSAO / shadow map).
* `quality="pretty"` — SSAO + 128² sun shadow map (viewers / dumps; no sray).
  Do not stack sray + smap; if both are enabled, sray wins.
* Defaults (Round 7 @ 192×144): `compile_mode="max-autotune-no-cudagraphs"`,
  `fxaa_impl="torch"`, `ssao_impl="torch"` (compiled shade+SSAO+FXAA graph).
* No-compile path: `compile_mode=None, fxaa_impl="tiled"` (tiled SSAO still loses).
* `RaycastSSGICamera` overrides `_compose_indirect`; SSGI knobs stay on the subclass.
* Script default resolution is **192×144**. Headline RL batch is **N=256**.
* Optional UV albedo: `set_ground_albedo("forest_ground")` (planar XZ UVs +
  Poly Haven diffuse under `pbr/assets/textures/`). Flat table albedo when
  `mesh_tex_id[m] == -1`. Needs ImageMagick `convert`/`identify` to decode JPGs.

## Which Raycaster to Use

### `MultiMeshRaycaster` (`raycaster.py`)

* Construct with mesh list or `from_prim_paths` / `from_MjModel`.
* Caller supplies `mesh_pos_w` `[N, n_meshes, 3]` and `mesh_quat_w` `[N, n_meshes, 4]` on every `raycast` / `raycast_fused` call.
* Supports `add_mesh`, `add_from_path` for incremental setup.

### `MultiMeshRaycasterV2` (`raycaster_v2.py`)

* Construct with `device` only; register geometry via Isaac helpers:
  * `add_isaac_static(prim_path)` — one combined static mesh, identity pose
  * `add_isaac_entity(entity)` — one mesh per body, poses from `entity.data.body_link_pose_w`
* `raycast` / `raycast_fused` take **only rays** (plus optional `enabled`, `mesh_indices`).
* Internal helpers `_add_mesh`, `_add_from_path` are private; always pair mesh loading with an entity registration entry.
* `_validate_registration()` runs after each `add_isaac_*` and again in `initialize()` to ensure entity/mesh counts match.
* Batch size `N` is validated at raycast time against `entity.num_instances`.
* TODO in source: add non-Isaac backends — do not assume V2 works outside Isaac Sim today.

### `MeshProximitySensor` (`proximity.py`)

* Closest-point queries via Warp `mesh_query_point_no_sign` (or `sign_normal` when `signed=True`).
* Reuses `MultiMeshRaycasterV2` registration (`add_isaac_static` / `add_isaac_entity`), or accepts an existing raycaster via `meshes=` to share geometry with raycasting.
* `query(positions_w [N, n_points, 3], max_dist=...)` → `(closest_pos_w, distances)`.
* Same `enabled` / `mesh_indices` kwargs as the raycasters. Misses return `max_dist` with `closest_pos_w == positions_w`.
* Observation wiring (active-adaptation): build the sensor in an obs term `_initialize`, sample probe points each step, call `query`.

## Raycast Methods

Both raycaster classes implement the same two entry points:

| Method | Pipeline | When to prefer |
| --- | --- | --- |
| `raycast` | GPU PyTorch transform (`quat_rotate_inverse`) + Warp `raycast_kernel` | Parity checks, debugging kernel split |
| `raycast_fused` | Single Warp `transform_and_raycast_kernel` | Production / training loops |

Both paths expect CUDA tensors in normal use. The non-fused path is still fully on GPU; it just launches transform and raycast as separate steps instead of one fused kernel.

Shared kwargs (keyword-only after ray args):

* `enabled: Bool[N]` — disabled batches get `inf` distance before the min reduction
* `mesh_indices: Int[N, n_subset]` — index into the global mesh array per batch row

Returns `(hit_positions_w [N, n_rays, 3], hit_distances [N, n_rays], hit_normals_w [N, n_rays, 3])`. Closest hit across meshes is taken with `.min(dim=1)`; the normal of the winning mesh is gathered with the argmin indices. Normals are world-frame face normals (fused kernels rotate them in-kernel; the non-fused path rotates in Python). Misses and disabled batches get zero normals.

## Conventions

* **Quaternions**: WXYZ everywhere in Python; fused kernels reorder to XYZW for Warp internally.
* **Devices**: use `device="cuda"` and keep all tensors on GPU; `device="cpu"` exists only for special debugging and is impractically slow at batch scale.
* **Warp init**: call `wp.init()` once in the process before first `wp.launch`.
* **Mesh IDs**: `initialize()` builds `meshes_array` of `wp.uint64` mesh handles; any mesh add invalidates initialization (`initialized = False`).
* **Static meshes in V2**: geometry must already be in world frame in the trimesh data; runtime pose is identity.
* **G1 assets**: never hardcode machine paths. Use `scripts/g1_assets.py`
  (`SIMPLE_RAYCASTER_G1_XML`, `--xml`, or `assets/unitree_g1/…`).
* **Style**: 4-space indent, type hints on public APIs (`jaxtyping` shapes in docstrings), keep docstrings focused on non-obvious behavior.

## Common Pitfalls

1. **V2 mesh/entity mismatch** — calling internal add helpers without `add_isaac_*` leaves `entities` out of sync with `meshes_wp`.
2. **Prim path regex** — `add_isaac_entity` uses `entity.root_physx_view.prim_paths[0]` as a template; matched visual count must equal `entity.num_bodies`.
3. **USD import timing** — `from pxr import Usd` fails before Isaac `AppLauncher` unless standalone `usd-core` is installed.
4. **Empty raycast** — raycasting with zero registered meshes raises in `_get_mesh_poses_w`.
5. **Selective indices shape** — with `mesh_indices`, pose tensors and indices must share the same `[N, n_subset]` mesh dimension.
6. **Editing Lambert camera for PBR** — put PBR work under `pbr/`; leave `raycast_camera.py` kernels alone unless fixing a shared bug.

## Warp versioning

Declare `warp-lang>=1.7,<2` in `pyproject.toml`. **Do not** add a 1.x upper bound (`<1.13` etc.). Isaac Sim 5.1, mjlab, and Isaac Sim 6 ship different Warp 1.x releases in the same process as this library; an upper bound makes the package uninstallable in newer stacks. Host `pyproject.toml` files pin the exact Warp. Optional extras per Warp version are not useful — a venv can have only one `warp-lang`.

When adopting a Warp feature that needs a newer minimum (cuBQL BVH, graph capture, …), raise the **floor** (`>=1.13`) and note which host stacks that drops. Tiled FXAA/SSGI need Warp **1.16**-class `wp.launch_tiled` / shared tiles.

## Build & Validation Commands

```bash
pip install -e .

# Standalone demo (requires scene file)
python scripts/example.py --usd path/to/scene.usd
python scripts/example.py --mjcf path/to/scene.xml

# Kernel strategy benchmark (synthetic scene, no USD/MJCF file)
python scripts/advanced.py

# PBR smoke (skips G1 preview if MJCF unresolved)
python scripts/smoke_pbr_camera.py

# G1 benches / orbit (need SIMPLE_RAYCASTER_G1_XML or assets/)
python scripts/bench_pbr_camera.py
python scripts/orbit_g1_video.py --frames 36
```

There is no `tests/` suite yet. When changing kernels or pose gathering, run `scripts/advanced.py` for numerical parity and prefer adding a small reproducible script snippet to PR notes. For PBR, run `smoke_pbr_camera.py` and note PNG paths under `scripts/_pbr_smoke/` (gitignored).

## Making Changes Safely

* **Kernel edits** — update both fused and non-fused paths in `kernels.py`; keep `raycaster.py` and `raycaster_v2.py` call sites consistent.
* **V2 registration** — if adding a new registration path, append to `self.entities`, load meshes, then call `_validate_registration()`.
* **Performance** — prefer `raycast_fused` for fewer intermediate buffers; both methods run on GPU in normal use.
* **New mesh sources** — add extraction in `utils_*.py`, wire through `MultiMeshRaycaster` factories first; only add V2 support if poses can be sourced automatically.
* **PBR** — prefer subclassing / `_compose_indirect` hooks over growing `RaycastPBRCamera` kwargs; keep SSGI knobs on `RaycastSSGICamera`.

## Planned Improvements (Warp Changelog)

Backlog of performance work to resume later. Re-read the [Warp changelog](https://nvidia.github.io/warp/latest/user_guide/changelog.html) before starting — pin a minimum `warp-lang` version once a feature is adopted. Validate with `scripts/advanced.py` (parity + timing) and note Isaac Sim / Warp versions in the PR.

### Current bottlenecks

* Intermediate buffer `[N, n_meshes, n_rays]` plus PyTorch `.min(dim=1)` after every launch
* Per-step Python overhead: `wp.from_torch`, empty tensor alloc, V2 pose concat
* One separate `wp.Mesh` BVH per body; default BVH build settings in `trimesh2wp()`
* No CUDA graph capture for fixed-shape training loops

### Priority 1 — do first

- [ ] **Fuse cross-mesh min into the kernel** (Warp `wp.atomic_min`)
  - Launch `(N, n_rays)` (or keep mesh loop inside kernel) and write directly to `[N, n_rays]`
  - Removes ~`n_meshes×` memory and the post-launch PyTorch reduction
  - Touch: `kernels.py`, `raycaster.py`, `raycaster_v2.py`
  - Note: Warp 1.15+ changes float `wp.min` / `wp.atomic_min` NaN semantics (GH-1376)

- [ ] **Experimental cuBQL mesh BVH** (Warp 1.13+, GH-1286)
  - Add optional `bvh_constructor="cubql"` to `helpers.trimesh2wp()` / mesh factories
  - Only supports `wp.mesh_query_ray()` — matches our kernels
  - Benchmark vs default; keep fallback while experimental
  - Unreleased fix for cuBQL on devices without memory pools (GH-1430)

- [ ] **CUDA graph capture for training loops** (Warp 1.13+ graph serialization, unreleased `ScopedCapture` mode)
  - Preallocate ray / pose / hit buffers; capture `wp.launch` once, replay each sim step
  - Best fit: V2 with fixed `N`, `n_rays`, `n_meshes` in Isaac Lab
  - Use staged capture if buffer addresses change (`GraphMode.WARP_STAGED`, 1.11+)
  - `array.fill_()` graph composability fixed in 1.13 (GH-207 area)

### Priority 2 — tune after P1

- [ ] **`bvh_leaf_size` on `wp.Mesh`** (Warp 1.9+, GH-994)
  - Expose in `trimesh2wp()`; sweep in `advanced.py` (default is 4)

- [ ] **Launch tuning** (Warp 1.11+ / 1.13+)
  - `@wp.kernel(launch_bounds=...)` (GH-1049)
  - `wp.get_suggested_block_size()` for occupancy (GH-1270)
  - Launch dim is `(N, n_meshes, n_rays)` — occupancy may be poor when one axis is small

- [ ] **RMM / shared CUDA allocator with PyTorch** (Warp 1.13+, GH-781)
  - `wp.set_cuda_allocator(wp.utils.AllocatorRmm(...))` in Isaac Lab training
  - Reduces fragmentation when V2 gathers poses every step

### Priority 3 — larger refactors / niche paths

- [ ] **Group-aware BVH for multi-env Isaac** (Warp 1.11+, GH-1074 / GH-1097)
  - `wp.Mesh(..., groups=...)`, group-root queries via `wp.mesh_get_group_root()`
  - Could replace flat per-body mesh list in V2; needs registration redesign

- [ ] **`mesh_query_ray_anyhit()` fast path** (Warp 1.11+, GH-1097)
  - Only if we add binary hit/mask observations (not closest range)

- [ ] **Memory profiling while tuning** (Warp 1.13+, GH-1269)
  - `wp.ScopedMemoryTracker` — labels include `(native:bvh)`, `(native:mesh)`

### Not planned (poor fit today)

| Feature | Reason |
| --- | --- |
| Tile BVH queries (`mesh_query_aabb_tiled`, etc.) | Block-cooperative AABB iteration, not closest-hit range finding |
| Textures / `texture_sample` | No texture-based sensing in this repo |
| `mesh_query_ray_count_intersections` | Multi-hit / parity, not lidar range |
| Differentiable raycasting | All kernels use `enable_backward=False` |
| CPU device optimizations | Production path is CUDA-only |

### Suggested implementation order

```
1. atomic_min fused output  →  2. bvh_constructor=cubql benchmark
         ↓
3. CUDA graph capture (V2)  →  4. bvh_leaf_size + launch_bounds tuning
         ↓
5. grouped BVH redesign (optional)
```

When implementing, extend `scripts/advanced.py` to compare old vs new paths and require numerical parity within existing tolerance before merging.

## Commit & PR Notes

Keep commit subjects short and imperative (see repo history). PRs should list:

* which raycaster class / kernel path changed
* validation command run (`scripts/example.py`, `scripts/advanced.py`, or `scripts/smoke_pbr_camera.py`)
* Isaac Sim version if V2 behavior changed
* for PBR: Warp / Torch versions and whether compile defaults were exercised
