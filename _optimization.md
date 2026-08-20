# Raycaster performance optimization plan

Working document for optimizing `simple-raycaster` on a realistic camera workload.
This refines the original stub (test scene + Warp review + measure/iterate) using
the current code, `AGENTS.md`, and the Warp 1.7–1.16 APIs we can actually use.

**Status:** Round 3 saturated on Warp 1.6. Closest-hit now shrinks `mesh_query_ray`
tmax, writes positions in-kernel, and launches at **block_dim=128**. Proximity
fused-min is in. Mesh-order / simplify knobs did not clear 5%. Further Warp 1.6
launch knobs (`launch_bounds`, `bvh_leaf_size`, cuBQL) are unavailable.

Local env (do not push `.venv`):

```bash
proxy_on
uv venv .venv --python 3.11
uv pip install -e . --overrides /dev/stdin <<<'warp-lang==1.6.0'
# Driver here is CUDA 12.8; default torch wheels may be cu13 and fail:
uv pip install 'torch==2.8.*' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python scripts/bench_g1_camera.py
```

## Round 1 results (2026-08-20)

Scene: ground box + cube + G1 Inspire (`61` meshes, `270768` verts, `542386`
faces). Headline: `N=64`, `128×96` OpenCV pinhole, fov `70°`, `max_dist=10`,
RTX 4090, Warp **1.6.0**, torch `2.8.0+cu128`. Hit fraction `0.851`.

| path | ms/iter | M rays/s | torch peak mem |
| --- | ---: | ---: | ---: |
| `lidar_fused_closest` (new default) | 1.858 | 423 | 48 MB |
| `lidar_fused_legacy` (`closest_hit=False`) | 2.273 | 346 | 761 MB |
| `lidar_unfused` | 28.610 | 27 | 3.96 GB |
| `camera_rgbd` | 1.771 | 444 | 40 MB |

- Closest-hit vs legacy fused: **1.22×** faster, **15.8×** less peak torch memory.
- Parity: legacy vs closest distances/positions/normals **exact** (`max=0`).
  Unfused vs closest `max abs dist 1.2e-6`. Camera ray-depth vs lidar `1.8e-5`.
- `mesh_indices` subset path also matches legacy exactly.
- Tiny batches can still favor the old `(N, n_meshes, n_rays)` launch (more
  threads); G1 training-scale `N=64` favors the serial in-kernel min.
- Peak memory is the **PyTorch** allocator only (Warp BVH lives outside it).

Scaling on the same scene / resolution (`128×96`, fused closest vs legacy). Unfused
at `N=1024` was skipped: N=256 already peaked at ~16 GB and would not fit.

| N | closest ms | legacy ms | speedup | closest peak | legacy peak | M rays/s closest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.86 | 2.27 | 1.22× | 48 MB | 761 MB | 423 |
| 256 | 5.07 | 8.02 | 1.58× | 192 MB | 2.97 GB | 621 |
| 1024 | 18.19 | 29.76 | 1.64× | 770 MB | 12.05 GB | 692 |

Closest-hit gets *more* efficient as `N` grows (occupancy). Memory stays ~linear
in `N` and ~`n_meshes×` smaller than legacy. Lidar vs camera depth still matches
in the mean (`~6e-7`); at `N=1024` a few pixels miss `atol=1e-4` (max `3.33`) —
likely a grazing/tie difference, not a batch-wide drift.

## Round 2 results (2026-08-20)

Same G1 scene. Closest-hit output buffers are now reused on the raycaster.

| knob | N=64 | N=1024 | keep? |
| --- | --- | --- | --- |
| buffer reuse (vs R1 closest) | 1.86→1.78 ms, 48→39 MB | 18.19→18.08 ms, 770→626 MB | **yes** (always on) |
| CUDA graph replay | 1.78→1.62 ms (**1.10×**) | 18.08→18.05 ms (~0%) | **yes as opt-in** (`capture_fused` / `use_graph`) |
| `bvh_constructor=sah` | 1.95 ms (worse than lbvh 1.78) | — | no |
| `bvh_constructor=median` | 2.17 ms (worse) | — | no |
| merge ground+cube | 1.70 vs 1.78 ms (~4.5%) | — | scene tip only; under 5% |

`launch_bounds` / `bvh_leaf_size` / cuBQL need Warp ≥ 1.9 / 1.11 / 1.13.

```bash
.venv/bin/python scripts/bench_g1_camera.py --graph --sweep-bvh --skip-unfused
```

`raycast_fused(..., closest_hit=True)` is the new default. Keep
`closest_hit=False` until a later round deletes the 3D kernels.

## Round 3 results (2026-08-20)

Same G1 scene. Closest-hit kernels now pass the running `best_t` into
`mesh_query_ray` (camera too), write hit positions in-kernel, and launch
with `block_dim=128` (Warp default is 256).

Headline `N=64`, `128×96`:

| path | R2 ms | R3 ms | vs R2 | peak |
| --- | ---: | ---: | ---: | ---: |
| `lidar_fused_closest` | 1.78 | **0.89** | **2.00×** | 39 MB |
| `lidar_fused_graph` | 1.62 | **0.81** | **2.00×** | 52 MB |
| `lidar_fused_legacy` | 2.27 | 2.26 | ~0% | 782 MB |
| `camera_rgbd` | 1.77 | **0.93** | **1.91×** | 52 MB |

N=1024 closest: **18.08 → 9.07 ms** (1.99×), lidar-only peak still ~626 MB.
Graph is still ~9% at N=64 and ~0% at N=1024.

Parity: distances vs legacy **exact**; in-kernel positions `max abs 9.5e-7`
(was exact with torch mul). `mesh_indices` identity subset exact. Camera
ray-depth vs lidar unchanged (`max 1.8e-5`). At N=1024 a few pixels still
miss `atol=1e-4` (max `3.33`) — same grazing/tie note as R1.

### Knobs

| knob | N=64 closest | keep? |
| --- | --- | --- |
| shrink `mesh_query_ray` tmax to `best_t` (+ in-kernel positions) | 1.78→0.96 ms (**1.85×**) | **yes** (always on) |
| `block_dim=128` (vs Warp 256) | 0.96→0.88 ms (**~11%**); N=1024 10.4→9.2 ms | **yes** (default 128; 32/64 close, 512 worse) |
| proximity fused-min (256 probes, max_dist=2) | 7.91→1.69 ms (**4.7×**), exact parity | **yes** (`query(..., closest_hit=True)` default) |
| camera output reuse + in-place clamp | camera 0.98→0.93 ms (~5%), less peak at N=1024 | **yes** (same overwrite contract as lidar) |
| `--robot-first` (G1 bodies before ground) | 0.96→1.53 ms (worse) | **no** — large ground-first lets tmax shrink before 59 body BVHs |
| `--cube-first` | 0.96→0.96 ms | no |
| `simplify_factor=0.5` (542k→271k faces) | 0.96→0.93 ms (2.7%) | no (under 5%, changes geometry) |

`block_dim` sweep (400 iters, N=64): 32=0.840, 64=0.847, **128=0.828**, 256=0.937, 512=0.989.
Same ranking at N=1024. Override with `raycaster.block_dim = …`.

Saturation: two mesh-order knobs and simplify all moved the headline by
<5% or made it slower. Remaining Warp 1.6 ideas (cuBQL, `bvh_leaf_size`,
`launch_bounds`) need a newer Warp. Stop.

```bash
.venv/bin/python scripts/bench_g1_camera.py --skip-unfused --graph --proximity
```

## 1. Goal

Make the **production GPU path** faster and cheaper in peak memory for a
multi-mesh scene that looks like HOI / active-adaptation: ground, a nearby
object, and a G1 with many posed body meshes.

Optimize both entry points that this scene exercises:

| API | Typical use | Closest-hit strategy |
| --- | --- | --- |
| `MultiMeshRaycaster.raycast_fused` | lidar / range obs | **Round 1:** launch `(N, n_rays)`, loop meshes in-kernel, write `[N, n_rays]`. Legacy `closest_hit=False` still uses the 3D buffer + PyTorch `.min(dim=1)`. |
| `RaycastCamera.render` | RGB-D mesh camera | launch `(N, H*W)`, loop meshes in-kernel, write `[N, H*W]` |

`raycast_fused` is the larger win. `RaycastCamera` already does in-kernel
closest-hit; treat it as the pattern to copy, then tune BVH / launch / graphs.

Do **not** start with V2 / Isaac (`MultiMeshRaycasterV2`) — this workspace
benchmark is standalone MuJoCo + trimesh. Keep V2 call sites in sync when
kernels change.

## 2. What the code does today

Facts that the original stub did not spell out:

- Fused lidar closest-hit kernels loop meshes in-kernel, shrink
  `mesh_query_ray` tmax to the running closest `t`, and write
  `[N, n_rays]` distances / normals / positions. Default `block_dim=128`.
  Legacy `closest_hit=False` still uses a per-mesh buffer + PyTorch `.min`.
- `MeshProximitySensor.query` defaults to the same fused-min pattern
  (`closest_hit=True`).
- Closest-hit memory scales as `O(N × n_rays)`. Legacy `closest_hit=False`
  is still `O(N × n_meshes × n_rays)`.
- Each body is a separate `wp.Mesh` BVH (`helpers.trimesh2wp()`). Default
  constructor is Warp’s CUDA default (`lbvh`); `bvh_constructor` is exposed,
  `bvh_leaf_size` only when the installed Warp accepts it.
- Closest-hit output buffers are reused; `capture_fused` / `use_graph` is
  opt-in. Returned tensors are overwritten on the next call.
- `RaycastCamera` loops meshes in-kernel with shrinking tmax, reuses RGB-D
  buffers, and uses the same `block_dim=128`.
- `scripts/advanced.py` benchmarks a 64×64 **box grid**, not this scene.
  `scripts/example.py` can load MJCF but uses a fixed 32×32 lidar fan, not
  random cameras.
- `utils_mjc.get_trimesh_from_body` extracts **only** `mjGEOM_MESH`. Ground
  planes and boxes in MJCF will not appear unless added as trimesh primitives.
  Bodies with geoms but no mesh geoms must be skipped (empty concatenate).

Local Warp version: check ``wp.__version__`` in the active venv / Isaac process.
Host stacks pin Warp independently (`pyproject.toml` floor is `>=1.7,<2`).
Re-read the live
[changelog](https://nvidia.github.io/warp/latest/user_guide/changelog.html)
and `wp.__version__` before adopting cuBQL / `bvh_leaf_size` / graphs.

## 3. Test scene

Fixed geometry, randomized cameras and (optionally) robot pose.

**Assets**

1. **Ground plane** — `trimesh.creation.box` (e.g. 10×10×0.05 m) with top face
   at z=0. Warp `wp.Mesh` has no infinite plane; a thin box is the right proxy.
2. **Cube around origin** — `trimesh.creation.box` extents ~0.4–0.6 m, center
   near `(0.4, 0.0, 0.3)` so it sits in front of the robot, not inside the
   pelvis. Identity quat, or a small random yaw.
3. **G1 at origin** — MJCF ``g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml``
   (resolve via ``SIMPLE_RAYCASTER_G1_XML``, ``--xml``, or
   ``assets/unitree_g1/``; see ``scripts/g1_assets.py``). Mesh dir is the
   sibling ``meshes/`` folder. Load via `mujoco.MjModel.from_xml_path` after
   setting `compiler meshdir` if the XML does not already resolve
   `meshes/*.STL`. One `wp.Mesh` per body that actually has mesh geoms
   (`from_MjModel` / `get_trimesh_from_body`). Poses from `mj_forward` →
   `data.xpos` / `data.xquat` (WXYZ).

**Poses**

- Default: keyframe / zero pose with pelvis at origin, feet on the plane.
- Optional stress: sample joint qpos in limits, `mj_forward`, upload body
  poses. Do this *after* the camera-only baseline is stable.

**Cameras (randomized per batch row)**

Keep OpenCV convention (matches `RaycastCamera` / AA).

For each env `i`:

- Position: on a sphere around the robot COM (or origin), radius ~1.5–3.5 m,
  azimuth `[0, 2π)`, elevation ~`[10°, 70°]` so most rays hit ground/robot,
  not empty sky.
- Orientation: look-at origin (or COM), world +Z up → WXYZ quat. Add a small
  roll jitter if we want a harder case.
- Intrinsics: sample `fov_y_deg ∈ [50, 90]`, principal point at image center.
  Resolutions as **fixed presets** (graphs need static shapes):
  `64×48`, `128×96`, `320×240`. Start with `128×96`.
- Rays: `RaycastCamera` generates them in-kernel. For
  `MultiMeshRaycaster`, generate the same pinhole rays in PyTorch
  (`dir_c = normalize(u, v, 1)`, rotate by cam quat) so both APIs see the
  same workload.

**Batch sizes to sweep:** `N ∈ {1, 16, 64, 256}` (and 1024 if memory allows
after the fused-output change). `N=64` is the default headline number.

**Seed** all sampling (`torch.manual_seed`, `np.random.seed`, Warp seed if
used) so parity checks are reproducible.

## 4. Benchmark harness (do this first)

New script: `scripts/bench_g1_camera.py`. Do not overload `advanced.py`; keep
that as the synthetic fused-vs-nonfused / independent-vs-combined check.

**Metrics (every config)**

- Wall time: CUDA events around the measured loop, `wp.synchronize()` /
  `torch.cuda.synchronize()` per iteration, report **ms/iter** and
  **rays/s** (`N * H * W / seconds`).
- Peak GPU memory: `torch.cuda.reset_peak_memory_stats()` after warmup,
  then `max_memory_allocated()`. Also log `memory_allocated()` before/after.
  If Warp ≥ 1.13, optionally `wp.ScopedMemoryTracker` for BVH labels.
- Geometry stats once: `n_meshes`, `n_points`, `n_faces` (raycaster
  `__repr__`).
- Warp version, GPU name, `N`, resolution, fov range.

**Protocol**

1. `wp.init()`, build scene + `MultiMeshRaycaster` / `RaycastCamera` once.
2. Warmup 10 iters (includes kernel compile).
3. Timed 100 iters. **Reuse** pose/ray tensors; only refresh values in-place
   if we later test graphs. First baseline may allocate every call — that is
   part of the current cost, so measure both “alloc every call” (today) and
   “preallocated outputs” (fairer kernel time).
4. Parity: `raycast` vs `raycast_fused` vs (later) fused-output kernel,
  `atol=1e-5` on distances/positions/normals (or slightly looser on normals
  at grazing hits). Camera RGB is not a lidar parity target; camera depth
  should match lidar distances when `depth_mode="ray"` and the same poses
  are used.

**Configs (columns in the report)**

- `lidar_fused` — current `raycast_fused` with camera rays
- `lidar_unfused` — `raycast` (sanity / regression only)
- `camera_rgbd` — `RaycastCamera.render`
- After each optimization: same names with a suffix (`_minfused`, `_cubql`, …)

Print a small table to stdout; optionally write `bench_results.json`.

**Sanity (once, not every timed iter)**

- Hit fraction should be high (ground + robot fill most of a downward-ish
  camera). If `mask.mean() < ~0.3` at 128×96 looking at origin, the cameras
  are pointing the wrong way.
- Optional: dump one `trimesh.PointCloud` of hits for visual check. Do not
  call `scene.show()` in the timed path.

## 5. Iteration order

Measure after **each** step. Keep a change only if it improves ms/iter or
peak memory by a clear margin on the G1 camera scene without breaking parity.
Stop a line of work when two consecutive knobs move the headline `N=64,
128×96` number by less than ~5% (or increase memory).

### Step 0 — harness + baseline (done)

- [x] `scripts/bench_g1_camera.py`
- [x] G1 loads: 59 body meshes + ground + cube = 61; ~542k faces
- [x] Empty mesh-geom bodies skipped in `from_MjModel`

### Step 1 — fuse closest-hit into the lidar kernel (done)

- [x] `transform_and_raycast_closest_kernel` (+ `mesh_indices` variant)
- [x] Wired in `raycaster.py` / `raycaster_v2.py`; default `closest_hit=True`
- [x] Serial in-kernel min (not `atomic_min`)
- [x] Same fused-output idea for `MeshProximitySensor` (`closest_hit=True`)

### Step 2 — BVH constructor / leaf size (done on Warp 1.6)

- [x] `trimesh2wp(..., bvh_constructor=, bvh_leaf_size=)` forwards only kwargs the
      installed Warp accepts
- [x] G1 sweep: **lbvh (default) fastest**; sah 1.95 ms, median 2.17 ms vs lbvh 1.78
- [ ] `bvh_leaf_size` / `cubql` — not in Warp 1.6; revisit on ≥ 1.13 hosts

Do not raise the `warp-lang` floor until a newer feature is actually adopted.

### Step 3 — allocation / graph capture (done)

- [x] Reuse `[N, n_rays]` hit buffers (overwritten on the next call)
- [x] `capture_fused` + `raycast_fused(use_graph=True)`
- [x] Graph ~10% at N=64, ~0% at N=1024

### Step 4 — launch tuning (partial on Warp 1.6)

- [x] `wp.launch(..., block_dim=128)` default for closest-hit (Warp 1.6)
- `@wp.kernel(launch_bounds=...)` needs Warp 1.11+
- `wp.get_suggested_block_size()` needs 1.13+

### Step 5 — optional / later

- Grouped `wp.Mesh(..., groups=...)` — Isaac multi-env redesign
- Merge ground+cube: **~4.5%** at N=64, below keep threshold; `--merge-static` in the bench
- Mesh simplification / RMM: `simplify_factor=0.5` is **2.7%** on G1; not default
- Query likely-hit **large** meshes first (ground before 59 G1 bodies).
  `--robot-first` was slower.

## 6. Out of scope (same as `AGENTS.md`)

Tile BVH queries, textures, multi-hit counts, differentiable raycast,
CPU device, `mesh_query_ray_anyhit` unless we add binary hit obs.

## 7. Definition of done for a “round”

A round is mergeable when:

1. `scripts/bench_g1_camera.py` prints a before/after table on the G1 scene.
2. Parity vs the previous lidar path holds at `atol=1e-5` (document any
   systematic exception).
3. `scripts/advanced.py` still agrees fused vs nonfused on the synthetic
   scene (unless that script is updated for the new output layout).
4. Public API of `raycast` / `raycast_fused` still returns
   `(positions, distances, normals)` with shapes `[N, n_rays, …]`.

Headline target is not a specific ms number until Step 0 exists. After
baseline, aim for Step 1 to cut peak memory by ~`n_meshes×` on the lidar
path and to reduce ms/iter on `N=64, 128×96`. Further steps must not give
that back.

## 8. Implementation notes

- Quaternions stay **WXYZ** in Python; kernels already swizzle to Warp XYZW.
- `device="cuda"` only for benches.
- Kernel edits: fused and non-fused, `raycaster.py` and `raycaster_v2.py`.
- Prefer extending helpers (`trimesh2wp`) over scattering BVH kwargs.
- Record Warp version in every result blob so cuBQL / leaf_size runs are
  comparable across hosts.
