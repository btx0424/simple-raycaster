# Raycaster performance optimization plan

Working document for optimizing `simple-raycaster` on a realistic camera workload.
This refines the original stub (test scene + Warp review + measure/iterate) using
the current code, `AGENTS.md`, and the Warp 1.7–1.16 APIs we can actually use.

**Status:** plan only. Next step is the benchmark harness in §4, then a measured
baseline before any kernel change.

## 1. Goal

Make the **production GPU path** faster and cheaper in peak memory for a
multi-mesh scene that looks like HOI / active-adaptation: ground, a nearby
object, and a G1 with many posed body meshes.

Optimize both entry points that this scene exercises:

| API | Typical use | Current closest-hit strategy |
| --- | --- | --- |
| `MultiMeshRaycaster.raycast_fused` | lidar / range obs | launch `(N, n_meshes, n_rays)`, write `[N, n_meshes, n_rays]`, then PyTorch `.min(dim=1)` |
| `RaycastCamera.render` | RGB-D mesh camera | launch `(N, H*W)`, loop meshes **in-kernel**, write `[N, H*W]` |

`raycast_fused` is the larger win. `RaycastCamera` already does in-kernel
closest-hit; treat it as the pattern to copy, then tune BVH / launch / graphs.

Do **not** start with V2 / Isaac (`MultiMeshRaycasterV2`) — this workspace
benchmark is standalone MuJoCo + trimesh. Keep V2 call sites in sync when
kernels change.

## 2. What the code does today

Facts that the original stub did not spell out:

- Fused lidar kernels (`transform_and_raycast_kernel` and
  `transform_and_raycast_against_meshes_kernel` in `kernels.py`) still allocate
  a **per-mesh** distance/normal buffer. Closest hit is a host-side
  `hit_distances.min(dim=1)` plus a `gather` for the winning normal.
- Memory scales as `O(N × n_meshes × n_rays)`. For G1 this is painful:
  ~60–70 mesh-bearing bodies (full body + Inspire fingers) × camera-resolution
  rays × env batch.
- Each body is a separate `wp.Mesh` BVH (`helpers.trimesh2wp()`). Default
  constructor is Warp’s CUDA default (`lbvh`); `bvh_constructor` and
  `bvh_leaf_size` are **not** exposed.
- Per-call Python overhead: `torch.empty` for those 3D buffers, several
  `wp.from_torch(..., return_ctype=True)`, no buffer reuse, no CUDA graph.
- `RaycastCamera` already loops `for m in range(n_meshes)` and keeps
  `best_t` / `best_n` in registers. Copy this for lidar instead of jumping
  straight to `wp.atomic_min` (atomics fight with “also keep the winning
  normal”; packing `t` + mesh id is extra complexity).
- `scripts/advanced.py` benchmarks a 64×64 **box grid**, not this scene.
  `scripts/example.py` can load MJCF but uses a fixed 32×32 lidar fan, not
  random cameras.
- `utils_mjc.get_trimesh_from_body` extracts **only** `mjGEOM_MESH`. Ground
  planes and boxes in MJCF will not appear unless added as trimesh primitives.
  Bodies with geoms but no mesh geoms must be skipped (empty concatenate).

Local Warp checkout under `/mnt/workspace/btx0424/references/warp` is **1.6**.
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
3. **G1 at origin** — MJCF
   `/mnt/workspace/btx0424/aa-projects/object_hoi/src/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml`
   Mesh dir is `.../unitree_g1/meshes/` (STLs are present; ~160 files). Load
   via `mujoco.MjModel.from_xml_path` after setting `compiler meshdir` if the
   XML does not already resolve `meshes/*.STL`. One `wp.Mesh` per body that
   actually has mesh geoms (`from_MjModel` / `get_trimesh_from_body`).
   Poses from `mj_forward` → `data.xpos` / `data.xquat` (WXYZ).

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

### Step 0 — harness + baseline (no kernel changes)

- Implement §3–§4.
- Record baseline table: lidar fused/unfused + camera, `N` × resolution.
- Confirm G1 mesh count / triangle count. If `from_MjModel` blows up on a
  body with no mesh geoms, skip those bodies.

### Step 1 — fuse closest-hit into the lidar kernel (highest leverage)

Copy the camera pattern: launch `(N, n_rays)`, loop meshes (or `mesh_indices`
subset) inside the kernel, write `[N, n_rays]` distances and `[N, n_rays, 3]`
world normals.

- Touch: `kernels.py`, `raycaster.py`, `raycaster_v2.py` (same launch/output
  contract). Keep the old 3D-output kernels behind a flag or as the unfused
  debug path until parity is proven.
- Prefer **serial min in-kernel** over `wp.atomic_min`. Atomics need a second
  channel for the winning normal; Warp 1.15 also changed float `atomic_min`
  NaN semantics. Revisit atomics only if a profiler shows the serial mesh
  loop is occupancy-bound *and* `n_meshes` is large.
- Deletes the `O(n_meshes)` intermediate buffer and the PyTorch
  `min` + `gather`. Expected: large memory drop; time drop grows with
  `n_meshes` (G1, not the 2-mesh `advanced.py` case).
- Apply the same fused-output idea to `MeshProximitySensor` only after lidar
  lands (same 3D-buffer smell).

### Step 2 — BVH constructor / leaf size

Expose on `trimesh2wp()` (and therefore every factory):

- `bvh_constructor`: `None` (Warp default), `"lbvh"`, `"sah"`, `"cubql"` if
  `wp.__version__ >= 1.13` and the build has cuBQL. Keep default as today.
- `bvh_leaf_size` if Warp ≥ 1.9 (default 4). Sweep `{1, 4, 8}` on the G1
  scene.

cuBQL is **ray-query only** until Warp 1.15 (point queries needed for
proximity). Do not turn it on globally if proximity shares those `wp.Mesh`
objects. Optional per-mesh flag: lidar/camera meshes `cubql`, proximity
meshes default.

If cuBQL is missing or errors (no mempool, GH-1430 / 1.15 fix), skip and
document the Warp version.

**Do not** raise the `warp-lang` floor until a feature is actually adopted
and host stacks that ship older Warp are called out in the PR.

### Step 3 — allocation / graph capture

After output shape is `[N, n_rays]`:

- Preallocate hit buffers on the raycaster; skip `torch.empty` every call
  when shapes match.
- Optional: persistent `enabled` ones tensor.
- CUDA graph (`wp.ScopedCapture` / `wp.capture_launch`) for the fixed-shape
  fused launch. Capture **after** warmup/compile. Only replay when `N`,
  `n_rays`, `n_meshes` are unchanged. This is the training-loop win; the
  random-camera bench can still retarget poses in-place (same storage).

Isaac V2 pose concat stays Python; graphs help most once poses already live
in GPU tensors (this standalone bench).

### Step 4 — launch tuning (only if Nsight / occupancy says so)

- `@wp.kernel(launch_bounds=...)` (Warp 1.11+)
- `wp.get_suggested_block_size()` (1.13+)
- After Step 1, lidar launch dim is `(N, n_rays)` — much healthier than
  `(N, n_meshes, n_rays)` with a tiny mesh axis.

Do not sweep this until Steps 1–2 are in.

### Step 5 — optional / later

- Grouped `wp.Mesh(..., groups=...)` to replace the per-body list in
  multi-env Isaac. Registration redesign; out of scope for the first pass.
- Combine static ground+cube into one world-frame mesh (identity pose).
  Cheap, but do not merge G1 bodies (they move independently).
- Mesh simplification (`simplify_factor`) as a **quality/perf trade**, not
  a kernel optimization. Report a 0.0 vs 0.5 pair if triangle count dominates.
- RMM / shared CUDA allocator with PyTorch: Isaac training only.

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
