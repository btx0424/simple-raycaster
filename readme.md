# Introduction

This repo implements a GPU raycaster built on Nvidia [Warp](https://nvidia.github.io/warp/). Compared to the `RayCaster` in [IsaacLab](https://github.com/isaac-sim/IsaacLab), it supports **multiple meshes** and **per-mesh rigid transforms** at batch scale.

Intended usage:

* Lidar-style range sensors
* Efficient depth / occupancy sensing
* Voxelization of hit points (`helpers.voxelize_wp`)

## Raycaster APIs

| Class | Use when |
| --- | --- |
| `MultiMeshRaycaster` | You manage mesh geometry and poses yourself (standalone scripts, MuJoCo, offline USD). |
| `MultiMeshRaycasterV2` | You run inside Isaac Sim / Isaac Lab and want poses pulled automatically from `Articulation` / `RigidObject` instances. |

Both classes expose:

* `raycast(...)` — GPU PyTorch world→mesh transform, then a separate Warp `raycast_kernel` (two GPU steps)
* `raycast_fused(...)` — single fused Warp kernel for transform + raycast (preferred; fewer temporaries)

Use `device="cuda"` in production. `device="cpu"` is supported only for special debugging and is too slow for real workloads.

Quaternions are **WXYZ** (scalar first). Ray directions should be normalized. Misses return `max_dist` as the hit distance.

# Installation

```bash
git clone https://github.com/btx0424/simple-raycaster
cd simple-raycaster
pip install -e .
```

Initialize Warp once before raycasting:

```python
import warp as wp
wp.init()
```

## OpenUSD

* With **Isaac Sim**, OmniUSD is already available after launching via `AppLauncher`. `from pxr import Usd` works once the app is running.
* **Without Isaac Sim**, install standalone USD for USD loading utilities:

```bash
pip install usd-core types-usd
```

Note: `usd-core` may conflict with OmniUSD shipped with Isaac Sim.

## Isaac Sim (for V2 only)

`MultiMeshRaycasterV2` imports `isaacsim` and `isaaclab` at runtime when registering entities. Install/use it inside an Isaac Lab environment where those packages are available.

# Examples

## Basic usage (`MultiMeshRaycaster`)

```python
import warp as wp
import torch
import trimesh
from simple_raycaster import MultiMeshRaycaster

wp.init()

mesh1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
mesh2 = trimesh.creation.icosphere(radius=0.5)
raycaster = MultiMeshRaycaster([mesh1, mesh2], device="cuda")

N, n_rays = 10, 100
ray_starts = torch.randn(N, n_rays, 3, device="cuda")
ray_dirs = torch.randn(N, n_rays, 3, device="cuda")
ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)

mesh_pos = torch.zeros(N, 2, 3, device="cuda")
mesh_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").expand(N, 2, 4)

hit_positions, hit_distances = raycaster.raycast_fused(
    mesh_pos,
    mesh_quat,
    ray_starts,
    ray_dirs,
    min_dist=0.0,
    max_dist=10.0,
)
```

## Isaac Lab integration (`MultiMeshRaycasterV2`)

V2 registers meshes once, then reads live body poses from Isaac entities on each raycast. You do **not** pass `mesh_pos_w` / `mesh_quat_w`.

```python
import warp as wp
import torch
from simple_raycaster import MultiMeshRaycasterV2

wp.init()

raycaster = MultiMeshRaycasterV2(device="cuda")

# Static geometry baked into world-frame mesh data (pose stays identity)
raycaster.add_isaac_static("/World/ground")

# Dynamic robot / object: one mesh per body, poses from entity.data.body_link_pose_w
robot = env.scene.articulations["robot"]
raycaster.add_isaac_entity(robot)

N = env.num_envs
ray_starts = ...  # [N, n_rays, 3]
ray_dirs = ...    # [N, n_rays, 3], normalized

hit_positions, hit_distances = raycaster.raycast_fused(
    ray_starts,
    ray_dirs,
    min_dist=0.1,
    max_dist=5.0,
)
```

Registration rules for V2:

* Call `add_isaac_static` / `add_isaac_entity` only — do not call internal `_add_mesh` / `_add_from_path`.
* Each `add_isaac_static` adds one combined mesh at a fixed world pose (identity).
* Each `add_isaac_entity` adds one mesh per body; visual prim count must match `entity.num_bodies`.
* Batch size `N` in ray tensors must equal `entity.num_instances`.

## Loading from USD

```python
from pxr import Usd
from simple_raycaster import MultiMeshRaycaster

stage = Usd.Stage.Open("scene.usd")
raycaster = MultiMeshRaycaster.from_prim_paths(
    paths=["World/.*/visuals"],
    stage=stage,
    device="cuda",
    simplify_factor=0.5,
)
```

## Loading from MuJoCo

```python
import mujoco
from simple_raycaster import MultiMeshRaycaster

model = mujoco.MjModel.from_xml_path("scene.xml")
raycaster = MultiMeshRaycaster.from_MjModel(
    body_names=["robot_base", "robot_arm"],
    model=model,
    device="cuda",
    simplify_factor=0.5,
)
```

## Selective mesh raycasting

Use `mesh_indices` to test each batch element against a subset of meshes. The second dimension of `mesh_indices` must match the number of mesh pose slots for that batch element.

```python
mesh_indices = torch.tensor([
    [0, 1],
    [1, 2],
], device="cuda")

hit_positions, hit_distances = raycaster.raycast_fused(
    mesh_pos,
    mesh_quat,
    ray_starts,
    ray_dirs,
    mesh_indices=mesh_indices,
    min_dist=0.0,
    max_dist=10.0,
)
```

With V2, omit `mesh_pos` / `mesh_quat` and pass only rays plus optional `mesh_indices`.

# Scripts

* `scripts/example.py` — USD or MJCF scene, benchmark / visualize ray hits and voxels
* `scripts/advanced.py` — compares fused vs non-fused kernels and independent vs combined mesh layouts

Run:

```bash
python scripts/example.py --usd path/to/scene.usd
python scripts/example.py --mjcf path/to/scene.xml --benchmark
```
