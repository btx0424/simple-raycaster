# Introduction

This repo implements a simple raycaster using Nvidia's [Warp](https://nvidia.github.io/warp/) library. Compared to the `Raycaster` in [IsaacLab](https://github.com/isaac-sim/IsaacLab), this implementation supports multiple and dynamic meshes.

Intended Usage:
* Lidar sensors.
* Efficient depth camera.

# Installation

```bash
git clone https://github.com/btx0424/simple-raycaster
cd simple-raycaster
pip install -e .
```

OpenUSD Installation:
* When used with Isaac Sim, which ships with OmniUSD, standalone installation of OpenUSD is unnecessary. However, `from pxr import Usd` is only available after invoking the `AppLauncher`.
* When used without Isaac Sim or before invoking the `AppLauncher`, you need to install OpenUSD via:
    ```bash
    pip install usd-core types-usd
    ```
  where `usd-core` is the core library and `types-usd` is the type stubs. Note that `usd-core` may conflict with the OmniUSD shipped with Isaac Sim.

# Examples

## Basic Usage

```python
import torch
import trimesh
from simple_raycaster.raycaster import MultiMeshRaycaster

# Create meshes
mesh1 = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
mesh2 = trimesh.creation.icosphere(radius=0.5)

# Initialize raycaster
raycaster = MultiMeshRaycaster([mesh1, mesh2], device="cuda")

# Define rays
N = 10  # batch size
n_rays = 100
ray_starts = torch.randn(N, n_rays, 3, device="cuda")
ray_dirs = torch.randn(N, n_rays, 3, device="cuda")
ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)  # normalize

# Mesh positions and orientations (identity)
mesh_pos = torch.zeros(N, 2, 3, device="cuda")  # 2 meshes
mesh_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").expand(N, 2, 4)

# Raycast
hit_positions, hit_distances = raycaster.raycast_fused(
    mesh_pos_w=mesh_pos,
    mesh_quat_w=mesh_quat,
    ray_starts_w=ray_starts,
    ray_dirs_w=ray_dirs,
    min_dist=0.0,
    max_dist=10.0,
)
```

## Loading from USD

```python
from pxr import Usd
from simple_raycaster.raycaster import MultiMeshRaycaster

stage = Usd.Stage.Open("scene.usd")
raycaster = MultiMeshRaycaster.from_prim_paths(
    paths=["World/.*/visuals"],  # regex pattern
    stage=stage,
    device="cuda",
    simplify_factor=0.5,  # optional mesh simplification
)
```

## Loading from MuJoCo

```python
import mujoco
from simple_raycaster.raycaster import MultiMeshRaycaster

model = mujoco.MjModel.from_xml_path("scene.xml")
body_names = ["robot_base", "robot_arm"]  # list of body names
raycaster = MultiMeshRaycaster.from_MjModel(
    body_names=body_names,
    model=model,
    device="cuda",
    simplify_factor=0.5,
)
```

## Selective Mesh Raycasting

```python
# Raycast against specific meshes using mesh_indices
mesh_indices = torch.tensor([
    [0, 1],  # batch 0: test against mesh 0 and 1
    [1, 2],  # batch 1: test against mesh 1 and 2
], device="cuda")

hit_positions, hit_distances = raycaster.raycast_fused(
    mesh_pos_w=mesh_pos,
    mesh_quat_w=mesh_quat,
    ray_starts_w=ray_starts,
    ray_dirs_w=ray_dirs,
    mesh_indices=mesh_indices,  # selective mesh testing
    min_dist=0.0,
    max_dist=10.0,
)
```

