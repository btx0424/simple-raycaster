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

