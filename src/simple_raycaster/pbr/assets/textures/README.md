# Bundled albedo textures

Diffuse (1K JPG) from [Poly Haven](https://polyhaven.com/textures),
[CC0 1.0](https://polyhaven.com/license). **Not stored in git** — downloaded on
first use when you call `load_albedo_texture("forest_ground")` or
`set_ground_albedo(...)`. Cached in this folder.

| file | scene | alias |
| --- | --- | --- |
| `forest_ground_04_diff_1k.jpg` | [Forest Ground 04](https://polyhaven.com/a/forest_ground_04) | `forest_ground` |
| `mud_forest_diff_1k.jpg` | [Mud Forest](https://polyhaven.com/a/mud_forest) | `mud_forest` |
| `rocky_trail_02_diff_1k.jpg` | [Rocky Trail 02](https://polyhaven.com/a/rocky_trail_02) | `rocky_trail` |
| `concrete_floor_diff_1k.jpg` | [Concrete Floor](https://polyhaven.com/a/concrete_floor) | `concrete_floor` |

JPEG decode uses ImageMagick `convert` / `identify` on PATH (no Pillow). Pass a
`.npy` path to skip both download and ImageMagick.
