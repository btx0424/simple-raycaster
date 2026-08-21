# Bundled albedo textures

Diffuse (1K JPG) from [Poly Haven](https://polyhaven.com/textures),
[CC0 1.0](https://polyhaven.com/license). Attribution not required; listed for
provenance. Used by the minimal UV→albedo ground path (`pbr/textures.py`).

| file | scene | alias |
| --- | --- | --- |
| `forest_ground_04_diff_1k.jpg` | [Forest Ground 04](https://polyhaven.com/a/forest_ground_04) | `forest_ground` |
| `mud_forest_diff_1k.jpg` | [Mud Forest](https://polyhaven.com/a/mud_forest) | `mud_forest` |
| `rocky_trail_02_diff_1k.jpg` | [Rocky Trail 02](https://polyhaven.com/a/rocky_trail_02) | `rocky_trail` |
| `concrete_floor_diff_1k.jpg` | [Concrete Floor](https://polyhaven.com/a/concrete_floor) | `concrete_floor` |

Load via `load_albedo_texture("forest_ground")` (needs ImageMagick `convert` /
`identify` on PATH). Roughness / normal maps are intentionally omitted for the
minimal path.
