# Bundled HDRIs

Radiance RGBE (1K) from [Poly Haven](https://polyhaven.com/hdris),
[CC0 1.0](https://polyhaven.com/license). **Not stored in git** — the library
downloads them on first use (see `pbr/assets_fetch.py`). Cached under this folder
after the first `RaycastPBRCamera()` or `resolve_hdri_path(...)`.

| file | scene | alias |
| --- | --- | --- |
| `poly_haven_studio_1k.hdr` | [Poly Haven Studio](https://polyhaven.com/a/poly_haven_studio) | `studio` (default) |
| `studio_small_09_1k.hdr` | [Studio Small 09](https://polyhaven.com/a/studio_small_09) | `studio_small` |
| `venice_sunset_1k.hdr` | [Venice Sunset](https://polyhaven.com/a/venice_sunset) | `venice_sunset` |

Albedo test textures (diffuse JPG) live under [`textures/`](textures/README.md).
