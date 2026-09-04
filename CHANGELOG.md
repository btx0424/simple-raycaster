# Changelog

All notable changes to this project are documented here.

## [0.4.2] - 2026-09-04

### Added

- **`RaycastCamera(closest_hit=False)`** — parallel-over-mesh launch
  ``(N, n_meshes, H*W)`` via ``raycast_camera_per_mesh_kernel``, then PyTorch
  ``min`` + Lambert shade. Default remains in-kernel mesh loop with shrinking
  tmax (``closest_hit=True``). Higher peak memory; for A/B vs the serial path.
- **`simple_raycaster.__version__`** — from package metadata (fallback ``0.4.2``
  when the distribution is not installed).

## [0.4.1] - 2026-09-03

### Added

- **USD primitive tessellation** — ``utils_usd`` now converts ``Sphere``,
  ``Cylinder``, ``Capsule``, and ``Cone`` (in addition to ``Mesh`` / ``Cube``)
  via ``trimesh.creation.*``, with USD ``axis`` remapping for cylinders/capsules/cones.
- **``UsdGeomPart`` / ``get_geom_parts_from_prim``** — per-child geom descriptors with
  body-local pose, tessellated mesh, and optional Viser-native params
  (``box`` / ``sphere`` / ``cylinder``). Capsule and Cone remain tessellation-only.

## [0.4.0] - 2026-09-03

### Added

- **`MESH_PROXIMITY_UPDATE_REQUIRED`** — migration hint for consumers moving off
  automatic Isaac pose gathering.

### Changed

- **`RaycastCamera.render`** — requires explicit ``mesh_pos_w`` / ``mesh_quat_w``
  (no implicit mesh pose gathering). Resolution is fixed at construction.
- **`RaycastPBRCamera` / `RaycastSSGICamera`** — ``render`` / ``render_gbuffer`` /
  sparse paths require explicit mesh poses via ``_require_mesh_poses()``.
- **`mesh_rgbd` renderers** — gather entity poses in the wrapper and pass explicit
  mesh poses into ``RaycastCamera.render``.

### Removed

- **`MultiMeshRaycasterV2`** — removed Isaac-specific auto pose gathering. Use
  ``MultiMeshRaycaster`` with explicit poses (e.g. active-adaptation
  ``MeshRegistry``) or ``RaycastCamera`` for RGB-D.

### Breaking

- **`MeshProximitySensor`** — stub only; ``__init__`` raises
  ``NotImplementedError`` until rewritten for explicit mesh poses.
- Any import or use of ``MultiMeshRaycasterV2`` must migrate to
  ``MultiMeshRaycaster`` + explicit ``mesh_pos_w`` / ``mesh_quat_w``.

## [0.3.0] - 2026-08-30

### Added

- **Tile-sparse rendering** — `RaycastPBRCamera.render_sparse()` and
  `render_gbuffer_sparse()` for partial-frame updates (`tiles [B,T,2]`, `tile_size`).
- **Functional render API** — per-call pose + optional domain-randomization tables on
  `render()` (`albedo`, `roughness`, `metallic`, `sun_dir`, `sun_intensity`, `exposure`).
- **Batched material/light tables** — `set_materials()` / `set_lights()` with
  `[Nw, M, …]` sticky tables; per-call kwargs override without dropping the compiled shade graph.
- **UV + albedo ground path** — planar XZ UVs, bilinear albedo sampling,
  `set_ground_albedo()`, `set_albedo_textures()`, bundled Poly Haven diffuse aliases.
- **Lazy asset fetch** — HDRIs and albedo JPGs download from Poly Haven (CC0) on first
  use; binaries are no longer shipped in git or the PyPI wheel.
- **Perf gate** — `scripts/bench_perf_gate.py` with JSON baselines and `--compare`
  regression mode (`bench/baselines/rtx4090.json`).

### Changed

- **`quality="fast"`** now enables sun shadow rays; **`quality="pretty"`** keeps 128² shadow maps.
  If both are on, shadow rays win (no stacking).
- **`render()` / `render_gbuffer()`** no longer accept `width`, `height`, `fov_y_deg`, or
  `light_dir` — use the camera constructor, `set_resolution()`, and `set_lights()` instead.

### Breaking

- Call sites that passed resolution or `light_dir` into `render()` must move those to camera
  setup APIs (see `readme.md` ticket table).
- First PBR camera init or bundled texture load requires network access to Poly Haven CDN
  (cached locally afterward under `pbr/assets/`).

### Requirements

- Bundled albedo JPG aliases need **ImageMagick** (`convert`, `identify`) on PATH, or pass
  your own `.npy` / file paths.

## [0.2.1] - prior release

- PBR camera, HDRI IBL, FXAA/SSAO, SSGI camera, mesh table materials, and related benches.

[0.4.2]: https://github.com/btx0424/simple-raycaster/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/btx0424/simple-raycaster/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/btx0424/simple-raycaster/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/btx0424/simple-raycaster/compare/v0.2.1...v0.3.0
