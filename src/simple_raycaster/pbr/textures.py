"""Bundled albedo textures + planar UV helpers for ground meshes."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .assets_fetch import ensure_polyhaven_albedo

_ASSETS = Path(__file__).resolve().parent / "assets" / "textures"

# Poly Haven CC0 diffuse (1K JPG). Fetched on first use; decode via ImageMagick.
_BUNDLED_ALBEDO_IDS: dict[str, str] = {
    "forest_ground": "forest_ground_04",
    "mud_forest": "mud_forest",
    "rocky_trail": "rocky_trail_02",
    "concrete_floor": "concrete_floor",
}
BUNDLED_ALBEDOS: dict[str, Path] = {
    "forest_ground": _ASSETS / "forest_ground_04_diff_1k.jpg",
    "mud_forest": _ASSETS / "mud_forest_diff_1k.jpg",
    "rocky_trail": _ASSETS / "rocky_trail_02_diff_1k.jpg",
    "concrete_floor": _ASSETS / "concrete_floor_diff_1k.jpg",
}


def default_textures_dir() -> Path:
    return _ASSETS


def resolve_albedo_path(name_or_path: str | Path) -> Path:
    """Resolve a bundled alias or a ``.jpg`` / ``.npy`` path."""
    key = str(name_or_path).strip().lower()
    if key in BUNDLED_ALBEDOS:
        path = BUNDLED_ALBEDOS[key]
        return ensure_polyhaven_albedo(path, asset_id=_BUNDLED_ALBEDO_IDS[key])
    path = Path(name_or_path)
    if not path.is_file():
        raise FileNotFoundError(f"albedo texture missing: {path}")
    return path


def _load_jpeg_rgb(path: Path) -> np.ndarray:
    """Decode JPEG to float32 ``[H,W,3]`` using ImageMagick (stdlib-only project)."""
    wh = subprocess.check_output(
        ["identify", "-format", "%w %h", str(path)],
        text=True,
    ).strip()
    w, h = (int(x) for x in wh.split())
    with tempfile.NamedTemporaryFile(suffix=".raw") as tmp:
        subprocess.check_call(
            ["convert", str(path), "-depth", "8", f"rgb:{tmp.name}"],
        )
        arr = np.fromfile(tmp.name, dtype=np.uint8).reshape(h, w, 3)
    return arr.astype(np.float32) / 255.0


def load_albedo_texture(name_or_path: str | Path) -> np.ndarray:
    """Load one albedo map as float32 ``[H, W, 3]`` in ``[0, 1]``."""
    path = resolve_albedo_path(name_or_path)
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float32)
    elif suf in (".jpg", ".jpeg"):
        arr = _load_jpeg_rgb(path)
    else:
        raise ValueError(f"unsupported albedo format {suf} ({path})")
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"albedo shape {arr.shape}, expected [H,W,3]")
    return np.clip(arr, 0.0, 1.0)


def load_albedo_stack(names: Sequence[str | Path]) -> np.ndarray:
    """Load and stack textures into ``[T, H, W, 3]`` (same H×W required)."""
    if not names:
        raise ValueError("load_albedo_stack: empty names")
    maps = [load_albedo_texture(n) for n in names]
    h0, w0 = maps[0].shape[:2]
    for i, m in enumerate(maps):
        if m.shape[0] != h0 or m.shape[1] != w0:
            raise ValueError(
                f"texture {i} shape {m.shape[:2]} != {h0}x{w0} (stack requires equal size)"
            )
    return np.stack(maps, axis=0)


def planar_xz_uvs(
    points: np.ndarray,
    *,
    scale: float = 1.0,
    origin_xz: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Mesh-local planar UVs from XZ: ``u=(x-ox)/scale``, ``v=(z-oz)/scale``."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    s = float(scale)
    if s <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    ox, oz = float(origin_xz[0]), float(origin_xz[1])
    uvs = np.empty((pts.shape[0], 2), dtype=np.float32)
    uvs[:, 0] = (pts[:, 0] - ox) / s
    uvs[:, 1] = (pts[:, 2] - oz) / s
    return uvs


def pack_vertex_uvs_from_wp(
    meshes_wp: Sequence,
    *,
    textured_mesh_ids: Sequence[int] | None = None,
    scale: float = 1.0,
    origin_xz: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Packed ``[V, 2]`` UVs. Only ``textured_mesh_ids`` get planar XZ; others zero."""
    n = len(meshes_wp)
    if textured_mesh_ids is None:
        ids = set(range(n))
    else:
        ids = {int(i) for i in textured_mesh_ids}
    chunks: list[np.ndarray] = []
    for i, mesh in enumerate(meshes_wp):
        pts = np.asarray(mesh.points.numpy(), dtype=np.float32).reshape(-1, 3)
        if i in ids:
            chunks.append(planar_xz_uvs(pts, scale=scale, origin_xz=origin_xz))
        else:
            chunks.append(np.zeros((pts.shape[0], 2), dtype=np.float32))
    if not chunks:
        raise ValueError("pack_vertex_uvs_from_wp: no meshes")
    return np.concatenate(chunks, axis=0)


def pack_vertex_uvs_from_trimeshes(
    meshes: Sequence,
    *,
    textured_mesh_ids: Sequence[int] | None = None,
    scale: float = 1.0,
    origin_xz: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Same packing from trimesh list."""
    n = len(meshes)
    if textured_mesh_ids is None:
        ids = set(range(n))
    else:
        ids = {int(i) for i in textured_mesh_ids}
    chunks: list[np.ndarray] = []
    for i, mesh in enumerate(meshes):
        pts = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
        if i in ids:
            chunks.append(planar_xz_uvs(pts, scale=scale, origin_xz=origin_xz))
        else:
            chunks.append(np.zeros((pts.shape[0], 2), dtype=np.float32))
    if not chunks:
        raise ValueError("pack_vertex_uvs_from_trimeshes: no meshes")
    return np.concatenate(chunks, axis=0)
