"""Lazy download of bundled Poly Haven CC0 assets (1K HDR + diffuse JPG)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

_API = "https://api.polyhaven.com/files"
_RES = "1k"
_TIMEOUT = 120
_USER_AGENT = "simple-raycaster"

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _locks_guard:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _fetch_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = resp.read()
    except urllib.error.URLError as exc:
        raise FileNotFoundError(
            f"failed to download {dest.name} from Poly Haven ({url}): {exc}"
        ) from exc
    tmp.write_bytes(data)
    tmp.replace(dest)


def _api_json(asset_id: str) -> dict:
    url = f"{_API}/{asset_id}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.URLError as exc:
        raise FileNotFoundError(
            f"Poly Haven metadata unavailable for {asset_id!r}: {exc}"
        ) from exc


def ensure_polyhaven_hdri(path: Path, *, asset_id: str) -> Path:
    """Download a 1K Radiance HDR from Poly Haven if ``path`` is missing."""
    if path.is_file():
        return path
    with _lock_for(path):
        if path.is_file():
            return path
        meta = _api_json(asset_id)
        url = meta["hdri"][_RES]["hdr"]["url"]
        _fetch_url(url, path)
    if not path.is_file():
        raise FileNotFoundError(f"failed to fetch HDRI {asset_id} -> {path}")
    return path


def ensure_polyhaven_albedo(path: Path, *, asset_id: str) -> Path:
    """Download a 1K diffuse JPG from Poly Haven if ``path`` is missing."""
    if path.is_file():
        return path
    with _lock_for(path):
        if path.is_file():
            return path
        meta = _api_json(asset_id)
        url = meta["Diffuse"][_RES]["jpg"]["url"]
        _fetch_url(url, path)
    if not path.is_file():
        raise FileNotFoundError(f"failed to fetch albedo {asset_id} -> {path}")
    return path
