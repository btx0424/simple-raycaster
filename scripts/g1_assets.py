"""Resolve Unitree G1 MJCF without hardcoded machine paths.

Order:
1. Explicit ``path`` argument / ``--xml``
2. Env ``SIMPLE_RAYCASTER_G1_XML``
3. Common relative locations from the repo root or CWD
"""

from __future__ import annotations

import os
from pathlib import Path

G1_XML_NAME = "g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"

# Relative to repo root (parent of ``scripts/``) or process CWD.
_RELATIVE_CANDIDATES = (
    Path("assets/unitree_g1") / G1_XML_NAME,
    Path("assets") / "unitree_g1" / G1_XML_NAME,
    Path("..") / "object_hoi" / "src" / "assets" / "unitree_g1" / G1_XML_NAME,
    Path("..") / ".." / "object_hoi" / "src" / "assets" / "unitree_g1" / G1_XML_NAME,
    Path("..") / "aa-projects" / "object_hoi" / "src" / "assets" / "unitree_g1" / G1_XML_NAME,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_g1_xml(path: str | Path | None = None) -> Path:
    """Return an existing G1 MJCF path or raise ``FileNotFoundError``."""
    if path is not None and str(path).strip():
        p = Path(path).expanduser()
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"G1 MJCF not found: {p}")

    env = os.environ.get("SIMPLE_RAYCASTER_G1_XML", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(
            f"SIMPLE_RAYCASTER_G1_XML is set but not a file: {p}"
        )

    roots = (_repo_root(), Path.cwd())
    tried: list[str] = []
    for root in roots:
        for rel in _RELATIVE_CANDIDATES:
            cand = (root / rel).resolve()
            tried.append(str(cand))
            if cand.is_file():
                return cand

    raise FileNotFoundError(
        "G1 MJCF not found. Set SIMPLE_RAYCASTER_G1_XML, pass --xml, or place "
        f"{G1_XML_NAME} under assets/unitree_g1/. Tried:\n  - "
        + "\n  - ".join(dict.fromkeys(tried))
    )


def find_g1_xml(path: str | Path | None = None) -> Path | None:
    """Like :func:`resolve_g1_xml` but returns ``None`` instead of raising."""
    try:
        return resolve_g1_xml(path)
    except FileNotFoundError:
        return None
