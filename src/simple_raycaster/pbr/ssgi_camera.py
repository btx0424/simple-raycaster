"""PBR camera variant that uses tiled SSGI instead of SSAO."""

from __future__ import annotations

from typing import Any

import torch

from .camera import RaycastPBRCamera
from .ssgi import ssgi_tiled


class RaycastSSGICamera(RaycastPBRCamera):
    """Same G-buffer + PBR shade as :class:`RaycastPBRCamera`, but replaces SSAO
    with Warp tiled screen-space GI (short-range color bleeding + free AO).

    Keeps ``RaycastPBRCamera`` argument surface small — SSGI knobs live here.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        ssgi_radius: float = 0.25,
        ssgi_thickness: float = 0.08,
        ssgi_intensity: float = 0.85,
        ssgi_bias: float = 0.004,
        quality: str = "pretty",
        ssao_enabled: bool | None = False,
        **kwargs: Any,
    ) -> None:
        # SSGI subsumes SSAO; force it off unless the caller insists.
        if ssao_enabled is None:
            ssao_enabled = False
        super().__init__(
            width,
            height,
            quality=quality,
            ssao_enabled=ssao_enabled,
            **kwargs,
        )
        self.ssgi_radius = float(ssgi_radius)
        self.ssgi_thickness = float(ssgi_thickness)
        self.ssgi_intensity = float(ssgi_intensity)
        self.ssgi_bias = float(ssgi_bias)

    def _compose_indirect(
        self,
        hdr: torch.Tensor,
        gb: dict[str, torch.Tensor],
        planar: torch.Tensor,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        cam_quat_wxyz: torch.Tensor,
    ) -> torch.Tensor:
        if self.ssao_enabled:
            # Escape hatch: fall back to base SSAO if explicitly re-enabled.
            return super()._compose_indirect(
                hdr,
                gb,
                planar,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                cam_quat_wxyz=cam_quat_wxyz,
            )
        del cx, cy, cam_quat_wxyz
        indirect, ao = ssgi_tiled(
            planar,
            gb["mask"],
            hdr,
            fx=fx,
            fy=fy,
            radius_m=self.ssgi_radius,
            thickness=self.ssgi_thickness,
            bias=self.ssgi_bias,
            intensity=self.ssgi_intensity,
        )
        # Diffuse-ish: occlude direct, add bounced radiance scaled by albedo.
        bounce = indirect * gb["albedo"].clamp(0.0, 1.0)
        out = hdr * ao.unsqueeze(-1) + bounce
        return torch.where(gb["mask"].unsqueeze(-1), out, hdr)
