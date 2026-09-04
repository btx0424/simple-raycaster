"""Distributed smoke tests for MultiMeshRaycaster and RaycastCamera.

Decoupled from active-adaptation: only ``simple_raycaster``, Torch, Warp, and
trimesh. The regression this guards is Warp mesh work on each local CUDA device
followed by an NCCL collective (the hang mode seen in multi-GPU training).

Run via pytest (spawns 2 processes when ≥2 GPUs are visible)::

    pytest tests/test_distributed.py -v

Or under torchrun::

    torchrun --standalone --nproc_per_node=2 tests/test_distributed.py
"""

from __future__ import annotations

import os
import socket
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import trimesh
import warp as wp

from simple_raycaster import MultiMeshRaycaster, RaycastCamera

_REQUIRES_2_GPUS = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="need ≥2 CUDA devices for NCCL multi-process tests",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _identity_quat(n: int, m: int, device: str) -> torch.Tensor:
    return (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        .view(1, 1, 4)
        .expand(n, m, 4)
        .contiguous()
    )


def _unit_box() -> trimesh.Trimesh:
    # Full extents 0.5 → faces at ±0.25.
    return trimesh.creation.box(extents=(0.5, 0.5, 0.5))


def exercise_multimesh_raycaster(device: str, *, batch: int = 4) -> float:
    """Raycast a box; return mean hit distance of the center ray."""
    box = _unit_box()
    raycaster = MultiMeshRaycaster([box], device=device)
    raycaster.initialize()

    # Rays from z=-2 along +Z toward the box at the origin.
    ray_starts = torch.zeros(batch, 1, 3, device=device)
    ray_starts[:, :, 2] = -2.0
    ray_dirs = torch.zeros(batch, 1, 3, device=device)
    ray_dirs[:, :, 2] = 1.0
    mesh_pos = torch.zeros(batch, 1, 3, device=device)
    mesh_quat = _identity_quat(batch, 1, device)

    _hit_pos, hit_dist, hit_n = raycaster.raycast_fused(
        mesh_pos,
        mesh_quat,
        ray_starts,
        ray_dirs,
        max_dist=10.0,
        closest_hit=True,
    )
    assert hit_dist.shape == (batch, 1), hit_dist.shape
    assert hit_n.shape == (batch, 1, 3), hit_n.shape

    center = float(hit_dist[:, 0].mean().item())
    # Front face at z=-0.25 → distance from z=-2 is 1.75.
    assert 1.6 < center < 1.9, f"unexpected MultiMeshRaycaster depth {center}"
    return center


def exercise_raycast_camera(device: str, *, batch: int = 2) -> float:
    """Render a box in front of an OpenCV camera; return center depth."""
    box = _unit_box()
    cam = RaycastCamera(
        64,
        48,
        fov_y_deg=70.0,
        convention="opencv",
        device=device,
        closest_hit=True,
    )
    cam.bind_trimeshes([box], albedos=[(0.9, 0.2, 0.15)])
    cam.initialize()

    # OpenCV camera at origin looking +Z; box centered at z=2.
    cam_pos = torch.zeros(batch, 3, device=device)
    cam_quat = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(batch, 4).contiguous()
    )
    mesh_pos = (
        torch.tensor([[[0.0, 0.0, 2.0]]], device=device)
        .expand(batch, 1, 3)
        .contiguous()
    )
    mesh_quat = _identity_quat(batch, 1, device)

    rgb, depth, mask = cam.render(
        cam_pos, cam_quat, mesh_pos_w=mesh_pos, mesh_quat_w=mesh_quat
    )
    assert rgb.shape == (batch, 48, 64, 3), rgb.shape
    assert depth.shape == (batch, 48, 64), depth.shape
    assert mask.shape == (batch, 48, 64), mask.shape
    assert bool(mask[0, 24, 32]), "expected hit at image center"
    hit_frac = float(mask.float().mean().item())
    assert hit_frac >= 0.01, f"box covers too little FOV: {hit_frac}"

    center = float(depth[0, 24, 32].item())
    assert 1.6 < center < 1.9, f"unexpected RaycastCamera depth {center}"
    return center


def exercise_nccl_after_warp(rank: int, device: str, world_size: int) -> None:
    """NCCL collectives must succeed after Warp raycast/camera work."""
    torch.cuda.synchronize(device=device)
    assert torch.cuda.current_device() == torch.device(device).index

    # Scalar all_reduce — both ranks must participate.
    payload = torch.tensor([float(rank + 1)], device=device, dtype=torch.float32)
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    expected = float(world_size * (world_size + 1) // 2)
    assert abs(payload.item() - expected) < 1e-5, (
        f"all_reduce got {payload.item()}, expected {expected}"
    )

    # Same collective shape as train_ppo's post-env name sync.
    objects = [f"rank-{rank}-ok" if rank == 0 else None]
    dist.broadcast_object_list(
        objects,
        src=0,
        device=torch.device(device),
    )
    assert objects[0] == "rank-0-ok", objects


def _init_distributed(rank: int, world_size: int, master_port: int) -> str:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(device),
    )
    wp.init()
    return device


def _run_rank_suite(rank: int, world_size: int, master_port: int) -> None:
    device = _init_distributed(rank, world_size, master_port)
    try:
        # Bind current device again — Warp / CUDA context setup can reset it.
        torch.cuda.set_device(rank)

        lidar_depth = exercise_multimesh_raycaster(device)
        cam_depth = exercise_raycast_camera(device)

        # Share a float summary so every rank proves its Warp results exist
        # and NCCL still works after both raycasters.
        summary = torch.tensor(
            [lidar_depth, cam_depth], device=device, dtype=torch.float32
        )
        dist.all_reduce(summary, op=dist.ReduceOp.SUM)
        summary /= float(world_size)
        assert 1.6 < float(summary[0]) < 1.9
        assert 1.6 < float(summary[1]) < 1.9

        exercise_nccl_after_warp(rank, device, world_size)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_worker(rank: int, world_size: int, master_port: int, error_queue) -> None:
    try:
        _run_rank_suite(rank, world_size, master_port)
        error_queue.put((rank, None))
    except Exception as exc:  # noqa: BLE001 — re-raise in parent with rank tag
        error_queue.put((rank, repr(exc)))
        raise


def _spawn_distributed(world_size: int, target: Callable[..., None]) -> None:
    port = _free_port()
    ctx = mp.get_context("spawn")
    error_queue = ctx.SimpleQueue()
    processes = []
    for rank in range(world_size):
        proc = ctx.Process(
            target=target,
            args=(rank, world_size, port, error_queue),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    errors: list[tuple[int, str]] = []
    for _ in range(world_size):
        rank, err = error_queue.get()
        if err is not None:
            errors.append((rank, err))

    for proc in processes:
        proc.join(timeout=180)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)
            errors.append((proc.pid or -1, "worker timed out / hung"))
        elif proc.exitcode not in (0, None):
            errors.append((proc.pid or -1, f"exitcode={proc.exitcode}"))

    if errors:
        details = "; ".join(f"rank/pid {r}: {e}" for r, e in errors)
        pytest.fail(f"distributed worker failure(s): {details}")


@_REQUIRES_2_GPUS
def test_multimesh_raycaster_and_camera_with_nccl() -> None:
    """Each rank runs both raycasters on its GPU, then NCCL still works."""
    _spawn_distributed(world_size=2, target=_spawn_worker)


def _spawn_worker_collective_warp_collective(
    rank: int, world_size: int, master_port: int, error_queue
) -> None:
    """Collective → Warp → collective again (ordering that training hits)."""
    try:
        device = _init_distributed(rank, world_size, master_port)
        try:
            torch.cuda.set_device(rank)
            exercise_nccl_after_warp(rank, device, world_size)
            exercise_multimesh_raycaster(device)
            exercise_raycast_camera(device)
            torch.cuda.set_device(rank)
            exercise_nccl_after_warp(rank, device, world_size)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        error_queue.put((rank, None))
    except Exception as exc:  # noqa: BLE001
        error_queue.put((rank, repr(exc)))
        raise


@_REQUIRES_2_GPUS
def test_nccl_survives_warp_then_more_raycasts() -> None:
    """Collective → Warp → collective again (ordering that training hits)."""
    _spawn_distributed(
        world_size=2, target=_spawn_worker_collective_warp_collective
    )


def _torchrun_main() -> None:
    """Entry point for ``torchrun --nproc_per_node=N tests/test_distributed.py``."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(device),
    )
    wp.init()
    try:
        torch.cuda.set_device(local_rank)
        exercise_multimesh_raycaster(device)
        exercise_raycast_camera(device)
        exercise_nccl_after_warp(local_rank, device, world_size)
        dist.barrier()
        if local_rank == 0:
            print("ok: MultiMeshRaycaster + RaycastCamera + NCCL")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _torchrun_main()
    else:
        # Local spawn fallback (same as pytest).
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise SystemExit("need ≥2 CUDA devices")
        _spawn_distributed(world_size=2, target=_spawn_worker)
        print("ok: MultiMeshRaycaster + RaycastCamera + NCCL (spawn)")
