import warp as wp


@wp.kernel(enable_backward=False)
def raycast_kernel(
    meshes: wp.array(dtype=wp.uint64),
    ray_starts: wp.array(dtype=wp.vec3, ndim=3),
    ray_dirs: wp.array(dtype=wp.vec3, ndim=3),
    enabled: wp.array(dtype=wp.bool, ndim=1),
    min_dist: float,
    max_dist: float,
    hit_distances: wp.array(dtype=wp.float32, ndim=3),
):
    i, mesh_id, ray_id = wp.tid()
    if not enabled[i]:
        hit_distances[i, mesh_id, ray_id] = wp.INF
        return
    mesh = meshes[mesh_id]
    ray_start = ray_starts[i, mesh_id, ray_id]
    ray_dir = ray_dirs[i, mesh_id, ray_id]
    result = wp.mesh_query_ray(
        mesh,
        ray_start,
        ray_dir,
        max_dist,
    )
    t = max_dist
    if result.result and result.t >= min_dist:
        t = result.t
    hit_distances[i, mesh_id, ray_id] = t


@wp.kernel(enable_backward=False)
def raycast_against_meshes_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_indices: wp.array(dtype=wp.int64, ndim=2),
    ray_starts: wp.array(dtype=wp.vec3, ndim=3),
    ray_dirs: wp.array(dtype=wp.vec3, ndim=3),
    enabled: wp.array(dtype=wp.bool, ndim=1),
    min_dist: float,
    max_dist: float,
    hit_distances: wp.array(dtype=wp.float32, ndim=3),
):
    i, j, ray_id = wp.tid()
    mesh_id = mesh_indices[i, j]
    if not enabled[i]:
        hit_distances[i, j, ray_id] = wp.INF
        return
    mesh = meshes[mesh_id]
    ray_start = ray_starts[i, j, ray_id]
    ray_dir = ray_dirs[i, j, ray_id]
    result = wp.mesh_query_ray(
        mesh,
        ray_start,
        ray_dir,
        max_dist,
    )
    t = max_dist
    if result.result and result.t >= min_dist:
        t = result.t
    hit_distances[i, j, ray_id] = t


@wp.kernel(enable_backward=False)
def transform_and_raycast_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),
    ray_starts_w: wp.array(dtype=wp.vec3, ndim=2),
    ray_dirs_w: wp.array(dtype=wp.vec3, ndim=2),
    enabled: wp.array(dtype=wp.bool, ndim=1),
    min_dist: float,
    max_dist: float,
    hit_distances: wp.array(dtype=wp.float32, ndim=3),
):
    i, mesh_id, ray_id = wp.tid()
    if not enabled[i]:
        hit_distances[i, mesh_id, ray_id] = wp.INF
        return
    
    # transform ray starts and dirs to mesh frame
    quat_wxyz = mesh_quat_w[i, mesh_id]
    quat_xyzw = wp.quat(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0])
    ray_start_b = wp.quat_rotate_inv(
        quat_xyzw,
        ray_starts_w[i, ray_id] - mesh_pos_w[i, mesh_id],
    )
    ray_dir_b = wp.quat_rotate_inv(
        quat_xyzw,
        ray_dirs_w[i, ray_id],
    )
    
    result = wp.mesh_query_ray(
        meshes[mesh_id],
        ray_start_b,
        ray_dir_b,
        max_dist,
    )
    t = max_dist
    if result.result and result.t >= min_dist:
        t = result.t
    hit_distances[i, mesh_id, ray_id] = t


@wp.kernel(enable_backward=False)
def transform_and_raycast_against_meshes_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_indices: wp.array(dtype=wp.int64, ndim=2),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),
    ray_starts_w: wp.array(dtype=wp.vec3, ndim=2),
    ray_dirs_w: wp.array(dtype=wp.vec3, ndim=2),
    enabled: wp.array(dtype=wp.bool, ndim=1),
    min_dist: float,
    max_dist: float,
    hit_distances: wp.array(dtype=wp.float32, ndim=3),
):
    i, j, ray_id = wp.tid()
    mesh_id = mesh_indices[i, j]
    if not enabled[i]:
        hit_distances[i, j, ray_id] = wp.INF
        return
    
    # transform ray starts and dirs to mesh frame
    quat_wxyz = mesh_quat_w[i, j]
    quat_xyzw = wp.quat(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0])
    ray_start_b = wp.quat_rotate_inv(
        quat_xyzw,
        ray_starts_w[i, ray_id] - mesh_pos_w[i, j],
    )
    ray_dir_b = wp.quat_rotate_inv(
        quat_xyzw,
        ray_dirs_w[i, ray_id],
    )
    
    result = wp.mesh_query_ray(
        meshes[mesh_id],
        ray_start_b,
        ray_dir_b,
        max_dist,
    )
    t = max_dist
    if result.result and result.t >= min_dist:
        t = result.t
    hit_distances[i, j, ray_id] = t


@wp.func
def _query_closest_point(
    mesh: wp.uint64,
    point_b: wp.vec3,
    max_dist: float,
    sign_mode: int,
):
    """Closest point on ``mesh`` to ``point_b`` (mesh-local).

    Returns ``(found, closest_b, distance)``. When ``sign_mode != 0``, distance is
    signed (negative inside) via ``mesh_query_point_sign_normal``.
    """
    if sign_mode == 0:
        query = wp.mesh_query_point_no_sign(mesh, point_b, max_dist)
        sign = float(1.0)
    else:
        query = wp.mesh_query_point_sign_normal(mesh, point_b, max_dist)
        sign = query.sign

    if query.result:
        closest_b = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        dist = wp.length(point_b - closest_b) * sign
        return wp.bool(True), closest_b, dist
    return wp.bool(False), point_b, max_dist


@wp.kernel(enable_backward=False)
def transform_and_query_point_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),
    points_w: wp.array(dtype=wp.vec3, ndim=2),
    enabled: wp.array(dtype=wp.bool, ndim=1),
    max_dist: float,
    sign_mode: int,
    closest_pos_w: wp.array(dtype=wp.vec3, ndim=3),
    distances: wp.array(dtype=wp.float32, ndim=3),
):
    """Fused world→mesh transform + closest-point query for every mesh."""
    i, mesh_id, point_id = wp.tid()
    if not enabled[i]:
        distances[i, mesh_id, point_id] = wp.INF
        closest_pos_w[i, mesh_id, point_id] = points_w[i, point_id]
        return

    quat_wxyz = mesh_quat_w[i, mesh_id]
    quat_xyzw = wp.quat(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0])
    point_w = points_w[i, point_id]
    point_b = wp.quat_rotate_inv(quat_xyzw, point_w - mesh_pos_w[i, mesh_id])

    found, closest_b, dist = _query_closest_point(meshes[mesh_id], point_b, max_dist, sign_mode)
    if found:
        closest_pos_w[i, mesh_id, point_id] = mesh_pos_w[i, mesh_id] + wp.quat_rotate(
            quat_xyzw, closest_b
        )
        distances[i, mesh_id, point_id] = dist
    else:
        closest_pos_w[i, mesh_id, point_id] = point_w
        distances[i, mesh_id, point_id] = max_dist


@wp.kernel(enable_backward=False)
def transform_and_query_point_against_meshes_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_indices: wp.array(dtype=wp.int64, ndim=2),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),
    points_w: wp.array(dtype=wp.vec3, ndim=2),
    enabled: wp.array(dtype=wp.bool, ndim=1),
    max_dist: float,
    sign_mode: int,
    closest_pos_w: wp.array(dtype=wp.vec3, ndim=3),
    distances: wp.array(dtype=wp.float32, ndim=3),
):
    """Fused closest-point query against a per-batch mesh subset."""
    i, j, point_id = wp.tid()
    mesh_id = mesh_indices[i, j]
    if not enabled[i]:
        distances[i, j, point_id] = wp.INF
        closest_pos_w[i, j, point_id] = points_w[i, point_id]
        return

    quat_wxyz = mesh_quat_w[i, j]
    quat_xyzw = wp.quat(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0])
    point_w = points_w[i, point_id]
    point_b = wp.quat_rotate_inv(quat_xyzw, point_w - mesh_pos_w[i, j])

    found, closest_b, dist = _query_closest_point(meshes[mesh_id], point_b, max_dist, sign_mode)
    if found:
        closest_pos_w[i, j, point_id] = mesh_pos_w[i, j] + wp.quat_rotate(quat_xyzw, closest_b)
        distances[i, j, point_id] = dist
    else:
        closest_pos_w[i, j, point_id] = point_w
        distances[i, j, point_id] = max_dist