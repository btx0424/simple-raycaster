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