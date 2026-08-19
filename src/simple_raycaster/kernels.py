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
    hit_normals: wp.array(dtype=wp.vec3, ndim=3),
):
    i, mesh_id, ray_id = wp.tid()
    if not enabled[i]:
        hit_distances[i, mesh_id, ray_id] = wp.INF
        hit_normals[i, mesh_id, ray_id] = wp.vec3(0.0, 0.0, 0.0)
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
    # normal is in mesh-local frame; caller rotates it back to world
    normal = wp.vec3(0.0, 0.0, 0.0)
    if result.result and result.t >= min_dist:
        t = result.t
        normal = result.normal
    hit_distances[i, mesh_id, ray_id] = t
    hit_normals[i, mesh_id, ray_id] = normal


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
    hit_normals: wp.array(dtype=wp.vec3, ndim=3),
):
    i, j, ray_id = wp.tid()
    mesh_id = mesh_indices[i, j]
    if not enabled[i]:
        hit_distances[i, j, ray_id] = wp.INF
        hit_normals[i, j, ray_id] = wp.vec3(0.0, 0.0, 0.0)
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
    # normal is in mesh-local frame; caller rotates it back to world
    normal = wp.vec3(0.0, 0.0, 0.0)
    if result.result and result.t >= min_dist:
        t = result.t
        normal = result.normal
    hit_distances[i, j, ray_id] = t
    hit_normals[i, j, ray_id] = normal


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
    hit_normals: wp.array(dtype=wp.vec3, ndim=3),
):
    i, mesh_id, ray_id = wp.tid()
    if not enabled[i]:
        hit_distances[i, mesh_id, ray_id] = wp.INF
        hit_normals[i, mesh_id, ray_id] = wp.vec3(0.0, 0.0, 0.0)
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
    normal_w = wp.vec3(0.0, 0.0, 0.0)
    if result.result and result.t >= min_dist:
        t = result.t
        # rotate mesh-local normal back to world frame
        normal_w = wp.quat_rotate(quat_xyzw, result.normal)
    hit_distances[i, mesh_id, ray_id] = t
    hit_normals[i, mesh_id, ray_id] = normal_w


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
    hit_normals: wp.array(dtype=wp.vec3, ndim=3),
):
    i, j, ray_id = wp.tid()
    mesh_id = mesh_indices[i, j]
    if not enabled[i]:
        hit_distances[i, j, ray_id] = wp.INF
        hit_normals[i, j, ray_id] = wp.vec3(0.0, 0.0, 0.0)
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
    normal_w = wp.vec3(0.0, 0.0, 0.0)
    if result.result and result.t >= min_dist:
        t = result.t
        # rotate mesh-local normal back to world frame
        normal_w = wp.quat_rotate(quat_xyzw, result.normal)
    hit_distances[i, j, ray_id] = t
    hit_normals[i, j, ray_id] = normal_w


@wp.kernel(enable_backward=False)
def transform_and_raycast_closest_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),
    ray_starts_w: wp.array(dtype=wp.vec3, ndim=2),
    ray_dirs_w: wp.array(dtype=wp.vec3, ndim=2),
    enabled: wp.array(dtype=wp.bool, ndim=1),
    min_dist: float,
    max_dist: float,
    hit_distances: wp.array(dtype=wp.float32, ndim=2),
    hit_normals: wp.array(dtype=wp.vec3, ndim=2),
):
    """Fused world→mesh transform + closest-hit raycast. Writes ``[N, n_rays]``."""
    i, ray_id = wp.tid()
    if not enabled[i]:
        hit_distances[i, ray_id] = wp.INF
        hit_normals[i, ray_id] = wp.vec3(0.0, 0.0, 0.0)
        return

    n_meshes = int(mesh_pos_w.shape[1])
    ray_start_w = ray_starts_w[i, ray_id]
    ray_dir_w = ray_dirs_w[i, ray_id]
    best_t = float(max_dist)
    best_n = wp.vec3(0.0, 0.0, 0.0)

    for m in range(n_meshes):
        quat_wxyz = mesh_quat_w[i, m]
        quat_xyzw = wp.quat(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0])
        ray_start_b = wp.quat_rotate_inv(quat_xyzw, ray_start_w - mesh_pos_w[i, m])
        ray_dir_b = wp.quat_rotate_inv(quat_xyzw, ray_dir_w)
        result = wp.mesh_query_ray(meshes[m], ray_start_b, ray_dir_b, max_dist)
        if result.result and result.t >= min_dist and result.t < best_t:
            best_t = result.t
            best_n = wp.quat_rotate(quat_xyzw, result.normal)

    hit_distances[i, ray_id] = best_t
    hit_normals[i, ray_id] = best_n


@wp.kernel(enable_backward=False)
def transform_and_raycast_closest_against_meshes_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_indices: wp.array(dtype=wp.int64, ndim=2),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),
    ray_starts_w: wp.array(dtype=wp.vec3, ndim=2),
    ray_dirs_w: wp.array(dtype=wp.vec3, ndim=2),
    enabled: wp.array(dtype=wp.bool, ndim=1),
    min_dist: float,
    max_dist: float,
    hit_distances: wp.array(dtype=wp.float32, ndim=2),
    hit_normals: wp.array(dtype=wp.vec3, ndim=2),
):
    """Closest-hit fused raycast against a per-batch mesh subset."""
    i, ray_id = wp.tid()
    if not enabled[i]:
        hit_distances[i, ray_id] = wp.INF
        hit_normals[i, ray_id] = wp.vec3(0.0, 0.0, 0.0)
        return

    n_subset = int(mesh_indices.shape[1])
    ray_start_w = ray_starts_w[i, ray_id]
    ray_dir_w = ray_dirs_w[i, ray_id]
    best_t = float(max_dist)
    best_n = wp.vec3(0.0, 0.0, 0.0)

    for j in range(n_subset):
        mesh_id = int(mesh_indices[i, j])
        quat_wxyz = mesh_quat_w[i, j]
        quat_xyzw = wp.quat(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0])
        ray_start_b = wp.quat_rotate_inv(quat_xyzw, ray_start_w - mesh_pos_w[i, j])
        ray_dir_b = wp.quat_rotate_inv(quat_xyzw, ray_dir_w)
        result = wp.mesh_query_ray(meshes[mesh_id], ray_start_b, ray_dir_b, max_dist)
        if result.result and result.t >= min_dist and result.t < best_t:
            best_t = result.t
            best_n = wp.quat_rotate(quat_xyzw, result.normal)

    hit_distances[i, ray_id] = best_t
    hit_normals[i, ray_id] = best_n


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
def raycast_camera_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),  # [N, M]
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),  # [N, M] WXYZ
    mesh_albedo: wp.array(dtype=wp.vec3, ndim=1),  # [M]
    cam_pos_w: wp.array(dtype=wp.vec3, ndim=1),  # [N]
    cam_quat_w: wp.array(dtype=wp.vec4, ndim=1),  # [N] WXYZ
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float,
    far: float,
    convention: int,  # 0 = OpenCV (+Z fwd), 1 = OpenGL (−Z fwd)
    depth_mode: int,  # 0 = planar, 1 = ray t
    light_dir_w: wp.vec3,
    ambient: float,
    diffuse: float,
    # Out [N, H, W, …] — flattened as [N, H*W] then written with (y,x)
    rgb_out: wp.array(dtype=wp.vec3, ndim=2),  # [N, H*W]
    depth_out: wp.array(dtype=wp.float32, ndim=2),  # [N, H*W]
    mask_out: wp.array(dtype=wp.bool, ndim=2),  # [N, H*W]
):
    """Per-pixel closest-hit RGB-D over posed triangle meshes (nvdiffrast alternative)."""
    n, pix = wp.tid()
    n_meshes = mesh_pos_w.shape[1]
    px = pix % width
    py = pix // width

    u = (float(px) + 0.5 - cx) / fx
    v = (float(py) + 0.5 - cy) / fy
    if convention == 0:
        # OpenCV: +X right, +Y down, +Z forward
        dir_c = wp.normalize(wp.vec3(u, v, 1.0))
        optical_z = dir_c[2]
    else:
        # OpenGL: +X right, +Y up, −Z forward
        dir_c = wp.normalize(wp.vec3(u, -v, -1.0))
        optical_z = -dir_c[2]

    cq = cam_quat_w[n]
    cq_xyzw = wp.quat(cq[1], cq[2], cq[3], cq[0])
    origin = cam_pos_w[n]
    dir_w = wp.quat_rotate(cq_xyzw, dir_c)

    best_t = float(far)
    best_n = wp.vec3(0.0, 0.0, 0.0)
    best_albedo = wp.vec3(0.65, 0.65, 0.65)
    hit = wp.bool(False)

    for m in range(n_meshes):
        mq = mesh_quat_w[n, m]
        mq_xyzw = wp.quat(mq[1], mq[2], mq[3], mq[0])
        mpos = mesh_pos_w[n, m]
        ray_o_b = wp.quat_rotate_inv(mq_xyzw, origin - mpos)
        ray_d_b = wp.quat_rotate_inv(mq_xyzw, dir_w)
        result = wp.mesh_query_ray(meshes[m], ray_o_b, ray_d_b, far)
        if result.result and result.t >= near and result.t < best_t:
            best_t = result.t
            best_n = wp.quat_rotate(mq_xyzw, result.normal)
            best_albedo = mesh_albedo[m]
            hit = wp.bool(True)

    if hit:
        # Face normal toward camera
        if wp.dot(best_n, dir_w) > 0.0:
            best_n = -best_n
        nlen = wp.length(best_n)
        if nlen > 1.0e-8:
            best_n = best_n / nlen
        l = light_dir_w / wp.max(wp.length(light_dir_w), 1.0e-8)
        ndotl = wp.max(wp.dot(best_n, l), 0.0)
        rgb_out[n, pix] = best_albedo * (ambient + diffuse * ndotl)
        if depth_mode == 0:
            depth_out[n, pix] = best_t * optical_z
        else:
            depth_out[n, pix] = best_t
        mask_out[n, pix] = wp.bool(True)
    else:
        rgb_out[n, pix] = wp.vec3(0.0, 0.0, 0.0)
        depth_out[n, pix] = far
        mask_out[n, pix] = wp.bool(False)


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