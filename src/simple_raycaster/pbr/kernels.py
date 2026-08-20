import warp as wp


@wp.kernel(enable_backward=False)
def raycast_gbuffer_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),
    mesh_roughness: wp.array(dtype=wp.float32, ndim=1),
    mesh_metallic: wp.array(dtype=wp.float32, ndim=1),
    vert_colors: wp.array(dtype=wp.vec3, ndim=1),
    vert_normals: wp.array(dtype=wp.vec3, ndim=1),
    vert_offset: wp.array(dtype=wp.int32, ndim=1),
    cam_pos_w: wp.array(dtype=wp.vec3, ndim=1),
    cam_quat_w: wp.array(dtype=wp.vec4, ndim=1),
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float,
    far: float,
    convention: int,
    depth_mode: int,
    use_smooth: int,
    depth_out: wp.array(dtype=wp.float32, ndim=2),
    mask_out: wp.array(dtype=wp.bool, ndim=2),
    normal_out: wp.array(dtype=wp.vec3, ndim=2),
    albedo_out: wp.array(dtype=wp.vec3, ndim=2),
    roughness_out: wp.array(dtype=wp.float32, ndim=2),
    metallic_out: wp.array(dtype=wp.float32, ndim=2),
    view_out: wp.array(dtype=wp.vec3, ndim=2),
):
    """Closest-hit G-buffer. Barycentric vertex colors / smooth normals; no shade."""
    n, pix = wp.tid()
    n_meshes = mesh_pos_w.shape[1]
    px = pix % width
    py = pix // width

    u_px = (float(px) + 0.5 - cx) / fx
    v_px = (float(py) + 0.5 - cy) / fy
    if convention == 0:
        dir_c = wp.normalize(wp.vec3(u_px, v_px, 1.0))
        optical_z = dir_c[2]
    else:
        dir_c = wp.normalize(wp.vec3(u_px, -v_px, -1.0))
        optical_z = -dir_c[2]

    cq = cam_quat_w[n]
    cq_xyzw = wp.quat(cq[1], cq[2], cq[3], cq[0])
    origin = cam_pos_w[n]
    dir_w = wp.quat_rotate(cq_xyzw, dir_c)
    view_out[n, pix] = -dir_w

    best_t = float(far)
    best_n = wp.vec3(0.0, 0.0, 0.0)
    best_albedo = wp.vec3(0.65, 0.65, 0.65)
    best_r = float(0.45)
    best_m = float(0.0)
    hit = wp.bool(False)

    for m in range(n_meshes):
        mq = mesh_quat_w[n, m]
        mq_xyzw = wp.quat(mq[1], mq[2], mq[3], mq[0])
        mpos = mesh_pos_w[n, m]
        ray_o_b = wp.quat_rotate_inv(mq_xyzw, origin - mpos)
        ray_d_b = wp.quat_rotate_inv(mq_xyzw, dir_w)
        result = wp.mesh_query_ray(meshes[m], ray_o_b, ray_d_b, best_t)
        if result.result and result.t >= near and result.t < best_t:
            best_t = result.t
            # Barycentric weights for mesh_query_ray (Warp ≥1.6):
            # hit = u*v0 + v*v1 + (1-u-v)*v2 — not the mesh_eval_position convention.
            # (not the mesh_eval_position convention).
            w0 = result.u
            w1 = result.v
            w2 = float(1.0) - w0 - w1
            face = result.face
            mid = meshes[m]
            i0 = wp.mesh_get_index(mid, face * 3 + 0)
            i1 = wp.mesh_get_index(mid, face * 3 + 1)
            i2 = wp.mesh_get_index(mid, face * 3 + 2)
            base = vert_offset[m]
            best_albedo = (
                vert_colors[base + i0] * w0
                + vert_colors[base + i1] * w1
                + vert_colors[base + i2] * w2
            )
            if use_smooth != 0:
                n_local = (
                    vert_normals[base + i0] * w0
                    + vert_normals[base + i1] * w1
                    + vert_normals[base + i2] * w2
                )
                nlen = wp.length(n_local)
                if nlen > 1.0e-8:
                    best_n = wp.quat_rotate(mq_xyzw, n_local / nlen)
                else:
                    best_n = wp.quat_rotate(mq_xyzw, result.normal)
            else:
                best_n = wp.quat_rotate(mq_xyzw, result.normal)
            best_r = mesh_roughness[m]
            best_m = mesh_metallic[m]
            hit = wp.bool(True)

    if hit:
        if wp.dot(best_n, dir_w) > 0.0:
            best_n = -best_n
        nlen = wp.length(best_n)
        if nlen > 1.0e-8:
            best_n = best_n / nlen
        if depth_mode == 0:
            depth_out[n, pix] = best_t * optical_z
        else:
            depth_out[n, pix] = best_t
        mask_out[n, pix] = wp.bool(True)
        normal_out[n, pix] = best_n
        albedo_out[n, pix] = best_albedo
        roughness_out[n, pix] = best_r
        metallic_out[n, pix] = best_m
    else:
        depth_out[n, pix] = far
        mask_out[n, pix] = wp.bool(False)
        normal_out[n, pix] = wp.vec3(0.0, 0.0, 0.0)
        albedo_out[n, pix] = wp.vec3(0.0, 0.0, 0.0)
        roughness_out[n, pix] = float(1.0)
        metallic_out[n, pix] = float(0.0)


@wp.kernel(enable_backward=False)
def raycast_ortho_depth_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),
    center_w: wp.array(dtype=wp.vec3, ndim=1),
    right_w: wp.vec3,
    up_w: wp.vec3,
    forward_w: wp.vec3,
    half_extent: float,
    catch_dist: float,
    size: int,
    near: float,
    far: float,
    depth_out: wp.array(dtype=wp.float32, ndim=2),
):
    """Directional sun shadow map: ortho closest-hit depth along ``forward_w``."""
    n, pix = wp.tid()
    n_meshes = mesh_pos_w.shape[1]
    px = pix % size
    py = pix // size
    u = ((float(px) + 0.5) / float(size) - 0.5) * 2.0 * half_extent
    v = ((float(py) + 0.5) / float(size) - 0.5) * 2.0 * half_extent
    origin = center_w[n] - forward_w * catch_dist + right_w * u + up_w * v
    dir_w = forward_w

    best_t = float(far)
    hit = wp.bool(False)
    for m in range(n_meshes):
        mq = mesh_quat_w[n, m]
        mq_xyzw = wp.quat(mq[1], mq[2], mq[3], mq[0])
        mpos = mesh_pos_w[n, m]
        ray_o_b = wp.quat_rotate_inv(mq_xyzw, origin - mpos)
        ray_d_b = wp.quat_rotate_inv(mq_xyzw, dir_w)
        result = wp.mesh_query_ray(meshes[m], ray_o_b, ray_d_b, best_t)
        if result.result and result.t >= near and result.t < best_t:
            best_t = result.t
            hit = wp.bool(True)

    if hit:
        depth_out[n, pix] = best_t
    else:
        depth_out[n, pix] = far


@wp.kernel(enable_backward=False)
def shadow_ray_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),
    hit_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mask: wp.array(dtype=wp.bool, ndim=2),
    sun_dir_w: wp.vec3,
    bias: float,
    tmax: float,
    vis_out: wp.array(dtype=wp.float32, ndim=2),
):
    """Debug: one mesh ray from the hit toward the sun. ``vis=0`` if occluded."""
    n, pix = wp.tid()
    if not mask[n, pix]:
        vis_out[n, pix] = float(1.0)
        return

    llen = wp.length(sun_dir_w)
    l = sun_dir_w / wp.max(llen, 1.0e-8)
    origin = hit_pos_w[n, pix] + l * bias
    n_meshes = mesh_pos_w.shape[1]
    best_t = float(tmax)
    occluded = wp.bool(False)
    for m in range(n_meshes):
        mq = mesh_quat_w[n, m]
        mq_xyzw = wp.quat(mq[1], mq[2], mq[3], mq[0])
        mpos = mesh_pos_w[n, m]
        ray_o_b = wp.quat_rotate_inv(mq_xyzw, origin - mpos)
        ray_d_b = wp.quat_rotate_inv(mq_xyzw, l)
        result = wp.mesh_query_ray(meshes[m], ray_o_b, ray_d_b, best_t)
        if result.result and result.t >= 0.0 and result.t < best_t:
            best_t = result.t
            occluded = wp.bool(True)

    if occluded:
        vis_out[n, pix] = float(0.0)
    else:
        vis_out[n, pix] = float(1.0)
