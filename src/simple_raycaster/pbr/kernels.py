import warp as wp


@wp.kernel(enable_backward=False)
def raycast_gbuffer_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_pos_w: wp.array(dtype=wp.vec3, ndim=2),
    mesh_quat_w: wp.array(dtype=wp.vec4, ndim=2),
    mesh_albedo: wp.array(dtype=wp.vec3, ndim=1),
    mesh_roughness: wp.array(dtype=wp.float32, ndim=1),
    mesh_metallic: wp.array(dtype=wp.float32, ndim=1),
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
    depth_out: wp.array(dtype=wp.float32, ndim=2),
    mask_out: wp.array(dtype=wp.bool, ndim=2),
    normal_out: wp.array(dtype=wp.vec3, ndim=2),
    albedo_out: wp.array(dtype=wp.vec3, ndim=2),
    roughness_out: wp.array(dtype=wp.float32, ndim=2),
    metallic_out: wp.array(dtype=wp.float32, ndim=2),
    view_out: wp.array(dtype=wp.vec3, ndim=2),
):
    """Closest-hit G-buffer (no shade). Same ray math as ``raycast_camera_kernel``."""
    n, pix = wp.tid()
    n_meshes = mesh_pos_w.shape[1]
    px = pix % width
    py = pix // width

    u = (float(px) + 0.5 - cx) / fx
    v = (float(py) + 0.5 - cy) / fy
    if convention == 0:
        dir_c = wp.normalize(wp.vec3(u, v, 1.0))
        optical_z = dir_c[2]
    else:
        dir_c = wp.normalize(wp.vec3(u, -v, -1.0))
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
            best_n = wp.quat_rotate(mq_xyzw, result.normal)
            best_albedo = mesh_albedo[m]
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
