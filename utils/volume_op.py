import torch


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    This function is modified from https://github.com/kwea123/nerf_pl.

    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (W / (2. * focal)) * ox_oz
    o1 = -1. / (H / (2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def get_ndc_rays_fxfy(H, W, fxfy, near, rays_o, rays_d):
    """
    This function is modified from https://github.com/kwea123/nerf_pl.

    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (W / (2. * fxfy[0])) * ox_oz
    o1 = -1. / (H / (2. * fxfy[1])) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * fxfy[0])) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H / (2. * fxfy[1])) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d



def rayPlaneInter(n, p0, ray_o, ray_d):
    s1 = torch.sum(p0 * n, dim=1)
    s2 = torch.sum(ray_o * n, dim=1)
    s3 = torch.sum(ray_d * n, dim=1)
    
    dist = (s1 - s2) / s3

    dist_group = torch.broadcast_to(dist.unsqueeze(1), (dist.shape[0], 3))

    inter_point = ray_o + dist_group * ray_d

    return inter_point

def volume_sampling_ndc(c2w, ray_dir_cam, near, far, H, W, focal, perturb_t, st_scale):
    
    ray_dir_world = torch.matmul(c2w[:3, :3].view(1, 1, 3, 3),
                                 ray_dir_cam.unsqueeze(3)).squeeze(3)  # (1, 1, 3, 3) * (H, W, 3, 1) -> (H, W, 3)
    ray_ori_world = c2w[:3, 3]  # the translation vector (3, )

    ray_dir_world = ray_dir_world.reshape(-1, 3)  # (H, W, 3) -> (H*W, 3)
    ray_ori_world = ray_ori_world.view(1, 3).expand_as(ray_dir_world)  # (3, ) -> (1, 3) -> (H*W, 3)

    if isinstance(focal, float):
        ray_ori_world, ray_dir_world = get_ndc_rays(H, W, focal, 1.0, rays_o=ray_ori_world, rays_d=ray_dir_world)  # (H*W, 3)
    else:  # if focal is a tensor contains fxfy
        ray_ori_world, ray_dir_world = get_ndc_rays_fxfy(H, W, focal, 1.0, rays_o=ray_ori_world, rays_d=ray_dir_world)  # (H*W, 3)

    ray_dir_world = ray_dir_world.reshape(-1, 3)  # (H*W, 3)
    ray_ori_world = ray_ori_world.reshape(-1, 3)  # (H*W, 3)
    plane_normal = torch.tensor([0.0, 0.0, 1.0], device = ray_dir_world.device).expand(ray_ori_world.shape)
    p_uv = torch.tensor([0.0, 0.0, 1.0], device = ray_dir_world.device).expand(ray_ori_world.shape)
    p_st = torch.tensor([0.0, 0.0, 0.0], device = ray_dir_world.device).expand(ray_ori_world.shape)
    inter_uv = rayPlaneInter(plane_normal, p_uv, ray_ori_world, ray_dir_world)
    inter_st = rayPlaneInter(plane_normal, p_st, ray_ori_world, ray_dir_world)

    data_uvst = torch.stack(
        [
            inter_st[:, 0]*st_scale.item(),
            inter_st[:, 1]*st_scale.item(),
            near * torch.ones_like(inter_st[:, 0]),
            inter_uv[:, 0] - inter_st[:, 0],
            inter_uv[:, 1] - inter_st[:, 1],
            (far - near) * torch.ones_like(inter_st[:, 0]),
        ],
        axis=-1
    ).view(-1, 6)

    return data_uvst
