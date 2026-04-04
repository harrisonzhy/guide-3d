#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
sys.path.insert(0, "/orcd/scratch/zhanghy/guide-3d/SegAnyGAussians/submodules/diff-gaussian-rasterization_contrastive_f")

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def get_visible_gaussians(radii):
    """
    Returns indices of Gaussians that are within the camera frustum.
    """
    return (radii > 0).nonzero(as_tuple=False).squeeze(-1)


def filter_gaussians_by_mask(visible_gaussians, means3D, mask, viewpoint_camera, H, W):
    """
    Projects visible Gaussian centers into screen space and keeps only
    those whose projection falls within the 2D boolean mask.

    Args:
        visible_gaussians: (K,) int tensor of indices into means3D
        means3D: (N, 3) all Gaussian centers
        mask: (H, W) boolean numpy array from LangSAM
        viewpoint_camera: camera with full_proj_transform
        H, W: frame dimensions

    Returns:
        (M,) int tensor of indices (subset of visible_gaussians)
    """
    vis_xyz = means3D[visible_gaussians]  # (K, 3)

    ones = torch.ones(vis_xyz.shape[0], 1, device=vis_xyz.device)
    vis_xyz_h = torch.cat([vis_xyz, ones], dim=1)  # (K, 4)

    # Project into clip space
    clip = vis_xyz_h @ viewpoint_camera.full_proj_transform  # (K, 4)

    # Perspective divide -> NDC [-1, 1]
    ndc = clip[:, :2] / clip[:, 3:4]  # (K, 2)

    # NDC to pixel coords
    px = ((ndc[:, 0] + 1) * 0.5 * W).long().clamp(0, W - 1)
    py = ((ndc[:, 1] + 1) * 0.5 * H).long().clamp(0, H - 1)

    mask_tensor = torch.from_numpy(mask).to(vis_xyz.device)  # (H, W)
    inside_mask = mask_tensor[py, px]  # (K,) bool

    return visible_gaussians[inside_mask]


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, filtered_mask=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    filtered_mask: (N,) bool tensor — Gaussians marked True will be hidden (opacity zeroed).
    """

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Zero out opacity for masked-out Gaussians
    if filtered_mask is not None:
        new_opacity = opacity.detach().clone()
        new_opacity[filtered_mask, :] = 0
        opacity = new_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "visible_gaussians": get_visible_gaussians(radii),
    }


def render_mask(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, precomputed_mask = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # start_time  = time.time()
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # print("Render time checker: raster_settings", time.time() - start_time)
    # start_time  = time.time()


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    mask = pc.get_mask if precomputed_mask is None else precomputed_mask
    if len(mask.shape) == 1 or mask.shape[-1] == 1:
        mask = mask.squeeze().unsqueeze(-1).repeat([1,3]).cuda()

    shs = None
    colors_precomp = mask

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # print("Render time checker: prepare vars", time.time() - start_time)


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_mask, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # print("Render time checker: main render", time.time() - start_time)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"mask": rendered_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

from diff_gaussian_rasterization_depth import GaussianRasterizationSettings as GaussianRasterizationSettingsDepth, GaussianRasterizer as GaussianRasterizerDepth

def render_with_depth(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, override_mask = None, filtered_mask = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # start_time  = time.time()
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsDepth(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # print("Render time checker: raster_settings", time.time() - start_time)
    # start_time  = time.time()


    rasterizer = GaussianRasterizerDepth(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    if filtered_mask is not None:
        new_opacity = opacity.detach().clone()
        new_opacity[filtered_mask, :] = -1.
        opacity = new_opacity

    mask = pc.get_mask if override_mask is None else override_mask

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # print("Render time checker: prepare vars", time.time() - start_time)


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_mask, rendered_depth, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        mask = mask,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # print("Render time checker: main render", time.time() - start_time)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_mask,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

from diff_gaussian_rasterization_contrastive_f import GaussianRasterizationSettings as GaussianRasterizationSettingsContrastiveF
from diff_gaussian_rasterization_contrastive_f import GaussianRasterizer as GaussianRasterizerContrastiveF
from scene.gaussian_model_ff import FeatureGaussianModel


def render_contrastive_feature(viewpoint_camera, pc : FeatureGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, norm_point_features = False, smooth_type = None, smooth_weights = None, smooth_K = 16):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettingsContrastiveF(
        image_height=int(viewpoint_camera.feature_height),
        image_width=int(viewpoint_camera.feature_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizerContrastiveF(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if smooth_type is None:
        colors_precomp = pc.get_point_features
    elif smooth_type == 'multi_res':
        colors_precomp = pc.get_multi_resolution_smoothed_point_features(smooth_weights = smooth_weights)
    elif smooth_type == 'traditional':
        colors_precomp = pc.get_smoothed_point_features(K = smooth_K, dropout=0.5)
    
    if norm_point_features:
        colors_precomp = colors_precomp / (colors_precomp.norm(dim=1, keepdim=True) + 1e-9)
    # colors_precomp = torch.nn.functional.normalize(colors_precomp, dim=1)
    colors_precomp = colors_precomp.contiguous().float()

    # print("colors_precomp shape:", colors_precomp.shape, "dtype:", colors_precomp.dtype)
    # print("colors_precomp contiguous:", colors_precomp.is_contiguous())
    # print("means3D shape:", means3D.shape, "dtype:", means3D.dtype)
    # print("scales shape:", scales.shape, "dtype:", scales.dtype)
    # print("opacity shape:", opacity.shape, "dtype:", opacity.dtype)
    # print("feature_height:", viewpoint_camera.feature_height)
    # print("feature_width:", viewpoint_camera.feature_width)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            }

