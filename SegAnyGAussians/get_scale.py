import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace
import cv2

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureGaussianModel

import gaussian_renderer
import importlib
importlib.reload(gaussian_renderer)

import os
import gc

FEATURE_DIM = 32

DATA_ROOT = './data/nerf_llff_data_for_3dgs/'
ALLOW_PRINCIPLE_POINT_SHIFT = False


def get_combined_args(parser: ArgumentParser):
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()

    target_cfg_file = "cfg_args"
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        print("Type error")
        pass

    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v

    return Namespace(**merged_dict)


def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid


if __name__ == '__main__':
    parser = ArgumentParser(description="Get scales for SAM masks")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)

    parser.add_argument("--image_root", default='../mats/data/images1/bicycle', type=str)
    parser.add_argument("--source", default='../mats/data/images1/bicycle', type=str)

    args = get_combined_args(parser)

    dataset = model.extract(args)
    dataset.need_features = False
    dataset.need_masks = False
    dataset.allow_principle_point_shift = ALLOW_PRINCIPLE_POINT_SHIFT

    feature_gaussians = None
    scene_gaussians = GaussianModel(dataset.sh_degree)

    dataset.source_path = args.source
    print("Source path:", dataset.source_path)

    scene = Scene(
        dataset,
        scene_gaussians,
        feature_gaussians,
        load_iteration=-1,
        feature_load_iteration=-1,
        shuffle=False,
        mode='eval',
        target='scene'
    )

    assert os.path.exists(os.path.join(dataset.source_path, 'images')) and "Please specify a valid image root."
    assert os.path.join(dataset.source_path, 'sam_masks') and "Please run extract_segment_everything_masks first."

    OUTPUT_DIR = os.path.join(args.image_root, 'mask_scales')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cameras = scene.getTrainCameras()
    background = torch.zeros(scene_gaussians.get_mask.shape[0], 3, device='cuda')

    kernel = torch.full((1, 1, 3, 3), 1.0)
    CHUNK = 8

    from tqdm import tqdm
    with torch.no_grad():
        for it, view in tqdm(enumerate(cameras)):
            rendered_pkg = gaussian_renderer.render_with_depth(
                view, scene_gaussians, pipeline.extract(args), background
            )
            depth = rendered_pkg['depth'].cpu().squeeze()

            H, W = depth.shape
            grid_index = generate_grid_index(depth)

            points_in_3D = torch.zeros(H, W, 3).cpu()
            points_in_3D[:, :, -1] = depth

            cx = W / 2
            cy = H / 2
            fx = cx / np.tan(cameras[0].FoVx / 2)
            fy = cy / np.tan(cameras[0].FoVy / 2)

            points_in_3D[:, :, 0] = (grid_index[:, :, 0] - cx) * depth / fx
            points_in_3D[:, :, 1] = (grid_index[:, :, 1] - cy) * depth / fy

            name = view.image_name
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                mask_name = name.rsplit(".", 1)[0] + ".pt"
            else:
                mask_name = name + ".pt"

            mask_file = os.path.join(dataset.source_path, "sam_masks", mask_name)
            corresponding_masks = torch.load(mask_file, map_location="cpu")

            num_masks = len(corresponding_masks)
            scale = torch.zeros(num_masks)

            for start in range(0, num_masks, CHUNK):
                end = min(start + CHUNK, num_masks)
                cm = corresponding_masks[start:end].cpu().float()

                upsampled_mask = torch.nn.functional.interpolate(
                    cm.unsqueeze(1),
                    mode='bilinear',
                    size=(H, W),
                    align_corners=False
                )

                eroded_masks = torch.conv2d(
                    upsampled_mask.float(),
                    kernel,
                    padding=1,
                )
                eroded_masks = (eroded_masks >= 5).squeeze(1)

                for j in range(end - start):
                    mask_bool = (eroded_masks[j] == 1)
                    point_in_3D_in_mask = points_in_3D[mask_bool]
                    scale[start + j] = (point_in_3D_in_mask.std(dim=0) * 2).norm()

                del cm, upsampled_mask, eroded_masks
                gc.collect()

            torch.save(scale, os.path.join(OUTPUT_DIR, view.image_name + '.pt'))

            del rendered_pkg, depth, grid_index, points_in_3D, corresponding_masks, scale
            gc.collect()
            torch.cuda.empty_cache()
