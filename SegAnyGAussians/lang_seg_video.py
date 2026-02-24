"""
Language-driven 3DGS Segmentation + video stitching of camera views.
"""

import os
import argparse
from typing import Callable

import numpy as np
import torch

original_set_default_dtype = torch.set_default_dtype
def patched_set_default_dtype(dtype):
    if dtype != torch.float32:
        import traceback
        print(f"WARNING: someone is setting default dtype to {dtype}")
        traceback.print_stack()
    original_set_default_dtype(dtype)
torch.set_default_dtype = patched_set_default_dtype

from hdbscan import HDBSCAN
import imageio.v2 as iio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from argparse import ArgumentParser, Namespace

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureGaussianModel
from gaussian_renderer import render, render_contrastive_feature

import importlib
import clip_utils
importlib.reload(clip_utils)
from clip_utils import get_scores_with_template
from clip_utils.clip_utils import load_clip

from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer

from PIL import Image

sam_dir = "../lang-segment-anything/checkpoints/sam/sam2.1_hiera_small.pt"
gdino_dir = "../lang-segment-anything/checkpoints/gdino/grounding-dino-base"

ALLOW_PRINCIPLE_POINT_SHIFT = False
FEATURE_DIM = 32
RENDER_DOWNSCALE = 8


def get_combined_args(parser: ArgumentParser, model_path, target_cfg_file=None):
    cmdlne_string = ["--model_path", model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    if target_cfg_file is None:
        if args_cmdline.target == "seg":
            target_cfg_file = "seg_cfg_args"
        elif args_cmdline.target in ("scene", "xyz"):
            target_cfg_file = "cfg_args"
        elif args_cmdline.target in ("feature", "coarse_seg_everything", "contrastive_feature"):
            target_cfg_file = "feature_cfg_args"

    try:
        cfgfilepath = os.path.join(model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found:", cfgfilepath)
            cfgfile_string = cfg_file.read()
    except TypeError:
        pass

    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def _to_uint8_rgb(rendered_image_3hw: torch.Tensor) -> np.ndarray:
    """[3,H,W] float tensor -> uint8 [H,W,3] numpy array for video writing."""
    img = rendered_image_3hw.detach().permute(1, 2, 0).contiguous()
    img = torch.clamp(img, 0.0, 1.0)
    return (img * 255.0).round().to(torch.uint8).cpu().numpy()


def cluster_id_to_scales(cluster_labels, flattened_scales, cluster_idx, scores):
    mask = cluster_labels == cluster_idx
    max_score_mask_scale_id = scores[mask].argmax()
    return flattened_scales[mask][max_score_mask_scale_id].item(), max_score_mask_scale_id


def get_similarity_map(
    point_features: torch.Tensor,
    scale: float,
    scale_gate: Callable,
    clip_query_feature: torch.Tensor,
    q_trans: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    scale_t = torch.full((1,), scale).cuda()
    scale_t = q_trans(scale_t)
    gates = scale_gate(scale_t).detach().squeeze()
    scale_conditioned = point_features * gates.unsqueeze(0)
    normed = torch.nn.functional.normalize(scale_conditioned, dim=-1, p=2)
    return torch.einsum("C,NC->N", clip_query_feature, normed)


def get_best_cluster_query(cluster_labels, flattened_mask_features, flattened_scales, scores, exclude_clusters=None):
    """Return the mask feature and scale of the highest-scoring mask, excluding specified clusters."""
    non_noise = cluster_labels.unique()
    non_noise = non_noise[non_noise != -1]
    if exclude_clusters is not None:
        for exc in exclude_clusters:
            non_noise = non_noise[non_noise != exc]
    if len(non_noise) == 0:
        raise ValueError("No clusters left after exclusion for negative prompt.")
    cluster_mean_scores = torch.stack([scores[cluster_labels == c].mean() for c in non_noise])
    best_c = non_noise[cluster_mean_scores.argmax()]
    s, ind = cluster_id_to_scales(cluster_labels, flattened_scales, best_c, scores)
    feat = torch.nn.functional.normalize(
        flattened_mask_features[cluster_labels == best_c][ind], dim=-1, p=2
    )
    return feat, s

def get_quantile_func(scales: torch.Tensor, distribution="normal"):
    """Fit a quantile transformer on 3D scale statistics."""
    scales_np = scales.flatten().detach().cpu().numpy()
    print(f"Scale max: {scales_np.max():.4f}")
    qt = QuantileTransformer(output_distribution=distribution)
    qt.fit(scales_np.reshape(-1, 1))

    def quantile_transformer_func(scales: torch.Tensor) -> torch.Tensor:
        shape = scales.shape
        return torch.tensor(
            qt.transform(scales.reshape(-1, 1).detach().cpu().numpy()),
            dtype=torch.float32,
        ).to(scales.device).reshape(shape)

    return quantile_transformer_func, qt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True,
                        help='Positive prompt e.g. "bicycle"')
    parser.add_argument("--nprompt", type=str, default=None,
                        help='Negative prompt e.g. "bench". Suppresses matching regions.')
    parser.add_argument("--nprompt_weight", type=float, default=1.0,
                        help="How strongly to subtract negative similarity (default: 1.0).")
    parser.add_argument("--video_path", type=str, default="output_videos/bicycle_segmented_views.mp4")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--similarity_threshold", type=float, default=0.35,
                        help="Threshold for segmenting 3D Gaussians by similarity.")
    parser.add_argument("--cluster_score_threshold", type=float, default=0.45,
                        help="Min CLIP score to consider a cluster relevant.")
    args_cli = parser.parse_args()

    MODEL_PATH = "../gaussian-splatting/output/a1c12f3b-d/"
    FEATURE_GAUSSIAN_ITERATION = 30000
    SCALE_GATE_PATH = os.path.join(
        MODEL_PATH, f"point_cloud/iteration_{FEATURE_GAUSSIAN_ITERATION}/scale_gate.pt"
    )

    # Load scale gate
    scale_gate = torch.nn.Sequential(torch.nn.Linear(1, 32, bias=True), torch.nn.Sigmoid())
    scale_gate.load_state_dict(torch.load(SCALE_GATE_PATH))
    scale_gate = scale_gate.cuda().eval()

    # Load scene
    parser2 = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser2, sentinel=True)
    pipeline = PipelineParams(parser2)
    parser2.add_argument("--target", default="scene", type=str)
    args = get_combined_args(parser2, MODEL_PATH)

    dataset = model.extract(args)
    dataset.need_features = True
    dataset.need_masks = True
    dataset.allow_principle_point_shift = ALLOW_PRINCIPLE_POINT_SHIFT
    dataset.source_path = "../mats/data/images1/bicycle"
    print("Source path:", dataset.source_path)

    scene_gaussians = GaussianModel(dataset.sh_degree)
    feature_gaussians = FeatureGaussianModel(FEATURE_DIM)
    scene = Scene(
        dataset, scene_gaussians, feature_gaussians,
        load_iteration=-1,
        feature_load_iteration=FEATURE_GAUSSIAN_ITERATION,
        shuffle=False, mode="eval", target="contrastive_feature",
    )

    # Build quantile transformer from all training camera scales
    all_scales = torch.cat([cam.mask_scales for cam in scene.getTrainCameras()])
    q_trans, _ = get_quantile_func(all_scales, "uniform")

    # Sample anchor points in 3D feature space
    pf = feature_gaussians.get_point_features
    keep = torch.rand(pf.shape[0], device=pf.device) > 0.99
    anchor_point_features = pf[keep]
    print(f"Anchor points: {len(anchor_point_features)}")

    cameras = scene.getTrainCameras()
    print(f"Views in dataset: {len(cameras)}")

    # Per-view feature extraction
    seg_features = []
    clip_features = []
    scales_list = []
    mask_identifiers = []
    camera_id_mask_id = []

    feature_background = torch.zeros(FEATURE_DIM, dtype=torch.float32, device="cuda")

    for i, view in enumerate(tqdm(cameras, desc="Extracting features")):
        torch.cuda.empty_cache()

        orig_h, orig_w = view.original_image.shape[-2:]
        old_fh, old_fw = getattr(view, "feature_height", None), getattr(view, "feature_width", None)
        view.feature_height, view.feature_width = orig_h, orig_w

        rendered_feature = render_contrastive_feature(
            view, feature_gaussians, pipeline.extract(args),
            feature_background, norm_point_features=True,
        )["render"]
        view.feature_height, view.feature_width = old_fh, old_fw

        feature_h, feature_w = rendered_feature.shape[-2:]

        with torch.no_grad():
            rendered_feature = torch.nn.functional.interpolate(
                rendered_feature.unsqueeze(0),
                (feature_h // 2, feature_w // 2),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

            # Truncate masks and scales to matching length
            n_masks = min(view.original_masks.shape[0], view.mask_scales.shape[0])

            clip_features.append(view.original_features[:n_masks])

            sam_masks = view.original_masks[:n_masks].cuda().unsqueeze(1).float()
            sam_masks = torch.nn.functional.interpolate(
                sam_masks, (feature_h // 2, feature_w // 2),
                mode="bilinear", align_corners=False,
            )
            sam_masks = torch.nn.functional.conv2d(
                sam_masks.cpu(), torch.full((1, 1, 3, 3), 1.0), padding=1,
            ).cuda()
            sam_masks = sam_masks >= 2

            mask_scales = q_trans(view.mask_scales[:n_masks].cuda().unsqueeze(-1))
            scale_gates_view = scale_gate(mask_scales)

            scale_cond_anchors = scale_gates_view.unsqueeze(1) * anchor_point_features.unsqueeze(0)
            scale_cond_anchors = torch.nn.functional.normalize(scale_cond_anchors, dim=-1, p=2)

            scale_cond_feat = rendered_feature.unsqueeze(0) * scale_gates_view.unsqueeze(-1).unsqueeze(-1)
            scale_cond_feat = torch.nn.functional.normalize(scale_cond_feat, dim=1, p=2)

            mask_features = (sam_masks * scale_cond_feat).sum(dim=-1).sum(dim=-1) / (
                sam_masks.sum(dim=-1).sum(dim=-1) + 1e-9
            )
            mask_features = torch.nn.functional.normalize(mask_features, dim=-1, p=2)

            mask_identifier = (
                torch.einsum("nmc,nc->nm", scale_cond_anchors, mask_features) > 0.5
            )

            mask_identifiers.append(mask_identifier.cpu())
            seg_features.append(mask_features)
            scales_list.append(view.mask_scales[:n_masks].cuda().unsqueeze(-1))

            for j in range(len(mask_features)):
                camera_id_mask_id.append((i, j))

    torch.cuda.empty_cache()

    # Flatten across views
    flattened_mask_features = torch.cat(seg_features, dim=0)
    flattened_clip_features = torch.cat(clip_features, dim=0)
    flattened_clip_features = torch.nn.functional.normalize(
        flattened_clip_features.float(), dim=-1, p=2
    )
    flattened_scales = torch.cat(scales_list, dim=0)
    flattened_mask_identifiers = (
        torch.cat(mask_identifiers, dim=0).to(torch.float16).cuda()
    )

    print(
        "Shapes — mask_features:", flattened_mask_features.shape,
        "| clip_features:", flattened_clip_features.shape,
        "| scales:", flattened_scales.shape,
        "| mask_identifiers:", flattened_mask_identifiers.shape,
        "| total masks:", len(camera_id_mask_id),
    )

    # Cluster masks by Jaccard distance on anchor identity vectors
    with torch.no_grad():
        inter = torch.einsum("mc,nc->mn", flattened_mask_identifiers, flattened_mask_identifiers)
        union = (
            flattened_mask_identifiers.sum(-1).unsqueeze(1)
            + flattened_mask_identifiers.sum(-1).unsqueeze(0)
            - inter
            + 1e-6
        )
        distance_map = (1 - inter / union).detach().cpu().numpy().astype(np.float64)

    clusterer = HDBSCAN(
        min_cluster_size=30, cluster_selection_epsilon=0.25, metric="precomputed"
    )
    cluster_labels_np = clusterer.fit_predict(distance_map)
    cluster_labels = torch.from_numpy(cluster_labels_np).to(
        device=flattened_clip_features.device, dtype=torch.long
    )

    # Positive prompt scoring
    clip_model = load_clip()
    clip_model.eval()

    scores = get_scores_with_template(
        clip_model, flattened_clip_features.cuda(), args_cli.prompt
    ).squeeze()

    unique_clusters = cluster_labels.unique()
    cluster_scores = torch.zeros(len(unique_clusters), device=cluster_labels.device)
    for cluster_idx in unique_clusters:
        if cluster_idx == -1:
            continue
        cluster_scores[cluster_idx + 1] = scores[cluster_labels == cluster_idx].mean()

    good_cluster_indices = torch.where(cluster_scores > args_cli.cluster_score_threshold)[0]

    if len(good_cluster_indices) > 0:
        good_clusters = [unique_clusters[i] for i in good_cluster_indices]
        print(f"Good clusters ({len(good_clusters)}): scores = {cluster_scores[good_cluster_indices]}")
    else:
        non_noise_scores = cluster_scores.clone()
        non_noise_scores[0] = -1
        best_idx = non_noise_scores.argmax()
        good_clusters = [unique_clusters[best_idx]]
        print(f"No cluster above threshold; using best cluster with score {cluster_scores[best_idx]:.4f}")

    clip_query_features = []
    corresponding_scales = []
    for g in good_clusters:
        s, ind = cluster_id_to_scales(cluster_labels, flattened_scales, g, scores)
        clip_query_features.append(
            torch.nn.functional.normalize(
                flattened_mask_features[cluster_labels == g][ind], dim=-1, p=2
            )
        )
        corresponding_scales.append(s)

    # Positive similarity map
    index = 0
    similarities = get_similarity_map(
        feature_gaussians.get_point_features,
        corresponding_scales[index],
        scale_gate,
        clip_query_features[index],
        q_trans,
    )

    # Negative prompt scoring
    if args_cli.nprompt is not None:
        print(f"Applying negative prompt: '{args_cli.nprompt}' (weight={args_cli.nprompt_weight})")

        neg_scores = get_scores_with_template(
            clip_model, flattened_clip_features.cuda(), args_cli.nprompt
        ).squeeze()

        # Print top clusters by negative prompt score
        for c in cluster_labels.unique():
            if c == -1:
                continue
            print(f"Cluster {c.item()}: neg_score={neg_scores[cluster_labels == c].mean():.4f}, "
                f"pos_score={scores[cluster_labels == c].mean():.4f}, "
                f"size={( cluster_labels == c).sum().item()}")

        # Find the cluster that best matches the negative prompt
        neg_query_feat, neg_scale = get_best_cluster_query(
            cluster_labels, flattened_mask_features, flattened_scales, neg_scores,
            exclude_clusters=good_clusters,  # don't pick the positive cluster as negative
        )

        neg_similarities = get_similarity_map(
            feature_gaussians.get_point_features,
            neg_scale,
            scale_gate,
            neg_query_feat,
            q_trans,
        )
        neg_similarities = torch.clamp(neg_similarities, min=0)
        similarities = similarities - args_cli.nprompt_weight * neg_similarities
        print(f"Similarity range after negative suppression: [{similarities.min():.3f}, {similarities.max():.3f}]")
    else:
        print(f"Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")

    # Segment scene Gaussians
    try:
        scene_gaussians.roll_back()
    except Exception:
        pass

    # Crop out negatives
    xyz = scene_gaussians.get_xyz.detach()
    selected = similarities > args_cli.similarity_threshold
    print(f"Selected {selected.sum().item()} Gaussians")
    print("Selected X:", xyz[selected,0].min().item(), "to", xyz[selected,0].max().item())
    print("Selected Y:", xyz[selected,1].min().item(), "to", xyz[selected,1].max().item())
    print("Selected Z:", xyz[selected,2].min().item(), "to", xyz[selected,2].max().item())

    scene_gaussians.segment(similarities > args_cli.similarity_threshold)

    # Render all views to video
    rgb_background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    out_dir = os.path.dirname(args_cli.video_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    first_cam = scene.getTrainCameras()[0]
    h = first_cam.image_height
    w = first_cam.image_width
    h = h if h % 2 == 0 else h - 1
    w = w if w % 2 == 0 else w - 1

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lang-segment-anything'))
    from lang_sam import LangSAM

    lang_sam = LangSAM(
        sam_ckpt_path=sam_dir,
        gdino_model_ckpt_path=gdino_dir,
        gdino_processor_ckpt_path=gdino_dir,
    )

    gaussian_vote_counts = torch.zeros(scene_gaussians.get_xyz.shape[0], device="cuda")

    writer = iio.get_writer(args_cli.video_path, fps=args_cli.fps)
    try:
        for cam in tqdm(scene.getTrainCameras(), desc="Masking Gaussians"):
            torch.cuda.empty_cache()

            # Render full scene
            render_output = render(cam, scene_gaussians, pipeline.extract(args), rgb_background)
            rendered = render_output["render"]
            frame = _to_uint8_rgb(rendered)[:h, :w, :]

            def predict_combined_mask(prompt_text: str):
                """Return a single (H,W) boolean numpy mask for a prompt, or None if no masks."""
                if prompt_text is None:
                    return None
                prompt_text = str(prompt_text).strip()
                if prompt_text == "":
                    return None

                try:
                    results = lang_sam.predict([pil_frame], [prompt_text])
                    masks = results[0].get("masks") if results else None
                except Exception as e:
                    print(f"LangSAM failed on frame for prompt={prompt_text!r}: {e}")
                    return None

                if masks is None or len(masks) == 0:
                    return None

                if isinstance(masks[0], torch.Tensor):
                    return torch.stack(masks).any(dim=0).detach().cpu().numpy().astype(bool)
                else:
                    return np.stack(masks).any(axis=0).astype(bool)

            def filter_gaussians_by_mask(visible_gaussians, means3D, mask, viewpoint_camera, H, W):
                """
                Projects visible Gaussian centers into screen space and keeps only
                those whose projection falls within the 2D boolean mask.
                """
                vis_xyz = means3D[visible_gaussians]  # (K, 3)

                # Homogeneous world coords
                ones = torch.ones(vis_xyz.shape[0], 1, device=vis_xyz.device)
                vis_xyz_h = torch.cat([vis_xyz, ones], dim=1)  # (K, 4)

                # Project into clip space using the full projection matrix
                clip = vis_xyz_h @ viewpoint_camera.full_proj_transform  # (K, 4)

                # Perspective divide -> NDC [-1, 1]
                ndc = clip[:, :2] / clip[:, 3:4]  # (K, 2)

                # NDC to pixel coords
                px = ((ndc[:, 0] + 1) * 0.5 * W).long()  # (K,)
                py = ((ndc[:, 1] + 1) * 0.5 * H).long()  # (K,)

                # Clamp to image bounds
                px = px.clamp(0, W - 1)
                py = py.clamp(0, H - 1)

                # Look up mask at each projected position
                mask_tensor = torch.from_numpy(mask).to(vis_xyz.device)  # (H, W)
                inside_mask = mask_tensor[py, px]  # (K,) bool

                return visible_gaussians[inside_mask]


            visible_gaussians = render_output["visible_gaussians"]  # indices into scene_gaussians
            pil_frame = Image.fromarray(frame)

            pos_mask = predict_combined_mask(args_cli.prompt)
            neg_mask = predict_combined_mask(args_cli.nprompt)

            if pos_mask is not None:
                final_mask = pos_mask & (~neg_mask) if neg_mask is not None else pos_mask
                masked_gaussians = filter_gaussians_by_mask(
                    visible_gaussians, scene_gaussians.get_xyz, final_mask, cam, h, w
                )
                gaussian_vote_counts[masked_gaussians] += 1 

        total_frames = len(scene.getTrainCameras())
        threshold = total_frames * 0.3
        keep = gaussian_vote_counts >= threshold
        exclude_mask = ~keep

        for cam in tqdm(scene.getTrainCameras(), desc="Rendering video"):
            rendered = render(cam, scene_gaussians, pipeline.extract(args), rgb_background, filtered_mask=exclude_mask)["render"]
            frame = _to_uint8_rgb(rendered)[:h, :w, :]
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"Saved video: {args_cli.video_path}")


if __name__ == "__main__":
    main()