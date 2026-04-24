# %%
import sys
from pathlib import Path

# Make gaussian-splatting visible for imports like `from utils...`
PROJECT_ROOT = Path(__file__).resolve().parent
gs_root = PROJECT_ROOT / "gaussian-splatting"
sys.path.append(str(gs_root))
print(sys.executable)

import os
import cv2
import numpy as np
import torch
import pycolmap
from PIL import Image
import contextlib, io
import argparse

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
import pandas as pd
from torch.cuda.amp import autocast

torch.set_default_dtype(torch.float32)

import lpips

lpips_model = lpips.LPIPS(net='vgg').to("cuda").eval()


"""
INTENDED USAGE PATTERN:

from video import initialization, import_path, segment, write_video, update_pickle
from pathlib import Path

initialization()

model_path = Path(".../models/scene1")
data_root  = Path(".../colmap_project")
output_dir = Path(".../outputs")
iteration  = 7000
text_prompt = "tractor"

scene, train_cams, dataset, pipe = import_path(model_path, data_root, output_dir, iteration) <- Load cameras and create scene when users specify paths

masks_by_name = segment(scene, train_cams, dataset, text_prompt) <- computes the masks for all pics by user text prompt

rows, out_path = write_video(output_dir, train_cams, dataset, text_prompt, masks_by_name, scene, pipe) <- writes the video to output folder, out_path is where the video is, including its name, rows is data for lineup

update_pickle(rows, output_dir) <- given new rows, update pickle file if there already exists one, create one if doesn't exist

pickle file is at output_dir / "data.pkl"

other functions in here are helpers.

If any bugs, tell me.
"""
def get_intrinsics(cam):
    W = cam.image_width
    H = cam.image_height
    fx = W / (2.0 * np.tan(cam.FoVx / 2.0))
    fy = H / (2.0 * np.tan(cam.FoVy / 2.0))
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy


def initialization():
    # This is code to fix path for me so imports are good, and designed be called upon initialization
    PROJECT_ROOT = Path(__file__).resolve().parent
    gs_root = PROJECT_ROOT / "gaussian-splatting"
    knn_root     = gs_root / "submodules" / "simple-knn"  # top of that repo
    sys.path.append(str(knn_root))
    sys.path.append(str(gs_root))
    print("last sys.path entries:", sys.path[-2:])
    return None

def compute_neighbors(train_cams, k=3):
    """
    Returns: dict[image_name] -> list of neighbor indices in train_cams
    """
    centers = []
    for cam in train_cams:
        C = cam.camera_center.detach().cpu().numpy()  # (3,)
        centers.append(C)
    centers = np.stack(centers, axis=0)  # (N,3)

    neighbors = {}
    N = len(train_cams)
    for i, cam in enumerate(train_cams):
        Ci = centers[i]
        dists = np.linalg.norm(centers - Ci, axis=1)
        order = np.argsort(dists)
        # skip self (0th), take next k
        nn = [int(j) for j in order[1:k+1] if j < N]
        neighbors[cam.image_name] = nn
    return neighbors


def import_path(model_path, data_root, output_dir, iteration):

    # check these two paths exists before putting them in
    # here I assume they exist.
    # model_path: scene output dir, iteration: which iteration inside the model_path,  
    # data_root: images that include sparce and images folder, output_dir: where the video will be

    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams
    device = "cuda:0"
    sparse_dir  = data_root / "sparse/0"
    output_dir.mkdir(exist_ok=True, parents=True)
    recon = pycolmap.Reconstruction(sparse_dir)
    images = list(recon.images.values())
    images.sort(key=lambda im: im.image_id)  # deterministic order

    print(f"Loaded {len(images)} registered images.")

    parser = argparse.ArgumentParser(description="Inference params")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    # minimal argv; all other options take defaults
    argv = [
        "--source_path", str(data_root),
        "--model_path",  str(model_path),
        "--images",      "images",
    ]

    args = parser.parse_args(argv)

    dataset = lp.extract(args)   # same object they call "dataset" in training()
    pipe    = pp.extract(args)
        
    gaussians = GaussianModel(dataset.sh_degree, optimizer_type="adam")
    scene = Scene(dataset, gaussians,
              load_iteration=iteration,
              shuffle=False,
              resolution_scales=[1.0])

    # Now scene.gaussians and scene.getTrainCameras() / getTestCameras() are ready
    train_cams = scene.getTrainCameras(scale=1.0)   # list of Camera
    test_cams  = scene.getTestCameras(scale=1.0)

    print("train cams:", len(train_cams), "test cams:", len(test_cams))

    # Ensure gaussians are float32 for rasterizer
    attrs = ["_xyz", "_features_dc", "_features_rest",
             "_scaling", "_rotation", "_opacity"]

    for name in attrs:
        t = getattr(scene.gaussians, name, None)
        if isinstance(t, torch.Tensor) and t.dtype != torch.float32:
            print(f"Converting {name} from {t.dtype} to float32")
            setattr(scene.gaussians, name, t.float())

    neighbors = compute_neighbors(train_cams, k=3)

    return scene, train_cams, dataset, pipe, neighbors


def quiet_predict(model, images_pil, texts_prompt):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return model.predict(images_pil, texts_prompt)

def predict_masks_safe_batch(langsam_model, images_pil, texts_prompt):
    """
    images_pil   : list of PIL images
    texts_prompt : list of strings, same length
    returns      : list of results (one per image), where a result is either
                   the LangSAM result dict or None if prediction failed
    """
    assert len(images_pil) == len(texts_prompt)
    n = len(images_pil)

    # Try one batched call for the whole chunk
    try:
        batch_results = quiet_predict(langsam_model, images_pil, texts_prompt)
        # If success, just return
        return list(batch_results)
    except AssertionError:
        # SAM2 got confused; fall back to per-image calls
        results = []
        for img_pil, txt in zip(images_pil, texts_prompt):
            try:
                single_res = quiet_predict(langsam_model, [img_pil], [txt])
                results.append(single_res[0])   # index 0 from list
            except AssertionError:
                # this particular image failed → no mask
                results.append(None)
            except RuntimeError as e:
                # e.g. OOM or other model errors; also treat as no mask
                results.append(None)
                torch.cuda.empty_cache()
        return results

def segment(scene, train_cams, dataset, text_prompt):
    torch.cuda.empty_cache()
    from lang_sam import LangSAM
    device = "cuda:0"
    langsam_model = LangSAM(device=device)

    batch_size = 2
    masks_by_name = {}

    gt_pils  = []
    gt_names = []

    for cam in train_cams:
        # cam.original_image: (3, H, W) float in [0,1]
        gt = torch.clamp(cam.original_image.to("cpu"), 0.0, 1.0)   # (3,H,W)

        # to HxWx3 uint8 RGB
        gt_np = (gt.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

        gt_pils.append(Image.fromarray(gt_np))   # PIL RGB image
        gt_names.append(cam.image_name)          # keep name to index masks_by_name

    for i in range(0, len(gt_pils), batch_size):
        batch_imgs  = gt_pils[i:i+batch_size]
        batch_names = gt_names[i:i+batch_size]
        batch_texts = [text_prompt] * len(batch_imgs)

        batch_results = predict_masks_safe_batch(langsam_model, batch_imgs, batch_texts)

        for name, res in zip(batch_names, batch_results):
            if res is None:
                masks_by_name[name] = None
                continue

            masks = res["masks"]
            if isinstance(masks, torch.Tensor):
                masks_np = masks.cpu().numpy()
            else:
                masks_np = np.asarray(masks)

            if masks_np.shape[0] == 0:
                masks_by_name[name] = None
                continue

            mask = (masks_np[0] > 0.5).astype(np.uint8)
            masks_by_name[name] = mask

        torch.cuda.empty_cache()

    return masks_by_name


def render_metrics_frame(cam, gaussians, pipe, background, masks_by_name, depth_cache, train_test_exp=False):
    """
    Render gaussians from `cam`, compute metrics vs cam.original_image,
    and return a labeled [GT | render] frame (H, 2W, 3, uint8) plus (L1, SSIM, PSNR).
    """
    
    from gaussian_renderer import render
    try:
        from fused_ssim import fused_ssim
        FUSED_SSIM_AVAILABLE = True
    except ImportError:
        FUSED_SSIM_AVAILABLE = False

# --- ensure ALL gaussians tensors are float32 before render ---
    for attr in dir(gaussians):
        t = getattr(gaussians, attr, None)
        if isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16:
            # print once if you want to see what got fixed
            print("Casting", attr, "from bfloat16 to float32")
            setattr(gaussians, attr, t.float())

    '''
    print("=== DTYPE CHECK ===")
    for name in ["_xyz", "_features_dc", "_features_rest",
             "_scaling", "_rotation", "_opacity"]:
        t = getattr(gaussians, name, None)
        print(name, t.dtype if isinstance(t, torch.Tensor) else type(t))
        '''

    # --- render (once) ----

    with torch.no_grad():
        with autocast(enabled=False):
            pkg = render(cam, gaussians, pipe, background,
                     use_trained_exp=train_test_exp)
            img = torch.clamp(pkg["render"], 0.0, 1.0)  # (3,H,W)
            depth = pkg.get("depth", None)
            if depth is not None:
                # store on CPU, (H,W)
                depth_cache[cam.image_name] = depth.squeeze().detach().cpu()

    gt = torch.clamp(cam.original_image.to(img.device), 0.0, 1.0)

    # train_test_exp cropping if used
    if train_test_exp:
        img = img[..., img.shape[-1] // 2:]
        gt  = gt[...,  gt.shape[-1] // 2:]

    # alpha mask if present
    if cam.alpha_mask is not None:
        alpha = cam.alpha_mask.to(img.device)
        img = img * alpha
        gt  = gt  * alpha

    gt_np_for_sam  = (gt.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    img_np_for_sam = (img.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

    mask = masks_by_name.get(cam.image_name, None)
    if mask is None:
        return None, None, None, None, None, None
    else:
        mask = (mask > 0).astype(np.uint8)  # ensure 0/1
        gt_np_masked  = apply_mask(gt_np_for_sam,  mask)
        img_np_masked = apply_mask(img_np_for_sam, mask)  # reuse GT mask

    gt  = torch.from_numpy(gt_np_masked).to(img.device).permute(2, 0, 1).float() / 255.0
    img = torch.from_numpy(img_np_masked).to(img.device).permute(2, 0, 1).float() / 255.0
    

    # --- metrics ---
    L1_val = l1_loss(img, gt).mean().double()

    if FUSED_SSIM_AVAILABLE:
        ssim_val = fused_ssim(img.unsqueeze(0), gt.unsqueeze(0)).double()
    else:
        ssim_val = ssim(img, gt).double()

    psnr_val = psnr(img, gt).mean().double()

    # --- LPIPS (masked) ---
    # LPIPS expects [-1,1] range, NCHW
    img_lp = (img * 2.0 - 1.0).unsqueeze(0)  # (1,3,H,W)
    gt_lp  = (gt  * 2.0 - 1.0).unsqueeze(0)

    with torch.no_grad():
        lpips_val = lpips_model(img_lp, gt_lp)   # shape [1,1] or [1]

    lpips_val = float(lpips_val.mean().item())

    mask_bool = mask.astype(bool)
    num_masked = int(mask_bool.sum())

    # --- convert to numpy for visualization ---
    img_vis = img.detach().cpu()   # (3,H,W)
    gt_vis  = gt.detach().cpu()

    if img_vis.ndim == 3 and img_vis.shape[0] == 3:
        img_vis = img_vis.permute(1, 2, 0)  # (H,W,3)
    if gt_vis.ndim == 3 and gt_vis.shape[0] == 3:
        gt_vis = gt_vis.permute(1, 2, 0)

    img_np = (img_vis.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    gt_np  = (gt_vis.numpy()  * 255.0).clip(0, 255).astype(np.uint8)

    rend_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gt_bgr   = cv2.cvtColor(gt_np,  cv2.COLOR_RGB2BGR)
    
    H, W = cam.image_height, cam.image_width
    rend_bgr = cv2.resize(rend_bgr, (W, H))
    gt_bgr   = cv2.resize(gt_bgr,   (W, H))
    
    frame = cv2.hconcat([gt_bgr, rend_bgr])

    # overlay text *after* concat
    name_text    = cam.image_name
    metric_text1 = f"L1 {L1_val:.4f}  SSIM {ssim_val.item():.3f}  PSNR {psnr_val.item():.2f} dB"
    metric_text2 = f"LPIPS {lpips_val:.3f}"
    
    cv2.putText(frame, name_text,    (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, metric_text1, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, metric_text2, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return frame, float(L1_val), float(ssim_val), float(psnr_val), num_masked, lpips_val

# %%
def apply_mask(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    img_rgb : HxWx3 uint8 (RGB)
    mask    : HxW (0/1 or bool)
    returns : HxWx3 uint8, background set to 0 (black) outside mask
    """
    # ensure mask is boolean and same size
    mask_bool = mask.astype(bool)
    if img_rgb.shape[:2] != mask_bool.shape:
        raise ValueError("img and mask size mismatch")

    out = img_rgb.copy()
    out[~mask_bool] = 0  # or any background color you prefer
    return out

# %%

def compute_depth_errs(train_cams, depth_cache, masks_by_name, neighbors, max_samples_per_view=5000):
    """
    Compute masked cross-view depth consistency error per view.

    Returns: dict[image_name] -> float or None
    """
    depth_errs = {}

    for i, camA in enumerate(train_cams):
        nameA = camA.image_name
        maskA = masks_by_name.get(nameA)
        depthA = depth_cache.get(nameA)
        if maskA is None or depthA is None:
            depth_errs[nameA] = None
            continue

        maskA_bool = maskA.astype(bool)
        ys, xs = np.where(maskA_bool)
        if ys.size == 0:
            depth_errs[nameA] = None
            continue

        # optionally subsample pixels for speed
        if ys.size > max_samples_per_view:
            idx = np.random.choice(ys.size, max_samples_per_view, replace=False)
            ys = ys[idx]
            xs = xs[idx]

        depthA_np = depthA.numpy()  # (H,W)

        fxA, fyA, cxA, cyA = get_intrinsics(camA)

        # camera A: world<->cam matrices
        W2C_A = camA.world_view_transform.detach().cpu().numpy()  # 4x4, world -> cam
        C2W_A = np.linalg.inv(W2C_A)                              # cam -> world

        errs = []
        # loop over neighbor views
        for j_idx in neighbors.get(nameA, []):
            camB = train_cams[j_idx]
            nameB = camB.image_name
            depthB = depth_cache.get(nameB)
            if depthB is None:
                continue
            depthB_np = depthB.numpy()
            H_B, W_B = depthB_np.shape

            fxB, fyB, cxB, cyB = get_intrinsics(camB)
            W2C_B = camB.world_view_transform.detach().cpu().numpy()
            # no need for C2W_B for this metric; we use z in B's frame and depthB for comparison

            # for each masked pixel in A
            for y, x in zip(ys, xs):
                zA = depthA_np[y, x]
                if zA <= 0:
                    continue

                # 1) backproject p in A to 3D point in camera A
                X_camA = np.array([
                    (x - cxA) / fxA * zA,
                    (y - cyA) / fyA * zA,
                    zA,
                    1.0,
                ], dtype=np.float32)

                # cam A -> world
                X_world = C2W_A @ X_camA  # (4,)
                X_world3 = X_world[:3] / X_world[3]

                # world -> cam B
                X_camB = W2C_B @ np.concatenate([X_world3, np.array([1.0], dtype=np.float32)])
                X_camB3 = X_camB[:3]

                z_pred = X_camB3[2]
                if z_pred <= 0:
                    continue

                # project into B's image
                u_pred = fxB * X_camB3[0] / z_pred + cxB
                v_pred = fyB * X_camB3[1] / z_pred + cyB

                if not (0 <= u_pred < W_B - 1 and 0 <= v_pred < H_B - 1):
                    continue

                # bilinear sample B's depth
                x0 = int(np.floor(u_pred))
                x1 = x0 + 1
                y0 = int(np.floor(v_pred))
                y1 = y0 + 1
                dx = u_pred - x0
                dy = v_pred - y0

                d00 = depthB_np[y0, x0]
                d10 = depthB_np[y0, x1]
                d01 = depthB_np[y1, x0]
                d11 = depthB_np[y1, x1]
                if (d00 <= 0) and (d10 <= 0) and (d01 <= 0) and (d11 <= 0):
                    continue

                dB = (
                    d00 * (1 - dx) * (1 - dy) +
                    d10 * dx       * (1 - dy) +
                    d01 * (1 - dx) * dy       +
                    d11 * dx       * dy
                )

                if dB <= 0:
                    continue

                # reprojection depth error in B's camera frame (z-coordinate)
                err = abs(z_pred - dB)
                errs.append(err)

        if len(errs) == 0:
            depth_errs[nameA] = None
        else:
            depth_errs[nameA] = float(np.mean(errs))

    return depth_errs

def write_video(output_dir, train_cams, dataset, text_prompt, masks_by_name, scene, pipe, scene_name, neighbors):
    rows = []
    depth_cache = {}
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Use resolution from first camera to configure VideoWriter
    H0, W0 = train_cams[0].image_height, train_cams[0].image_width
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    prompt_str = str(text_prompt).replace(" ", "_")      # replace spaces with _
    prompt_str = prompt_str.replace("/", "_")       # avoid path separators

    scene_str = str(scene_name).replace(" ", "_").replace("/", "_")
    out_path = output_dir / f"comparison_{scene_str}_{prompt_str}.avi"
    
    writer = cv2.VideoWriter(str(out_path), fourcc, 5, (2*W0, H0))
    print("writer opened:", writer.isOpened())

    for i, cam in enumerate(train_cams):
        frame, L1, ssim_val, psnr_val, total_pixels, lpips_val = render_metrics_frame(
            cam,
            scene.gaussians,
            pipe,
            background,
            masks_by_name,
            depth_cache,
            train_test_exp=dataset.train_test_exp
        )
        if frame is None:
            continue

        rows.append({
        "image_name": cam.image_name,
        "object": text_prompt,
        "L1": L1,
        "SSIM": ssim_val,
        "PSNR_dB": psnr_val,
        "total_pixels": total_pixels,
        "LPIPS": lpips_val,
        })

        # sanity: show pixel stats so we know frame content changes
        #print(f"{i}: {cam.image_name}, mean={frame.mean():.2f}, std={frame.std():.2f}, "
            #f"SSIM={ssim_val:.3f}, PSNR={psnr_val:.2f}")

        writer.write(frame)

    writer.release()
    print("wrote", out_path)
    depth_errs = compute_depth_errs(train_cams, depth_cache, masks_by_name, neighbors)
    for row in rows:
        name = row["image_name"]
        row["depth_err"] = depth_errs.get(name)
    return rows, out_path

# %%
def update_pickle(rows, output_dir, scene_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_str = str(scene_name).replace(" ", "_").replace("/", "_")
    pkl_path = output_dir / f"{scene_str}.pkl"

    new_df = pd.DataFrame(rows)
    if new_df.empty:
        return  # nothing to update
        
    # New data as DataFrame
    if pkl_path.exists():
        # Load existing and append
        old_df = pd.read_pickle(pkl_path)
        objs = new_df["object"].unique()

        # drop any rows with these objects from the old df
        old_df = old_df[~old_df["object"].isin(objs)]

        # append new rows
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        # First time: just use new_df
        df = new_df

    df.to_pickle(pkl_path)

def delete_object(output_dir, scene_name, text_prompt):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_str = str(scene_name).replace(" ", "_").replace("/", "_")
    pkl_path = output_dir / f"{scene_str}.pkl"

    if not pkl_path.exists():
        return

    df = pd.read_pickle(pkl_path)
    df = df[df["object"] != text_prompt]
    df.to_pickle(pkl_path)

