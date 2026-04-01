# guide-3d

### Setup

Request GPU:
```sh
salloc --mem=80G --time=06:00:00 -p mit_normal_gpu --gres=gpu:h200:1
```

### Environment
Initialize environment:
```sh
git clone https://github.com/harrisonzhy/guide-3d && cd guide-3d
source env.sh
```
We use Python 3.11 and CUDA 12.4. Set up conda environment and install `pytorch3d`:
```sh
conda create --name "guide-3d-env" python=3.11
conda activate guide-3d-env
conda install pytorch3d-0.7.8-py311_cu121_pyt241.tar.bz2
```
Below, use `--no-build-isolation` for `pip install`ing modules if needed.
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install plyfile
pip install tqdm
pip install opencv-python
pip install joblib
pip install open_clip_torch
pip install wheel
pip install scikit-learn

export CPLUS_INCLUDE_PATH="$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])"):${CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])"):${C_INCLUDE_PATH}"
export LD_LIBRARY_PATH=/orcd/software/core/001/spack/pkg/gcc/12.2.0/yt6vabm/lib64:$LD_LIBRARY_PATH
```

You may also try installing Python requirements (the ones above and below) like so, but it is not guaranteed to work:
```sh
pip install -r requirements.txt
```

### Dependencies

Set up LangSAM:
```sh
pushd lang-segment-anything
pip install -e .
popd
```

For offline LangSAM inference, download SAM and GroundingDINO:
```sh
pushd lang-segment-anything/checkpoints/sam
wget -O checkpoints/sam2.1_hiera_small.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
popd
pushd lang-segment-anything/checkpoints/gdino
hf download IDEA-Research/grounding-dino-base --local-dir checkpoints/gdino/grounding-dino-base
popd
```

Set up 3DGS:
```sh
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
pushd gaussian-splatting
pushd ./submodules/diff-gaussian-rasterization
pip install -e . --no-build-isolation
popd
pushd ./submodules/simple-knn
pip install -e . --no-build-isolation
popd
pushd ./submodules/fused-ssim
pip install -e . --no-build-isolation
popd
popd
```

Set up (our modified) SAGA:
```sh
pushd SegAnyGAussians/third_party/segment-anything
git clone https://github.com/facebookresearch/segment-anything.git
pip install -e .
popd
pushd SegAnyGAussians
pushd ./submodules/diff-gaussian-rasterization
pip install -e . --no-build-isolation
popd
pushd ./submodules/simple-knn
pip install -e . --no-build-isolation
popd
pushd ./submodules/diff-gaussian-rasterization_contrastive_f
pip install -e . --no-build-isolation
pushd ./submodules/diff-gaussian-rasterization-depth
pip install -e . --no-build-isolation
popd
popd
```

Set up LineUp:
```sh
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
```
Restart kernel
```sh
nvm install --lts
node -v
npm -v
npm install --save lineupjs
```

## Data

Download pre-trained scenes from the [3DGS repo](https://github.com/graphdeco-inria/gaussian-splatting) and Mip-NeRF 360 dataset:
```sh
pushd gaussian-splatting/models
wget -O models.zip "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip"
unzip models.zip
popd
pushd gaussian-splatting/data
wget -O images1.zip "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
wget -O images2.zip "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"
unzip images1.zip -d images1
unzip images2.zip -d images2
popd
```

## Pre-training

Follow the instructions in the [3DGS repo](https://github.com/graphdeco-inria/gaussian-splatting) for pre-training on COLMAP dataset. Namely:
```
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

## Open-vocabulary 3D scene segmentation
```sh
pushd SegAnyGAussians
python extract_segment_everything_masks.py --image_root "/home/zhanghy/orcd/scratch/zhanghy/guide-3d/mats/data/images1/bicycle" --sam_checkpoint_path "third_party/segment-anything-model/sam_vit_h_4b8939.pth" --downsample 4

python get_scale.py --image_root "../mats/data/images1/bicycle" --source "../mats/data/images1/bicycle" --model_path "../gaussian-splatting/output/a1c12f3b-d"
popd
sh clip.sh
pushd SegAnyGAussians
python get_clip_features.py --image_root "../mats/data/images1/bicycle"
python train_contrastive_feature.py -m "../gaussian-splatting/output/a1c12f3b-d" --iteration 30000 --num_sampled_rays 1000
popd
```

## Video generation
Create video of prompt-segmented Gaussians from all camera views:
```sh
python lang_seg_video.py --prompt "bicycle"
```
