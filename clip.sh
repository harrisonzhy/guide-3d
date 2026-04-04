export CKPT_DIR="./third_party/clip_ckpt"
mkdir -p "$CKPT_DIR"

python - <<'PY'
import os
import open_clip
from huggingface_hub import snapshot_download

dst = os.environ["CKPT_DIR"]
snap = snapshot_download(
    repo_id="laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    local_dir=dst,
    local_dir_use_symlinks=False,
)
print("downloaded to:", snap)
# open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
print("ok")
PY

mkdir -p ../sagav2/clip_ckpt

WEIGHT=$(find "$CKPT_DIR" -type f \( -name "*.bin" -o -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" \) \
  -exec ls -l {} \; | sort -k5 -n | tail -n 1 | awk '{print $NF}')

echo "Using weights: $WEIGHT"
# mv "$WEIGHT" ../sagav2/clip_ckpt/ViT-B-16-laion2b_s34b_b88k.bin
# ls -lah ../sagav2/clip_ckpt/ViT-B-16-laion2b_s34b_b88k.bin
mv "$WEIGHT" ../sagav2/clip_ckpt/ViT-L-14-laion2b_s32b_b82k.bin
ls -lah ../sagav2/clip_ckpt/ViT-L-14-laion2b_s32b_b82k.bin
