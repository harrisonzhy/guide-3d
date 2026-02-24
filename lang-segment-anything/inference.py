from PIL import Image
from lang_sam import LangSAM

sam_dir = "./checkpoints/sam/sam2.1_hiera_small.pt"
gdino_dir = "./checkpoints/gdino/grounding-dino-base"

model = LangSAM(
    sam_ckpt_path=sam_dir,
    gdino_model_ckpt_path=gdino_dir,
    gdino_processor_ckpt_path=gdino_dir,
)
image_pil = Image.open("./assets/car.jpeg").convert("RGB")
text_prompt = "wheel."
results = model.predict([image_pil], [text_prompt])
