from lang_sam import LangSAM

sam_ckpt = "checkpoints/sam2.1_hiera_small.pt"
gdino_dir = "ckpts/gdino/grounding-dino-base"

model = LangSAM(
    sam_ckpt_path=sam_ckpt,
    gdino_model_ckpt_path=gdino_dir,
    gdino_processor_ckpt_path=gdino_dir,
)

