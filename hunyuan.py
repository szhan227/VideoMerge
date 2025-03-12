import torch
import torchvision

from diffusers.models import HunyuanVideoTransformer3DModel
from pipeline_hunyuan_video import HunyuanVideoPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm
import time
import os

def long_video_gen():

    device = "cuda"

    model_id = "hunyuanvideo-community/HunyuanVideo"
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16).to(device)
    pipe.vae.enable_tiling()

    prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",

    output = pipe.long_inference(
        prompt=prompt,
        height=320,
        width=512,
        num_chunks=5,
        num_inference_steps=30,
        shuffle=True,
        noise_refine=False,
    ).frames[0]

    output_path = "./output.mp4"
    export_to_video(output, output_path, fps=14)


if __name__ == "__main__":
    long_video_gen()


