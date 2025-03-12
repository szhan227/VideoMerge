import torch
import torchvision

from diffusers.models import HunyuanVideoTransformer3DModel
from pipeline_hunyuan_video import HunyuanVideoPipeline
from diffusers.utils import export_to_video
from tqdm import tqdm
import time
import os

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt_file", type=str, default="./prompt.txt")
    parser.add_argument("--output_path", type=str, default="./output.mp4")
    parser.add_argument("--resolution", type=str, default="320p", choices=["320p", "720p"])
    parser.add_argument("--num_chunks", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--high_freq_merge", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    
    return parser.parse_args()

def long_video_gen():

    args = parse_args()

    device = "cuda"

    model_id = "hunyuanvideo-community/HunyuanVideo"
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.float16).to(device)
    pipe.vae.enable_tiling()


    with open(args.prompt_file, "r") as f:
        prompt = f.read().strip()

    if args.resolution == "320p":
        height, width = 320, 512
    elif args.resolution == "720p":
        height, width = 720, 1280

    output = pipe.long_inference(
        prompt=prompt,
        height=height,
        width=width,
        num_chunks=args.num_chunks,
        num_inference_steps=args.num_inference_steps,
        shuffle=True,
        high_freq_merge=args.high_freq_merge,
    ).frames[0]

    export_to_video(output, args.output_path, fps=args.fps)


if __name__ == "__main__":
    long_video_gen()


