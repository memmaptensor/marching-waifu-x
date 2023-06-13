import argparse
import gc
import glob
import json
import os
import pathlib
import sys

import PIL.Image

sys.path.append("..")

import torch

from src.pipelines.realesrgan_pipeline import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings_path",
        type=str,
        required=True,
        help="Path to file containing inference settings",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    with open(args.settings_path, "r") as f:
        conf = json.load(f)

    realesrgan = realesrgan_pipeline(
        conf["upscale"]["outscale"],
        conf["upscale"]["tile"],
        conf["upscale"]["tile_pad"],
        conf["upscale"]["pre_pad"],
        conf["upscale"]["face_enhance"],
        conf["upscale"]["fp32"],
        conf["upscale"]["gpu_id"],
    )

    for filepath in sorted(glob.glob(os.path.join(conf["paths"]["diffusion_output_path"], "*.png"))):
        pl = pathlib.Path(filepath)
        output_image = realesrgan(PIL.Image.open(filepath))
        output_image.save(os.path.join(conf["paths"]["upscaled_path"], f"{pl.stem}.png"))

    del realesrgan
    gc.collect()
    torch.cuda.empty_cache()
