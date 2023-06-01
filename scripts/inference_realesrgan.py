import argparse
import gc
import glob
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
        "--in_path",
        type=str,
        required=True,
        help="Directory containing frames to upscale",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Directory to contain the output frames",
    )
    parser.add_argument(
        "--outscale", type=float, default=4.0, help="Factor to scale input by"
    )
    parser.add_argument("--tile", type=int, default=192, help="Tile size")
    parser.add_argument(
        "--tile_pad", type=int, default=16, help="Size to pad when upscaling"
    )
    parser.add_argument(
        "--pre_pad", type=int, default=16, help="Size to pad the input before upscaling"
    )
    parser.add_argument(
        "--face_enhance", type=bool, default=True, help="Enhance face using GFPGAN"
    )
    parser.add_argument("--fp32", type=bool, default=False, help="Use fp32 precision")
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU device ID to run inference on"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    realesrgan = realesrgan_pipeline(
        args.outscale,
        args.tile,
        args.tile_pad,
        args.pre_pad,
        args.face_enhance,
        args.fp32,
        args.gpu_id,
    )

    for filepath in sorted(glob.glob(os.path.join(args.in_path, "*.png"))):
        pl = pathlib.Path(filepath)
        output_image = realesrgan(PIL.Image.open(filepath))
        output_image.save(os.path.join(args.out_path, pl.stem, ".png"))

    del realesrgan
    gc.collect()
    torch.cuda.empty_cache()
