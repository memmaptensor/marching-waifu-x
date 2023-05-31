import argparse
import gc
import sys

sys.path.append("..")

import torch

from src.pipelines.realesrgan_pipeline import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    args = parser.parse_args()

    realesrgan = realesrgan_pipeline()
    image = realesrgan(args.in_path)
    image.save(args.out_path)

    del realesrgan
    gc.collect()
    torch.cuda.empty_cache()
