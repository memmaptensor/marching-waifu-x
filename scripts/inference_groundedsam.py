import argparse
import gc
import json
import os
import sys

import PIL.Image
import torch

sys.path.append("..")
sys.path.append("../ext/Grounded-Segment-Anything/")

from src.pipelines.groundedsam_pipeline import *
from src.utils.image_wrapper import *


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

    frames = [None] * conf["groundedsam"]["length"]
    for detection in conf["groundedsam"]["detection"]:
        for frame in detection["frames"]:
            frames[frame] = detection

    groundedsam = groundedsam_pipeline(conf["paths"]["checkpoints_path"])

    # Load images
    for i in range(conf["groundedsam"]["length"]):
        torch.cuda.empty_cache()
        
        # Load image
        filepath = os.path.join(conf["paths"]["diffusion_output_path"], f"{(i+1):04}.png")
        image = PIL.Image.open(filepath)

        # Scale image
        scaled_image = image_wrapper(image, "pil")
        scaled_image = scaled_image.scale(conf["groundedsam"]["scale"]).to_pil()

        # Run GroundedSAM
        mask = groundedsam(
            scaled_image,
            frames[i]["prompt"],
            frames[i]["box_threshold"],
            frames[i]["text_threshold"],
        )

        # Mask alpha
        mask = image_wrapper(mask.convert("L"), "pil")
        mask = mask.scale(1.0 / conf["groundedsam"]["scale"]).to_pil()
        masked = image
        masked.putalpha(mask)

        # Save results
        masked.save(
            os.path.join(
                conf["paths"]["masked_path"],
                f"{(i+1):04}.png",
            )
        )

    del groundedsam
    gc.collect()
    torch.cuda.empty_cache()
