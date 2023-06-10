import argparse
import gc
import json
import os
import sys

import PIL.Image
import torch

sys.path.append("..")

from src.pipelines.groundedsam_pipeline import *


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

    # Load images
    images = []
    for i in range(conf["groundedsam"]["length"]):
        filepath = os.path.join(conf["paths"]["in_path"], f"{(i+1):04}.png")
        image = PIL.Image.open(filepath)
        images.append(image)

    # Scale images
    scaled_images = []
    for i in range(conf["groundedsam"]["length"]):
        scaled_image = image_wrapper(images[i], "pil")
        scaled_image = scaled_image.scale(conf["groundedsam"]["scale"]).to_pil()
        scaled_images.append(scaled_image)

    # Load & Run GroundedSAM
    groundedsam = groundedsam_pipeline(
        conf["paths"]["cache_dir"], conf["groundedsam"]["device"]
    )
    for clip in conf["groundedsam"]["clips"]:
        for frame in clip["clip_frames"]:
            # Run GroundedSAM
            mask = groundedsam(
                scaled_images[frame],
                clip["det_prompt"],
                clip["box_threshold"],
                clip["text_threshold"],
                clip["merge_masks"],
            )
            mask = image_wrapper(mask.convert("L"), "pil")
            mask = mask.scale(1.0 / conf["groundedsam"]["scale"]).to_pil()

            # Mask alpha
            masked = images[frame]
            masked.putalpha(mask)

            # Save results
            masked.save(
                os.path.join(
                    conf["paths"]["out_path"],
                    f"{conf['paths']}{(frame+1):04}.png",
                )
            )

    del groundedsam
    gc.collect()
    torch.cuda.empty_cache()
