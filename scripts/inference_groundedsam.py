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
        image = image_wrapper(PIL.Image.open(filepath), "pil")
        image = image.scale(conf["groundedsam"]["scale"]).to_pil()
        images.append(image)

    # Load & Run GroundedSAM
    groundedsam = groundedsam_pipeline(
        conf["paths"]["cache_dir"], conf["groundedsam"]["device"]
    )
    for clip in conf["groundedsam"]["clips"]:
        for frame in clip["clip_frames"]:
            mask = groundedsam(
                images[frame],
                clip["det_prompt"],
                clip["box_threshold"],
                clip["text_threshold"],
                clip["merge_masks"],
            )
            mask = image_wrapper(mask, "pil")
            mask = mask.scale(1.0 / conf["groundedsam"]["scale"]).to_pil()
            mask.save(
                os.path.join(
                    conf["paths"]["out_path"],
                    f"{conf['paths']['file_prefix']}{(frame+1):04}.png",
                )
            )

    del groundedsam
    gc.collect()
    torch.cuda.empty_cache()
