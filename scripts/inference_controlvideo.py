import argparse
import json
import os
import sys

import PIL.Image
import torch

sys.path.append("..")

from src.pipelines.controlvideo_pipeline import *


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

    # Load ControlNet conditioning images
    controlnet_prefixes = conf["diffusion"]["controlnets"]["pipe"].keys()
    controlnet_repositories = conf["diffusion"]["controlnets"]["pipe"].values()
    controlnet_conditions = []
    for i in range(conf["diffusion"]["length"]):
        controlnet_condition = []
        for controlnet_prefix in controlnet_prefixes:
            filepath = os.path.join(
                conf["paths"]["conditions_path"], f"{controlnet_prefix}{(i+1):04}.png"
            )
            controlnet_condition.append(PIL.Image.open(filepath))
        controlnet_conditions.append(controlnet_condition)

    # Load pipeline
    pipe = controlvideo_pipeline(
        conf["repositories"]["sd"],
        conf["repositories"]["vae"],
        controlnet_repositories,
        conf["paths"]["checkpoints_path"],
        conf["diffusion"]["length"],
    )

    # Load video attributes
    additions = conf["diffusion"]["additions"]
    prompts = [None] * conf["diffusion"]["length"]
    negative_prompts = [None] * conf["diffusion"]["length"]
    for addition in additions:
        for frame in addition["frames"]:
            prompts[frame] = conf["diffusion"]["base_prompt"] % tuple(
                addition["add_prompt"]
            )
            negative_prompts[frame] = conf["diffusion"]["base_neg_prompt"] % tuple(
                addition["add_neg_prompt"]
            )

    # Prepare generators
    seed = conf["diffusion"]["seed"]
    same_frame_noise = conf["diffusion"]["same_frame_noise"]
    assert not ((not same_frame_noise) and (seed is not None))
    if same_frame_noise:
        generator = torch.Generator("cuda")
        if seed is None:
            seed = generator.seed()
        else:
            generator = generator.manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        generators = []
        for i in range(conf["diffusion"]["length"]):
            generator = torch.Generator("cuda")
            seed = generator.seed()
            generators.append(generator)
            print(f"Using seed: {i}, {seed}")
        generator = generators

    # Inference
    video = pipe(
        conf["paths"]["embeddings_path"],
        prompts,
        negative_prompts,
        controlnet_conditions,
        conf["diffusion"]["controlnets"]["scales"],
        conf["diffusion"]["controlnets"]["exp"],
        conf["diffusion"]["length"],
        generator,
        conf["diffusion"]["num_inference_steps"],
        conf["diffusion"]["guidance_scale"]
    )

    # Save
    for i, frame in enumerate(video):
        frame.save(
            os.path.join(conf["paths"]["diffusion_output_path"], f"{(i+1):04}.png")
        )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
