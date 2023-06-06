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
    controlnet_prefixes = conf["controlnet"]["pipe"].keys()
    controlnet_repositories = conf["controlnet"]["pipe"].values()
    controlnet_conditions = []
    for i in range(conf["video"]["length"]):
        controlnet_condition = []
        for controlnet_prefix in controlnet_prefixes:
            filepath = os.path.join(
                conf["paths"]["condition_path"], f"{controlnet_prefix}{(i+1):04}.png"
            )
            controlnet_condition.append(PIL.Image.open(filepath))
        controlnet_conditions.append(controlnet_condition)

    # Load video attributes
    num_clips = len(conf["video"]["clips"])

    # Load pipeline
    pipe = controlvideo_pipeline(
        conf["repositories"]["sd"],
        conf["repositories"]["vae"],
        controlnet_repositories,
        conf["paths"]["ifnet_path"],
        conf["paths"]["cache_dir"],
        num_clips + 1,
    )

    # Load video attributes
    clips = [
        (clip["attn_frames"], clip["clip_frames"]) for clip in conf["video"]["clips"]
    ]
    clip_prompts = [clip["prompt"] for clip in conf["video"]["clips"]]
    clip_negative_prompts = [clip["negative_prompt"] for clip in conf["video"]["clips"]]

    # Prepare generators
    seed = conf["video"]["seed"]
    same_frame_noise = conf["video"]["same_frame_noise"]
    assert (seed is not None) and (not same_frame_noise)
    if same_frame_noise:
        generator = torch.Generator("cuda")
        if seed is None:
            seed = generator.seed()
        else:
            generator = generator.manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        generators = []
        for _ in range(conf["video"]["length"]):
            generator = torch.Generator("cuda")
            seed = generator.seed()
            generators.append(generator)
            print(f"Using seed: {seed}")
        generator = generators

    # Inference
    video = pipe(
        conf["paths"]["textual_inversion_path"],
        conf["video"]["keyframes"]["frames"],
        conf["video"]["keyframes"]["prompt"],
        conf["video"]["keyframes"]["negative_prompt"],
        clips,
        clip_prompts,
        clip_negative_prompts,
        controlnet_conditions,
        conf["controlnet"]["scales"],
        conf["controlnet"]["exp"],
        conf["video"]["length"],
        generator,
        conf["video"]["num_inference_steps"],
        conf["video"]["guidance_scale"],
        conf["video"]["smooth_steps"],
    ).video

    # Save
    for i, frame in enumerate(video):
        frame.save(os.path.join(conf["paths"]["out_path"], f"{(i+1):04}.png"))

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
