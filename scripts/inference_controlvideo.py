import argparse
import glob
import os
import pathlib
import sys

import PIL.Image

sys.path.append("..")

from src.pipelines.controlvideo_pipeline import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", type=str, required=True, help="Directory of output"
    )
    parser.add_argument(
        "--sd_repo",
        type=str,
        required=True,
        help="huggingface repository containing the main model weights",
    )
    parser.add_argument(
        "--vae_repo",
        type=str,
        required=True,
        help="huggingface repository containing the VAE weights",
    )
    parser.add_argument(
        "--controlnet_repo",
        type=str,
        required=True,
        help="huggingface repository containing the ControlNet weights",
    )
    parser.add_argument(
        "--ifnet_path", type=str, required=True, help="Path to the IFNet weights"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory to contain cached huggingface model weights",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to the target video in the Invoke.AI format",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        required=True,
        help="Negative prompt to the target video in the Invoke.AI format",
    )
    parser.add_argument(
        "--textual_inversion_path",
        type=str,
        required=True,
        help="Directory containing textual inversion embeddings",
    )
    parser.add_argument(
        "--controlnet_conditioning_path",
        type=str,
        required=True,
        help="Directory containing controlnet conditioning images",
    )
    parser.add_argument(
        "--video_length", type=int, default=16, help="Length of synthesized video"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of timesteps to take during DPM sampling",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--smoother_steps",
        type=str,
        default="19,20",
        help="Timesteps at which using interleaved-frame smoother",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Gamma, controlling the window size in hierachical sampling, defaults to sqrt(video_length)",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="Coefficient controlling how much the ControlNet weights affect the main SD UNet",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed of generator, defaults to a random seed",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    pipe = controlvideo_pipeline(
        args.sd_repo,
        args.vae_repo,
        args.controlnet_repo,
        args.ifnet_path,
        args.cache_dir,
    )

    # Load the output paths and ControlNet conditioning images
    output_paths = []
    controlnet_conditions = []
    for filepath in sorted(
        glob.glob(os.path.join(args.controlnet_conditioning_path, "*.png"))
    ):
        pl = pathlib.Path(filepath)
        output_paths.append(os.path.join(args.out_path, pl.stem, ".png"))
        controlnet_conditions.append(PIL.Image.open(filepath))

    # Inferrence
    output_video = pipe(
        args.prompt,
        args.negative_prompt,
        args.textual_inversion_path,
        controlnet_conditions,
        args.video_length,
        args.num_inference_steps,
        args.guidance_scale,
        args.smoother_steps.split(","),
        args.window_size,
        args.controlnet_conditioning_scale,
        args.seed,
    )

    # Save
    for i, output_frame in enumerate(output_video):
        output_frame.save(output_paths[i])
