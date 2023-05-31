import os
import sys
import subprocess

sys.path.append("..")

from src.utils.file_loader import *
from src.utils.image_wrapper import *
from src.workflows.controlnet_unet4_workflow import *

path = {
    "config_file": "inference.json",
    "prompt_additions_file": "../data/prompts/prompt_additions.json",
    "dd_prompt_file": "../data/prompts/deepdanbooru_prompt.txt",
    "textual_inversion_folder": "../data/embeddings/",
    "controlnet_conditions_folder": "../data/multi_controlnet/multi_controlnet_data/",
    "ngp_overview_folder": "../data/ngp/overview/",
    "ngp_train_folder": "../data/ngp/train/",
}


def process_results(image, seed, interim, filename):
    print(seed)

    image.save(os.path.join(path["ngp_train_folder"], filename))

    if interim is not None and len(interim) > 0:
        interim = [image_wrapper(x, "pil") for x in interim]

        interim_img = interim[0]
        for img in interim[1:]:
            interim_img.concatenate(img)

        interim_img.to_pil().save(os.path.join(path["ngp_overview_folder"], filename))


if __name__ == "__main__":
    # 0. Prepare all pipeline stages
    fl = file_loader()
    config = fl.load_json(path["config_file"])
    prompt_additions = [
        j["direction"]
        for j in fl.load_json(path["prompt_additions_file"])["prompt_addition"]
    ]

    controlnet_conditions = fl.load_controlnet_conditions(
        path["controlnet_conditions_folder"]
    )
    controlnet_scales = [
        config["controlnet"]["unit_scales"]["openpose"],
        config["controlnet"]["unit_scales"]["depth"],
        config["controlnet"]["unit_scales"]["normals"],
        config["controlnet"]["unit_scales"]["lineart"],
    ]

    unet4 = controlnet_unet4_workflow(
        config["models"]["vae_repo_id"],
        config["models"]["ldm_repo_id"],
        path["textual_inversion_folder"],
    )

    # 1. Generate prereq image
    image, seed, interim = unet4(
        config["pipeline"]["prereq"]["prompt"].format(prompt_additions[0]),
        config["pipeline"]["prereq"]["negative_prompt"],
        config["pipeline"]["prereq"]["steps"],
        config["pipeline"]["prereq"]["cfg_scale"],
        config["pipeline"]["prereq"]["denoising_strength"],
        config["pipeline"]["prereq"]["seed"],
        config["pipeline"]["prereq"]["callback_steps"],
        controlnet_conditions[0],
        controlnet_scales,
        config["controlnet"]["guidance"]["start"],
        config["controlnet"]["guidance"]["end"],
        config["controlnet"]["soft_exp"],
    )
    filename = "prereq.png"
    filepath = os.path.join(path["ngp_train_folder"], filename)
    process_results(image, seed, interim, filename)

    # 2. Run inference on DeepDanbooru
    subprocess.call(
        [
            "python",
            "inference_deepdanbooru.py",
            filepath,
            path["dd_prompt_file"],
        ]
    )

    # 3. Upscale prereq image
    subprocess.call(["python", "inference_realesrgan.py", filepath, filepath])
