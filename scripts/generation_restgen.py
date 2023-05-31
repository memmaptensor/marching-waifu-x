import os
import sys
import subprocess

sys.path.append("..")

import PIL.Image

from src.utils.file_loader import *
from src.utils.image_wrapper import *
from src.workflows.controlnet_unet9_workflow import *

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


# RestGen loop:
#   Construct parameters
#   Generate new image
#   Crop, save, upscale new image
def restgen_loop(
    image_filename,
    mask_relindex,
    controlnet_indices,
    unet9,
    prompt,
    prompt_additions,
    config,
    controlnet_conditions,
    controlnet_scales,
):
    # Construct stitched image
    images = [
        image_wrapper(
            PIL.Image.open(os.path.join(path["ngp_train_folder"], f)), "pil"
        ).scale(0.25)
        for f in image_filename
    ]
    for image in images[1:]:
        images[0].concatenate(image)
    images = images[0].to_pil()

    # Calculate crop area
    crop_area = (
        mask_relindex * int(images.width / len(image_filename)),
        0,
        (mask_relindex + 1) * int(images.width / len(image_filename)),
        int(images.height),
    )

    # Construct mask image
    mask = PIL.Image.new("1", (images.width, images.height))
    mask.paste(True, crop_area)

    # Construct ControlNet images
    controlnet_images = [
        [
            image_wrapper(condition.copy(), "pil")
            for condition in controlnet_conditions[i]
        ]
        for i in controlnet_indices
    ]
    for controlnet_image in controlnet_images[1:]:
        for i, condition in enumerate(controlnet_image):
            controlnet_images[0][i].concatenate(condition)
    controlnet_images = [condition.to_pil() for condition in controlnet_images[0]]

    # Get absolute index
    absolute = controlnet_indices[mask_relindex]

    # Generate, crop, save new image
    image, seed, interim = unet9(
        prompt.format(prompt_additions[absolute]),
        config["pipeline"]["restgen"]["negative_prompt"],
        config["pipeline"]["restgen"]["steps"],
        config["pipeline"]["restgen"]["cfg_scale"],
        config["pipeline"]["restgen"]["denoising_strength"],
        config["pipeline"]["restgen"]["seed"],
        config["pipeline"]["restgen"]["callback_steps"],
        controlnet_images,
        controlnet_scales,
        config["controlnet"]["guidance"]["start"],
        config["controlnet"]["guidance"]["end"],
        config["controlnet"]["soft_exp"],
        images,
        mask,
        config["pipeline"]["restgen"]["inpaint_method"],
    )
    image = image.crop(crop_area)
    filename = f"{(absolute+1):04}.png"
    filepath = os.path.join(path["ngp_train_folder"], filename)
    process_results(image, seed, interim, filename)

    # Upscale new image
    subprocess.call(["python", "inference_realesrgan.py", filepath, filepath])


if __name__ == "__main__":
    # Prepare all pipeline stages
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

    prompt = fl.load_text(path["dd_prompt_file"])

    unet9 = controlnet_unet9_workflow(
        config["models"]["vae_repo_id"],
        config["models"]["ldm_inpaint_repo_id"],
        path["textual_inversion_folder"],
    )

    # Generate first image
    restgen_loop(
        ["prereq.png", "prereq.png"],
        1,
        [0, 0],
        unet9,
        prompt,
        prompt_additions,
        config,
        controlnet_conditions,
        controlnet_scales,
    )

    # Generate the remaining image (index [1, dataset_size))
    for i in range(1, config["pipeline"]["restgen"]["dataset_size"]):
        restgen_loop(
            [f"{i:04}.png", f"{i:04}.png", "prereq.png"],
            1,
            [i - 1, i, 0],
            unet9,
            prompt,
            prompt_additions,
            config,
            controlnet_conditions,
            controlnet_scales,
        )
