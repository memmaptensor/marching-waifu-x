import torch

from src.pipelines.controlnet_unet4_pipeline import *
from src.workflows.controlnet_base_workflow import *


class controlnet_unet4_workflow(controlnet_base_workflow):
    @torch.no_grad()
    def __init__(self, vae_repo_id, ldm_repo_id, textual_inversion_folderpath):
        super().__init__(
            vae_repo_id,
            ldm_repo_id,
            controlnet_unet4_pipeline,
            textual_inversion_folderpath,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt,
        steps,
        cfg_scale,
        denoising_strength,
        seed,
        callback_steps,
        controlnet_conditions,
        controlnet_scales,
        controlnet_guidance_start,
        controlnet_guidance_end,
        controlnet_soft_exp,
        image=None,
    ):
        seed = self.load_generator(seed)
        self.load_weighted_embeds(prompt, negative_prompt)

        interim = []
        image = self.pipe(
            image=image,
            controlnet_conditioning_image=controlnet_conditions,
            strength=denoising_strength,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=self.generator,
            prompt_embeds=self.weighted_embeds[0],
            negative_prompt_embeds=self.weighted_embeds[1],
            callback=None
            if callback_steps == 0
            else (lambda i, t, img: interim.append(img)),
            callback_steps=callback_steps,
            controlnet_conditioning_scale=controlnet_scales,
            controlnet_guidance_start=controlnet_guidance_start,
            controlnet_guidance_end=controlnet_guidance_end,
            controlnet_soft_exp=controlnet_soft_exp,
        )[0]

        return (image, seed, interim)
