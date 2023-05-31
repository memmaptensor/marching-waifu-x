import gc
import glob
import os
import pathlib

import torch
from compel import Compel
from diffusers import AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler


class controlnet_base_workflow:
    def __init__(self, vae_repo_id, ldm_repo_id, pipe, textual_inversion_folderpath):
        self.pipe = pipe.from_pretrained(
            ldm_repo_id,
            vae=AutoencoderKL.from_pretrained(vae_repo_id, torch_dtype=torch.float16),
            controlnet=[
                ControlNetModel.from_pretrained(
                    controlnet_repo_id, torch_dtype=torch.float16
                )
                for controlnet_repo_id in [
                    "lllyasviel/control_v11p_sd15_openpose",
                    "lllyasviel/control_v11f1p_sd15_depth",
                    "lllyasviel/control_v11p_sd15_normalbae",
                    "lllyasviel/control_v11p_sd15_lineart",
                ]
            ],
            torch_dtype=torch.float16,
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, use_karras_sigmas=True
        )

        for filepath in sorted(
            glob.glob(os.path.join(textual_inversion_folderpath, "*"))
        ):
            pl = pathlib.Path(filepath)
            if pl.suffix in [".safetensors", ".ckpt", ".pt"]:
                use_safetensors = pl.suffix == ".safetensors"
                self.pipe.load_textual_inversion(
                    filepath, token=pl.stem, use_safetensors=use_safetensors
                )

        self.compel = Compel(
            self.pipe.tokenizer,
            self.pipe.text_encoder,
            truncate_long_prompts=False,
            device="cuda:0",
        )

        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_sequential_cpu_offload()

    def load_generator(self, seed):
        self.generator = torch.Generator("cpu")
        if seed == -1:
            seed = self.generator.seed()
        else:
            self.generator = self.generator.manual_seed(seed)

        return seed

    def load_weighted_embeds(self, prompt, negative_prompt):
        conditioning = self.compel.build_conditioning_tensor(prompt)
        negative_conditioning = self.compel.build_conditioning_tensor(negative_prompt)

        self.weighted_embeds = self.compel.pad_conditioning_tensors_to_same_length(
            [conditioning, negative_conditioning]
        )
