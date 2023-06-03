import gc
import glob
import os
import pathlib
from typing import List, Union

import numpy as np
import PIL.Image
import torch
from compel import Compel
from diffusers import AutoencoderKL
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer

from src.controlvideo.controlnet import ControlNetModel3D
from src.controlvideo.dpmsolver_multistep import DPMSolverMultistepScheduler
from src.controlvideo.pipeline_controlvideo import ControlVideoPipeline
from src.controlvideo.RIFE.IFNet_HDv3 import IFNet
from src.controlvideo.unet import UNet3DConditionModel


class controlvideo_pipeline:
    @torch.no_grad()
    def __init__(
        self, sd_repo, vae_repo, controlnet_repo, ifnet_path, cache_dir, gen_indices
    ):
        # Cache model weights
        sd_path = snapshot_download(
            sd_repo, cache_dir=cache_dir, local_dir_use_symlinks=False
        )
        vae_path = snapshot_download(
            vae_repo, cache_dir=cache_dir, local_dir_use_symlinks=False
        )
        controlnet_path = snapshot_download(
            controlnet_repo, cache_dir=cache_dir, local_dir_use_symlinks=False
        )

        # Load models
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_path, subfolder="text_encoder"
        ).to(dtype=torch.float16)
        self.vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=torch.float16)
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            sd_path,
            subfolder="scheduler",
            use_karras_sigmas=True,
            gen_indices=gen_indices,
        )
        self.compel = Compel(
            self.tokenizer,
            self.text_encoder,
            truncate_long_prompts=False,
            device="cuda",
        )
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            sd_path, subfolder="unet"
        ).to(dtype=torch.float16)
        self.controlnet = ControlNetModel3D.from_pretrained_2d(controlnet_path).to(
            dtype=torch.float16
        )
        self.interpolator = IFNet(ifnet_path).to(device="cuda:0", dtype=torch.float16)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        textual_inversion_path: str,
        frames: List[PIL.Image.Image],
        video_length: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 10,
        smoother_steps: List[int] = [19, 20],
        window_size: Union[None, int] = None,
        controlnet_conditioning_scale: float = 1.0,
        seed: Union[None, int] = None,
    ):
        # Load pipeline
        generator = torch.Generator(device="cuda")
        if seed is None:
            seed = generator.seed()
        else:
            generator = generator.manual_seed(seed)
        print(f"Using seed: {seed}")

        pipe = ControlVideoPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            interpolater=self.interpolator,
            scheduler=self.scheduler,
        )

        for filepath in sorted(glob.glob(os.path.join(textual_inversion_path, "*"))):
            pl = pathlib.Path(filepath)
            if pl.suffix in [".safetensors", ".ckpt", ".pt"]:
                pipe.load_textual_inversion(
                    filepath,
                    token=pl.stem,
                    use_safetensors=(pl.suffix == ".safetensors"),
                )

        pipe.enable_vae_slicing()
        pipe.enable_xformers_memory_efficient_attention()
        # pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()

        # Inference
        if window_size is None:
            window_size = int(np.sqrt(video_length))
        weighted_embeds = self.compel.pad_conditioning_tensors_to_same_length(
            [
                self.compel.build_conditioning_tensor(prompt),
                self.compel.build_conditioning_tensor(negative_prompt),
            ]
        )
        video = pipe.generate_long_video(
            prompt_embeds=weighted_embeds[0],
            negative_prompt_embeds=weighted_embeds[1],
            frames=frames,
            video_length=video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            smoother_steps=smoother_steps,
            window_size=window_size,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            output_type="pil",
        ).videos[0]

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        return video
