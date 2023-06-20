import gc
import glob
import os
import pathlib
from typing import List, Union

import PIL.Image
import torch
from compel import Compel
from diffusers import AutoencoderKL
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer

from src.controlvideo.controlnet import ControlNetModel3D
from src.controlvideo.pipeline_controlvideo import ControlVideoPipeline
from src.controlvideo.unet import UNet3DConditionModel


class controlvideo_pipeline:
    @torch.no_grad()
    def __init__(
        self,
        sd_repo,
        vae_repo,
        controlnet_repos,
        cache_dir,
        num_frames,
        optimizations,
        scheduler,
        textual_inversion_path,
    ):
        # Cache model weights
        sd_path = snapshot_download(
            sd_repo, cache_dir=cache_dir, local_dir_use_symlinks=False
        )
        vae_path = snapshot_download(
            vae_repo, cache_dir=cache_dir, local_dir_use_symlinks=False
        )
        controlnet_paths = []
        for controlnet_repo in controlnet_repos:
            controlnet_paths.append(
                snapshot_download(
                    controlnet_repo, cache_dir=cache_dir, local_dir_use_symlinks=False
                )
            )

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")

        # Load CLIPText
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_path, subfolder="text_encoder"
        ).to(dtype=torch.float16)

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=torch.float16)

        # Load pipeline
        self.pipe = ControlVideoPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
        )

        # Load scheduler
        if scheduler == "unipcsolver_multistep":
            from src.controlvideo.unipcsolver_multistep import (
                UniPCSolverMultistepScheduler,
            )

            self.scheduler = UniPCSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                num_frames=num_frames,
            )
        elif scheduler == "dpmsolver_multistep":
            from src.controlvideo.dpmsolver_multistep import DPMSolverMultistepScheduler

            self.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                use_karras_sigmas=True,
                num_frames=num_frames,
            )

        # Load main UNet
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            sd_path, subfolder="unet", use_safetensors=True
        ).to(dtype=torch.float16)

        # Load ControlNets
        self.controlnet = [
            ControlNetModel3D.from_pretrained_2d(
                controlnet_path, use_safetensors=True
            ).to(dtype=torch.float16)
            for controlnet_path in controlnet_paths
        ]

        # Load textual inversions
        for filepath in sorted(glob.glob(os.path.join(textual_inversion_path, "*"))):
            pl = pathlib.Path(filepath)
            if pl.suffix in [".safetensors", ".ckpt", ".pt"]:
                self.pipe.load_textual_inversion(
                    filepath,
                    token=pl.stem,
                    use_safetensors=(pl.suffix == ".safetensors"),
                )

        # Load pipeline optimizations
        assert not (
            "model_cpu_offload" in optimizations
            and "sequential_cpu_offload" in optimizations
        )
        for optimization in optimizations:
            if optimization == "vae_slicing":
                self.pipe.enable_vae_slicing()
            elif optimization == "xformers":
                self.pipe.enable_xformers_memory_efficient_attention()
            elif optimization == "cuda":
                self.pipe.to("cuda")
            elif optimization == "model_cpu_offload":
                self.pipe.enable_model_cpu_offload()
            elif optimization == "sequential_cpu_offload":
                self.pipe.enable_sequential_cpu_offload()

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        negative_prompts: List[str],
        controlnet_frames: List[List[PIL.Image.Image]],
        controlnet_scales: List[float],
        controlnet_exp: float,
        video_length: int,
        generator: Union[torch.Generator, List[torch.Generator]],
        num_inference_steps: int = 20,
        guidance_scale: float = 10.0,
        eta: float = 0.0,
    ):
        # Load text encoder
        self.pipe.text_encoder.to("cuda")
        compel = Compel(
            self.pipe.tokenizer,
            self.pipe.text_encoder,
            truncate_long_prompts=False,
            device="cuda",
        )
        frame_wembeds = [
            compel.pad_conditioning_tensors_to_same_length(
                [
                    compel.build_conditioning_tensor(prompt),
                    compel.build_conditioning_tensor(negative_prompt),
                ]
            )
            for prompt, negative_prompt in zip(prompts, negative_prompts)
        ]
        del compel
        self.pipe.text_encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        video = self.pipe.generate_long_video(
            frame_wembeds,
            controlnet_frames,
            controlnet_scales,
            controlnet_exp,
            generator,
            video_length,
            num_inference_steps,
            guidance_scale,
            eta=eta,
        ).video

        return video
