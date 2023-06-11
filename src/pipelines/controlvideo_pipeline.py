import gc
import glob
import os
import pathlib
from typing import List, Optional, Tuple, Union

import PIL.Image
import torch
from compel import Compel
from diffusers import AutoencoderKL
from huggingface_hub import snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer

from src.controlvideo.controlnet import ControlNetModel3D
from src.controlvideo.dpmsolver_multistep import DPMSolverMultistepScheduler
from src.controlvideo.pipeline_controlvideo import ControlVideoPipeline
from src.controlvideo.unet import UNet3DConditionModel


class controlvideo_pipeline:
    @torch.no_grad()
    def __init__(
        self, sd_repo, vae_repo, controlnet_repos, cache_dir, num_indices
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

        # Load scheduler
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            sd_path,
            subfolder="scheduler",
            use_karras_sigmas=True,
            num_indices=num_indices,
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

    @torch.no_grad()
    def __call__(
        self,
        textual_inversion_path: str,
        keyframes: List[int],
        keyframe_prompt: torch.FloatTensor,
        keyframe_negative_prompt: torch.FloatTensor,
        clips: List[Tuple[List[int], List[int]]],
        clip_prompts: List[torch.FloatTensor],
        clip_negative_prompts: List[torch.FloatTensor],
        controlnet_frames: List[List[PIL.Image.Image]],
        controlnet_scales: List[float],
        controlnet_exp: float,
        video_length: int,
        generator: Union[torch.Generator, List[torch.Generator]],
        num_inference_steps: int = 20,
        guidance_scale: float = 10.0,
        eta: float = 0.0,
    ):
        # Load pipeline
        pipe = ControlVideoPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
        )

        # Load textual inversions
        for filepath in sorted(glob.glob(os.path.join(textual_inversion_path, "*"))):
            pl = pathlib.Path(filepath)
            if pl.suffix in [".safetensors", ".ckpt", ".pt"]:
                pipe.load_textual_inversion(
                    filepath,
                    token=pl.stem,
                    use_safetensors=(pl.suffix == ".safetensors"),
                )

        # Load text encoder
        pipe.text_encoder.to("cuda")
        compel = Compel(
            pipe.tokenizer,
            pipe.text_encoder,
            truncate_long_prompts=False,
            device="cuda",
        )
        keyframe_wembeds = compel.pad_conditioning_tensors_to_same_length(
            [
                compel.build_conditioning_tensor(keyframe_prompt),
                compel.build_conditioning_tensor(keyframe_negative_prompt),
            ]
        )
        clip_wembeds = [
            compel.pad_conditioning_tensors_to_same_length(
                [
                    compel.build_conditioning_tensor(prompt),
                    compel.build_conditioning_tensor(negative_prompt),
                ]
            )
            for prompt, negative_prompt in zip(clip_prompts, clip_negative_prompts)
        ]
        del compel
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        # Load pipeline optimizations
        pipe.enable_vae_slicing()
        pipe.enable_xformers_memory_efficient_attention()
        # pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()

        video = pipe.generate_long_video(
            keyframes,
            keyframe_wembeds,
            clips,
            clip_wembeds,
            controlnet_frames,
            controlnet_scales,
            controlnet_exp,
            generator,
            video_length,
            num_inference_steps,
            guidance_scale,
            eta=eta,
        ).video

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        return video
