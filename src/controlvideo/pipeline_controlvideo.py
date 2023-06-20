import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKL, ModelMixin, logging
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import PIL_INTERPOLATION, BaseOutput, logging, randn_tensor
from einops import rearrange
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from src.controlvideo.controlnet import ControlNetModel3D, ControlNetOutput
from src.controlvideo.unet import UNet3DConditionModel

logger = logging.get_logger(__name__)


@dataclass
class ControlVideoPipelineOutput(BaseOutput):
    video: List[PIL.Image.Image]


class MultiControlNetModel3D(ModelMixin):
    def __init__(
        self, controlnets: Union[List[ControlNetModel3D], Tuple[ControlNetModel3D]]
    ):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.FloatTensor],
        conditioning_scale: List[float],
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        for i, (image, scale, controlnet) in enumerate(
            zip(controlnet_cond, conditioning_scale, self.nets)
        ):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states,
                image,
                scale,
                class_labels,
                timestep_cond,
                attention_mask,
                cross_attention_kwargs,
                return_dict,
            )

            # Merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(
                        down_block_res_samples, down_samples
                    )
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample


class ControlVideoPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin
):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        controlnet: Union[
            List[ControlNetModel3D],
            Tuple[ControlNetModel3D],
            MultiControlNetModel3D,
        ],
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel3D(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        from accelerate import cpu_offload

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [
            self.unet,
            self.text_encoder,
            self.vae,
            self.controlnet,
        ]:
            cpu_offload(cpu_offloaded_model, device)

        self.controlnet.enable_sequential_cpu_offload(gpu_id)

    def enable_model_cpu_offload(self, gpu_id=0):
        from accelerate import cpu_offload_with_hook

        device = torch.device(f"cuda:{gpu_id}")

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(
                cpu_offloaded_model, device, prev_module_hook=hook
            )

        cpu_offload_with_hook(self.controlnet, device)
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _decode_latents(self, latents, return_tensor=False):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        if return_tensor:
            return video
        video = video.cpu().float().numpy()
        return video

    def _prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def _prepare_image(
        self,
        image,
        width,
        height,
        device,
        dtype,
    ):
        image = [image]
        images = []

        for image_ in image:
            image_ = image_.convert("RGB")
            image_ = image_.resize(
                (width, height), resample=PIL_INTERPOLATION["lanczos"]
            )
            image_ = np.array(image_)
            image_ = image_[None, :]
            images.append(image_)

        image = images
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        image = image.repeat_interleave(1, dim=0)
        image = image.to(device=device, dtype=dtype)
        image = torch.cat([image] * 2)

        return image

    def _prepare_latents(
        self,
        num_channels_latents,
        video_length,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        same_frame_noise=True,
    ):
        if latents is None:
            if same_frame_noise:
                shape = (
                    1,
                    num_channels_latents,
                    1,
                    height // self.vae_scale_factor,
                    width // self.vae_scale_factor,
                )
                latents = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
                latents = latents.repeat(1, 1, video_length, 1, 1)
            else:
                shape = (
                    1,
                    num_channels_latents,
                    video_length,
                    height // self.vae_scale_factor,
                    width // self.vae_scale_factor,
                )
                latents = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
        else:
            shape = (
                1,
                num_channels_latents,
                video_length,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_alpha_prev(self, timestep):
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[max(0, prev_timestep)]
        return alpha_prod_t_prev

    @torch.no_grad()
    def generate_long_video(
        self,
        frame_wembeds: List[torch.FloatTensor],
        controlnet_frames: List[List[PIL.Image.Image]],
        controlnet_scales: List[float],
        controlnet_exp: float,
        generator: Union[torch.Generator, List[torch.Generator]],
        video_length: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latents: Optional[torch.FloatTensor] = None,
        eta: float = 0.0,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = (controlnet_frames[0][0].height // 8) * 8
        width = (controlnet_frames[0][0].width // 8) * 8

        # 1. Define call parameters
        assert guidance_scale > 1.0
        device = self._execution_device
        controlnet_block_scales = [controlnet_exp ** (12 - i) for i in range(13)]

        # 2. Encode input prompts
        frame_wembeds = [
            torch.cat(
                [
                    wembeds[1].to(device, self.text_encoder.dtype),
                    wembeds[0].to(device, self.text_encoder.dtype),
                ]
            )
            for wembeds in frame_wembeds
        ]

        # 3. Prepare image
        images = [[] for _ in range(len(controlnet_frames[0]))]
        for cnet_frame in controlnet_frames:
            for i, cnet_img in enumerate(cnet_frame):
                image = self._prepare_image(
                    image=cnet_img,
                    width=width,
                    height=height,
                    device=device,
                    dtype=self.controlnet.dtype,
                )
                images[i].append(image)
        controlnet_frames = [None] * len(controlnet_frames[0])
        for i, cnet_frames in enumerate(images):
            controlnet_frames[i] = torch.stack(cnet_frames, dim=2)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self._prepare_latents(
            num_channels_latents,
            video_length,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
            latents,
            same_frame_noise=isinstance(generator, torch.Generator),
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self._prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for CFG
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                noise_pred = torch.zeros_like(latents)
                pred_original_sample = torch.zeros_like(latents)

                for frame_n in range(video_length):
                    if frame_n == 0:
                        frames = [0]
                        focus_rel = 0
                    elif frame_n == 1:
                        frames = [0, 1]
                        focus_rel = 1
                    else:
                        frames = [frame_n - 1, frame_n, 0]
                        focus_rel = 1

                    # Inference on ControlNet
                    (
                        down_block_res_samples,
                        mid_block_res_sample,
                    ) = self.controlnet(
                        latent_model_input[:, :, frames],
                        t,
                        encoder_hidden_states=frame_wembeds[frame_n],
                        controlnet_cond=[
                            cnet_frames[:, :, frames]
                            for cnet_frames in controlnet_frames
                        ],
                        conditioning_scale=controlnet_scales,
                        return_dict=False,
                    )
                    block_res_samples = [
                        *down_block_res_samples,
                        mid_block_res_sample,
                    ]
                    block_res_samples = [
                        b * s
                        for b, s in zip(block_res_samples, controlnet_block_scales)
                    ]
                    down_block_res_samples = block_res_samples[:-1]
                    mid_block_res_sample = block_res_samples[-1]

                    # Inference on UNet
                    pred_noise_pred = self.unet(
                        latent_model_input[:, :, frames],
                        t,
                        encoder_hidden_states=frame_wembeds[frame_n],
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        inter_frame=False,
                    ).sample

                    # Perform CFG
                    noise_pred_uncond, noise_pred_text = pred_noise_pred[
                        :, :, focus_rel
                    ].chunk(2)
                    noise_pred[:, :, frame_n] = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    # Compute the previous noisy sample x_t -> x_t-1
                    step_dict = self.scheduler.step(
                        noise_pred[:, :, frame_n],
                        t,
                        latents[:, :, frame_n],
                        frame_n,
                        **extra_step_kwargs,
                    )
                    latents[:, :, frame_n] = step_dict.prev_sample
                    pred_original_sample[:, :, frame_n] = step_dict.pred_original_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # If we do sequential model offloading, let's offload UNet and ControlNet manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        # Post-processing
        video = self._decode_latents(latents)
        video = rearrange(video, "b c f h w -> b f h w c")[0]
        video = [self.numpy_to_pil(frame)[0] for frame in video]

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return ControlVideoPipelineOutput(video=video)
