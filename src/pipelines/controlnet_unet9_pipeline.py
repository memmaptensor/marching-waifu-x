import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    UNet2DConditionModel,
    logging,
)
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import (
    MultiControlNetModel,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import PIL_INTERPOLATION, randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer

from src.utils.image_wrapper import *

logger = logging.get_logger(__name__)


class controlnet_unet9_pipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin
):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[
            ControlNetModel,
            List[ControlNetModel],
            MultiControlNetModel,
        ],
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        if isinstance(controlnet, list):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=False)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        self.vae.disable_tiling()

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

    def _zeros_tensor(
        self,
        shape: Union[Tuple, List],
        generator: Optional[Union[List[torch.Generator], torch.Generator]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        layout: Optional[torch.layout] = None,
    ):
        rand_device = device
        batch_size = shape[0]

        layout = layout or torch.strided
        device = device or torch.device("cpu")

        if generator is not None:
            gen_device_type = (
                generator.device.type
                if not isinstance(generator, list)
                else generator[0].device.type
            )
            if gen_device_type != device.type and gen_device_type == "cpu":
                rand_device = "cpu"

        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [
                torch.zeros(
                    shape,
                    device=rand_device,
                    dtype=dtype,
                    layout=layout,
                )
                for _ in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.zeros(
                shape,
                device=rand_device,
                dtype=dtype,
                layout=layout,
            ).to(device)

        return latents

    def _pil_to_tensors(self, image):
        if isinstance(image, PIL.Image.Image):
            image = [image]
        image = [np.array(i.convert("RGB"))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        return image

    def _pil_to_mask_tensors(self, mask_image):
        if isinstance(mask_image, PIL.Image.Image):
            mask_image = [mask_image]
        mask_image = np.concatenate(
            [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0
        )
        mask_image = mask_image.astype(np.float32) / 255.0
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)

        return mask_image

    def _pil_to_controlnet_conditioning_tensors(
        self,
        controlnet_conditioning_image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance,
    ):
        if isinstance(controlnet_conditioning_image, PIL.Image.Image):
            controlnet_conditioning_image = [controlnet_conditioning_image]

        controlnet_conditioning_image = [
            np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]))[
                None, :
            ]
            for i in controlnet_conditioning_image
        ]
        controlnet_conditioning_image = np.concatenate(
            controlnet_conditioning_image, axis=0
        )
        controlnet_conditioning_image = (
            np.array(controlnet_conditioning_image).astype(np.float32) / 255.0
        )
        controlnet_conditioning_image = controlnet_conditioning_image.transpose(
            0, 3, 1, 2
        )
        controlnet_conditioning_image = torch.from_numpy(controlnet_conditioning_image)

        image_batch_size = controlnet_conditioning_image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt

        controlnet_conditioning_image = controlnet_conditioning_image.repeat_interleave(
            repeat_by, dim=0
        )

        controlnet_conditioning_image = controlnet_conditioning_image.to(
            device=device, dtype=dtype
        )

        if do_classifier_free_guidance:
            controlnet_conditioning_image = torch.cat(
                [controlnet_conditioning_image] * 2
            )

        return controlnet_conditioning_image

    def _encode_prompt(
        self,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        batch_size = prompt_embeds.shape[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

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

    def _get_timesteps(self, num_inference_steps, strength):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def _decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        return image

    def _prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        num_channels_latents,
        width,
        height,
        fill_mode,
        generator,
    ):
        batch_size = batch_size * num_images_per_prompt

        def _get_noised_latents(image, transformation=(lambda x: x)):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            image = [transformation(img) for img in image]
            image = self._pil_to_tensors(image)
            image = image.to(device=device, dtype=dtype)

            if isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents
            init_latents = torch.cat([init_latents], dim=0)
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
            latents = init_latents

            return latents

        def _get_latents_shape():
            return (
                batch_size,
                num_channels_latents,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )

        if fill_mode == "original_mean":
            return _get_noised_latents(
                image,
                lambda x: PIL.Image.new(
                    "RGB",
                    (width, height),
                    tuple(
                        np.average(
                            np.average(image_wrapper(x, "pil").to_np(), axis=0),
                            axis=0,
                        )
                    ),
                ),
            )
        elif fill_mode == "original":
            return _get_noised_latents(image)
        elif fill_mode == "latent_rand":
            latents = (
                randn_tensor(
                    _get_latents_shape(),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                * self.scheduler.init_noise_sigma
            )
            return latents
        elif fill_mode == "latent_none":
            latents = (
                self.zeros_tensor(
                    _get_latents_shape(),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                * self.scheduler.init_noise_sigma
            )
            return latents

    def _prepare_mask_latents(
        self,
        mask_image,
        batch_size,
        height,
        width,
        dtype,
        device,
        do_classifier_free_guidance,
    ):
        mask_image = F.interpolate(
            mask_image,
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
        )
        mask_image = mask_image.to(device=device, dtype=dtype)

        if mask_image.shape[0] < batch_size:
            mask_image = mask_image.repeat(batch_size // mask_image.shape[0], 1, 1, 1)

        mask_image = (
            torch.cat([mask_image] * 2) if do_classifier_free_guidance else mask_image
        )

        mask_image_latents = mask_image

        return mask_image_latents

    def _prepare_masked_image_latents(
        self,
        masked_image,
        batch_size,
        height,
        width,
        dtype,
        device,
        generator,
        do_classifier_free_guidance,
    ):
        masked_image = masked_image.to(device=device, dtype=dtype)

        if isinstance(generator, list):
            masked_image_latents = [
                self.vae.encode(masked_image[i : i + 1]).latent_dist.sample(
                    generator=generator[i]
                )
                for i in range(batch_size)
            ]
            masked_image_latents = torch.cat(masked_image_latents, dim=0)
        else:
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(
                generator=generator
            )
        masked_image_latents = self.vae.config.scaling_factor * masked_image_latents

        if masked_image_latents.shape[0] < batch_size:
            masked_image_latents = masked_image_latents.repeat(
                batch_size // masked_image_latents.shape[0], 1, 1, 1
            )

        masked_image_latents = (
            torch.cat([masked_image_latents] * 2)
            if do_classifier_free_guidance
            else masked_image_latents
        )
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        return masked_image_latents

    def _default_height_width(self, height, width, image):
        if isinstance(image, list):
            image = image[0]

        if height is None:
            height = image.height
            height = (height // 8) * 8

        if width is None:
            width = image.width
            width = (width // 8) * 8

        return height, width

    @torch.no_grad()
    def __call__(
        self,
        image: PIL.Image.Image = None,
        mask_image: PIL.Image.Image = None,
        controlnet_conditioning_image: Union[
            PIL.Image.Image,
            List[PIL.Image.Image],
        ] = None,
        strength: float = 0.8,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, PIL.Image.Image], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: float = 1.0,
        controlnet_guidance_start: float = 0.0,
        controlnet_guidance_end: float = 1.0,
        controlnet_soft_exp: float = 1.0,
        fill_mode: str = None,
    ):
        # 0. Default height and width to unet
        height, width = self._default_height_width(
            height, width, controlnet_conditioning_image
        )

        # 1. Define call parameters
        batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(self.controlnet, MultiControlNetModel) and isinstance(
            controlnet_conditioning_scale, float
        ):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(
                self.controlnet.nets
            )

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 3. Prepare mask, masked, and controlnet_conditioning_image
        mask_image = self._pil_to_mask_tensors(mask_image)
        masked_image = self._pil_to_tensors(image) * (mask_image < 0.5)
        if isinstance(self.controlnet, ControlNetModel):
            controlnet_conditioning_image = (
                self._pil_to_controlnet_conditioning_tensors(
                    controlnet_conditioning_image,
                    width,
                    height,
                    batch_size * num_images_per_prompt,
                    num_images_per_prompt,
                    device,
                    self.controlnet.dtype,
                    do_classifier_free_guidance,
                )
            )
        elif isinstance(self.controlnet, MultiControlNetModel):
            controlnet_conditioning_images = []
            for image_ in controlnet_conditioning_image:
                image_ = self._pil_to_controlnet_conditioning_tensors(
                    image_,
                    width,
                    height,
                    batch_size * num_images_per_prompt,
                    num_images_per_prompt,
                    device,
                    self.controlnet.dtype,
                    do_classifier_free_guidance,
                )
                controlnet_conditioning_images.append(image_)
            controlnet_conditioning_image = controlnet_conditioning_images

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps, strength
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self._prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            num_channels_latents,
            width,
            height,
            fill_mode,
            generator,
        )
        mask_image_latents = self._prepare_mask_latents(
            mask_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            do_classifier_free_guidance,
        )
        masked_image_latents = self._prepare_masked_image_latents(
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self._prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                non_inpainting_latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                non_inpainting_latent_model_input = self.scheduler.scale_model_input(
                    non_inpainting_latent_model_input, t
                )
                inpainting_latent_model_input = torch.cat(
                    [
                        non_inpainting_latent_model_input,
                        mask_image_latents,
                        masked_image_latents,
                    ],
                    dim=1,
                )

                current_sampling_percent = i / len(timesteps)

                if (
                    current_sampling_percent < controlnet_guidance_start
                    or current_sampling_percent > controlnet_guidance_end
                ):
                    down_block_res_samples = None
                    mid_block_res_sample = None
                else:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        non_inpainting_latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=controlnet_conditioning_image,
                        conditioning_scale=controlnet_conditioning_scale,
                        return_dict=False,
                    )

                    scales = list(
                        map(lambda i: controlnet_soft_exp ** (12 - i), range(13))
                    )
                    block_res_samples = [*down_block_res_samples, mid_block_res_sample]
                    new_block_res_samples = [
                        bw * s for bw, s in zip(block_res_samples, scales)
                    ]

                    down_block_res_samples = new_block_res_samples[:-1]
                    mid_block_res_sample = new_block_res_samples[-1]

                noise_pred = self.unet(
                    inpainting_latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(
                            i, t, self.numpy_to_pil(self._decode_latents(latents))[0]
                        )

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        # 8. Post-processing
        image = self._decode_latents(latents)

        # 9. Convert to PIL
        image = self.numpy_to_pil(image)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return image
