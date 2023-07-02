# marching-waifu-x 💖

[![License](https://img.shields.io/github/license/rossiyareich/marching-waifu-x)](https://github.com/rossiyareich/marching-waifu-x/blob/main/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rossiyareich/marching-waifu-x/blob/main/ipynb/end2end_colab.ipynb)

Medium write-up: [Diffusion models are zero-shot 3D character generators, too](https://medium.com/@rwussiya/diffusion-models-are-zero-shot-3d-character-generators-too-6261c264755c)

## Table of Contents
- [Introduction](#introduction)
- [Preliminary works](#preliminary-works)
- [Pipeline and architecture](#pipeline-and-architecture)
- [Related resources](#related-resources)

## Introduction
Source repository of "marching-waifu-x 💖: Diffusion models are zero-shot 3D character generators, too"

Exploring state-of-the-art latent diffusion techniques and assembling a pipeline for end-to-end 3D character generation 😊

## Preliminary works
We first explore existing Text-to-video and Text-to-3D works, along with additional prior works used in **marching-waifu-x**

**DreamFusion** uses a text-to-image generative model called Imagen to optimize a 3D scene. It proposed a way to optimize Neural Radience Fields by generating samples from a diffusion model using a technique dubbed "Score Distillation Sampling (SDS)"; SDS allows optimization of samples of a network in an arbitrary parameter space as long as said space can be mapped back to images differentiably, such as the renderings from a 3D Neural Volume represented by Neural Radiance Fields.

**ControlVideo** adapts ControlNet to the video counterpart by adding fully cross-frame interaction into self-attention modules. ControlVideo also utilizes an additional network for interleave-frame smoothing to smooth all inter-frame transitions via the interleaved interpolation (RIFE)

**GroundingDINO and SegmentAnything (GroundedSAM)** GroundingDINO is a string zero-shot detector which is capable of generating high quality AABBs and labels with free-form text; SegmentAnything is a string foundation model aiming to segment objects in an image, given text prompts and AABBs/Points to generate high-quality object masks

**nvdiffrec** is a multi-stage pipeline for joint optimization of topology, materials and lighting from multi-view image observations. Unlike other multi-view reconstruction approaches, which typically produce entangled 3D representations encoded in neural networks, **nvdiffrec** outputs triangle meshes with spatially-varying materials and environment lighting that can be deployed in any traditional graphics engine unmodified.

## Pipeline and architecture
**marching-waifu-x** utilizes a multi-stage pipeline for 3D character generation; mainly consisting of two stages-- Text-to-Video and Video-to-3D.

**(1) Text-to-Video**<br>
We adapt the following techniques to achieve generations from latent diffusion models with temporal consistency from ControlVideo:
- With current limitations on LDMs, we can not feasably achieve a high level of control on the generations only with the hidden states derived from the CLIP text encoder alone
- Thus, we leverage ControlNet along with pre-rendered conditioning images of different classes for synthesizing a dataset consisting of sparse views of the same character and pose from different camera views; We used blender to render out 100 images of a base female character and cleaned up the resulting OpenPose conditioning images to be compatible with pretrained OpenPose controlnet models.
- However, the combination of ControlNet and text descriptions was not enough to achieve a level of consistency needed for a NeRF or NGP dataset
- To achieve temporal consistency between frames of generated image sequences, **ControlVideo** modifies stable diffusion's main U-Net and auxiliary U-Net (ControlNet) architecture by adding fully cross-frame interaction into the self-attention modules. It does this in a similiar fashion to **Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation**; by replacing the 2D convolutional residual blocks with their 3D counterpart. The learned weights for each of the expanded dimension's layers are simply filled with the same learned weights from the initial layer (dubbed `Network Inflation`). The transformer blocks are also modified to support an input of multiple latents by first concatenating all the latents along the batch dimension before presenting it to the multi-head self-attention module.
- Improving memory efficiency was a main concern of ours, as the implementation of **ControlVideo** only supports two generation schemes:
    - Fully cross-frame generation
        - All frames are generated at once 
        - The memory complexity is similiar to that of generating all frames at the same time in a batch
        - While this generation scheme produces the most consistent result, and without requiring the retraining of network weights, the memory complexity grows exponentially over time and generation of videos beyond ~32 frames at 768x1024 with no ControlNet guidance is not possible on a V100 with 16GiB of VRAM.
    - Sparse-causal generation
        - The keyframes are first generated
        - Then, each clip is generated, where each clip consists of the in-between frames and the pair of surrounding keyframes. The intuition behind this generation scheme is that the in-between frames will pertain to the surrounding keyframes
        - The memory complexity is similiar to that of generating all the keyframes and a single clip for all clips (with each clip size being a controllable generation parameter)
        - **ControlVideo** utilizes a modified sparse-causal generation scheme to reduce VRAM usages during generation. Although it does reduce the memory complexity, there was a lack of consistency across multiple clips 
- We explore a generation scheme inspired by **Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation**
    - Sliding-window cross-frame generation
        - The first frame is generated independently
        - The second frame is generated along with the first frame
        - The rest of the frames are generated with the first frame and the previous frame
        - The memory complexity is similiar to that of generating 1, 2, and 3 frames in a batch. This complexity, however, does not depend on the video length
        - We found this generation scheme to be as consistent as fully cross-frame generation, while allowing generations of infinitely long videos
- We adopt two higher-order schedulers-- UniPC and Multistep DPM++
    - Using less timesteps reduces total generation time for the pipeline
    - **ControlVideo** utilizes a DDIM scheduler, requiring at least 50-100 timesteps for producing high-quality samples
    - Multistep DPM++ allows generation of high-quality samples using 20 timesteps
    - UniPC allows generation of high-quality samples using 10-15 timesteps, although using 20 timesteps, the sample results are not comparable to that of Multistep DPM++
- We remove the IF-Net/RIFE model from the denoising loop
    - The authors of **ControlVideo** introduced the interleaved-frame smoother model to reduce flickering within the results
    - With the IF-Net, visible flickers are mediated-- though, this has the side effect of desaturating the output results
    - We found the effects of deflickering to not impact the training of **InstantNGP** by a large factor

**(2) Video-to-3D**<br>
- instant-ngp
    - Training parameters are as follows:
        ```
        sharpen = 1.0
        n_steps = 5000
        ```
- nvdiffrec
    - We utilize nvdiffrec for extracting diffuse, specular, and normal maps along with the character mesh from the synthetic 2D image dataset
    - Default training parameters are as follows:
        ```json
        {
            "ref_mesh": "data/nerf",
            "random_textures": true,
            "iter": 2500,
            "save_interval": 500,
            "texture_res": [2048, 2048],
            "train_res": [1024, 768],
            "batch": 4,
            "learning_rate": [0.03, 0.01],
            "ks_min" : [0, 0.08, 0.0],
            "dmtet_grid" : 128,
            "mesh_scale" : 2.0,
            "laplace_scale" : 2500,
            "display": [
                {"latlong" : true}, 
                {"bsdf" : "kd"}, 
                {"bsdf" : "ks"}, 
                {"bsdf" : "normal"}
            ],
            "background" : "white",
            "out_dir": "nerf_output"
        }
        ```

## Related resources
- [huggingface/diffusers](https://github.com/huggingface/diffusers)
    - [rossiyareich/abyssorangemix3-popupparade-fp16](https://huggingface.co/rossiyareich/abyssorangemix3-popupparade-fp16)
    - [rossiyareich/anything-v4.0-vae](https://huggingface.co/rossiyareich/anything-v4.0-vae)
    - [gsdf/EasyNegative](https://huggingface.co/datasets/gsdf/EasyNegative)
    - [AsciiP/badhandv4](https://huggingface.co/AsciiP/badhandv4)
    - [veryBadImageNegative](https://civitai.com/models/11772)
- [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [maximeraafat/BlenderNeRF](https://github.com/maximeraafat/BlenderNeRF)
- [genshin-style-anime-female-base-mesh-for-blender](https://sketchfab.com/3d-models/genshin-style-anime-female-base-mesh-for-blender-c2d6727e8c9742feb9a4a3bccac6e0e0)
- [OpenPoseBones_v9](https://toyxyz.gumroad.com/l/ciojz)