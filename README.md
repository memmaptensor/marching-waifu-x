# marching-waifu-x

<p>
    <a href="https://github.com/rossiyareich/marching-waifu-x/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/rossiyareich/marching-waifu-x">
    </a>
</p>

Complete 3D character + animation generation based on ControlVideo, GroundingDINO, SegmentAnything, InstantNGP, nvdiffrec, and T2M-GPT

- End-To-End
<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rossiyareich/marching-waifu-x/blob/main/ipynb/end2end_colab.ipynb)

## Related resources:
- [huggingface/diffusers](https://github.com/huggingface/diffusers)
    - [rossiyareich/aniflatmixAnimeFlatColorStyle_v20-fp16](https://huggingface.co/rossiyareich/aniflatmixAnimeFlatColorStyle_v20-fp16)
    - [rossiyareich/ClearVAE-diffusers](https://huggingface.co/rossiyareich/ClearVAE-diffusers)
    - [gsdf/EasyNegative](https://huggingface.co/datasets/gsdf/EasyNegative)
    - [AsciiP/badhandv4](https://huggingface.co/AsciiP/badhandv4)
    - [veryBadImageNegative](https://civitai.com/models/11772)
    - [Pop Up Parade](https://civitai.com/models/78997)
- [YBYBZhang/ControlVideo](https://github.com/YBYBZhang/ControlVideo)
- [rossiyareich/ldm-ckpt-conversion](https://github.com/rossiyareich/ldm-ckpt-conversion.git)
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [NVlabs/instant-ngp](https://github.com/NVlabs/instant-ngp)
- [NVlabs/nvdiffrec](https://github.com/NVlabs/nvdiffrec)
- [Mael-zys/T2M-GPT](https://github.com/Mael-zys/T2M-GPT.git)
- [softcat477/SMPL-to-FBX](https://github.com/softcat477/SMPL-to-FBX)
- [maximeraafat/BlenderNeRF](https://github.com/maximeraafat/BlenderNeRF)
- [genshin-style-anime-female-base-mesh-for-blender](https://sketchfab.com/3d-models/genshin-style-anime-female-base-mesh-for-blender-c2d6727e8c9742feb9a4a3bccac6e0e0)
- [OpenPoseBones_v9](https://toyxyz.gumroad.com/l/ciojz)

## To-do:
- [x] ControlVideo + RealESRGAN
- [x] 2nd-order MultistepDPM++ Karras
- [x] MultiControlNet
- [x] Multiprompt
- [x] Separate notebooks
- [ ] Increase pipeline resolution back to 768x1024
- [ ] Fix OpenPose conditioning images
- [ ] Switch to ClearVAE
- [ ] LoRA loading (load LoRA before inflating the UNet)
- [ ] Rewrite first stage (t2i) 
    - [ ] Use sliding (P,C,F)-Attn instead of CF-Attn + SC-Attn
- [ ] Rewrite second stage (t2i)
    - [ ] Apply LoRA
    - [ ] Use sliding (P,C,F)-Attn instead of CF-Attn + SC-Attn
    - [ ] SoftEdge HED ControlNet from first stage
    - [ ] Upscale by 2x
- [ ] Rewrite third stage (i2i)
    - [ ] Apply LoRA
    - [ ] No special attention mechanism (standard diffusers pipeline)
    - [ ] Upscale by 4x
- [ ] Fix GroundingDINO + SegmentAnything
- [ ] Fix InstantNGP prior
- [ ] Add nvdiffrec
- [ ] Add T2M-GPT + SMPL-to-FBX
- [ ] bpy processing + rigging + retargeting
- [ ] Evaluation against stable-dreamfusion with DeepDanbooru
- [ ] Add Gradio UI
- [ ] Medium article