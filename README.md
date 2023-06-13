# marching-waifu-x

<p>
    <a href="https://github.com/rossiyareich/marching-waifu-x/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/rossiyareich/marching-waifu-x">
    </a>
</p>

Complete 3D character generation based on ControlVideo, GroundingDINO, SegmentAnything, and InstantNGP

- End-To-End
<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rossiyareich/marching-waifu-x/blob/main/ipynb/end2end_colab.ipynb)

## Related resources:
- [huggingface/diffusers](https://github.com/huggingface/diffusers)
    - [rossiyareich/abyssorangemix3-popupparade-fp16](https://huggingface.co/rossiyareich/abyssorangemix3-popupparade-fp16)
    - [rossiyareich/anything-v4.0-vae](https://huggingface.co/rossiyareich/anything-v4.0-vae)
    - [gsdf/EasyNegative](https://huggingface.co/datasets/gsdf/EasyNegative)
    - [AsciiP/badhandv4](https://huggingface.co/AsciiP/badhandv4)
    - [veryBadImageNegative](https://civitai.com/models/11772)
- [YBYBZhang/ControlVideo](https://github.com/YBYBZhang/ControlVideo)
- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [NVLabs/instant-ngp](https://github.com/NVlabs/instant-ngp)
- [maximeraafat/BlenderNeRF](https://github.com/maximeraafat/BlenderNeRF)
- [genshin-style-anime-female-base-mesh-for-blender](https://sketchfab.com/3d-models/genshin-style-anime-female-base-mesh-for-blender-c2d6727e8c9742feb9a4a3bccac6e0e0)
- [OpenPoseBones_v9](https://toyxyz.gumroad.com/l/ciojz)

## To-do:
- [x] ControlVideo + RealESRGAN
- [x] 2nd-order MultistepDPM++ Karras
- [x] MultiControlNet
- [x] Multiprompt
- [x] Separate notebooks
- [x] Increase pipeline resolution back to 768x1024
- [x] Fix OpenPose conditioning images
- [x] Merge config files
- [x] Rewrite first stage (t2i) 
    - [x] Use sliding (P,C,F)-Attn instead of CF-Attn + SC-Attn
    - [x] New ControlNet modules
        - [x] OpenPose
        - [x] Pix2Pix
- [x] Fix GroundingDINO + SegmentAnything
- [x] Add InstantNGP
- [x] Evaluation with CLIP R-Precision