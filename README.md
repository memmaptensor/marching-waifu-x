# marching-waifu-x

<p>
    <a href="https://github.com/rossiyareich/marching-waifu-x/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/rossiyareich/marching-waifu-x">
    </a>
</p>

Complete 3D character + animation generation based on ControlVideo, Grounding DINO, Segment Anything, InstantNGP, and T2M-GPT

- End-To-End
<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rossiyareich/marching-waifu-x/blob/main/ipynb/end_to_end.ipynb)

## Related resources:
- [huggingface/diffusers](https://github.com/huggingface/diffusers) models
    - [rossiyareich/Nabylon-v1.0-fp16](https://huggingface.co/rossiyareich/Nabylon-v1.0-fp16)

## To-do:
- [x] ControlVideo + RealESRGAN pipeline
- [x] 1st-order MultistepDPM++ Karras
- [x] 2nd-order MultistepDPM++ Karras
- [x] New generation scheme + format
- [x] MultiControlNet
- [x] Multiprompt
- [x] Upscaled preview video
- [ ] Grounding DINO + Segment Anything
- [ ] Fix blender base model
- [ ] Update inference parameters
- [ ] InstantNGP
- [ ] T2M-GPT + SMPL-to-FBX
- [ ] bpy processing + rigging + retargeting
- [ ] Evaluation
- [ ] Deployment on Replicate
- [ ] Medium article
