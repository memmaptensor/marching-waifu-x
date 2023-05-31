import os

import PIL.Image
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

from src.utils.image_wrapper import *


class realesrgan_pipeline:
    @torch.no_grad()
    def __init__(
        self,
        outscale=4.0,
        tile=192,
        tile_pad=16,
        pre_pad=16,
        face_enhance=True,
        fp32=False,
        gpu_id=0,
    ):
        self.outscale = outscale

        # x4 RRDBNet model with 6 blocks
        model_name = "RealESRGAN_x4plus_anime_6B"
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ]

        # Determine model paths
        model_path = os.path.join("weights", model_name + ".pth")
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                model_path = load_file_from_url(
                    url=url,
                    model_dir=os.path.join(ROOT_DIR, "weights"),
                    progress=True,
                    file_name=None,
                )

        # Restorer
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32,
            gpu_id=gpu_id,
        )

        # Use GFPGAN for face enhancement
        self.face_enhancer = None
        if face_enhance:
            from gfpgan import GFPGANer

            self.face_enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.upsampler,
            )

    @torch.no_grad()
    def __call__(self, img):
        img = image_wrapper(PIL.Image.open(img), "pil").to_cv2()

        if self.face_enhancer is not None:
            _, _, output = self.face_enhancer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            output, _ = self.upsampler.enhance(img, outscale=self.outscale)

        return image_wrapper(output, "cv2").to_pil()
