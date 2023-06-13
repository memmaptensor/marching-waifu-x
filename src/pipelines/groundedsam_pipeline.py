import GroundingDINO.groundingdino.datasets.transforms as T
import numpy as np
import PIL.Image
import torch
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.inference import predict
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from segment_anything import SamPredictor, build_sam


class groundedsam_pipeline:
    @torch.no_grad()
    def __init__(self, save_path, device="cuda"):
        self.device = device
        self.groundingdino_model = self._load_model_hf(
            "ShilongLiu/GroundingDINO",
            "groundingdino_swinb_cogcoor.pth",
            "GroundingDINO_SwinB.cfg.py",
            save_path,
        )
        sam_checkpoint = hf_hub_download(
            "camenduru/ovseg",
            "sam_vit_h_4b8939.pth",
            cache_dir=save_path,
            local_dir_use_symlinks=True,
        )
        self.sam_predictor = SamPredictor(build_sam(sam_checkpoint).to(device))

    def _load_model_hf(self, repo_id, filename, ckpt_config_filename, save_path):
        cache_config_file = hf_hub_download(
            repo_id=repo_id,
            filename=ckpt_config_filename,
            cache_dir=save_path,
            local_dir_use_symlinks=True,
        )

        args = SLConfig.fromfile(cache_config_file)
        args.device = self.device
        model = build_model(args)

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=self.device)
        log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model

    def _load_image(self, image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = image_pil.convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed

    def _detect(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25):
        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        return boxes

    def _segment(self, image, boxes):
        self.sam_predictor.set_image(image)
        H, W, _ = image.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy.to(self.device), image.shape[:2]
        )
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks.cpu()

    @torch.no_grad()
    def __call__(self, image_pil, prompt, box_threshold, text_threshold):
        torch.cuda.empty_cache()

        image_source, image = self._load_image(image_pil)
        detected_boxes = self._detect(image, prompt, box_threshold, text_threshold)
        segmented_frame_masks = self._segment(image_source, detected_boxes)

        mask = segmented_frame_masks[0][0].cpu().numpy()
        mask_pil = PIL.Image.fromarray(mask)

        return mask_pil
