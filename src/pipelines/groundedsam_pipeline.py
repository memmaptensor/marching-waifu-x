import cv2
import GroundingDINO.groundingdino.datasets.transforms as T
import torch
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from huggingface_hub import hf_hub_download
from PIL import Image
from segment_anything import SamPredictor, build_sam

from src.utils.image_wrapper import *


class groundedsam_pipeline:
    @torch.no_grad()
    def __init__(self, cache_dir, device):
        # Cache model weights
        grounding_dino_conf_checkpoint = hf_hub_download(
            "ShilongLiu/GroundingDINO",
            "GroundingDINO_SwinB.cfg.py",
            cache_dir=cache_dir,
            local_dir_use_symlinks=True,
        )
        grounding_dino_checkpoint = hf_hub_download(
            "ShilongLiu/GroundingDINO",
            "groundingdino_swinb_cogcoor.pth",
            cache_dir=cache_dir,
            local_dir_use_symlinks=True,
        )
        sam_checkpoint = hf_hub_download(
            "camenduru/ovseg",
            "sam_vit_h_4b8939.pth",
            cache_dir=cache_dir,
            local_dir_use_symlinks=True,
        )

        # Load models
        self.device = device
        self.grounding = self._load_model(
            grounding_dino_conf_checkpoint, grounding_dino_checkpoint
        )
        self.sam = SamPredictor(build_sam(sam_checkpoint).to(device))

    def _load_image(self, image):
        img_pil = image.convert("RGB")

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img, _ = transform(img_pil, None)  # 3, h, w

        return img_pil, img

    def _load_model(self, model_config_path, model_checkpoint_path):
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        print(load_res)
        _ = model.eval()

        return model

    def _get_grounding_output(
        self, image, caption, box_threshold, text_threshold, with_logits=True
    ):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        grounding = self.grounding.to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = grounding(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # Filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # Get phrase
        tokenlizer = grounding.tokenizer
        tokenized = tokenlizer(caption)

        # Build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer
            )
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    @torch.no_grad()
    def __call__(
        self, image, det_prompt, box_threshold, text_threshold, merge_masks=True
    ):
        torch.cuda.empty_cache()

        img_pil, img = self._load_image(image)
        boxes_filt, pred_phrases = self._get_grounding_output(
            img,
            det_prompt,
            box_threshold,
            text_threshold,
        )

        img_cv2 = image_wrapper(img_pil).to_cv2()
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        self.sam.set_image(img_cv2)

        size = img_pil.size()
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam.transform.apply_boxes_torch(
            boxes_filt, img.shape[:2]
        ).to(self.device)

        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        if merge_masks:
            masks = torch.sum(masks, dim=0).unsqueeze(0)
            masks = torch.where(masks > 0, True, False)

        # Simply choose the first mask, which will be refined in the future release
        mask = masks[0][0].cpu().numpy()
        mask_pil = Image.fromarray(mask)

        return mask_pil
