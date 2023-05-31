import gc
import sys

sys.path.append("..")

import torch

from src.pipelines.convert_from_ckpt import *
from src.utils.file_loader import *

path = {"config_file": "ldm.json"}

if __name__ == "__main__":
    fl = file_loader()
    configs = fl.load_json(path["config_file"])

    for config in configs:
        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path=config["checkpoint_path"],
            original_config_file=config["original_config_file"],
            prediction_type="epsilon",
            extract_ema=config["extract_ema"],
            scheduler_type="dpm-karras",
            device=config["device"],
            from_safetensors=config["from_safetensors"],
        )

        if config["half"]:
            pipe.to(torch_dtype=torch.float16)

        pipe.save_pretrained(
            config["dump_path"], safe_serialization=config["to_safetensors"]
        )

        del pipe
        gc.collect()
        torch.cuda.empty_cache()
