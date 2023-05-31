import glob
import json
import os
import pathlib

import PIL.Image


class file_loader:
    def load_json(self, filepath):
        with open(filepath, "r") as f:
            j = json.load(f)

        return j

    def load_text(self, filepath):
        with open(filepath, "r") as f:
            t = f.read()

        return t

    def load_controlnet_conditions(self, folderpath):
        prefixes = [
            "openpose_full",
            "depth",
            "normals",
            "lineart",
        ]

        num_controlnet_conditions = len(glob.glob(os.path.join(folderpath, "*"))) // 4
        controlnet_conditions = [[None] * 4 for _ in range(num_controlnet_conditions)]

        for j, prefix in enumerate(prefixes):
            controlnet_files = sorted(glob.glob(os.path.join(folderpath, f"{prefix}*")))
            for i, filepath in enumerate(controlnet_files):
                pl = pathlib.Path(filepath)
                if pl.stem[:-4] in prefixes:
                    controlnet_conditions[i][j] = PIL.Image.open(filepath).convert(
                        "RGB"
                    )

        return controlnet_conditions
