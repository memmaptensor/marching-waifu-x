import argparse
import glob
import json

import PIL.Image
from sentence_transformers import SentenceTransformer, util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings_path",
        type=str,
        required=True,
        help="Path to file containing inference settings",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to directory containing images",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    with open(args.settings_path, "r") as f:
        conf = json.load(f)

    # Load CLIP model
    model = SentenceTransformer(
        conf["evaluation"]["clip"], cache_folder=conf["paths"]["checkpoints_path"]
    )

    # Encode text descriptions
    text_emb = model.encode([conf["evaluation"]["text"]])

    # Calculate R-Precision
    r_precision = []
    for i, filepath in enumerate(sorted(glob.glob(args.path))):
        img_emb = model.encode(PIL.Image.open(filepath))
        cos_scores = util.cos_sim(img_emb, text_emb)
        r_precision.append(cos_scores[0][0].cpu().numpy())

    final_r_precision = 0
    for r in r_precision:
        final_r_precision += r
        print(f"CLIP R-Precision: {i}, {r}")
    final_r_precision /= len(r_precision)
    print(f"Final CLIP R-Precision: {r_precision}")
