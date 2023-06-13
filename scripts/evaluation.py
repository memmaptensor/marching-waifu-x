import argparse
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

    # Encode an image
    img_emb = model.encode(PIL.Image.open(conf["paths"]["image"]))

    # Encode text descriptions
    text_emb = model.encode([f"{args.text}"])

    # Compute cosine similarities
    cos_scores = util.cos_sim(img_emb, text_emb)
    print("The final CLIP R-Precision is:", cos_scores[0][0].cpu().numpy())
