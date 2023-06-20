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

    # Output R-Precision for training set images
    training_set_glob = sorted(glob.glob(conf["paths"]["training_images"]))
    training_set_r_precision = 0
    for i, filepath in enumerate(training_set_glob):
        img_emb = model.encode(PIL.Image.open(filepath))
        cos_scores = util.cos_sim(img_emb, text_emb)
        r_precision = cos_scores[0][0].cpu().numpy()
        print(f"Training set CLIP R-Precision: {i}, {r_precision}")
        training_set_r_precision += r_precision
    training_set_r_precision /= len(training_set_glob)
    print(f"Training set final CLIP R-Precision: {training_set_r_precision}")

    # Output R-Precision for rendered images
    rendered_set_glob = sorted(glob.glob(conf["paths"]["rendered_images"]))
    rendered_set_r_precision = 0
    for i, filepath in enumerate(rendered_set_glob):
        img_emb = model.encode(PIL.Image.open(filepath))
        cos_scores = util.cos_sim(img_emb, text_emb)
        r_precision = cos_scores[0][0].cpu().numpy()
        print(f"Rendered set CLIP R-Precision: {i}, {r_precision}")
        rendered_set_r_precision += r_precision
    rendered_set_r_precision /= len(rendered_set_glob)
    print(f"Rendered set final CLIP R-Precision: {rendered_set_r_precision}")
