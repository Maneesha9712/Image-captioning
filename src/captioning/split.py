"Apply the official Flickr8k split(Hodosh et al., 2013) as used by Xu et al. (2015)."
"All five captions of an image stay in the same set"

import sys
from pathlib import Path

import pandas as pd
import yaml

SPLITS = ("train", "val", "test")

def read_split_file(path):
    "Return the image file namesin one split file, skipping empty lines."
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def assign_splits(captions, split_ids):
    """Add a 'split' column and drop images that are in no split list.

    Raises an error if any image is listed in more than one split.
    Returns the labelled table and the number of dropped images.
    """
    image_to_split = {}
    for split, names in split_ids.items():
        for name in names:
            if name in image_to_split:
                raise ValueError(f"{name} is in both {image_to_split[name]} and {split}")
            image_to_split[name] = split

    labelled = captions.assign(split=captions["image"].map(image_to_split))
    dropped = labelled.loc[labelled["split"].isna(), "image"].nunique()
    return labelled.dropna(subset=["split"]).reset_index(drop=True), dropped


def main(config_path="config.yaml"):
    with open(config_path, encoding="utf-8") as f:
        paths = yaml.safe_load(f)["paths"]

    split_ids = {s: read_split_file(paths[f"split_{s}"]) for s in SPLITS}
    captions = pd.read_csv(paths["raw_captions"])  # keeps quoted captions with commas intact
    labelled, dropped = assign_splits(captions, split_ids)

    out_path = Path(paths["processed"]) / "captions_split.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(out_path, index=False)

    print("Caption rows read:", len(captions))
    print("Images in no split list (dropped):", dropped)
    print(
        labelled.groupby("split")
        .agg(images=("image", "nunique"), captions=("image", "size"))
        .reindex(SPLITS)
    )
    print("Saved to", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:]) 
    