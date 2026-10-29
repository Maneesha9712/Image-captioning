import hashlib
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import yaml
from PIL import Image


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def check_captions(df, short_words=3):
    """Count images without 5 captions, and empty or very short captions."""
    counts = df.groupby("image").size()
    not_five = counts[counts != 5]

    captions = df["caption"].fillna("").str.strip()
    n_words = captions.str.split().str.len()
    empty = n_words == 0
    short = (n_words > 0) & (n_words < short_words)

    print("--- Caption checks ---")
    print(f"Images without exactly 5 captions: {len(not_five)}")
    print(f"Empty captions: {empty.sum()}")
    print(f"Very short captions (fewer than {short_words} words): {short.sum()}")
    if short.sum() > 0:
        print(df.loc[short, ["image", "caption"]].head(20).to_string())


def check_images(df, image_dir):
    """Check every image opens, record its size and colour mode,
    and find exact duplicate files using an MD5 fingerprint."""
    names = sorted(df["image"].unique())
    unreadable = []
    sizes = Counter()
    modes = Counter()
    fingerprints = {}

    print("\n--- Image checks ---")
    for i, name in enumerate(names, start=1):
        path = Path(image_dir) / name
        try:
            data = path.read_bytes()
            with Image.open(path) as img:
                img.verify()          # detects corrupted files
            with Image.open(path) as img:
                img.load()            # fully decodes the pixels
                sizes[img.size] += 1
                modes[img.mode] += 1
        except Exception as error:
            unreadable.append((name, str(error)))
            continue
        fingerprint = hashlib.md5(data).hexdigest()
        fingerprints.setdefault(fingerprint, []).append(name)
        if i % 1000 == 0:
            print(f"  checked {i} / {len(names)} images...")

    print(f"Images checked: {len(names)}")
    print(f"Unreadable or corrupted images: {len(unreadable)}")
    for name, error in unreadable:
        print(f"  {name}: {error}")

    print(f"Colour modes: {dict(modes)}")

    widths = [w for (w, h) in sizes.elements()]
    heights = [h for (w, h) in sizes.elements()]
    landscape = sum(n for (w, h), n in sizes.items() if w > h)
    portrait = sum(n for (w, h), n in sizes.items() if h > w)
    square = sum(n for (w, h), n in sizes.items() if w == h)
    print(f"Width range: {min(widths)} to {max(widths)} pixels")
    print(f"Height range: {min(heights)} to {max(heights)} pixels")
    print(f"Distinct image sizes: {len(sizes)}")
    print(f"Most common sizes: {sizes.most_common(5)}")
    print(f"Landscape: {landscape}, Portrait: {portrait}, Square: {square}")

    duplicates = [group for group in fingerprints.values() if len(group) > 1]
    print(f"Exact duplicate groups: {len(duplicates)}")
    for group in duplicates:
        print(f"  {group}")


def check_near_duplicates(df, image_dir, max_distance=5):
    """Find visually similar image pairs using a perceptual hash (phash).
    A pair is flagged if their 64-bit hashes differ in max_distance bits or fewer."""
    image_split = df.drop_duplicates("image").set_index("image")["split"]
    names = list(image_split.index)

    print("\n--- Near-duplicate checks ---")
    hashes = []
    for i, name in enumerate(names, start=1):
        with Image.open(Path(image_dir) / name) as img:
            hashes.append(imagehash.phash(img).hash.flatten())
        if i % 1000 == 0:
            print(f"  hashed {i} / {len(names)} images...")
    hashes = np.array(hashes)  # shape (number of images, 64), True/False values

    pairs = []
    for i in range(len(names)):
        distances = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        for offset in np.nonzero(distances <= max_distance)[0]:
            j = i + 1 + offset
            pairs.append((names[i], names[j], int(distances[offset]),
                          image_split[names[i]], image_split[names[j]]))

    crossing = [p for p in pairs if p[3] != p[4]]
    print(f"Near-duplicate pairs (distance <= {max_distance}): {len(pairs)}")
    print(f"  in the same split: {len(pairs) - len(crossing)}")
    print(f"  crossing splits: {len(crossing)}")
    for a, b, distance, split_a, split_b in pairs:
        print(f"  {a} ({split_a})  {b} ({split_b})  distance={distance}")


def main():
    cfg = load_config()
    csv_path = Path(cfg["paths"]["processed"]) / "captions_split.csv"
    df = pd.read_csv(csv_path)
    print(f"Caption rows loaded: {len(df)}")
    check_captions(df)
    check_images(df, cfg["paths"]["raw_images"])
    check_near_duplicates(df, cfg["paths"]["raw_images"])


if __name__ == "__main__":
    main()