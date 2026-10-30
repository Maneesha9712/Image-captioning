import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from captioning.clean import load_config
from captioning.vocab import encode

AUTOTUNE = tf.data.AUTOTUNE

# Each pretrained encoder expects its own ImageNet preprocessing.
PREPROCESS = {
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "mobilenet_v3": lambda x: x,   # Keras MobileNetV3 rescales inside the model
}


def load_vocab(cfg):
    path = Path(cfg["paths"]["processed"]) / "vocab.json"
    return json.loads(path.read_text(encoding="utf-8"))


def encode_split(df, vocab, max_len):
    """Group the 5 captions of each image as padded token IDs."""
    seq_len = max_len + 2                              # <start> + words + <end>
    names, captions = [], []
    for image, group in df.groupby("image", sort=True):
        ids = np.zeros((len(group), seq_len), dtype=np.int32)   # 0 = <pad>
        for row, caption in enumerate(group["caption"]):
            encoded = encode(caption, vocab, max_len)
            ids[row, :len(encoded)] = encoded
        names.append(image)
        captions.append(ids)
    return names, np.stack(captions)                   # (images, 5, seq_len)


def load_image(path, image_size):
    """Read a JPEG and squash it to image_size x image_size (kept as uint8)."""
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    return tf.cast(tf.round(image), tf.uint8)


def make_dataset(split, cfg, encoder="resnet50", shuffle=None):
    """Return a batched tf.data.Dataset of ((image, caption_in), caption_out)."""
    df = pd.read_csv(Path(cfg["paths"]["processed"]) / "captions_split.csv")
    df = df[df["split"] == split]
    vocab = load_vocab(cfg)
    names, captions = encode_split(df, vocab, cfg["vocab"]["max_caption_length"])
    paths = [str(Path(cfg["paths"]["raw_images"]) / n) for n in names]

    size = cfg["image_size"]
    seed = cfg["seed"]
    preprocess = PREPROCESS[encoder]
    if shuffle is None:
        shuffle = split == "train"

    ds = tf.data.Dataset.from_tensor_slices((paths, captions))
    ds = ds.map(lambda p, c: (load_image(p, size), c), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()                                    # decode each image only once
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    # One example per caption: each image is paired with each of its 5 captions.
    ds = ds.flat_map(lambda img, caps: tf.data.Dataset.from_tensor_slices(caps)
                     .map(lambda cap: (img, cap)))
    if shuffle:
        ds = ds.shuffle(1000, seed=seed, reshuffle_each_iteration=True)

    # Same preprocessing for every split. Teacher forcing: input = all but last
    # token, target = all but first token.
    ds = ds.map(lambda img, cap: ((preprocess(tf.cast(img, tf.float32)), cap[:-1]),
                                  cap[1:]),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg["batch_size"])
    return ds.prefetch(AUTOTUNE)


def main():
    cfg = load_config()
    tf.random.set_seed(cfg["seed"])
    ds = make_dataset("train", cfg)
    (images, caption_in), caption_out = next(iter(ds))
    print("images     ", images.shape, images.dtype)
    print("caption_in ", caption_in.shape, caption_in.dtype)
    print("caption_out", caption_out.shape, caption_out.dtype)
    print(f"image value range: {float(tf.reduce_min(images)):.1f} "
          f"to {float(tf.reduce_max(images)):.1f}")


if __name__ == "__main__":
    main()