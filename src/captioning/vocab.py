import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from captioning.clean import load_config

SPECIAL = ["<pad>", "<start>", "<end>", "<unk>"]   # <pad> gets ID 0


def tokenize(caption):
    """Lowercase and keep only words made of letters (same as the EDA)."""
    return re.findall(r"[a-z]+", str(caption).lower())


def build_vocab(captions, min_count):
    counts = Counter(w for c in captions for w in tokenize(c))
    words = sorted(w for w, n in counts.items() if n >= min_count)
    return {w: i for i, w in enumerate(SPECIAL + words)}


def encode(caption, vocab, max_len):
    """Caption -> list of IDs: <start> + words (cut to max_len) + <end>."""
    unk = vocab["<unk>"]
    words = tokenize(caption)[:max_len]
    return [vocab["<start>"]] + [vocab.get(w, unk) for w in words] + [vocab["<end>"]]


def coverage(captions, vocab, max_len, name):
    tokens = captions.apply(tokenize)
    total = tokens.str.len().sum()
    unknown = sum(1 for t in tokens for w in t if w not in vocab)
    truncated = (tokens.str.len() > max_len).mean() * 100
    print(f"{name}: {unknown / total * 100:.2f}% of tokens become <unk>, "
          f"{truncated:.2f}% of captions truncated")


def main():
    cfg = load_config()
    df = pd.read_csv(Path(cfg["paths"]["processed"]) / "captions_split.csv")
    train = df[df["split"] == "train"]

    vocab = build_vocab(train["caption"], cfg["vocab"]["min_word_count"])
    max_len = cfg["vocab"]["max_caption_length"]

    out = Path(cfg["paths"]["processed"]) / "vocab.json"
    out.write_text(json.dumps(vocab, indent=1), encoding="utf-8")

    print(f"Vocabulary size: {len(vocab)} (including {len(SPECIAL)} special tokens)")
    for split in ["train", "val"]:
        coverage(df.loc[df["split"] == split, "caption"], vocab, max_len, split)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()