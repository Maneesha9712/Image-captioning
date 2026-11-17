"""Non-deep baselines (M4 Section 3).
Run:
  py -m captioning.models.baseline --model constant
  py -m captioning.models.baseline --model knn
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image

from captioning.clean import load_config
from captioning.evaluate import evaluate, load_references, save_run
from captioning.vocab import tokenize

FEATURES = ["hist", "tiny", "both"]
KS = [1, 5, 10, 25]


# ---------- Constant caption ----------

def constant_caption(cfg, n_candidates=300, n_images=500):
    """Pick ONE training caption with the best BLEU-4 on training images.
    No validation data is used to choose it."""
    rng = random.Random(cfg["seed"])
    train_refs = load_references(cfg, "train")
    images = rng.sample(sorted(train_refs), n_images)
    refs = [train_refs[i] for i in images]
    pool = [caption for captions in train_refs.values() for caption in captions]
    candidates = rng.sample(pool, n_candidates)

    best, best_score = None, -1.0
    for candidate in candidates:
        score = corpus_bleu(refs, [candidate] * len(refs))
        if score > best_score:
            best, best_score = candidate, score
    return " ".join(best), best_score * 100


# ---------- Nearest neighbours on hand-made features ----------

def image_features(path):
    """Colour histogram (8 x 8 x 8 RGB bins) and a 16 x 16 tiny image."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        small = np.asarray(img.resize((64, 64)), dtype=np.uint8)
        tiny = np.asarray(img.resize((16, 16)), dtype=np.float32).ravel() / 255.0
    bins = (small // 32).reshape(-1, 3).astype(np.int64)
    index = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    hist = np.bincount(index, minlength=512).astype(np.float32)
    return hist / hist.sum(), tiny


def load_features(cfg, split):
    """Compute features once per split and cache them in data/processed/."""
    cache = Path(cfg["paths"]["processed"]) / f"knn_features_{split}.npz"
    if cache.exists():
        data = np.load(cache)
        return list(data["names"]), data["hist"], data["tiny"]
    df = pd.read_csv(Path(cfg["paths"]["processed"]) / "captions_split.csv")
    names = sorted(df.loc[df["split"] == split, "image"].unique())
    image_dir = Path(cfg["paths"]["raw_images"])
    hists, tinys = zip(*(image_features(image_dir / n) for n in names))
    hist, tiny = np.stack(hists), np.stack(tinys)
    np.savez(cache, names=np.array(names), hist=hist, tiny=tiny)
    return names, hist, tiny


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def prepare(train, val, kind):
    """Scale features using TRAINING statistics only, then L2-normalise."""
    parts_train, parts_val = [], []
    if kind in ("hist", "both"):
        parts_train.append(l2(np.sqrt(train[0])))       # square root: Hellinger distance
        parts_val.append(l2(np.sqrt(val[0])))
    if kind in ("tiny", "both"):
        mean, std = train[1].mean(0), train[1].std(0) + 1e-6
        parts_train.append(l2((train[1] - mean) / std))
        parts_val.append(l2((val[1] - mean) / std))
    return l2(np.hstack(parts_train)), l2(np.hstack(parts_val))


def consensus(captions, vocab):
    """Return the caption sharing the most words with the other candidates."""
    if len(captions) == 1:
        return captions[0]
    counts = np.zeros((len(captions), len(vocab)), dtype=np.float32)
    for row, caption in enumerate(captions):
        for word in tokenize(caption):
            counts[row, vocab.get(word, vocab["<unk>"])] += 1
    counts = l2(counts)
    similarity = counts @ counts.T
    np.fill_diagonal(similarity, 0)
    return captions[int(similarity.sum(1).argmax())]


def knn_predict(train_x, val_x, train_names, val_names, train_captions, vocab, k):
    similarity = val_x @ train_x.T                     # cosine similarity
    top = np.argsort(-similarity, axis=1)[:, :k]
    predictions = {}
    for row, image in enumerate(val_names):
        candidates = [c for j in top[row] for c in train_captions[train_names[j]]]
        predictions[image] = consensus(candidates, vocab)
    return predictions


def run_knn(cfg, val_refs):
    vocab = json.loads((Path(cfg["paths"]["processed"]) / "vocab.json").read_text(encoding="utf-8"))
    df = pd.read_csv(Path(cfg["paths"]["processed"]) / "captions_split.csv")
    train_captions = {img: list(g["caption"]) for img, g in df[df["split"] == "train"].groupby("image")}

    print("Computing features (cached after the first run)...")
    train_names, train_hist, train_tiny = load_features(cfg, "train")
    val_names, val_hist, val_tiny = load_features(cfg, "val")

    tuning, best = [], None
    for kind in FEATURES:
        train_x, val_x = prepare((train_hist, train_tiny), (val_hist, val_tiny), kind)
        for k in KS:
            preds = knn_predict(train_x, val_x, train_names, val_names, train_captions, vocab, k)
            scores = evaluate(preds, val_refs)
            tuning.append({"features": kind, "k": k, **{n: round(v, 2) for n, v in scores.items()}})
            print(f"  features={kind:5s} k={k:2d}  BLEU-4 {scores['bleu4']:.2f}  "
                  f"hallucination {scores['hallucination_rate']:.1f}%")
            if best is None or scores["bleu4"] > best[2]["bleu4"]:
                best = (kind, k, scores, preds)

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(tuning).to_csv("results/knn_tuning.csv", index=False)
    kind, k, scores, preds = best
    print(f"Best on validation: features={kind}, k={k}")
    settings = {"model": "knn", "features": kind, "k": k, "distance": "cosine",
                "selection": "consensus", "tuned_over": {"features": FEATURES, "k": KS}}
    return settings, scores, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["constant", "knn"], required=True)
    args = parser.parse_args()

    cfg = load_config()
    val_refs = load_references(cfg, "val")

    if args.model == "constant":
        caption, train_score = constant_caption(cfg)
        print(f'Chosen caption: "{caption}" (training BLEU-4 {train_score:.2f})')
        predictions = {image: caption for image in val_refs}
        settings = {"model": "constant", "caption": caption,
                    "n_candidates": 300, "n_images": 500, "seed": cfg["seed"]}
        scores = evaluate(predictions, val_refs)
    else:
        settings, scores, predictions = run_knn(cfg, val_refs)
        pd.DataFrame(sorted(predictions.items()), columns=["image", "caption"]).to_csv(
            "results/knn_val_predictions.csv", index=False)

    print("Validation scores:")
    for name, value in scores.items():
        print(f"  {name:20s} {value:.2f}")
    save_run(args.model, settings, scores)
    print("Saved to configs/ and results/results.csv")


if __name__ == "__main__":
    main()