import csv
import json
import time
from pathlib import Path

import nltk
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu

from captioning.clean import load_config
from captioning.vocab import tokenize

MALE = {"man", "men", "boy", "boys", "he", "his", "him", "guy", "guys",
        "gentleman", "father", "son", "husband"}
FEMALE = {"woman", "women", "girl", "girls", "she", "her", "lady", "ladies",
          "mother", "daughter", "wife"}
WATCH = ["man", "dog"]          # the M3 shortcut words

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def load_references(cfg, split="val"):
    """image name -> list of its 5 tokenized human captions."""
    df = pd.read_csv(Path(cfg["paths"]["processed"]) / "captions_split.csv")
    df = df[df["split"] == split]
    return {image: [tokenize(c) for c in group["caption"]]
            for image, group in df.groupby("image", sort=True)}


def nouns(tokens):
    return {w for w, tag in nltk.pos_tag(tokens) if tag.startswith("NN")} if tokens else set()


def evaluate(predictions, references):
    """predictions: image name -> one caption (string). Returns a dict of scores."""
    images = sorted(references)
    missing = [i for i in images if i not in predictions]
    if missing:
        raise ValueError(f"{len(missing)} images have no prediction, e.g. {missing[0]}")

    hyps = [tokenize(predictions[i]) for i in images]
    refs = [references[i] for i in images]

    scores = {}
    for n in range(1, 5):
        scores[f"bleu{n}"] = corpus_bleu(refs, hyps, weights=tuple([1 / n] * n)) * 100

    generated_nouns = hallucinated = wrong_gender = 0
    false_word = {w: 0 for w in WATCH}
    for hyp, ref in zip(hyps, refs):
        ref_words = {w for caption in ref for w in caption}
        hyp_nouns = nouns(hyp)
        generated_nouns += len(hyp_nouns)
        hallucinated += len(hyp_nouns - ref_words)
        for w in WATCH:
            if w in hyp and w not in ref_words:
                false_word[w] += 1
        if (MALE & set(hyp) and not MALE & ref_words) or \
           (FEMALE & set(hyp) and not FEMALE & ref_words):
            wrong_gender += 1

    n_images = len(images)
    scores["hallucination_rate"] = 100 * hallucinated / max(generated_nouns, 1)
    for w in WATCH:
        scores[f"false_{w}"] = 100 * false_word[w] / n_images
    scores["wrong_gender"] = 100 * wrong_gender / n_images
    scores["mean_length"] = sum(len(h) for h in hyps) / n_images
    scores["distinct_captions"] = len({" ".join(h) for h in hyps})
    return scores


def save_run(run_name, settings, scores, results_dir="results"):
    """Save the run's settings in configs/ and add one row to results/results.csv."""
    Path("configs").mkdir(exist_ok=True)
    Path(results_dir).mkdir(exist_ok=True)
    Path("configs", f"{run_name}.json").write_text(json.dumps(settings, indent=2))
    row = {"run": run_name, "time": time.strftime("%Y-%m-%d %H:%M"),
           **{k: round(v, 3) if isinstance(v, float) else v for k, v in scores.items()},
           "settings": json.dumps(settings)}
    path = Path(results_dir) / "results.csv"
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    """Human reference score: caption 1 of each image judged against the other 4."""
    cfg = load_config()
    refs = load_references(cfg, "val")
    predictions = {img: " ".join(caps[0]) for img, caps in refs.items()}
    others = {img: caps[1:] for img, caps in refs.items()}
    for name, value in evaluate(predictions, others).items():
        print(f"{name:20s} {value:.2f}")


if __name__ == "__main__":
    main()