"""Non-deep baselines (M4).
  py -m captioning.models.baseline --model constant
"""
import argparse
import random

from nltk.translate.bleu_score import corpus_bleu

from captioning.clean import load_config
from captioning.evaluate import evaluate, load_references, save_run


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["constant"], required=True)
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
    print("Validation scores:")
    for name, value in scores.items():
        print(f"  {name:20s} {value:.2f}")
    save_run(args.model, settings, scores)
    print("Saved to configs/ and results/results.csv")


if __name__ == "__main__":
    main()