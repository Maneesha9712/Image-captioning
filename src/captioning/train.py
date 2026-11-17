"""Train and evaluate the simple deep baseline with one command (M4 Sections 4-5)."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from captioning.clean import load_config
from captioning.evaluate import evaluate, load_references, save_run
from captioning.models.simple_deep import build_model, greedy_captions
from captioning.pipeline import load_vocab, make_dataset, make_image_dataset


class EpochTimer(keras.callbacks.Callback):
    """Record how long each epoch takes."""
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self.start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize", choices=["squash", "pad"], default="squash")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true", help="tiny local test; saves nothing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed                     # seed also controls the shuffle order
    keras.utils.set_random_seed(args.seed)      # Python, NumPy and TensorFlow seeds
    run = f"simple_deep_{args.resize}_d{args.dropout}_s{args.seed}"

    vocab = load_vocab(cfg)
    id_to_word = {i: w for w, i in vocab.items()}
    seq_len = cfg["vocab"]["max_caption_length"] + 1          # 23

    train_ds = make_dataset("train", cfg, encoder="raw", resize_mode=args.resize)
    val_ds = make_dataset("val", cfg, encoder="raw", resize_mode=args.resize)
    epochs = args.epochs
    if args.quick:
        train_ds, val_ds, epochs = train_ds.take(20), val_ds.take(5), 1

    model = build_model(len(vocab), seq_len, cfg["image_size"], dropout=args.dropout)
    model.compile(optimizer=keras.optimizers.Adam(args.lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                  ignore_class=0))

    checkpoint_dir = Path("checkpoints") / run
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timer = EpochTimer()
    callbacks = [
        keras.callbacks.BackupAndRestore(str(checkpoint_dir / "backup")),
        keras.callbacks.ModelCheckpoint(str(checkpoint_dir / "best.keras"),
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                      restore_best_weights=True),
        keras.callbacks.TensorBoard(log_dir=str(Path("logs") / run)),
        timer,
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=callbacks, verbose=2)

    curves = pd.DataFrame(history.history)
    curves["epoch"] = range(1, len(curves) + 1)
    curves["seconds"] = timer.times[-len(curves):]
    best_epoch = int(curves["val_loss"].idxmin()) + 1

    print("Generating validation captions...")
    names, image_ds = make_image_dataset("val", cfg, encoder="raw", resize_mode=args.resize)
    captions = greedy_captions(model, image_ds, id_to_word, seq_len)
    scores = evaluate(dict(zip(names, captions)), load_references(cfg, "val"))
    scores["best_epoch"] = best_epoch
    scores["epochs_run"] = len(curves)
    later_epochs = timer.times[1:] or timer.times                 # epoch 1 also fills the cache
    scores["seconds_per_epoch"] = float(np.median(later_epochs))

    print("Validation scores:")
    for name, value in scores.items():
        print(f"  {name:20s} {value:.2f}")
    print("Example captions:")
    for name, caption in list(zip(names, captions))[:5]:
        print(f"  {name}: {caption}")

    if args.quick:
        print("Quick test finished (nothing saved).")
        return

    Path("results").mkdir(exist_ok=True)
    curves.to_csv(f"results/history_{run}.csv", index=False)
    pd.DataFrame({"image": names, "caption": captions}).to_csv(
        f"results/predictions_{run}.csv", index=False)
    settings = {**vars(args), "run": run, "model": "simple_deep (3-layer CNN + LSTM-256)",
                "parameters": model.count_params(), "batch_size": cfg["batch_size"],
                "optimizer": "Adam", "loss": "sparse categorical cross-entropy, <pad> ignored",
                "early_stopping": "val_loss, restore best weights"}
    save_run(run, settings, scores)
    print(f"Saved results for {run}")


if __name__ == "__main__":
    main()