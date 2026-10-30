# Lightweight On-Device Image Captioning for Screen-Reader Users

IE7615 Deep Learning for AI, Fall 2026, Northeastern University Vancouver
Author: Maneesha Devindi

## Problem and intended user

Blind and low-vision people who use screen readers need short, accurate descriptions of photos.
This project measures how small and fast an image-captioning model can become while its captions
stay useful, so captioning can run **privately and offline on a phone**. Four model versions are
compared, from ResNet-50 + LSTM with attention down to a quantized MobileNetV3 model.

## Data

- **Dataset:** Flickr8k (Hodosh, Young and Hockenmaier, 2013): 8,091 photos, 5 English captions each,
  written by Amazon Mechanical Turk crowd workers.
- **Images and captions:** Kaggle, `adityajn105/flickr8k`, https://www.kaggle.com/datasets/adityajn105/flickr8k
- **Official split lists** (not included in the Kaggle copy):
  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
- **Downloaded:** 29 October 2026
- **License:** the Kaggle copy is listed as CC0 (Public Domain). The original Flickr8k dataset was
  released for research use, and the photos belong to their Flickr owners. Data is used for research
  only and is **not** committed to this repository.

### How to obtain the data

1. Download the Kaggle dataset and extract `Images/` and `captions.txt` into `data/raw/`.
2. Download `Flickr8k_text.zip` and extract its contents into `data/raw/`.

You should then have:

```
data/raw/Images/                      (8,091 .jpg files)
data/raw/captions.txt
data/raw/Flickr_8k.trainImages.txt
data/raw/Flickr_8k.devImages.txt
data/raw/Flickr_8k.testImages.txt
```

`data/raw/` is treated as read-only. All outputs go to `data/processed/`.

## Installation

Python 3.13 (Kaggle uses 3.13.15).

```
git clone https://github.com/Maneesha9712/Image-captioning.git
cd Image-captioning
pip install -r requirements.txt
pip install -e .
```

On Windows, use `py -m pip` and `py -m ...` instead of `pip` and `python -m ...`.

## Reproducing the results

Run from the repository folder, in this order:

| Command | Produces | Report section |
|---|---|---|
| `python -m captioning.split` | `data/processed/captions_split.csv`; split counts | 3 |
| `python -m pytest` | 4 tests: no image in two sets | 3 |
| `python -m captioning.clean` | cleaning counts (corrupt files, duplicates, sizes, captions) | 2 |
| `python -m captioning.audit` | 60-image audit sheet; results in `reports/label_audit.csv` | 2 |
| `python -m captioning.vocab` | `data/processed/vocab.json`; % unknown and % truncated | 5 |
| `python -m captioning.pipeline` | one batch: shapes and value range | 6 |
| `notebooks/01_eda.ipynb` | Figures 1 to 6 in `reports/figures/` | 4 |
| `notebooks/02_smoke_test.ipynb` | Figures 7 to 9, smoke test, timing | 6 |

All settings (paths, seed 42, image size 224, batch size 64, vocabulary settings) are in `config.yaml`.

## Running on Kaggle

1. Create a notebook, set **Accelerator** to a GPU, and turn **Internet** on.
2. Add the input dataset `adityajn105/flickr8k`.
3. Clone this repository, download `Flickr8k_text.zip` into `/kaggle/working/`, and update the
   `paths` section of `config.yaml` to the Kaggle folders
   (images: `/kaggle/input/datasets/adityajn105/flickr8k/Images`; processed: `/kaggle/working/processed`).
4. Run commands with `PYTHONPATH=src`, for example `!PYTHONPATH=src python -m captioning.split`.

Measured on a Kaggle T4 GPU: one pass of the training data takes 36.3 s from disk, then 18.5 s from the memory cache.

## Repository structure

```
app/                  microproduct (later milestones)
config.yaml           all paths and settings
data/                 not committed (see "How to obtain the data")
notebooks/            01_eda.ipynb, 02_smoke_test.ipynb
reports/figures/      figures used in the report
reports/label_audit.csv
src/captioning/       installable package
    split.py          applies the official split
    clean.py          cleaning checks
    audit.py          label-audit sample
    vocab.py          vocabulary from training captions only
    pipeline.py       tf.data input pipeline
    stats.py          not used at M3 (encoders use ImageNet preprocessing)
tests/test_split.py   fails if any image appears in more than one set
```

## Generative AI use

Parts of the code and documentation were drafted with Claude Opus 5.5 (Anthropic), with TA approval.
Affected files are marked "AI-assisted" in their docstrings. Details are in the report's disclosure note.