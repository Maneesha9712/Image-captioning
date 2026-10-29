
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from captioning.split import SPLITS, assign_splits

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = REPO_ROOT / "data" / "processed" / "captions_split.csv"
EXPECTED_IMAGES = {"train": 6000, "val": 1000, "test": 1000}


@pytest.fixture(scope="module")
def labelled():
    """Load the split-labelled captions, or skip if the split hasn't been run."""
    if not SPLIT_CSV.exists():
        pytest.skip("Split file not found. Run: py -m captioning.split")
    return pd.read_csv(SPLIT_CSV)


def test_no_image_in_more_than_one_split(labelled):
    image_sets = {s: set(labelled.loc[labelled["split"] == s, "image"]) for s in SPLITS}
    for a, b in combinations(SPLITS, 2):
        shared = image_sets[a] & image_sets[b]
        assert not shared, f"{len(shared)} images are in both {a} and {b}"


def test_all_captions_of_an_image_share_one_split(labelled):
    assert (labelled.groupby("image")["split"].nunique() == 1).all()


def test_official_split_sizes(labelled):
    assert labelled.groupby("split")["image"].nunique().to_dict() == EXPECTED_IMAGES


def test_overlapping_split_lists_are_rejected():
    captions = pd.DataFrame({"image": ["a.jpg", "b.jpg"], "caption": ["x", "y"]})
    with pytest.raises(ValueError):
        assign_splits(captions, {"train": ["a.jpg"], "val": ["a.jpg"], "test": ["b.jpg"]})