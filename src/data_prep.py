"""Data loading, cleaning, near-duplicate grouping and leakage-free splitting.

Owner: Faisal (Team Leader)
Dataset: UCI SMS Spam Collection
https://archive.ics.uci.edu/dataset/228/sms+spam+collection
"""

import io
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_FILE = "SMSSpamCollection"
SEED = 42

_URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|co\.uk|net|org|biz)\S*)", re.I)
_PHONE_RE = re.compile(r"\+?\d[\d\s-]{6,}\d")
_DIGIT_RE = re.compile(r"\d+")
_SPACE_RE = re.compile(r"\s+")


# Identical copy of the UCI file (same 5,574 rows), used only if UCI is unreachable.
MIRROR_URL = ("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
              "master/data/sms.tsv")


def download_dataset(raw_dir="data/raw"):
    """Download and extract the dataset if it is not already present.
    Tries the official UCI archive first, then the mirror copy."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_FILE
    if target.exists():
        return target
    try:
        with urlopen(UCI_ZIP_URL, timeout=30) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                zf.extract(RAW_FILE, raw_dir)
        print("Downloaded from UCI.")
    except Exception as err:  # network or server problem
        print(f"UCI download failed ({type(err).__name__}); using mirror copy.")
        with urlopen(MIRROR_URL, timeout=30) as resp:
            target.write_bytes(resp.read())
    return target


def load_raw(path="data/raw/SMSSpamCollection"):
    """Read the tab-separated file into a DataFrame with columns label, text."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     quoting=3, encoding="utf-8")
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def clean(df):
    """Trim whitespace, drop empty rows and exact duplicates. Returns (df, report)."""
    report = {"raw_rows": len(df)}
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    report["after_empty_removed"] = len(df)
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    report["after_exact_duplicates_removed"] = len(df)
    report["ham"] = int((df["label"] == 0).sum())
    report["spam"] = int((df["label"] == 1).sum())
    return df, report


def template_key(text):
    """Normalise a message so near-duplicate templates share one key."""
    t = text.lower()
    t = _URL_RE.sub(" <url> ", t)
    t = _PHONE_RE.sub(" <phone> ", t)
    t = _DIGIT_RE.sub(" <num> ", t)
    return _SPACE_RE.sub(" ", t).strip()


def add_groups(df):
    """Attach a group id: messages with the same template key share a group."""
    df = df.copy()
    df["template"] = df["text"].map(template_key)
    df["group"] = df["template"].factorize()[0]
    return df


def split(df, seed=SEED):
    """Stratified, group-aware split of about 70 / 15 / 15.

    Test: 1 of 7 folds (~14.3%). Dev: 1 of 6 folds of the rest (~14.3%).
    A template group never appears in more than one split.
    """
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, df["label"], df["group"]))
    rest, test = df.iloc[rest_idx], df.iloc[test_idx]

    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_idx, dev_idx = next(inner.split(rest, rest["label"], rest["group"]))
    train, dev = rest.iloc[train_idx], rest.iloc[dev_idx]

    assert not set(train["group"]) & set(dev["group"])
    assert not set(train["group"]) & set(test["group"])
    assert not set(dev["group"]) & set(test["group"])
    return (train.reset_index(drop=True), dev.reset_index(drop=True),
            test.reset_index(drop=True))


def split_summary(train, dev, test):
    """Table of sizes and class counts per split."""
    rows = []
    total = len(train) + len(dev) + len(test)
    for name, part in [("train", train), ("dev", dev), ("test", test)]:
        rows.append({"split": name, "n": len(part),
                     "share": round(len(part) / total, 3),
                     "ham": int((part["label"] == 0).sum()),
                     "spam": int((part["label"] == 1).sum())})
    return pd.DataFrame(rows)
