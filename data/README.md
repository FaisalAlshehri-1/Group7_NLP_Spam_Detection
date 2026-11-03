# Data

| File | What it is | Used for |
|---|---|---|
| `raw/SMSSpamCollection` | UCI SMS Spam Collection, downloaded by the notebook (not committed) | Train / Dev / Test |
| `processed/train.csv`, `dev.csv`, `test.csv` | Saved splits after cleaning and template grouping | Reproducibility |
| `processed/spam_words.csv` | Spam-associated words, built from **train only** | Test subsets (a) and (b) |
| `challenge_set.csv` | ~60 hand-written legitimate messages (label 0) containing spam-associated words, ~20 per member | Challenge set (c), **never** used for training or tuning |
| `original_examples.csv` | Two original labelled examples per member (brief requirement) | Shown separately from test scores |

## Rules for `challenge_set.csv`
- `label` is always `0` (legitimate).
- `author`: Faisal, Saad or Nawaf.
- `uses_words`: comma-separated spam-associated words the message contains (must come from `processed/spam_words.csv`).
- Write real-sounding messages; do not copy from the dataset.

Source: Almeida, Gómez Hidalgo & Yamakami (2011), https://archive.ics.uci.edu/dataset/228/sms+spam+collection
