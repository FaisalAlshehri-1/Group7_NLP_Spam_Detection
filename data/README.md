# Data

| File | What it is | Used for |
|---|---|---|
| `raw/SMSSpamCollection` | UCI SMS Spam Collection, downloaded by the notebook (not committed) | Train / Dev / Test |
| `processed/train.csv`, `dev.csv`, `test.csv` | Saved splits after cleaning and template grouping | Reproducibility |
| `processed/spam_words.csv` | Spam-associated words, built from **train only** | Test subsets (a) and (b) |
| `challenge_set.csv` | 60 legitimate messages (label 0) containing spam-associated words, 20 per member; drafting assisted by ChatGPT, each member responsible for reviewing his messages and labels | Challenge set (c), **never** used for training or tuning |
| `original_examples.csv` | Two labelled examples per member, one ham and one spam (brief requirement); drafting assisted by ChatGPT, reviewed by the member | Shown separately from test scores |

## Rules for `challenge_set.csv`
- `label` is always `0` (legitimate).
- `author`: Faisal, Saad or Nawaf.
- `uses_words`: comma-separated spam-associated words the message contains (must come from `processed/spam_words.csv`).
- Messages must not be copied from the dataset (checked: no overlap).

Source: Almeida, Gómez Hidalgo & Yamakami (2011), https://archive.ics.uci.edu/dataset/228/sms+spam+collection
