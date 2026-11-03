# When "Free" Is Not Spam: TF-IDF vs. Sentence Embeddings on Spam-Like Legitimate SMS

**Natural Language Processing — Project (Option 1: Spam Detection)**
Imam Abdulrahman Bin Faisal University · College of Computer Science & IT
Section 7MA2 · Group 7 · Instructor: Dr. Mohammed Talal Ahmed Mousalli · 2026–2027, First Semester

| Name | Student ID | Role |
|---|---|---|
| Faisal Abdullah Ali Al Shehri | 2240001737 | Team Leader — data, splits, spam-word list, repo, report integration |
| Saad Abdulaziz Saad Al Shehri | 2240006495 | Method 1 (TF-IDF + LR), evaluation of both methods |
| Nawaf Rabie Hassan bin Shahbal | 2240009025 | Method 2 (Sentence embeddings + LR), model saving, Try it cell |

## Task
Input: a short, unseen English SMS. Output: **Spam** or **Not Spam**.

## Experimental question
> Which method better avoids false spam alerts on legitimate messages that contain spam-associated words such as "free", "win" or "call"?

## Methods
1. **Traditional:** TF-IDF word features + Logistic Regression (class-weighted), scikit-learn.
2. **Pretrained:** Sentence-Transformers `all-MiniLM-L6-v2` embeddings + Logistic Regression.

Settings are chosen on the **development set only** using Macro F1. Both methods are tested on identical examples.

## Evaluation
- Spam-associated words: top 20 by smoothed spam/ham log-ratio on **train only**, keeping words with 3+ letters that appear in at least 10 spam and 10 ham training messages (words that never occur in ham cannot cause false alarms).
- Main comparison: Accuracy and Macro F1 on the unchanged test set.
- False-positive rate on (a) test ham **with** spam-associated words, (b) test ham **without** them, (c) a challenge set of 60 legitimate messages prepared by the team, 20 per member, with ChatGPT assisting the drafting (`data/challenge_set.csv`).
- Error analysis: at least three errors per method.

## Repository layout
```
notebooks/Group7_Spam_Detection.ipynb   Main notebook: Setup, Data, Method 1, Method 2, Results, Try it, Sources
src/data_prep.py                        Download, clean, template grouping, 70/15/15 group-aware split
src/spam_words.py                       Spam-associated word list (train only) and ham test subsets
src/evaluation.py                       Shared metrics: accuracy, macro F1, false-positive rate
src/models.py                           Method 1 and Method 2: build, tune on dev, TF-IDF word contributions
src/analysis.py                         Evaluate both models on test + subsets, collect errors
src/predict.py                          Load saved models and classify new text (Try it)
data/                                   Splits, challenge set, original examples (see data/README.md)
models/                                 Saved trained classifiers (joblib)
results/                                Saved metric tables and error-analysis outputs
docs/evidence_log.md                    Dated evidence of each member's work
```

## How to run (Google Colab)
1. Open `notebooks/Group7_Spam_Detection.ipynb` in Colab (File → Open notebook → GitHub, paste the repo URL).
2. Runtime → **Restart and run all**. The Setup cell clones the repo, installs `requirements.txt` and downloads the dataset.
3. Use the **Try it** cell to classify a new message without retraining.

## Sources
- T. A. Almeida, J. M. G. Hidalgo, A. Yamakami, "Contributions to the study of SMS spam filtering: new collection and results," ACM DocEng, 2011.
- N. Reimers, I. Gurevych, "Sentence-BERT," EMNLP-IJCNLP, 2019.
- M. T. Ribeiro et al., "Beyond accuracy: Behavioral testing of NLP models with CheckList," ACL, 2020.
- Libraries: scikit-learn, pandas, sentence-transformers. Pretrained weights: `sentence-transformers/all-MiniLM-L6-v2`.
- AI assistance: Claude (Anthropic) assisted with code structure and implementation, and with drafting the report, slides and appendix. ChatGPT (OpenAI) assisted with drafting the challenge-set messages, labelled examples and contribution-documentation templates. Each member was assigned one part of the system (data, Method 1, Method 2) and is responsible for reviewing, validating and explaining that part. The team is responsible for the submitted work.
