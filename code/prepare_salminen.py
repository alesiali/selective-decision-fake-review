#!/usr/bin/env python3
"""
prepare_salminen.py - second dataset: prepares the fake product reviews of Salminen et al. (2022)

Prepares the fake product review data of Salminen et al. (2022) (Amazon reviews, 10 product
categories; label CG = computer-generated with GPT-2, OR = original) for the same leave-one-
domain-out (LODO) protocol as the main corpus, with the product category as the domain.

Steps (all deterministic):
  1. Read the raw CSV (columns category, rating, label, text_) and check the published counts
     (40,432 reviews; 20,216 CG and 20,216 OR; 10 categories).
  2. Exact de-duplication, the same rule as for the main corpus: texts that are identical after
     strip() and lower() are duplicates; the first occurrence is kept; a text that occurs with
     both labels is removed entirely.
  3. Text cleaning, the same as the clean_review field of the main corpus: lowercase, delete
     every character that is not a letter a-z or white space, lemmatise each token with the
     WordNet lemmatiser of NLTK (default part of speech = noun) and join with single spaces.
     With --main_dir, the function is first checked against the main corpus (it reproduces
     the distributed clean_review field exactly for all 2,830 reviews).
  4. Truncation control: many generated reviews are cut off in the middle of a sentence or a
     word. So that a classifier cannot use this artefact, the last token of every cleaned
     review (both classes) is removed -> column model_text. Reviews with fewer than two
     cleaned tokens are dropped. Exact duplicates of model_text are then removed (same rule).
  5. Near-duplicate audit, as for the main corpus: TF-IDF on character 3-5-grams within word
     boundaries (min_df 2, at most 50,000 features), cosine similarity >= 0.90, within and
     across categories, on model_text. Pairs across categories are removed (the later review,
     or both if their labels differ); pairs within a category are only reported.
  6. Balanced training sample: --n_per_class reviews per label and category (seed --seed),
     column in_sample. Models are trained on the sample of the source categories and tested
     on ALL reviews of the held-out category.

Outputs (in --out_dir): salminen_processed.csv, salminen_prep_report.json

Usage (needs NLTK and its WordNet data: python -m nltk.downloader wordnet):
  python code/prepare_salminen.py --raw fake_reviews_dataset.csv --out_dir data_salminen_new \
         [--main_dir FOLDER_WITH_THE_THREE_MAIN_CSVs] [--nltk_data DIR]
  fake_reviews_dataset.csv is the data file of Salminen et al. (2022), https://osf.io/tyue9/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0 (2026-10-07)"
EXPECTED = {"n": 40432, "CG": 20216, "OR": 20216, "categories": 10}
SHORT = {  # category -> short domain name used in file names
    "Books_5": "books", "Clothing_Shoes_and_Jewelry_5": "clothing", "Electronics_5": "electronics",
    "Home_and_Kitchen_5": "home", "Kindle_Store_5": "kindle", "Movies_and_TV_5": "movies",
    "Pet_Supplies_5": "pets", "Sports_and_Outdoors_5": "sports",
    "Tools_and_Home_Improvement_5": "tools", "Toys_and_Games_5": "toys",
}
MAIN_FILES = ["LIWC_Hotel_readability_cleaned_dedup.csv", "LIWC_Restaurant_readability_cleaned_dedup.csv",
              "LIWC_doctor_readability_cleaned_dedup.csv"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_cleaner(nltk_data=None):
    import nltk
    if nltk_data:
        nltk.data.path.insert(0, nltk_data)
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    cache = {}

    def lemma(tok):
        if tok not in cache:
            cache[tok] = lem.lemmatize(tok)  # default pos = "n", as in the main corpus
        return cache[tok]

    def clean(text):
        text = re.sub(r"[^a-z\s]", "", str(text).lower())
        return " ".join(lemma(t) for t in text.split())

    return clean, nltk.__version__


def dedup(df, key_col, label_col):
    """Keep the first occurrence of each key; remove keys that occur with both labels."""
    key = df[key_col]
    n_labels = df.groupby(key)[label_col].transform("nunique")
    conflict = n_labels > 1
    keep = ~key.duplicated(keep="first") & ~conflict
    return df[keep].copy(), int((key.duplicated(keep="first") & ~conflict).sum()), int(conflict.sum())


def near_duplicate_pairs(texts, threshold=0.90, chunk=500):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000,
                          dtype=np.float32)
    X = vec.fit_transform(texts)  # rows are L2-normalised, so X @ X.T is the cosine similarity
    XT = X.T.tocsr()
    pairs = []
    for s in range(0, X.shape[0], chunk):
        S = (X[s:s + chunk] @ XT).toarray()
        for i_loc, j in zip(*np.nonzero(S >= threshold)):
            i = s + i_loc
            if j > i:
                pairs.append((int(i), int(j), float(S[i_loc, j])))
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--main_dir", default=None, help="folder with the three main-corpus CSVs (cleaning check)")
    ap.add_argument("--nltk_data", default=None)
    ap.add_argument("--n_per_class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    clean, nltk_version = make_cleaner(args.nltk_data)
    rep = {"script": "prepare_salminen.py", "script_version": SCRIPT_VERSION, "nltk": nltk_version}

    if args.main_dir:  # 0. the cleaning function must reproduce the main corpus exactly
        tot = ok = 0
        for f in MAIN_FILES:
            m = pd.read_csv(os.path.join(args.main_dir, f))
            out = m["review"].astype(str).map(clean)
            ok += int((out == m["clean_review"].astype(str).str.strip()).sum())
            tot += len(m)
        rep["cleaning_check_main_corpus"] = {"exact_matches": ok, "reviews": tot}
        print(f"Cleaning check on the main corpus: {ok}/{tot} identical to clean_review")
        if ok != tot:
            raise SystemExit("The cleaning function does not reproduce the main corpus - stop.")

    # 1. raw data and published counts
    df = pd.read_csv(args.raw)
    df["text_"] = df["text_"].fillna("").astype(str)
    counts = {"n": int(len(df)), "CG": int((df.label == "CG").sum()), "OR": int((df.label == "OR").sum()),
              "categories": int(df.category.nunique())}
    rep["raw"] = {"file": os.path.basename(args.raw), "md5": md5(args.raw), **counts,
                  "matches_published_counts": counts == EXPECTED}
    print("Raw:", rep["raw"])
    df["orig_row"] = np.arange(len(df))
    df["y"] = (df["label"] == "CG").astype(int)  # 1 = fake (computer-generated)
    df["domain"] = df["category"].map(SHORT)
    if df["domain"].isna().any():
        raise SystemExit(f"Unknown categories: {sorted(df.loc[df.domain.isna(), 'category'].unique())}")

    # 2. exact de-duplication on the raw text
    df["key_raw"] = df["text_"].str.strip().str.lower()
    df, n_dup, n_conf = dedup(df, "key_raw", "y")
    rep["dedup_raw"] = {"removed_duplicates": n_dup, "removed_conflicting_label_rows": n_conf, "remaining": int(len(df))}
    print("Exact de-duplication (raw text):", rep["dedup_raw"])

    # 3-4. cleaning and truncation control
    df["clean_review"] = df["text_"].map(clean)
    toks = df["clean_review"].str.split()
    df["model_text"] = toks.map(lambda t: " ".join(t[:-1]))
    df["n_tokens_model"] = toks.map(len) - 1
    short = df["n_tokens_model"] < 1
    rep["dropped_too_short"] = int(short.sum())
    df = df[~short].copy()
    df, n_dup2, n_conf2 = dedup(df, "model_text", "y")
    rep["dedup_model_text"] = {"removed_duplicates": n_dup2, "removed_conflicting_label_rows": n_conf2,
                               "remaining": int(len(df))}
    print("Dropped (fewer than 2 cleaned tokens):", rep["dropped_too_short"],
          "| exact de-duplication (model_text):", rep["dedup_model_text"])

    # cue check: share of raw reviews ending with punctuation, by label
    ends = df["text_"].str.strip().str[-1:].isin(list(".!?)\"'"))
    rep["raw_ends_with_punctuation_share"] = {lab: round(float(ends[df.label == lab].mean()), 4) for lab in ["CG", "OR"]}

    # 5. near-duplicate audit
    df = df.reset_index(drop=True)
    pairs = near_duplicate_pairs(df["model_text"].tolist())
    dom, yy = df["domain"].to_numpy(), df["y"].to_numpy()
    cross = [(i, j, s) for i, j, s in pairs if dom[i] != dom[j]]
    within = [(i, j, s) for i, j, s in pairs if dom[i] == dom[j]]
    drop = set()
    for i, j, _ in cross:
        drop.add(j)
        if yy[i] != yy[j]:
            drop.add(i)
    rep["near_duplicates"] = {
        "method": "char_wb 3-5-gram TF-IDF, min_df 2, max_features 50000, cosine >= 0.90, on model_text",
        "pairs_across_categories": len(cross), "pairs_within_categories": len(within),
        "pairs_within_with_different_labels": int(sum(yy[i] != yy[j] for i, j, _ in within)),
        "removed_rows": len(drop)}
    print("Near-duplicate audit:", rep["near_duplicates"])
    df = df.drop(index=sorted(drop)).reset_index(drop=True)

    # 6. balanced training sample per category
    rng = np.random.RandomState(args.seed)
    df["in_sample"] = False
    for (d, y), g in df.groupby(["domain", "y"]):
        if len(g) < args.n_per_class:
            raise SystemExit(f"Not enough reviews in {d}/{y}: {len(g)}")
        pick = rng.choice(g.index.to_numpy(), size=args.n_per_class, replace=False)
        df.loc[pick, "in_sample"] = True
    rep["sample"] = {"n_per_class": args.n_per_class, "seed": args.seed, "n_in_sample": int(df.in_sample.sum())}
    final = df.groupby("domain").agg(n=("y", "size"), fake=("y", "sum")).reset_index()
    final["original"] = final["n"] - final["fake"]
    rep["final_counts"] = {r.domain: {"n": int(r.n), "fake_CG": int(r.fake), "original_OR": int(r.original)}
                           for r in final.itertuples()}
    rep["final_total"] = int(len(df))
    print("Final counts per category:\n", final.to_string(index=False))

    cols = ["orig_row", "category", "domain", "rating", "label", "y", "clean_review", "model_text",
            "n_tokens_model", "in_sample"]
    out_csv = os.path.join(args.out_dir, "salminen_processed.csv")
    df[cols].to_csv(out_csv, index=False)
    rep["output"] = {"file": "salminen_processed.csv", "md5": md5(out_csv)}
    with open(os.path.join(args.out_dir, "salminen_prep_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("Written:", out_csv, rep["output"]["md5"])


if __name__ == "__main__":
    main()
