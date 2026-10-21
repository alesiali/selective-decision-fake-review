#!/usr/bin/env python3
"""
train_classical.py - second model family: TF-IDF + logistic regression (CPU, deterministic)

TF-IDF (word 1-2 grams) + logistic regression, trained under exactly the same LODO
splits and repetitions as train_save_logits.py (same validation splits, same ids),
and saved in the same .npz format so that analyze_logits.py can be reused unchanged.
Logistic regression is deterministic, so there is one model per repetition (no seeds).

Usage (from the repository root, with the three main-corpus CSV files in the root folder):
  python code/train_classical.py --data_dir . --out_dir logits_new/tfidf_logreg
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import time

import sys

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # train_save_logits.py lives next to this file
import train_save_logits as T  # noqa: E402

MODEL_NAME = "tfidf-logreg"
TFIDF = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
LOGREG = {"C": 1.0, "solver": "lbfgs", "max_iter": 2000}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_size", type=float, default=0.15)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    domains, data_info = T.load_domains(args.data_dir, "clean_review", "label")

    for sources, target in T.LODO_SPLITS:
        for rep, (split_seed, _) in T.REPETITIONS.items():
            path = os.path.join(args.out_dir, T.run_name(MODEL_NAME, sources, target, rep, 0))
            if os.path.exists(path):
                continue
            tr, va, te = T.make_split(domains, sources, target, split_seed, args.val_size)
            t0 = time.time()
            vec = TfidfVectorizer(**TFIDF)
            Xtr = vec.fit_transform(tr["text"])          # fitted on source training data only
            clf = LogisticRegression(**LOGREG).fit(Xtr, tr["y"])
            dv = clf.decision_function(vec.transform(va["text"]))
            dtst = clf.decision_function(vec.transform(te["text"]))
            val_logits = np.column_stack([np.zeros_like(dv), dv]).astype(np.float32)
            test_logits = np.column_stack([np.zeros_like(dtst), dtst]).astype(np.float32)
            val_f1 = float(f1_score(va["y"], (dv >= 0).astype(int), average="macro"))
            meta = {
                "script": "train_classical.py", "script_version": "1.0 (2026-10-07)",
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "model_name": MODEL_NAME, "train_domains": list(sources), "test_domain": target,
                "repetition": rep, "split_seed": split_seed, "model_seed": 0,
                "hyperparameters": {"tfidf": {**TFIDF, "ngram_range": list(TFIDF["ngram_range"])},
                                    "logistic_regression": LOGREG, "val_size": args.val_size,
                                    "val_stratified_by": "domain and label"},
                "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                "train_strata": T.strata_counts(tr), "val_strata": T.strata_counts(va),
                "vocabulary_size": int(len(vec.vocabulary_)),
                "data_files": data_info,
                "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "numpy": np.__version__},
                "runtime_sec": round(time.time() - t0, 2),
                "sanity": {"val_macro_f1_at_0.5": round(val_f1, 4),
                           "test_share_predicted_deceptive_at_0.5": round(float((dtst >= 0).mean()), 4)},
            }
            arrays = {
                "val_logits": val_logits, "val_labels": va["y"].to_numpy().astype(np.int8),
                "val_domain": np.asarray(va["domain"].astype(str).tolist()),
                "val_row": va["row_id"].to_numpy().astype(np.int32),
                "val_ids": np.asarray([f"{d}:{r}" for d, r in zip(va["domain"], va["row_id"])]),
                "test_logits": test_logits, "test_labels": te["y"].to_numpy().astype(np.int8),
                "test_row": te["row_id"].to_numpy().astype(np.int32),
                "test_ids": np.asarray([f"{target}:{r}" for r in te["row_id"]]),
                "meta": np.asarray(json.dumps(meta)),
            }
            T.save_npz_atomic(path, arrays)
            print(f"train={'+'.join(sources):<18} test={target:<10} rep={rep}  val macro-F1={val_f1:.3f}  "
                  f"test share predicted deceptive={(dtst >= 0).mean():.3f}")


if __name__ == "__main__":
    main()
