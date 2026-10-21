#!/usr/bin/env python3
"""
train_salminen.py - second dataset: LODO over the product categories of Salminen et al. (2022)

Leave-one-domain-out (LODO) experiment on the fake product reviews of Salminen et al. (2022),
prepared by prepare_salminen.py (file salminen_processed.csv). The product category is the domain.

Protocol (every value is also written into each output file)
  * 10 LODO splits: each category is the target once; the other nine are the sources.
  * Training and validation data: the balanced sample (column in_sample: 200 fake + 200 original
    reviews per category) of the nine source categories; validation = 15 % of it, stratified by
    category and label. Test data: ALL reviews of the target category (never used for training,
    tuning or selection; their labels are stored for scoring only).
  * 5 repetitions; repetition r uses validation-split seed 42+r and ONE model seed 42+r (50 runs).
  * roberta-base with the same settings as train_save_logits.py: AdamW (lr 2e-5, PyTorch default
    weight decay 0.01), batch 16, 3 epochs, linear warm-up over 10 % of the steps then linear decay,
    maximum length 256. Batches are padded to their longest review (dynamic padding) to save time;
    this does not change the model.
  * --model_name tfidf-logreg: TF-IDF (word 1-2 grams, min_df 2, sublinear tf) + logistic regression
    (C = 1), deterministic, CPU only. --train_pool all trains it on all source reviews instead of the sample.
  * Output: one .npz per run in the format of train_save_logits.py, so analyze_logits.py reads it unchanged.

Usage (from the repository root, after decompressing data/salminen_processed.csv.gz; GPU for RoBERTa):
  1) check:     python code/train_salminen.py --data data/salminen_processed.csv --out_dir logits_new/salminen_roberta --reps 0 1 2 --dry_run
  2) all runs:  the same command without --dry_run. After an interruption, run it again: finished runs are skipped.
  The article uses repetitions 0-2 for RoBERTa; TF-IDF (CPU): --model_name tfidf-logreg [--train_pool all], all
  five repetitions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # train_save_logits.py lives next to this file
import train_save_logits as T  # noqa: E402

SCRIPT_VERSION = "1.0 (2026-10-07)"
DOMAINS = ["books", "clothing", "electronics", "home", "kindle", "movies", "pets", "sports", "tools", "toys"]
REPETITIONS = {r: (42 + r, 42 + r) for r in range(5)}  # repetition -> (split seed, model seed)
TFIDF = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
LOGREG = {"C": 1.0, "solver": "lbfgs", "max_iter": 2000}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="LODO over the Salminen et al. (2022) product categories.")
    ap.add_argument("--data", required=True, help="salminen_processed.csv from prepare_salminen.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="roberta-base", help='a Hugging Face model, or "tfidf-logreg"')
    ap.add_argument("--train_pool", choices=["sample", "all"], default="sample")
    ap.add_argument("--text_col", default="model_text")
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--reps", nargs="+", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args(argv)


def load(path, text_col):
    df = pd.read_csv(path)
    df["text"] = df[text_col].fillna("").astype(str)
    if (df["text"].str.strip() == "").any():
        raise ValueError("Empty texts in the processed file.")
    if sorted(df["domain"].unique()) != DOMAINS:
        raise ValueError(f"Unexpected domains: {sorted(df['domain'].unique())}")
    df["row_id"] = df["orig_row"].astype(int)  # row in the raw CSV, for traceability
    return df


def make_split(df, target, split_seed, val_size, pool):
    src = df[(df["domain"] != target) & (df["in_sample"] if pool == "sample" else True)].reset_index(drop=True)
    strata = src["domain"].astype(str) + "|" + src["y"].astype(str)
    tr_idx, va_idx = train_test_split(np.arange(len(src)), test_size=val_size, random_state=split_seed,
                                      stratify=strata)
    tr = src.iloc[np.sort(tr_idx)].reset_index(drop=True)
    va = src.iloc[np.sort(va_idx)].reset_index(drop=True)
    te = df[df["domain"] == target].reset_index(drop=True)
    return tr, va, te


def strata_counts(d):
    s = d["domain"].astype(str) + "|" + d["y"].astype(str)
    return {k: int(v) for k, v in s.value_counts().sort_index().items()}


def run_name(model_name, pool, target, rep, seed):
    tag = model_name.replace("/", "-") + ("" if pool == "sample" else "-alltrain")
    return f"{tag}__salminen__test-{target}__rep{rep}__seed{seed}.npz"


# ------------------------------------------------------------------ transformer
def make_loader(tokenizer, texts, labels, batch_size, max_len, shuffle, seed=None):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    enc = tokenizer(list(texts), truncation=True, max_length=max_len)
    feats = [{"input_ids": i, "attention_mask": a, "labels": int(l)}
             for i, a, l in zip(enc["input_ids"], enc["attention_mask"], labels)]
    gen = None
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return DataLoader(feats, batch_size=batch_size, shuffle=shuffle, generator=gen,
                      collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"))


def predict(model, loader, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for b in loader:
            logits = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def train_transformer(args, tokenizer, tr, va, te, seed, device):
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    T.set_seed(seed)
    train_loader = make_loader(tokenizer, tr["text"], tr["y"], args.batch_size, args.max_len, True, seed)
    val_loader = make_loader(tokenizer, va["text"], va["y"], args.batch_size, args.max_len, False)
    test_loader = make_loader(tokenizer, te["text"], np.zeros(len(te), dtype=int),  # no target labels
                              args.batch_size, args.max_len, False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total), total)
    model.train()
    for epoch in range(args.epochs):
        loss_sum, nb = 0.0, 0
        for b in train_loader:
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += float(out.loss.item())
            nb += 1
        print(f"      epoch {epoch + 1}/{args.epochs}  mean train loss {loss_sum / max(nb, 1):.4f}", flush=True)
    v, t = predict(model, val_loader, device), predict(model, test_loader, device)
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return v, t


def train_tfidf(tr, va, te):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(**TFIDF)
    clf = LogisticRegression(**LOGREG).fit(vec.fit_transform(tr["text"]), tr["y"])  # source training data only
    dv, dt_ = clf.decision_function(vec.transform(va["text"])), clf.decision_function(vec.transform(te["text"]))
    z = lambda d: np.column_stack([np.zeros_like(d), d]).astype(np.float32)
    return z(dv), z(dt_), int(len(vec.vocabulary_))


# --------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    is_tfidf = args.model_name == "tfidf-logreg"
    os.makedirs(args.out_dir, exist_ok=True)
    df = load(args.data, args.text_col)
    data_info = {"file": os.path.basename(args.data), "md5": T.md5(args.data), "n": int(len(df)),
                 "n_in_sample": int(df["in_sample"].sum()), "text_col": args.text_col}
    print(f"train_salminen.py {SCRIPT_VERSION} | model={args.model_name} | pool={args.train_pool} | "
          f"data={data_info}")
    targets = [t for t in DOMAINS if not args.targets or t in [x.lower() for x in args.targets]]
    plan = [(t, r) for t in targets for r in REPETITIONS if args.reps is None or r in args.reps]
    seed_of = (lambda r: 0) if is_tfidf else (lambda r: REPETITIONS[r][1])
    todo = [p for p in plan if not os.path.exists(os.path.join(
        args.out_dir, run_name(args.model_name, args.train_pool, p[0], p[1], seed_of(p[1]))))]
    print(f"Runs selected: {len(plan)}  already finished: {len(plan) - len(todo)}  to do: {len(todo)}")

    if args.dry_run:
        for t, r in plan:
            tr, va, te = make_split(df, t, REPETITIONS[r][0], args.val_size, args.train_pool)
            print(f"  test={t:<12} rep={r}  n_train={len(tr)} n_val={len(va)} n_test={len(te)} "
                  f"(test fake share {te['y'].mean():.3f})")
        print("Dry run finished - no training was done.")
        return

    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    import sklearn
    versions["scikit-learn"] = sklearn.__version__
    device, gpu_name, tokenizer = None, "cpu", None
    if not is_tfidf:
        import torch
        import transformers
        from transformers import AutoTokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        if device.type != "cuda":
            print("WARNING: no GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")
        versions.update({"torch": torch.__version__, "transformers": transformers.__version__})
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"Device: {gpu_name} | versions: {versions}")

    for i, (t, r) in enumerate(plan, 1):
        split_seed, seed = REPETITIONS[r][0], seed_of(r)
        path = os.path.join(args.out_dir, run_name(args.model_name, args.train_pool, t, r, seed))
        tag = f"[{i}/{len(plan)}] test={t} rep={r} split_seed={split_seed} seed={seed}"
        if os.path.exists(path):
            print(f"{tag}  -> already done, skipped")
            continue
        tr, va, te = make_split(df, t, split_seed, args.val_size, args.train_pool)
        print(f"{tag}  n_train={len(tr)} n_val={len(va)} n_test={len(te)}", flush=True)
        t0 = time.time()
        if is_tfidf:
            val_logits, test_logits, vocab = train_tfidf(tr, va, te)
            hp = {"tfidf": {**TFIDF, "ngram_range": list(TFIDF["ngram_range"])}, "logistic_regression": LOGREG,
                  "vocabulary_size": vocab}
        else:
            val_logits, test_logits = train_transformer(args, tokenizer, tr, va, te, seed, device)
            hp = {"lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
                  "warmup_ratio": args.warmup_ratio, "max_len": args.max_len, "padding": "dynamic (per batch)",
                  "optimizer": "torch.optim.AdamW (default weight_decay=0.01)",
                  "scheduler": "linear warm-up, then linear decay to 0"}
        runtime = time.time() - t0
        val_f1 = float(f1_score(va["y"], (T.softmax_pos(val_logits) >= 0.5).astype(int), average="macro"))
        test_rate = float((T.softmax_pos(test_logits) >= 0.5).mean())
        sources = [d for d in DOMAINS if d != t]
        meta = {
            "script": "train_salminen.py", "script_version": SCRIPT_VERSION, "dataset": "salminen2022",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "model_name": args.model_name + ("" if args.train_pool == "sample" else "-alltrain"),
            "train_domains": sources, "test_domain": t, "repetition": r, "split_seed": split_seed,
            "model_seed": seed, "train_pool": args.train_pool,
            "hyperparameters": {**hp, "val_size": args.val_size, "val_stratified_by": "category and label"},
            "text_col": args.text_col, "label_col": "y (1 = computer-generated)",
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "train_strata": strata_counts(tr), "val_strata": strata_counts(va),
            "data_files": {"salminen": data_info}, "versions": versions, "device": gpu_name,
            "runtime_sec": round(runtime, 1),
            "sanity": {"val_macro_f1_at_0.5": round(val_f1, 4),
                       "test_share_predicted_deceptive_at_0.5": round(test_rate, 4)},
        }
        arrays = {
            "val_logits": val_logits.astype(np.float32),
            "val_labels": va["y"].to_numpy().astype(np.int8),
            "val_domain": np.asarray(va["domain"].astype(str).tolist()),
            "val_row": va["row_id"].to_numpy().astype(np.int32),
            "val_ids": np.asarray([f"{d}:{x}" for d, x in zip(va["domain"], va["row_id"])]),
            "test_logits": test_logits.astype(np.float32),
            "test_labels": te["y"].to_numpy().astype(np.int8),
            "test_row": te["row_id"].to_numpy().astype(np.int32),
            "test_ids": np.asarray([f"{t}:{x}" for x in te["row_id"]]),
            "meta": np.asarray(json.dumps(meta)),
        }
        T.save_npz_atomic(path, arrays)
        print(f"      saved {os.path.basename(path)}  ({runtime / 60:.1f} min)  val macro-F1@0.5={val_f1:.4f}  "
              f"test share predicted fake={test_rate:.3f}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
