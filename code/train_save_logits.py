#!/usr/bin/env python3
"""
train_save_logits.py - step 1: fine-tune a transformer under LODO and save its logits

Trains a transformer classifier under the Leave-One-Domain-Out (LODO) protocol
and saves the raw logits of every run (one .npz file per run). Nothing else is
computed here: calibration, ensembling, thresholds, costs, baselines,
risk-coverage curves and the Hotel analysis are all done afterwards from the
saved files, so no model ever has to be retrained.

Protocol (defaults; every value is also written into each output file)
  * LODO splits: hotel+restaurant -> doctor
                 hotel+doctor     -> restaurant
                 doctor+restaurant -> hotel
  * 5 repetitions per split. Repetition r has its own validation split
    (15 % of the source domains, stratified by domain AND label) and three
    model seeds (table REPETITIONS below; repetition 0 keeps the original
    model seeds 42/123/2024).  3 splits x 5 repetitions x 3 seeds = 45 runs.
  * roberta-base, AdamW (lr 2e-5, PyTorch default weight decay 0.01),
    batch 16, 3 epochs, linear warm-up over 10 % of the steps, max length 256.
  * Target-domain labels are stored for later scoring only. They never enter
    training, prediction or any selection step.

Usage (GPU; from the repository root, with the three main-corpus CSV files in the root folder; in Google
Colab choose Runtime > Change runtime type > T4 GPU and put "!" in front of the command):
  1) check data and splits, no training:
     python code/train_save_logits.py --data_dir . --out_dir logits_new/roberta --dry_run
  2) a single test run:   add  --targets hotel --reps 0 --seeds 42
  3) all runs: the command of step 1 without --dry_run (BERT: add --model_name bert-base-uncased and use
     another --out_dir). After an interruption, run the same command again: finished runs are skipped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import platform
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

SCRIPT_VERSION = "1.0 (2026-10-07)"

DOMAIN_FILES = {
    "doctor": "LIWC_doctor_readability_cleaned_dedup.csv",
    "hotel": "LIWC_Hotel_readability_cleaned_dedup.csv",
    "restaurant": "LIWC_Restaurant_readability_cleaned_dedup.csv",
}
# (genuine, deceptive) in the de-duplicated files used in the article
EXPECTED_COUNTS = {"doctor": (200, 355), "hotel": (796, 1079), "restaurant": (200, 200)}
LABEL_MAP = {"truthful": 0, "genuine": 0, "deceptive": 1, "fake": 1}

LODO_SPLITS = [
    (("hotel", "restaurant"), "doctor"),
    (("hotel", "doctor"), "restaurant"),
    (("doctor", "restaurant"), "hotel"),
]

# repetition -> (validation-split seed, model seeds)
REPETITIONS = {
    0: (42, (42, 123, 2024)),  # original model seeds
    1: (43, (43, 143, 243)),
    2: (44, (44, 144, 244)),
    3: (45, (45, 145, 245)),
    4: (46, (46, 146, 246)),
}


# ------------------------------------------------------------------ helpers
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train under LODO and save logits (Step 1).")
    ap.add_argument("--data_dir", required=True, help="folder with the three de-duplicated CSV files")
    ap.add_argument("--out_dir", required=True, help="folder where the .npz files are written")
    ap.add_argument("--model_name", default="roberta-base")
    ap.add_argument("--text_col", default="clean_review")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--splits", nargs="+", default=None,
                    help='e.g. "hotel+restaurant>doctor" (default: the three LODO splits)')
    ap.add_argument("--targets", nargs="+", default=None, help="only these test domains")
    ap.add_argument("--reps", nargs="+", type=int, default=None, help="only these repetitions")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="only these model seeds")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--dry_run", action="store_true", help="check data and splits only; no training")
    return ap.parse_args(argv)


def parse_split(spec: str):
    src, sep, tgt = spec.partition(">")
    sources = tuple(s.strip().lower() for s in src.split("+") if s.strip())
    target = tgt.strip().lower()
    if not sep or not sources or not target:
        raise ValueError(f'Bad split "{spec}". Use e.g. "hotel+restaurant>doctor".')
    for d in (*sources, target):
        if d not in DOMAIN_FILES:
            raise ValueError(f'Unknown domain "{d}" in split "{spec}".')
    if target in sources:
        raise ValueError(f'Target "{target}" is also a source in "{spec}".')
    return sources, target


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_csv(data_dir: str, domain: str) -> str:
    exact = os.path.join(data_dir, DOMAIN_FILES[domain])
    if os.path.isfile(exact):
        return exact
    cands = [p for p in glob.glob(os.path.join(data_dir, "*.csv"))
             if domain in os.path.basename(p).lower() and "dedup" in os.path.basename(p).lower()]
    if len(cands) == 1:
        return cands[0]
    raise FileNotFoundError(
        f'Could not find a unique de-duplicated CSV for "{domain}" in {data_dir}. '
        f"Expected {DOMAIN_FILES[domain]}; candidates found: {cands}")


def to_labels(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        y = series.to_numpy()
    else:
        y = series.astype(str).str.strip().str.lower().map(LABEL_MAP).to_numpy()
    if pd.isna(y).any() or not set(np.unique(y)) <= {0, 1}:
        raise ValueError(f"Labels must be 0/1 (or truthful/deceptive); found {sorted(set(map(str, np.unique(y))))}")
    return y.astype(int)


def load_domains(data_dir: str, text_col: str, label_col: str):
    domains, info = {}, {}
    print("Data files")
    for d in DOMAIN_FILES:
        path = resolve_csv(data_dir, d)
        df = pd.read_csv(path)
        for c in (text_col, label_col):
            if c not in df.columns:
                raise KeyError(f'Column "{c}" not found in {path}. Columns: {list(df.columns)}')
        y = to_labels(df[label_col])
        text = df[text_col].fillna("").astype(str)
        n_empty = int((text.str.strip() == "").sum())
        domains[d] = pd.DataFrame({"text": text.tolist(), "y": y,
                                   "row_id": np.arange(len(df)), "domain": d})
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        info[d] = {"file": os.path.basename(path), "md5": md5(path),
                   "n": int(len(df)), "genuine": n0, "deceptive": n1}
        flag = "" if (n0, n1) == EXPECTED_COUNTS[d] else "   <-- differs from the counts in the article!"
        print(f"  {d:<10} {os.path.basename(path):<48} n={len(df):5d}  genuine={n0:4d}  "
              f"deceptive={n1:4d}  empty_texts={n_empty}{flag}")
    return domains, info


def make_split(domains, sources, target, split_seed: int, val_size: float):
    src = pd.concat([domains[d] for d in sources], ignore_index=True)
    strata = src["domain"].astype(str) + "|" + src["y"].astype(str)
    idx = np.arange(len(src))
    tr_idx, va_idx = train_test_split(idx, test_size=val_size, random_state=split_seed, stratify=strata)
    tr = src.iloc[np.sort(tr_idx)].reset_index(drop=True)
    va = src.iloc[np.sort(va_idx)].reset_index(drop=True)
    te = domains[target].reset_index(drop=True)
    return tr, va, te


def strata_counts(df: pd.DataFrame) -> dict:
    s = df["domain"].astype(str) + "|" + df["y"].astype(str)
    return {k: int(v) for k, v in s.value_counts().sort_index().items()}


def run_name(model_name: str, sources, target: str, rep: int, seed: int) -> str:
    tag = model_name.replace("/", "-")
    return f"{tag}__train-{'+'.join(sources)}__test-{target}__rep{rep}__seed{seed}.npz"


def softmax_pos(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def save_npz_atomic(path: str, arrays: dict):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)  # a disconnect never leaves a half-written .npz


# ------------------------------------------------------------- torch parts
def make_loader(tokenizer, texts, labels, batch_size, max_len, shuffle, seed=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    enc = tokenizer(list(texts), truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(np.array(labels, dtype=np.int64)))
    gen = None
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


def predict(model, loader, device) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in loader:
            logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def train_and_predict(args, tokenizer, tr, va, te, seed, device):
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    set_seed(seed)
    train_loader = make_loader(tokenizer, tr["text"], tr["y"], args.batch_size, args.max_len, True, seed)
    val_loader = make_loader(tokenizer, va["text"], va["y"], args.batch_size, args.max_len, False)
    # test labels are NOT passed to the model loop: dummy zeros only
    test_loader = make_loader(tokenizer, te["text"], np.zeros(len(te), dtype=int),
                              args.batch_size, args.max_len, False)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    model.train()
    for epoch in range(args.epochs):
        loss_sum, n_batches = 0.0, 0
        for input_ids, attention_mask, labels in train_loader:
            out = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                        labels=labels.to(device))
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += float(out.loss.item())
            n_batches += 1
        print(f"      epoch {epoch + 1}/{args.epochs}  mean train loss {loss_sum / max(n_batches, 1):.4f}",
              flush=True)

    val_logits = predict(model, val_loader, device)
    test_logits = predict(model, test_loader, device)
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_logits, test_logits


# --------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    splits = [parse_split(s) for s in args.splits] if args.splits else LODO_SPLITS
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"train_save_logits.py {SCRIPT_VERSION} | model={args.model_name} | out_dir={args.out_dir}")

    domains, data_info = load_domains(args.data_dir, args.text_col, args.label_col)

    plan = []
    for sources, target in splits:
        if args.targets and target not in [t.lower() for t in args.targets]:
            continue
        for rep, (split_seed, model_seeds) in REPETITIONS.items():
            if args.reps is not None and rep not in args.reps:
                continue
            for seed in model_seeds:
                if args.seeds is not None and seed not in args.seeds:
                    continue
                plan.append((sources, target, rep, split_seed, seed))
    done = [p for p in plan if os.path.exists(os.path.join(args.out_dir, run_name(args.model_name, p[0], p[1], p[2], p[4])))]
    print(f"Runs selected: {len(plan)}  already finished: {len(done)}  to do: {len(plan) - len(done)}")
    if not plan:
        print("Nothing selected - check --targets/--reps/--seeds.")
        return

    if args.dry_run:
        seen = set()
        for sources, target, rep, split_seed, _ in plan:
            if (sources, target, split_seed) in seen:
                continue
            seen.add((sources, target, split_seed))
            tr, va, te = make_split(domains, sources, target, split_seed, args.val_size)
            print(f"  train={'+'.join(sources):<18} test={target:<10} rep={rep} split_seed={split_seed}  "
                  f"n_train={len(tr)} n_val={len(va)} n_test={len(te)}  val strata={strata_counts(va)}")
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU"
        except ImportError:
            gpu = "torch not installed"
        print(f"GPU: {gpu}\nDry run finished - no training was done.")
        return

    import sklearn
    import torch
    import transformers
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    if device.type != "cuda":
        print("WARNING: no GPU found - training will be very slow. In Colab: Runtime > Change runtime type > T4 GPU.")
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": transformers.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    print(f"Device: {gpu_name} | versions: {versions}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    split_cache = {}
    for i, (sources, target, rep, split_seed, seed) in enumerate(plan, 1):
        path = os.path.join(args.out_dir, run_name(args.model_name, sources, target, rep, seed))
        tag = f"[{i}/{len(plan)}] train={'+'.join(sources)} test={target} rep={rep} split_seed={split_seed} seed={seed}"
        if os.path.exists(path):
            print(f"{tag}  -> already done, skipped")
            continue
        key = (sources, target, split_seed)
        if key not in split_cache:
            split_cache[key] = make_split(domains, sources, target, split_seed, args.val_size)
        tr, va, te = split_cache[key]
        print(f"{tag}  n_train={len(tr)} n_val={len(va)} n_test={len(te)}", flush=True)

        t0 = time.time()
        val_logits, test_logits = train_and_predict(args, tokenizer, tr, va, te, seed, device)
        runtime = time.time() - t0

        val_f1 = float(f1_score(va["y"], (softmax_pos(val_logits) >= 0.5).astype(int), average="macro"))
        test_rate = float((softmax_pos(test_logits) >= 0.5).mean())
        meta = {
            "script": "train_save_logits.py", "script_version": SCRIPT_VERSION,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "model_name": args.model_name, "train_domains": list(sources), "test_domain": target,
            "repetition": rep, "split_seed": split_seed, "model_seed": seed,
            "hyperparameters": {
                "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
                "warmup_ratio": args.warmup_ratio, "max_len": args.max_len,
                "optimizer": "torch.optim.AdamW (default weight_decay=0.01)",
                "scheduler": "linear warm-up, then linear decay to 0",
                "val_size": args.val_size, "val_stratified_by": "domain and label"},
            "text_col": args.text_col, "label_col": args.label_col,
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "train_strata": strata_counts(tr), "val_strata": strata_counts(va),
            "data_files": data_info, "versions": versions, "device": gpu_name,
            "cudnn_deterministic": True, "runtime_sec": round(runtime, 1),
            "sanity": {"val_macro_f1_at_0.5": round(val_f1, 4),
                       "test_share_predicted_deceptive_at_0.5": round(test_rate, 4)},
        }
        arrays = {
            "val_logits": val_logits.astype(np.float32),
            "val_labels": va["y"].to_numpy().astype(np.int8),
            "val_domain": np.asarray(va["domain"].astype(str).tolist()),
            "val_row": va["row_id"].to_numpy().astype(np.int32),
            "val_ids": np.asarray([f"{d}:{r}" for d, r in zip(va["domain"], va["row_id"])]),
            "test_logits": test_logits.astype(np.float32),
            "test_labels": te["y"].to_numpy().astype(np.int8),
            "test_row": te["row_id"].to_numpy().astype(np.int32),
            "test_ids": np.asarray([f"{target}:{r}" for r in te["row_id"]]),
            "meta": np.asarray(json.dumps(meta)),
        }
        save_npz_atomic(path, arrays)
        print(f"      saved {os.path.basename(path)}  ({runtime / 60:.1f} min)  "
              f"val macro-F1@0.5={val_f1:.4f}  test share predicted deceptive={test_rate:.3f}", flush=True)

    print("Finished.")


if __name__ == "__main__":
    main()
