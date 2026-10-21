#!/usr/bin/env python3
"""
published_protocol.py - base classifiers under the published cross-domain protocol (Supplementary Table S15)

Protocol of Li et al. (2014, Section 5.2, Table 4) and Ren & Zhang (2016, Section 4.4, Table 4):
train on hotel reviews, truthful (Customer) versus deceptive (Turker); test on restaurant and doctor reviews.
Neither paper states whether both hotel polarities were used, so two variants are run:
  --variant all       Turker 800 vs Customer 796 (both polarities; the literal reading)   [primary]
  --variant positive  Turker 400 vs Customer 400 (positive only; the test domains are positive)
Employee-written ("expert") hotel reviews are excluded (identified from data/hotel_meta_v2.csv).
Test sets are the full restaurant (200/200) and doctor (355/200) files of the main corpus. Li et al. tested doctor on
200 Turker vs 200 Customer reviews, so for doctor the balanced accuracy (mean of the two class recalls)
is the closer comparator to their accuracy.

As in the main experiments, 15% of the training pool (stratified by label and polarity, split seeds 42-46) is
held out as source validation for temperature scaling and ensemble weights; transformers use the same
seeds and hyper-parameters as train_save_logits.py (three models per repetition).

Sub-commands
  classical  (CPU)  TF-IDF + logistic regression (5 repetitions) and a linear SVM on unigrams trained on
                    the whole pool, as in Li et al.'s "SVM unigram" row (needs the main-corpus CSV files)
  train      (GPU)  fine-tune a transformer; one .npz per run (resumable; needs the main-corpus CSV files)
  analyze    (CPU)  ensembles as in analyze_logits.py; metrics at p = 0.5 per test domain (saved logits only)

Examples (from the repository root; the main-corpus CSV files in the root folder for classical and train)
  python code/published_protocol.py classical --data_dir . --meta data/hotel_meta_v2.csv --out results/published
  python code/published_protocol.py train --data_dir . --meta data/hotel_meta_v2.csv \
         --out_dir logits_new/published_roberta --model_name roberta-base --variant all
  python code/published_protocol.py analyze --logits_dir logits/published_roberta --model_name roberta-base \
         --variant all --out results/published
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_save_logits as T  # noqa: E402  (same data loading, seeds, hyper-parameters and training loop)

SCRIPT_VERSION = "1.0 (2026-10-15)"
VARIANTS = ("all", "positive")
TEST_DOMAINS = ("restaurant", "doctor")
PUBLISHED = {  # accuracy under this protocol (verified 2026-10-15 on the ACL Anthology PDFs)
    "Li et al. 2014, SVM unigram": {"restaurant": 0.785, "doctor": 0.550},
    "Li et al. 2014, SAGE LIWC (best doctor)": {"restaurant": 0.742, "doctor": 0.647},
    "Ren & Zhang 2016, integrated neural model": {"restaurant": 0.837, "doctor": 0.573},
}


# ------------------------------------------------------------------ data
def build_pool(domains, meta_path, variant):
    """Hotel Turker (deceptive) vs Customer (genuine); employee reviews excluded."""
    h = domains["hotel"].copy()
    meta = pd.read_csv(meta_path)
    if len(meta) != len(h) or not np.array_equal(meta["row_id"].to_numpy(), h["row_id"].to_numpy()):
        raise ValueError("hotel_meta does not match the hotel file row by row")
    h["writer"], h["polarity"] = meta["writer"].to_numpy(), meta["polarity"].to_numpy()
    ok = ((h.writer == "turker") & (h.y == 1)) | ((h.writer == "customer") & (h.y == 0))
    if not ((h.writer != "expert") == ok).all():
        raise ValueError("writer and label disagree in the hotel file")
    pool = h[h.writer.isin(["turker", "customer"])]
    if variant == "positive":
        pool = pool[pool.polarity == "positive"]
    return pool.reset_index(drop=True)


def split_pool(pool, split_seed, val_size=0.15):
    strata = pool["y"].astype(str) + "|" + pool["polarity"].astype(str)
    idx = np.arange(len(pool))
    tr_idx, va_idx = train_test_split(idx, test_size=val_size, random_state=split_seed, stratify=strata)
    return pool.iloc[np.sort(tr_idx)].reset_index(drop=True), pool.iloc[np.sort(va_idx)].reset_index(drop=True)


def test_set(domains):
    return pd.concat([domains[d] for d in TEST_DOMAINS], ignore_index=True)


# --------------------------------------------------------------- metrics
def metrics(y, pred):
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    prec0 = tn / (tn + fn) if tn + fn else 0.0
    f1_0 = 2 * prec0 * spec / (prec0 + spec) if prec0 + spec else 0.0
    return {"accuracy": (tp + tn) / len(y), "balanced_accuracy": (rec + spec) / 2, "precision_dec": prec,
            "recall_dec": rec, "f1_dec": f1, "macro_f1": (f1 + f1_0) / 2, "share_pred_dec": float(pred.mean()),
            "n": int(len(y)), "n_dec": int(y.sum())}


def summarise(rows, label):
    df = pd.DataFrame(rows)
    out = {}
    for d in TEST_DOMAINS:
        sub = df[df.domain == d]
        out[d] = {k: [float(sub[k].mean()), float(sub[k].std(ddof=1)) if len(sub) > 1 else 0.0]
                  for k in ["accuracy", "balanced_accuracy", "precision_dec", "recall_dec", "f1_dec", "macro_f1",
                            "share_pred_dec"]}
        out[d]["repetitions"] = int(len(sub))
    print(f"\n{label}")
    for d in TEST_DOMAINS:
        o = out[d]
        print(f"  {d:<10} acc {o['accuracy'][0]:.3f} ± {o['accuracy'][1]:.3f}   balanced acc "
              f"{o['balanced_accuracy'][0]:.3f}   P/R/F1(dec) {o['precision_dec'][0]:.3f}/{o['recall_dec'][0]:.3f}/"
              f"{o['f1_dec'][0]:.3f}   macro-F1 {o['macro_f1'][0]:.3f}   predicted deceptive {o['share_pred_dec'][0]:.3f}")
    return out


# --------------------------------------------------------------- classical
def cmd_classical(a):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    import train_classical as C  # same TF-IDF + LR settings as the main experiments
    os.makedirs(a.out, exist_ok=True)
    domains, info = T.load_domains(a.data_dir, "clean_review", "label")
    raw, _ = T.load_domains(a.data_dir, "review", "label")
    te, te_raw = test_set(domains), test_set(raw)
    results = {"script": "published_protocol.py classical", "version": SCRIPT_VERSION, "data_files": info,
               "published": PUBLISHED, "variants": {}}
    for variant in VARIANTS:
        pool = build_pool(domains, a.meta, variant)
        res = {"pool": {"n": int(len(pool)), "deceptive": int(pool.y.sum()),
                        "by_polarity": {k: int(v) for k, v in pool.polarity.value_counts().items()}}}
        rows = []
        for rep in range(5):
            tr, va = split_pool(pool, 42 + rep)
            vec = TfidfVectorizer(**C.TFIDF)
            clf = LogisticRegression(**C.LOGREG).fit(vec.fit_transform(tr["text"]), tr["y"])
            pred = (clf.decision_function(vec.transform(te["text"])) > 0).astype(int)  # p >= 0.5
            for d in TEST_DOMAINS:
                m = te.domain == d
                rows.append({"rep": rep, "domain": d, **metrics(te.y[m], pred[m])})
        res["tfidf_logreg"] = summarise(rows, f"[{variant}] TF-IDF + LR (clean_review; 85% of the pool; 5 repetitions)")
        # Li et al.-style linear SVM on unigrams of the raw review text, trained on the whole pool
        praw = raw["hotel"].set_index("row_id").loc[pool["row_id"]].reset_index()
        vec = CountVectorizer(lowercase=True, ngram_range=(1, 1))
        Xtr = vec.fit_transform(praw["text"])
        svm = LinearSVC(C=1.0, max_iter=20000).fit(Xtr, praw["y"])
        pred = svm.predict(vec.transform(te_raw["text"]))
        rows = [{"rep": 0, "domain": d, **metrics(te_raw.y[te_raw.domain == d], pred[te_raw.domain == d])}
                for d in TEST_DOMAINS]
        res["svm_unigram_raw"] = summarise(rows, f"[{variant}] linear SVM, unigram counts of the raw text, whole pool")
        results["variants"][variant] = res
    json.dump(results, open(os.path.join(a.out, "classical_published_protocol.json"), "w"), indent=1)
    print("\nwritten", os.path.join(a.out, "classical_published_protocol.json"))


# ------------------------------------------------------------------ train
def run_file(model_name, variant, rep, seed):
    return f"{model_name.replace('/', '-')}__published-{variant}__rep{rep}__seed{seed}.npz"


def cmd_train(a):
    os.makedirs(a.out_dir, exist_ok=True)
    domains, info = T.load_domains(a.data_dir, "clean_review", "label")
    pool, te = build_pool(domains, a.meta, a.variant), test_set(domains)
    plan = [(rep, ss, s) for rep, (ss, seeds) in T.REPETITIONS.items() for s in seeds]
    done = [p for p in plan if os.path.exists(os.path.join(a.out_dir, run_file(a.model_name, a.variant, p[0], p[2])))]
    print(f"published_protocol.py {SCRIPT_VERSION} | model={a.model_name} | variant={a.variant} | pool n={len(pool)} "
          f"(deceptive {int(pool.y.sum())}) | test n={len(te)}")
    print(f"Runs selected: {len(plan)}  already finished: {len(done)}  to do: {len(plan) - len(done)}")
    if a.dry_run:
        for rep, ss, s in plan:
            tr, va = split_pool(pool, ss)
            print(f"  rep={rep} split_seed={ss} seed={s}  n_train={len(tr)} n_val={len(va)} n_test={len(te)}")
        print("Dry run finished - no training was done.")
        return
    import torch
    import sklearn
    import transformers
    from transformers import AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU found - training will be very slow. In Colab: Runtime > Change runtime type > T4 GPU.")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    versions = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    print(f"Device: {gpu} | versions: {versions}")
    tok = AutoTokenizer.from_pretrained(a.model_name)
    hp = argparse.Namespace(model_name=a.model_name, lr=2e-5, batch_size=16, epochs=3, warmup_ratio=0.10, max_len=256)
    for i, (rep, ss, seed) in enumerate(plan, 1):
        path = os.path.join(a.out_dir, run_file(a.model_name, a.variant, rep, seed))
        tag = f"[{i}/{len(plan)}] variant={a.variant} rep={rep} split_seed={ss} seed={seed}"
        if os.path.exists(path):
            print(f"{tag}  -> already done, skipped")
            continue
        tr, va = split_pool(pool, ss)
        print(f"{tag}  n_train={len(tr)} n_val={len(va)} n_test={len(te)}", flush=True)
        t0 = time.time()
        vl, tl = T.train_and_predict(hp, tok, tr, va, te, seed, device)
        meta = {"script": "published_protocol.py", "script_version": SCRIPT_VERSION, "model_name": a.model_name,
                "variant": a.variant, "repetition": rep, "split_seed": ss, "model_seed": seed,
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "hyperparameters": vars(hp), "data_files": info, "versions": versions, "device": gpu,
                "runtime_sec": round(time.time() - t0, 1), "n_train": int(len(tr)), "n_val": int(len(va))}
        T.save_npz_atomic(path, {"val_logits": vl.astype(np.float32), "val_labels": va["y"].to_numpy(np.int64),
                                 "val_row": va["row_id"].to_numpy(np.int64), "test_logits": tl.astype(np.float32),
                                 "test_labels": te["y"].to_numpy(np.int64), "test_row": te["row_id"].to_numpy(np.int64),
                                 "test_domain": te["domain"].to_numpy().astype("U10"), "meta": json.dumps(meta)})
        acc = {d: float(((tl[(te.domain == d).to_numpy()].argmax(1)) == te.y[te.domain == d].to_numpy()).mean())
               for d in TEST_DOMAINS}
        print(f"      saved {os.path.basename(path)}  ({(time.time() - t0) / 60:.1f} min)  "
              f"accuracy at 0.5: restaurant {acc['restaurant']:.3f}  doctor {acc['doctor']:.3f}", flush=True)
    print("Finished.")


# ---------------------------------------------------------------- analyze
def cmd_analyze(a):
    import analyze_logits as A
    files = sorted(f for f in os.listdir(a.logits_dir)
                   if f.startswith(f"{a.model_name.replace('/', '-')}__published-{a.variant}__") and f.endswith(".npz"))
    by_rep = {}
    for f in files:
        z = np.load(os.path.join(a.logits_dir, f), allow_pickle=False)
        m = json.loads(str(z["meta"]))
        run = {k: z[k] for k in z.files if k != "meta"}
        run["meta"] = m
        pv = (run["val_logits"][:, 1] > run["val_logits"][:, 0]).astype(int)
        if pv.min() == pv.max():
            print(f"excluded as degenerate (single-class validation predictions): {f}")
            continue
        by_rep.setdefault(m["repetition"], []).append(run)
    if not by_rep:
        print("No usable runs found (check --logits_dir, --model_name and --variant).")
        return
    rows = []
    for rep, runs in sorted(by_rep.items()):
        g = A.ensemble_group(runs)  # member temperature scaling, F1 weights, ensemble temperature scaling
        dom = runs[0]["test_domain"]
        pred = (g["pt"] >= 0.5).astype(int)
        for d in TEST_DOMAINS:
            m = dom == d
            rows.append({"rep": rep, "domain": d, "models": len(runs), **metrics(g["yt"][m], pred[m])})
    print(f"{len(files)} files, repetitions: { {r: len(v) for r, v in sorted(by_rep.items())} }")
    out = summarise(rows, f"[{a.variant}] {a.model_name} ensemble (5 repetitions)")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"model": a.model_name, "variant": a.variant, "summary": out, "per_repetition": rows},
              open(os.path.join(a.out, f"{a.model_name.replace('/', '-')}_published_{a.variant}.json"), "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classical"); c.add_argument("--data_dir", required=True); c.add_argument("--meta", required=True)
    c.add_argument("--out", required=True)
    t = sub.add_parser("train"); t.add_argument("--data_dir", required=True); t.add_argument("--meta", required=True)
    t.add_argument("--out_dir", required=True); t.add_argument("--model_name", default="roberta-base")
    t.add_argument("--variant", choices=VARIANTS, default="all"); t.add_argument("--dry_run", action="store_true")
    n = sub.add_parser("analyze"); n.add_argument("--logits_dir", required=True); n.add_argument("--out", required=True)
    n.add_argument("--model_name", default="roberta-base"); n.add_argument("--variant", choices=VARIANTS, default="all")
    a = ap.parse_args(argv)
    {"classical": cmd_classical, "train": cmd_train, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
