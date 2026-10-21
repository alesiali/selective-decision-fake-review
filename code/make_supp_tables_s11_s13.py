#!/usr/bin/env python3
"""
make_supp_tables_s11_s13.py - Supplementary Tables S11-S13

S11  expected cost of every rule on the second dataset: RoBERTa (3 repetitions) and TF-IDF on the same repetitions
     (taken from make_tables_salminen.py output, passed with --s11_md)
S12  post-hoc diagnostic: cost-derived rule after Platt scaling instead of temperature scaling, five settings
     (computed by diagnostic_platt_intercept.py, which must be in the same folder as this script)
S13  excess cost of each rule over the cheapest rule, all model-target-scenario settings (descriptive)

Usage (from the repository root, after extracting the logits and running analyze_logits.py; LOGITS_ROOT and
RESULTS_ROOT contain the folders roberta, bert, tfidf_logreg, salminen_roberta and salminen_tfidf_sample):
  python code/make_supp_tables_s11_s13.py --logits_root logits --results_root results \
         --s11_md tables/second_dataset_S11_roberta_vs_tfidf.md --out tables/second_dataset_S11_S13.md
"""
from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
import contextlib
import io
import json
import os
import runpy

import numpy as np
import pandas as pd

DIAG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostic_platt_intercept.py")
RULES = {"Forced, F1-tuned": lambda s: "FORCED-F1", "Forced, cost-optimal": lambda s: f"FORCED-COST_{s}",
         "Cost-derived three-way": lambda s: f"CHOW_{s}", "Three-way, tuned": lambda s: f"CHOW-TUNED_{s}",
         "MSP, tuned": lambda s: f"MSP-TUNED_{s}"}


def hu(x, d=3, sign=False):
    """Round half up and format with d decimals. The value is first rounded to 10 decimals, so exact ties
    are not decided by floating-point noise (same rule as make_tables.py)."""
    v = Decimal(str(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{v:+f}" if sign else f"{v:f}"


def platt_means(logits_dir, model, results_json):
    import sys
    argv = sys.argv
    sys.argv = [DIAG, logits_dir, model, results_json]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            g = runpy.run_path(DIAG, run_name="__main__")
    finally:
        sys.argv = argv
    return g["m"]


def rng(x, nd=3):
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    return hu(lo, nd) if abs(hi - lo) < 10 ** (-nd) / 2 else f"{hu(lo, nd)} to {hu(hi, nd)}"


def s12(logits_root, results_root, main_roberta_logits=None):
    L = lambda d: os.path.join(logits_root, d)
    R = lambda d: os.path.join(results_root, d)
    settings = [("Main corpus, RoBERTa ensemble", main_roberta_logits or L("roberta"), "roberta-base", R("roberta")),
                ("Main corpus, BERT ensemble", L("bert"), "bert-base-uncased", R("bert")),
                ("Main corpus, TF-IDF", L("tfidf_logreg"), "tfidf-logreg", R("tfidf_logreg")),
                ("Second dataset, RoBERTa (3 repetitions)", L("salminen_roberta"), "roberta-base",
                 R("salminen_roberta")),
                ("Second dataset, TF-IDF (5 repetitions)", L("salminen_tfidf_sample"), "tfidf-logreg",
                 R("salminen_tfidf_sample"))]
    out = ["\n## Supplementary Table S12 (post-hoc diagnostic). Cost-derived rule after Platt scaling instead of temperature "
           "scaling, both fitted on the source-validation set\n",
           "Platt scaling fits a slope a and an intercept b to the logit of the validation probability. Mean over repetitions. "
           "The last column compares the cost-derived rule after Platt scaling with the cheapest validation-tuned rule "
           "(three-way, MSP or F1-tuned forced). Added after the main results were seen; temperature scaling is the "
           "pre-specified method.\n",
           "| Setting | Target | Slope a | Intercept b | ECE target: TS → Platt | Share predicted fake on target after Platt | "
           "S2 cost of cost-derived rule: TS → Platt | Platt version ≤ cheapest tuned rule (S1 / S2 / S3) |",
           "|---|---|---|---|---|---|---|---|"]
    for name, ld, model, rd in settings:
        m = platt_means(ld, model, os.path.join(rd, "results_per_repetition.json"))
        best = {s: m[[f"tuned_{s}", f"msp_{s}", f"f1forced_{s}"]].min(axis=1) for s in ["S1", "S2", "S3"]}
        if len(m) <= 3:
            for t, r in m.iterrows():
                le = " / ".join("yes" if r[f"chow_platt_{s}"] <= best[s][t] + 1e-9 else "no" for s in ["S1", "S2", "S3"])
                out.append(f"| {name} | {t} | {hu(r.a, 2)} | {hu(r.b, 2)} | {hu(r.ece_t_ts, 3)} → {hu(r.ece_t_platt, 3)} | "
                           f"{hu(r.share_t_platt, 3)} | {hu(r.chow_ts_S2, 3)} → {hu(r.chow_platt_S2, 3)} | {le} |")
        else:
            cnt = " / ".join(f"{int((m[f'chow_platt_{s}'] <= best[s] + 1e-9).sum())} of {len(m)}" for s in ["S1", "S2", "S3"])
            out.append(f"| {name} | all {len(m)} categories (range) | {rng(m.a, 2)} | {rng(m.b, 2)} | "
                       f"{rng(m.ece_t_ts)} → {rng(m.ece_t_platt)} | {rng(m.share_t_platt)} | "
                       f"{rng(m.chow_ts_S2)} → {rng(m.chow_platt_S2)} | {cnt} |")
    return out


def s13(results_root):
    res = lambda d: os.path.join(results_root, d)
    sets = [("Main corpus, RoBERTa ensemble", res("roberta")), ("Main corpus, BERT ensemble", res("bert")),
            ("Main corpus, TF-IDF", res("tfidf_logreg")),
            ("Second dataset, TF-IDF (5 repetitions)", res("salminen_tfidf_sample")),
            ("Second dataset, RoBERTa (3 repetitions)", res("salminen_roberta"))]
    rows = []
    for name, d in sets:
        R = json.load(open(os.path.join(d, "results_per_repetition.json")))
        for k, r in R.items():
            t = k.split(">")[1].split("|")[0]
            for s in ["S1", "S2", "S3"]:
                for rule, f in RULES.items():
                    rows.append({"set": name, "t": t, "s": s, "rule": rule, "cost": r["decisions"][f(s)][f"cost_{s}"]})
                rows.append({"set": name, "t": t, "s": s, "rule": "Entropy-adjusted score", "cost": r["decisions"]["ORIGINAL-SDF"][f"cost_{s}"]})
    D6 = pd.DataFrame(rows).groupby(["set", "t", "s", "rule"]).cost.mean().unstack("rule")
    init_cells = list(D6.index[D6.idxmin(axis=1) == "Entropy-adjusted score"])
    s2_with_init = float((100 * (D6["Cost-derived three-way"] / D6.min(axis=1) - 1)).xs("S2", level="s").max())
    D = D6[list(RULES)]
    ex = 100 * (D.div(D.min(axis=1), axis=0) - 1)
    out = ["\n## Supplementary Table S13 (descriptive). Excess expected cost of each rule over the cheapest of the five rules (%)\n",
           f"Each setting is one model–target combination under one cost scenario, using mean cost over repetitions "
           f"({len(D.xs('S2', level='s'))} combinations per scenario). Largest excess / median excess. "
           "Computed after the main results were seen. The entropy-adjusted score is excluded; it was the cheapest rule in "
           f"{len(init_cells)} main-corpus cells, all on restaurant reviews ("
           + "; ".join(f"{a.replace('Main corpus, ', '')} {c}" for a, b, c in init_cells)
           + "), by coincidence of the bias direction (Section 6.3); including it, the largest S2 excess of the cost-derived "
           f"rule would be {hu(s2_with_init, 1)}%.\n",
           "| Scenario | " + " | ".join(RULES) + " |", "|---|" + "---|" * len(RULES)]
    for s in ["S1", "S2", "S3"]:
        e = ex.xs(s, level="s")
        out.append(f"| {s} | " + " | ".join(f"{hu(e[c].max(), 1)} / {hu(e[c].median(), 1)}" for c in RULES) + " |")
    out.append("| All | " + " | ".join(f"{hu(ex[c].max(), 1)} / {hu(ex[c].median(), 1)}" for c in RULES) + " |")
    wins = D.idxmin(axis=1)
    out.append("\nNumber of settings in which each rule was the cheapest: " +
               "; ".join(f"{c} {int((wins == c).sum())}" for c in RULES) + f" (of {len(D)}).")
    worst = ex["Cost-derived three-way"].sort_values(ascending=False).head(3)
    out.append("Largest excess of the cost-derived rule: " +
               "; ".join(f"{a} / {b} / {c}: {hu(v, 1)}%" for (a, b, c), v in worst.items()) + ".")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits_root", default="logits", help="folder with the extracted logits: roberta, bert, "
                    "tfidf_logreg, salminen_roberta, salminen_tfidf_sample")
    ap.add_argument("--results_root", default="results", help="folder with the analysis outputs (same folder names)")
    ap.add_argument("--main_roberta_logits", default=None, help="RoBERTa logits of the main corpus "
                    "(default: LOGITS_ROOT/roberta)")
    ap.add_argument("--s11_md", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s11 = open(a.s11_md).read().replace(
        "# Additional dataset: Salminen et al. (2022), LODO over 10 product categories (make_tables_salminen.py)",
        "## Supplementary Table S11. Second dataset: RoBERTa (3 repetitions) and TF-IDF on the same repetitions")
    s11 = s11.replace("\n## ", "\n### ").replace("\n### Supplementary Table S11", "\n## Supplementary Table S11")
    s11 = s11.replace("### A.", "#### A.").replace("### B.", "#### B.").replace("### C.", "#### C.")
    md = ["# Supplementary Tables S11–S13 (generated by make_supp_tables_s11_s13.py)\n", s11]
    md += s12(a.logits_root, a.results_root, a.main_roberta_logits) + s13(a.results_root)
    open(a.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md[2:]))


if __name__ == "__main__":
    main()
