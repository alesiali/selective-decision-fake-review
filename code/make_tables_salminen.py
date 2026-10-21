#!/usr/bin/env python3
"""
make_tables_salminen.py - summary tables for the additional dataset (Salminen et al. 2022)

Reads the results_per_repetition.json written by analyze_logits.py for the 10 product
categories and writes a Markdown file and a JSON file with per-category results.

Usage (from the repository root):
  python code/make_tables_salminen.py --results results/salminen_tfidf_sample --label "TF-IDF + LR" \
         [--results2 results/salminen_tfidf_all --label2 "TF-IDF + LR, all source reviews"] \
         [--reps 0 1 2] [--same_targets] --out tables --name NAME
Writes OUT/NAME.md and OUT/numbers_NAME.json.
"""
from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
import json
import os

import numpy as np
import pandas as pd

DOMAINS = ["books", "clothing", "electronics", "home", "kindle", "movies", "pets", "sports", "tools", "toys"]
SCEN = ["S1", "S2", "S3"]


def hu(x, d=3, sign=False):
    """Round half up and format with d decimals. The value is first rounded to 10 decimals, so exact ties
    are not decided by floating-point noise (same rule as make_tables.py)."""
    v = Decimal(str(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{v:+f}" if sign else f"{v:f}"


def load(results_dir, reps=None, targets=None):
    d = json.load(open(os.path.join(results_dir, "results_per_repetition.json")))
    by = {}
    for k, v in d.items():
        tgt = k.split(">")[1].split("|")[0]
        r = int(k.split("|rep")[1])
        if (reps is None or r in reps) and (targets is None or tgt in targets):
            by.setdefault(tgt, []).append(v)
    return by


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{hu(x.mean(), nd)} ± {hu(x.std(ddof=1), nd)}" if len(x) > 1 else hu(x.mean(), nd)


def summarise(by):
    rows = []
    for t in [d for d in DOMAINS if d in by]:
        reps = by[t]
        g = lambda f: [f(r) for r in reps]
        row = {"target": t, "reps": len(reps)}
        row["acc"] = g(lambda r: r["label_free"]["actual_test_accuracy"])
        row["share_val"] = g(lambda r: r["label_free"]["share_pred_deceptive_val"])
        row["share_test"] = g(lambda r: r["label_free"]["share_pred_deceptive_test"])
        row["ece_val"] = g(lambda r: r["calibration"]["ece_val_ens"])
        row["ece_test"] = g(lambda r: r["calibration"]["ece_test_ens"])
        row["aurc"] = g(lambda r: r["rc"]["MSP"]["AURC"])
        row["defer_val"] = g(lambda r: r["label_free"]["chow_S2_abstain_rate_val"])
        row["defer_test"] = g(lambda r: r["label_free"]["chow_S2_abstain_rate_test"])
        for s in SCEN:
            dec = lambda r, rule: r["decisions"][rule][f"cost_{s}"]
            row[f"chow_{s}"] = g(lambda r: dec(r, f"CHOW_{s}"))
            row[f"forced_{s}"] = g(lambda r: dec(r, f"FORCED-COST_{s}"))
            row[f"f1forced_{s}"] = g(lambda r: dec(r, "FORCED-F1"))
            row[f"tuned_{s}"] = g(lambda r: dec(r, f"CHOW-TUNED_{s}"))
            row[f"msp_{s}"] = g(lambda r: dec(r, f"MSP-TUNED_{s}"))
            row[f"reviewed_{s}"] = g(lambda r: dec(r, "ORIGINAL-SDF"))
            row[f"cov_{s}"] = g(lambda r: r["decisions"][f"CHOW_{s}"]["coverage"])
            row[f"recall_{s}"] = g(lambda r: r["decisions"][f"CHOW_{s}"]["recall_dec"])
        sens = [pd.DataFrame(r["sensitivity_cost"]) for r in reps]
        grid = pd.concat(sens).groupby(["C_miss", "C_reject"])[["cost_chow", "cost_forced"]].mean()
        grid["red"] = 100 * (grid.cost_forced - grid.cost_chow) / grid.cost_forced
        row["grid_min_red"], row["grid_max_red"] = float(grid.red.min()), float(grid.red.max())
        rows.append(row)
    return rows


def md_tables(rows, label):
    m = lambda v: float(np.mean(v))
    out = [f"\n## {label}\n",
           "### A. Base classifier on the target category (prediction at p = 0.5; mean ± SD over repetitions)\n",
           "| Target | Accuracy | Share predicted fake: val → target (true 0.50) | ECE val → target | AURC (MSP) |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['target']} | {ms(r['acc'])} | {hu(m(r['share_val']), 3)} → {hu(m(r['share_test']), 3)} | "
                   f"{hu(m(r['ece_val']), 3)} → {hu(m(r['ece_test']), 3)} | {ms(r['aurc'])} |")
    out += ["\n### B. Expected cost by rule (mean over repetitions), S1 / S2 / S3\n",
            "| Target | Forced, F1-tuned | Forced, cost-optimal | Cost-derived three-way | Three-way, tuned | MSP, tuned | Entropy-adjusted score |",
            "|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda k: " / ".join(hu(m(r[f'{k}_{s}'])) for s in SCEN)
        out.append(f"| {r['target']} | {f('f1forced')} | {f('forced')} | **{f('chow')}** | {f('tuned')} | {f('msp')} | {f('reviewed')} |")
    out += ["\n### C. Cost-derived rule: reduction vs cost-optimal forced (%), coverage and deceptive recall\n",
            "| Target | Reduction S1 / S2 / S3 | Coverage S1 / S2 / S3 | Fake recall S1 / S2 | Deferral rate S2: val → target | 28-setting grid: min / max reduction |",
            "|---|---|---|---|---|---|"]
    for r in rows:
        red = " / ".join(hu(100 * (m(r[f'forced_{s}']) - m(r[f'chow_{s}'])) / m(r[f'forced_{s}']), 1, True) for s in SCEN)
        cov = " / ".join(hu(m(r[f'cov_{s}'])) for s in SCEN)
        rec = " / ".join(hu(m(r[f'recall_{s}'])) for s in SCEN[:2])
        out.append(f"| {r['target']} | {red} | {cov} | {rec} | {hu(m(r['defer_val']), 3)} → {hu(m(r['defer_test']), 3)} | "
                   f"{hu(r['grid_min_red'], 1, True)} / {hu(r['grid_max_red'], 1, True)} |")
    # which rule is cheapest
    rules = ["f1forced", "forced", "chow", "tuned", "msp", "reviewed"]
    wins = {s: sum(min(rules, key=lambda k: m(r[f"{k}_{s}"])) == "chow" for r in rows) for s in SCEN}
    beats_all_tuned = {s: sum(m(r[f"chow_{s}"]) <= min(m(r[f"tuned_{s}"]), m(r[f"msp_{s}"]), m(r[f"f1forced_{s}"]))
                              for r in rows) for s in SCEN}
    out.append(f"\nCost-derived rule is the cheapest of the six rules in {wins['S1']} / {wins['S2']} / {wins['S3']} "
               f"of {len(rows)} categories (S1 / S2 / S3); cheaper than or equal to every validation-tuned rule in "
               f"{beats_all_tuned['S1']} / {beats_all_tuned['S2']} / {beats_all_tuned['S3']}.")
    return out, {"cheapest_counts": wins, "beats_all_tuned_counts": beats_all_tuned}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--results2")
    ap.add_argument("--label2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", nargs="+", type=int, default=None, help="only these repetitions")
    ap.add_argument("--same_targets", action="store_true",
                    help="restrict both results to the categories present in --results")
    ap.add_argument("--name", default="tables_salminen")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    md, js = ["# Additional dataset: Salminen et al. (2022), LODO over 10 product categories (make_tables_salminen.py)"], {}
    first = load(args.results, args.reps)
    targets = sorted(first) if args.same_targets else None
    for res, lab in [(args.results, args.label), (args.results2, args.label2)]:
        if not res:
            continue
        rows = summarise(load(res, args.reps, targets))
        t, extra = md_tables(rows, lab)
        md += t
        js[lab] = {"rows": [{k: (np.round(np.asarray(v, float), 5).tolist() if isinstance(v, list) else v)
                             for k, v in r.items()} for r in rows], **extra}
    open(os.path.join(args.out, f"{args.name}.md"), "w").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(args.out, f"numbers_{args.name}.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
