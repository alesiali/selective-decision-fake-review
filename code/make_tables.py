#!/usr/bin/env python3
"""
make_tables.py - builds the result tables of the article (markdown + one JSON of every number) from the
outputs of analyze_logits.py, plus the mechanism analysis of the entropy-adjusted score (comparison rule,
Section 3.5).

Usage (from the repository root):
  python code/make_tables.py --results results/roberta --logits logits/roberta --model roberta-base \
                             --results2 results/tfidf_logreg --logits2 logits/tfidf_logreg --model2 tfidf-logreg \
                             --out tables --name main_corpus_roberta_tfidf
Writes OUT/NAME.md and OUT/numbers_NAME.json (default NAME = tables: OUT/tables.md and OUT/numbers.json).
"""
from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_logits as A  # noqa: E402

TARGETS = ["doctor", "restaurant", "hotel"]
SC = ["S1", "S2", "S3"]


def ms(df, rule, metric, tgt):
    r = df[(df.test == tgt) & (df.rule == rule)]
    return float(r[f"{metric}_mean"].iloc[0]), float(r[f"{metric}_sd"].iloc[0])


def hu(x, d=3, sign=False):
    """Round half up and format with d decimals. The value is first rounded to 10 decimals, so exact ties
    (e.g. 0.7695 = 1539/2000) are not decided by floating-point noise."""
    v = Decimal(str(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{v:+f}" if sign else f"{v:f}"


def fmt(m, s=None, d=3):
    return hu(m, d) if s is None or np.isnan(s) else f"{hu(m, d)} ± {hu(s, d)}"


def mechanism(logits_dir, model):
    groups, _ = A.load_runs(logits_dir, model)
    grid = np.linspace(0, 1, 100001)
    adj = grid * (1 - A.binary_entropy(grid))
    rows = []
    for key, runs in sorted(groups.items()):
        g = A.ensemble_group(runs)
        pv, pt, yv, yt = g["pv_orig"], g["pt_orig"], g["yv"], g["yt"]
        lo, hi = A.original_sdf_thresholds(pv, yv)
        d = A.original_sdf_apply(pt, lo, hi)
        ab = d == -1
        missed = (d == 0) & (yt == 1)
        rows.append({
            "test": key[1], "rep": key[2], "low": lo, "high": hi,
            "deceptive_only_if_p_at_least": float(grid[adj >= hi].min()),
            "abstained_share": float(ab.mean()),
            "genuine_share_among_abstained": float((yt[ab] == 0).mean()) if ab.any() else np.nan,
            "labelled_genuine_with_p_0.24_0.66_share": float(((pt > 0.24) & (pt < 0.66) & (d == 0)).mean()),
            "missed_deceptive": int(missed.sum()),
            "median_p_missed": float(np.median(pt[missed])) if missed.any() else np.nan,
        })
    return pd.DataFrame(rows)


def build(results, logits, model, label):
    t = {n: pd.read_csv(os.path.join(results, f"table_{n}.csv")) for n in
         ["main", "calibration", "label_free", "risk_coverage", "sensitivity_cost", "sensitivity_mincov",
          "single_models"]}
    sub_path = os.path.join(results, "table_hotel_subgroups.csv")
    t["hotel_subgroups"] = pd.read_csv(sub_path) if os.path.exists(sub_path) else None
    mech = mechanism(logits, model)
    out, md = {}, [f"\n## Model: {label}\n"]

    # Table A: base classifier transfer
    md.append("### A. Base classifier on the target domain (forced decision at p = 0.5; mean ± SD over 5 repetitions)\n")
    md.append("| Target | Accuracy | Macro-F1 | Deceptive F1 | Deceptive recall | Share predicted deceptive (true) | ECE source val | ECE target |")
    md.append("|---|---|---|---|---|---|---|---|")
    true_rate = {"doctor": 355 / 555, "restaurant": 0.5, "hotel": 1079 / 1875}
    for tg in TARGETS:
        acc = ms(t["main"], "FORCED-0.5", "acc_accepted", tg)
        mf1 = ms(t["main"], "FORCED-0.5", "macro_f1", tg)
        f1d = ms(t["main"], "FORCED-0.5", "f1_dec", tg)
        rec = ms(t["main"], "FORCED-0.5", "recall_dec", tg)
        lf = t["label_free"][t["label_free"].test == tg].iloc[0]
        cal = t["calibration"][t["calibration"].test == tg].iloc[0]
        out[f"A_{tg}"] = {"acc": acc, "macro_f1": mf1, "f1_dec": f1d, "recall_dec": rec,
                          "share_pred_dec_test": lf["share_pred_deceptive_test_mean"],
                          "share_pred_dec_val": lf["share_pred_deceptive_val_mean"], "true_rate": true_rate[tg],
                          "ece_val": (cal["ece_val_ens_mean"], cal["ece_val_ens_sd"]),
                          "ece_test": (cal["ece_test_ens_mean"], cal["ece_test_ens_sd"]),
                          "T_ens": cal["T_ens_mean"]}
        md.append(f"| {tg} | {fmt(*acc)} | {fmt(*mf1)} | {fmt(*f1d)} | {fmt(*rec)} | "
                  f"{hu(lf['share_pred_deceptive_test_mean'], 3)} ({hu(true_rate[tg], 3)}) | "
                  f"{fmt(cal['ece_val_ens_mean'], cal['ece_val_ens_sd'])} | {fmt(cal['ece_test_ens_mean'], cal['ece_test_ens_sd'])} |")

    # Table B: decision rules per scenario
    rules = [("FORCED-F1", "Forced, F1-tuned"),
             ("FORCED-COST_{s}", "Forced, cost-optimal"),
             ("CHOW_{s}", "Cost-derived three-way"),
             ("CHOW-TUNED_{s}", "Three-way, validation-tuned"),
             ("MSP-TUNED_{s}", "Confidence (MSP = entropy), tuned"),
             ("DISAGREE-TUNED_{s}", "Disagreement, tuned"),
             ("ORIGINAL-SDF", "Entropy-adjusted score")]
    for s in SC:
        md.append(f"\n### B-{s}. Decision rules on the target domain, scenario {s} (mean ± SD over repetitions)\n")
        md.append("| Target | Rule | Coverage | Accepted risk | Deceptive F1 | Deceptive recall | Expected cost |")
        md.append("|---|---|---|---|---|---|---|")
        for tg in TARGETS:
            for r, name in rules:
                rr = r.format(s=s)
                if rr.startswith("DISAGREE") and model == "tfidf-logreg":
                    continue
                vals = {m: ms(t["main"], rr, m, tg) for m in
                        ["coverage", "risk_accepted", "f1_dec", "recall_dec", f"cost_{s}"]}
                out[f"B_{s}_{tg}_{rr}"] = vals
                md.append(f"| {tg} | {name} | {fmt(*vals['coverage'])} | {fmt(*vals['risk_accepted'])} | "
                          f"{fmt(*vals['f1_dec'])} | {fmt(*vals['recall_dec'])} | {fmt(*vals[f'cost_{s}'])} |")
    # relative reductions
    md.append("\n### C. Relative change in expected cost of the cost-derived rule (CHOW)\n")
    md.append("| Target | Cost reduction vs cost-optimal forced, S1 / S2 / S3 (%) | Cost reduction vs F1-tuned forced, S1 / S2 / S3 (%) | CHOW coverage S1 / S2 / S3 |")
    md.append("|---|---|---|---|")
    # full-precision means from results_per_repetition.json (table_main.csv holds 4-decimal means,
    # which can shift the last digit of a percentage; fixed 2026-10-15)
    _R = json.load(open(os.path.join(results, "results_per_repetition.json")))

    def fm(rule, metric, tg):
        return float(np.mean([v["decisions"][rule][metric] for k, v in _R.items()
                              if k.split(">")[1].split("|")[0] == tg]))

    for tg in TARGETS:
        r1, r2, cv = [], [], []
        for s in SC:
            c = fm(f"CHOW_{s}", f"cost_{s}", tg)
            fc = fm(f"FORCED-COST_{s}", f"cost_{s}", tg)
            ff = fm("FORCED-F1", f"cost_{s}", tg)
            r1.append(100 * (fc - c) / fc)
            r2.append(100 * (ff - c) / ff)
            cv.append(ms(t["main"], f"CHOW_{s}", "coverage", tg)[0])
        out[f"C_{tg}"] = {"vs_cost_opt_pct": r1, "vs_f1_tuned_pct": r2, "coverage": cv}
        md.append(f"| {tg} | {' / '.join(f'{hu(x, 1, True)}' for x in r1)} | {' / '.join(f'{hu(x, 1, True)}' for x in r2)} | "
                  f"{' / '.join(f'{hu(x, 3)}' for x in cv)} |")
    md.append("\n(Positive = the cost-derived rule has lower expected cost than the comparator; negative = higher.)")

    # Matched coverage
    md.append("\n### D. Same coverage, different selection (S2)\n")
    md.append("| Target | Entropy-adjusted score: coverage / risk / deceptive recall / cost | MSP at the same coverage: risk / deceptive recall / cost |")
    md.append("|---|---|---|")
    for tg in TARGETS:
        o = {m: ms(t["main"], "ORIGINAL-SDF", m, tg)[0] for m in ["coverage", "risk_accepted", "recall_dec", "cost_S2"]}
        p = {m: ms(t["main"], "MSP@ORIGINAL-COVERAGE", m, tg)[0] for m in ["risk_accepted", "recall_dec", "cost_S2"]}
        out[f"D_{tg}"] = {"original": o, "msp_same_cov": p}
        md.append(f"| {tg} | {hu(o['coverage'], 3)} / {hu(o['risk_accepted'], 3)} / {hu(o['recall_dec'], 3)} / {hu(o['cost_S2'], 3)} | "
                  f"{hu(p['risk_accepted'], 3)} / {hu(p['recall_dec'], 3)} / {hu(p['cost_S2'], 3)} |")

    # Risk-coverage
    md.append("\n### E. Risk–coverage (prediction at p = 0.5)\n")
    md.append("| Target | Score | AURC | E-AURC | Risk @ 50% coverage | Risk @ 80% | Risk @ 100% |")
    md.append("|---|---|---|---|---|---|---|")
    for tg in TARGETS:
        for sc in ["MSP", "ENTROPY", "DISAGREEMENT"]:
            if sc == "DISAGREEMENT" and model == "tfidf-logreg":
                continue
            r = t["risk_coverage"][(t["risk_coverage"].test == tg) & (t["risk_coverage"].score == sc)].iloc[0]
            out[f"E_{tg}_{sc}"] = {k: float(r[k]) for k in ["AURC_mean", "AURC_sd", "E-AURC_mean", "risk@0.5", "risk@0.8", "risk@1.0"]}
            md.append(f"| {tg} | {sc} | {fmt(r['AURC_mean'], r['AURC_sd'])} | {hu(r['E-AURC_mean'], 3)} | "
                      f"{hu(r['risk@0.5'], 3)} | {hu(r['risk@0.8'], 3)} | {hu(r['risk@1.0'], 3)} |")
    ident = bool(t["calibration"]["msp_entropy_identical_all_reps"].all())
    out["msp_entropy_identical"] = ident
    md.append(f"\nMSP and entropy rankings identical in every repetition: {ident}")

    # Sensitivity
    md.append("\n### F. Cost-ratio sensitivity: % reduction in mean expected cost of CHOW vs cost-optimal forced (C_false = 1)\n")
    sens = t["sensitivity_cost"]
    for tg in TARGETS:
        st = sens[sens.test == tg].copy()
        # reduction in mean expected cost over the repetitions (the measure of Table C and of the text)
        st["reduction_of_mean_pct"] = 100 * (st.cost_forced_mean - st.cost_chow_mean) / st.cost_forced_mean
        pv = st.pivot(index="C_miss", columns="C_reject", values="reduction_of_mean_pct")
        out[f"F_{tg}"] = {"min": float(pv.values.min()), "max": float(pv.values.max())}
        md.append(f"\n{tg} (rows C_miss, columns C_reject):\n")
        md.append("| C_miss | " + " | ".join(str(c) for c in pv.columns) + " |")
        md.append("|---|" + "---|" * len(pv.columns))
        for i, row in pv.iterrows():
            md.append(f"| {i} | " + " | ".join(f"{hu(v, 1)}" for v in row.values) + " |")
    mc = t["sensitivity_mincov"]
    md.append("\n### G. Minimum-coverage sensitivity (validation-tuned variants, S2 cost)\n")
    md.append("| Target | Min coverage | Dual thresholds (F1-tuned): coverage / deceptive F1 / cost | Entropy-adjusted score: coverage / deceptive F1 / cost |")
    md.append("|---|---|---|---|")
    for tg in TARGETS:
        for cmin in sorted(mc.min_coverage.unique()):
            a = mc[(mc.test == tg) & (mc.min_coverage == cmin) & (mc.variant == "DUAL-F1")].iloc[0]
            b = mc[(mc.test == tg) & (mc.min_coverage == cmin) & (mc.variant == "ORIGINAL-SDF")].iloc[0]
            md.append(f"| {tg} | {cmin} | {hu(a['coverage_mean'], 3)} / {hu(a['f1_dec_mean'], 3)} / {hu(a['cost_S2_mean'], 3)} | "
                      f"{hu(b['coverage_mean'], 3)} / {hu(b['f1_dec_mean'], 3)} / {hu(b['cost_S2_mean'], 3)} |")

    # Label-free
    md.append("\n### H. Label-free indicators (no target labels used to compute them)\n")
    md.append("| Target | Share predicted deceptive: source val → target | CHOW-S2 abstention: val → target | DoC predicted accuracy | ATC predicted accuracy | Actual accuracy |")
    md.append("|---|---|---|---|---|---|")
    for tg in TARGETS:
        r = t["label_free"][t["label_free"].test == tg].iloc[0]
        out[f"H_{tg}"] = {k: float(r[k]) for k in r.index if k.endswith("_mean")}
        md.append(f"| {tg} | {hu(r['share_pred_deceptive_val_mean'], 3)} → {hu(r['share_pred_deceptive_test_mean'], 3)} | "
                  f"{hu(r['chow_S2_abstain_rate_val_mean'], 3)} → {hu(r['chow_S2_abstain_rate_test_mean'], 3)} | "
                  f"{hu(r['DoC_predicted_test_accuracy_mean'], 3)} | {hu(r['ATC_predicted_test_accuracy_mean'], 3)} | "
                  f"{hu(r['actual_test_accuracy_mean'], 3)} |")

    # Single models / variability
    sm = t["single_models"]
    md.append("\n### I. Single-model variability (all retained runs)\n")
    md.append("| Target | Runs | Temperature T | Val macro-F1 | Target macro-F1 at 0.5 | Target ECE raw → scaled |")
    md.append("|---|---|---|---|---|---|")
    for tg in TARGETS:
        s_ = sm[sm.test == tg]
        out[f"I_{tg}"] = {"n": int(len(s_)), "T": (s_["T"].mean(), s_["T"].std()),
                          "val_macro_f1": (s_["val_macro_f1"].mean(), s_["val_macro_f1"].std()),
                          "test_macro_f1": (s_["test_macro_f1_at_0.5"].mean(), s_["test_macro_f1_at_0.5"].std())}
        md.append(f"| {tg} | {len(s_)} | {fmt(s_['T'].mean(), s_['T'].std(), 2)} | "
                  f"{fmt(s_['val_macro_f1'].mean(), s_['val_macro_f1'].std())} | "
                  f"{fmt(s_['test_macro_f1_at_0.5'].mean(), s_['test_macro_f1_at_0.5'].std())} | "
                  f"{hu(s_['ece_test_raw'].mean(), 3)} → {hu(s_['ece_test_ts'].mean(), 3)} |")

    # Mechanism
    md.append("\n### J. Mechanism of the entropy-adjusted score, per repetition\n")
    md.append("| Target | Rep | Deceptive only if p ≥ | Abstained | Genuine among abstained | Labeled genuine with 0.24 < p < 0.66 | Missed deceptive | Median p of missed |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in mech.iterrows():
        md.append(f"| {r['test']} | {r['rep']} | {hu(r['deceptive_only_if_p_at_least'], 3)} | {hu(r['abstained_share'], 2)} | "
                  f"{hu(r['genuine_share_among_abstained'], 2)} | {hu(r['labelled_genuine_with_p_0.24_0.66_share'], 2)} | "
                  f"{r['missed_deceptive']} | {hu(r['median_p_missed'], 2)} |")
    out["J_mechanism"] = mech.to_dict(orient="records")

    # Hotel subgroups
    if t["hotel_subgroups"] is not None:
        md.append("\n### K. Hotel subgroups (writer/polarity labels from the file's block order, boundaries checked by reading; "
                  "hotel_meta_v2.csv; expert = employee-written)\n")
        md.append("| Writer | Polarity | n | Label | Error at 0.5 | Mean p(deceptive) | CHOW-S2 abstention | CHOW-S2 error on accepted |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, r in t["hotel_subgroups"].iterrows():
            md.append(f"| {r['writer']} | {r['polarity']} | {r['n']} | {'deceptive' if r['label'] == 1 else 'genuine'} | "
                      f"{hu(r['error_at_0.5_mean'], 3)} | {hu(r['mean_p_deceptive_mean'], 3)} | {hu(r['chow_S2_abstain_rate_mean'], 3)} | "
                      f"{hu(r['chow_S2_error_accepted_mean'], 3)} |")
        out["K_hotel_subgroups"] = t["hotel_subgroups"].to_dict(orient="records")

    # ablation without ensemble-level scaling
    md.append("\n### L. Ablation: without ensemble-level temperature scaling (S2 cost)\n")
    md.append("| Target | CHOW | CHOW without ensemble scaling | Cost-optimal forced | Same without ensemble scaling |")
    md.append("|---|---|---|---|---|")
    for tg in TARGETS:
        a = ms(t["main"], "CHOW_S2", "cost_S2", tg)[0]
        b = ms(t["main"], "CHOW-NOENSCAL_S2", "cost_S2", tg)[0]
        c = ms(t["main"], "FORCED-COST_S2", "cost_S2", tg)[0]
        d = ms(t["main"], "FORCED-COST-NOENSCAL_S2", "cost_S2", tg)[0]
        md.append(f"| {tg} | {hu(a, 3)} | {hu(b, 3)} | {hu(c, 3)} | {hu(d, 3)} |")
    return out, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--results2")
    ap.add_argument("--logits2")
    ap.add_argument("--model2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="tables", help="output names: OUT/NAME.md and OUT/numbers_NAME.json "
                    "(default: OUT/tables.md and OUT/numbers.json)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allnum, allmd = {}, ["# Result tables (generated by make_tables.py)\n"]
    o, m = build(a.results, a.logits, a.model, a.model)
    allnum[a.model], allmd = o, allmd + m
    if a.results2:
        o2, m2 = build(a.results2, a.logits2, a.model2, a.model2)
        allnum[a.model2], allmd = o2, allmd + m2
    md_name = f"{a.name}.md"
    num_name = "numbers.json" if a.name == "tables" else f"numbers_{a.name}.json"
    with open(os.path.join(a.out, md_name), "w") as f:
        f.write("\n".join(allmd) + "\n")
    with open(os.path.join(a.out, num_name), "w") as f:
        json.dump(allnum, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print("written", os.path.join(a.out, md_name))


if __name__ == "__main__":
    main()
