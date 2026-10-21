#!/usr/bin/env python3
"""
analyze_logits.py - step 2: the decision-layer analysis of the saved logits

Reads the .npz files written by train_save_logits.py (train_classical.py and train_salminen.py write
the same format) and computes every decision-layer result of the article. No training; CPU only.

For each LODO split and repetition (three models trained with different seeds):
  1. temperature scaling of each model on the SOURCE validation set (NLL);
  2. ensemble = average of the calibrated probabilities, weighted by each
     model's validation macro-F1; the ensemble is then temperature-scaled once more on the same set;
  3. decision rules applied to the TARGET domain. Target labels are used only
     to score the decisions, never to choose anything.

Decision rules (thresholds come from the costs alone, or from source validation); the names are the
keys used in results_per_repetition.json and table_main.csv:
  FORCED-0.5      deceptive if p >= 0.5
  FORCED-F1       one threshold maximising validation macro-F1 (forced, F1-tuned)
  FORCED-COST     cost-optimal threshold without abstention, t = C_false/(C_false + C_miss)
                  (Elkan 2001)
  CHOW            three-way rule with cost-derived thresholds (Chow 1970; Yao 2010):
                  genuine if p <= C_reject/C_miss, deceptive if p >= 1 - C_reject/C_false,
                  abstain otherwise. Needs no labels at all.
  CHOW-TUNED      two thresholds tuned on validation to minimise validation expected cost
  MSP-TUNED       abstain if max(p, 1-p) < c, otherwise predict at 0.5; c tuned on
                  validation (binary entropy gives exactly the same rule)
  DISAGREE-TUNED  abstain if the std of the three models' probabilities > d, otherwise
                  predict at 0.5; d tuned on validation
  ORIGINAL-SDF    entropy-adjusted score (comparison rule, Section 3.5): score p * (1 - H(p)),
                  two thresholds on a grid 0.05-0.95 step 0.025 maximising accepted macro-F1
                  subject to coverage >= 0.55

Also: risk-coverage curves with AURC / E-AURC, calibration (ECE), sensitivity to cost
ratios and to the minimum-coverage constraint, variability over repetitions,
label-free shift indicators, and Hotel subgroup analysis (with --hotel_meta).

Usage:
  python analyze_logits.py --logits_dir LOGITS --out_dir RESULTS [--model_name roberta-base] \
                           [--hotel_meta data/hotel_meta_v2.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, precision_score, recall_score

SCENARIOS = {  # C_miss : C_false : C_reject (Table 1 of the article)
    "S1": {"C_miss": 3.0, "C_false": 1.0, "C_reject": 0.5},
    "S2": {"C_miss": 8.0, "C_false": 1.0, "C_reject": 0.5},
    "S3": {"C_miss": 1.5, "C_false": 1.0, "C_reject": 0.5},
}
TARGET_ORDER = ["doctor", "restaurant", "hotel"]
ORIG_GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)  # grid of the F1-tuned rules
FINE_GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)
COVERAGE_POINTS = np.round(np.arange(0.1, 1.0001, 0.1), 1)
SENS_CMISS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
SENS_CREJECT = [0.1, 0.25, 0.5, 0.75]
MIN_COVERAGES = [0.5, 0.55, 0.6, 0.7, 0.8, 0.9]


# =========================================================== basic pieces
def softmax_pos(logits, T=1.0):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def logit(p, eps=1e-7):
    q = np.clip(p, eps, 1 - eps)
    return np.log(q) - np.log(1 - q)


def nll(logits, y, T):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def fit_temperature(logits, y):
    res = minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.05, 10.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def ece(p, y, n_bins=10):
    """Top-label expected calibration error with equal-width bins."""
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= hi))
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def binary_entropy(p, eps=1e-12):
    q = np.clip(p, eps, 1 - eps)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q)) / np.log(2)


def chow_thresholds(c):
    """(beta, alpha): genuine if p <= beta, deceptive if p >= alpha, abstain between.
    If the abstain region is empty, both equal the cost-optimal forced threshold."""
    beta = c["C_reject"] / c["C_miss"]
    alpha = 1.0 - c["C_reject"] / c["C_false"]
    if beta >= alpha:
        t = c["C_false"] / (c["C_false"] + c["C_miss"])
        return t, t
    return beta, alpha


def three_way(p, beta, alpha):
    """1 = deceptive, 0 = genuine, -1 = abstain. If beta == alpha it is a forced rule."""
    if beta >= alpha:  # empty abstain region: plain forced rule at alpha
        return (p >= alpha).astype(int)
    out = np.full(len(p), -1, dtype=int)
    out[p >= alpha] = 1
    out[p <= beta] = 0
    return out


def expected_cost(y, d, c):
    fn = np.sum((d == 0) & (y == 1))
    fp = np.sum((d == 1) & (y == 0))
    ab = np.sum(d == -1)
    return float((c["C_miss"] * fn + c["C_false"] * fp + c["C_reject"] * ab) / len(y))


def decision_metrics(y, d, scenarios=SCENARIOS):
    acc = d != -1
    out = {"coverage": float(acc.mean())}
    if acc.sum() > 0:
        ya, da = y[acc], d[acc]
        out.update({
            "acc_accepted": float((ya == da).mean()),
            "risk_accepted": float((ya != da).mean()),
            "f1_dec": float(f1_score(ya, da, pos_label=1, zero_division=0)),
            "recall_dec": float(recall_score(ya, da, pos_label=1, zero_division=0)),
            "precision_dec": float(precision_score(ya, da, pos_label=1, zero_division=0)),
            "macro_f1": float(f1_score(ya, da, average="macro", zero_division=0)),
        })
    else:
        out.update({k: np.nan for k in ["acc_accepted", "risk_accepted", "f1_dec", "recall_dec",
                                        "precision_dec", "macro_f1"]})
    for s, c in scenarios.items():
        out[f"cost_{s}"] = expected_cost(y, d, c)
    return out


# ================================================== threshold tuning (validation only)
def tune_forced_f1(p, y, grid=ORIG_GRID):
    best_t, best = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f > best:
            best, best_t = f, float(t)
    return best_t


def tune_dual_cost(p, y, c, grid=FINE_GRID):
    """Two thresholds beta < alpha minimising validation expected cost (ties: higher coverage)."""
    best = None
    for i, b in enumerate(grid):
        for a in grid[i + 1:]:
            d = three_way(p, b, a)
            key = (expected_cost(y, d, c), -float((d != -1).mean()))
            if best is None or key < best[0]:
                best = (key, float(b), float(a))
    return best[1], best[2]


def score_rule(score, threshold, p):
    """Abstain where score < threshold; otherwise predict at 0.5."""
    d = (p >= 0.5).astype(int)
    d[score < threshold] = -1
    return d


def tune_score_cost(score, p, y, c):
    cands = np.unique(np.concatenate([[-np.inf], score, [np.inf]]))
    best = None
    for t in cands:
        d = score_rule(score, t, p)
        key = (expected_cost(y, d, c), -float((d != -1).mean()))
        if best is None or key < best[0]:
            best = (key, float(t))
    return best[1]


def original_sdf_thresholds(p_val, y_val, min_coverage=0.55, grid=ORIG_GRID):
    """Entropy-adjusted score p * (1 - H(p)) (comparison rule, Section 3.5): maximise accepted macro-F1."""
    adj = p_val * (1 - binary_entropy(p_val))
    best = (-1.0, 0.3, 0.7)
    for low in grid:
        for high in grid:
            if high <= low:
                continue
            d = np.full(len(adj), -1, dtype=int)
            d[adj >= high] = 1
            d[adj <= low] = 0
            m = d != -1
            if m.mean() < min_coverage or m.sum() == 0:
                continue
            f = f1_score(y_val[m], d[m], average="macro")
            if f > best[0]:
                best = (f, float(low), float(high))
    return best[1], best[2]


def original_sdf_apply(p, low, high):
    adj = p * (1 - binary_entropy(p))
    d = np.full(len(adj), -1, dtype=int)
    d[adj >= high] = 1
    d[adj <= low] = 0
    return d


def tune_dual_f1_mincov(p, y, min_coverage, grid=ORIG_GRID):
    """Two thresholds on p maximising accepted macro-F1 subject to coverage >= min_coverage."""
    best = (-1.0, 0.5, 0.5)
    for i, b in enumerate(grid):
        for a in grid[i:]:
            d = three_way(p, b, a)
            m = d != -1
            if m.mean() < min_coverage or m.sum() == 0:
                continue
            f = f1_score(y[m], d[m], average="macro", zero_division=0)
            if f > best[0]:
                best = (f, float(b), float(a))
    return best[1], best[2]


# ======================================================= risk-coverage
def rc_curve(conf, correct):
    order = np.argsort(-conf, kind="mergesort")
    err = 1.0 - correct[order]
    k = np.arange(1, len(err) + 1)
    risk = np.cumsum(err) / k
    return k / len(err), risk


def aurc_eaurc(conf, correct):
    cov, risk = rc_curve(conf, correct)
    aurc = float(risk.mean())
    n, n_correct = len(correct), int(correct.sum())
    k = np.arange(1, n + 1)
    risk_opt = np.maximum(0, k - n_correct) / k
    return aurc, aurc - float(risk_opt.mean())


def risk_at_coverage(conf, correct, cov_points=COVERAGE_POINTS):
    cov, risk = rc_curve(conf, correct)
    return {f"{c:.1f}": float(risk[max(int(np.ceil(c * len(risk))) - 1, 0)]) for c in cov_points}


# ============================================================ loading
def is_degenerate(run):
    """Validation-only failure check: the model predicts one class for every validation review
    (a known instability of fine-tuning on small data). Uses no target information."""
    pred = (run["val_logits"][:, 1] > run["val_logits"][:, 0]).astype(int)
    return pred.min() == pred.max()


def load_runs(logits_dir, model_name, exclude_degenerate=True):
    groups, excluded = defaultdict(list), []
    for f in sorted(glob.glob(os.path.join(logits_dir, "*.npz"))):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta["model_name"] != model_name:
            continue
        key = (tuple(meta["train_domains"]), meta["test_domain"], int(meta["repetition"]))
        run = {"file": os.path.basename(f), "meta": meta, **{k: z[k] for k in z.files if k != "meta"}}
        if exclude_degenerate and is_degenerate(run):
            excluded.append(run["file"])
            continue
        groups[key].append(run)
    for key, runs in groups.items():
        runs.sort(key=lambda r: r["meta"]["model_seed"])
        ids0 = runs[0]["val_ids"]
        for r in runs[1:]:
            if not np.array_equal(r["val_ids"], ids0):
                raise ValueError(f"Validation ids differ inside {key}")
            if not np.array_equal(r["test_row"], runs[0]["test_row"]):
                raise ValueError(f"Test rows differ inside {key}")
    return dict(groups), excluded


def ensemble_group(runs):
    yv = runs[0]["val_labels"].astype(int)
    yt = runs[0]["test_labels"].astype(int)
    pv_list, pt_list, f1s, temps, singles = [], [], [], [], []
    for r in runs:
        T = fit_temperature(r["val_logits"], yv)
        pv_raw, pt_raw = softmax_pos(r["val_logits"]), softmax_pos(r["test_logits"])
        pv, pt = softmax_pos(r["val_logits"], T), softmax_pos(r["test_logits"], T)
        f1v = f1_score(yv, (pv >= 0.5).astype(int), average="macro")
        pv_list.append(pv), pt_list.append(pt), f1s.append(f1v), temps.append(T)
        singles.append({
            "seed": r["meta"]["model_seed"], "T": T, "val_macro_f1": float(f1v),
            "test_macro_f1_at_0.5": float(f1_score(yt, (pt >= 0.5).astype(int), average="macro")),
            "test_f1_dec_at_0.5": float(f1_score(yt, (pt >= 0.5).astype(int), pos_label=1)),
            "ece_val_raw": ece(pv_raw, yv), "ece_val_ts": ece(pv, yv),
            "ece_test_raw": ece(pt_raw, yt), "ece_test_ts": ece(pt, yt)})
    w = np.asarray(f1s) / np.sum(f1s)
    pv_avg = np.tensordot(w, np.vstack(pv_list), axes=(0, 0))  # ensemble average (no ensemble-level scaling)
    pt_avg = np.tensordot(w, np.vstack(pt_list), axes=(0, 0))
    # Averaging calibrated members can leave the ensemble mis-calibrated (usually under-confident),
    # and the cost-derived thresholds assume calibrated probabilities. So the ensemble itself is
    # temperature-scaled once more on the same source-validation set.
    lv, lt = logit(pv_avg), logit(pt_avg)
    zeros_v, zeros_t = np.zeros_like(lv), np.zeros_like(lt)
    T_ens = fit_temperature(np.column_stack([zeros_v, lv]), yv)
    pv_ens = softmax_pos(np.column_stack([zeros_v, lv]), T_ens)
    pt_ens = softmax_pos(np.column_stack([zeros_t, lt]), T_ens)
    dis_v = np.vstack(pv_list).std(axis=0)
    dis_t = np.vstack(pt_list).std(axis=0)
    return {"yv": yv, "yt": yt, "pv": pv_ens, "pt": pt_ens, "pv_orig": pv_avg, "pt_orig": pt_avg,
            "dis_v": dis_v, "dis_t": dis_t, "T_ens": T_ens,
            "weights": w.tolist(), "temperatures": temps, "singles": singles,
            "test_row": runs[0]["test_row"]}


# ===================================================== per-group analysis
def analyse_group(g):
    yv, yt, pv, pt = g["yv"], g["yt"], g["pv"], g["pt"]
    pv_o, pt_o = g["pv_orig"], g["pt_orig"]  # ensemble average without the ensemble-level scaling
    res = {"decisions": {}, "thresholds": {}}
    d_by_rule = {}

    # forced baselines
    d_by_rule["FORCED-0.5"] = (pt >= 0.5).astype(int)
    t_f1 = tune_forced_f1(pv_o, yv)  # forced, F1-tuned (comparison rule)
    res["thresholds"]["FORCED-F1"] = t_f1
    d_by_rule["FORCED-F1"] = (pt_o >= t_f1).astype(int)

    # entropy-adjusted score (comparison rule, Section 3.5)
    lo, hi = original_sdf_thresholds(pv_o, yv)
    res["thresholds"]["ORIGINAL-SDF"] = (lo, hi)
    d_by_rule["ORIGINAL-SDF"] = original_sdf_apply(pt_o, lo, hi)

    for r, d in d_by_rule.items():
        res["decisions"][r] = decision_metrics(yt, d)

    # cost-dependent rules, per scenario
    msp_v, msp_t = np.maximum(pv, 1 - pv), np.maximum(pt, 1 - pt)
    for s, c in SCENARIOS.items():
        t_cost = c["C_false"] / (c["C_false"] + c["C_miss"])
        b, a = chow_thresholds(c)
        bt, at = tune_dual_cost(pv, yv, c)
        tm = tune_score_cost(msp_v, pv, yv, c)
        td = tune_score_cost(-g["dis_v"], pv, yv, c)
        rules = {
            "FORCED-COST": (pt >= t_cost).astype(int),
            "CHOW": three_way(pt, b, a),
            # same two rules WITHOUT the ensemble-level temperature scaling (to show its effect)
            "FORCED-COST-NOENSCAL": (pt_o >= t_cost).astype(int),
            "CHOW-NOENSCAL": three_way(pt_o, b, a),
            "CHOW-TUNED": three_way(pt, bt, at),
            "MSP-TUNED": score_rule(msp_t, tm, pt),
            "DISAGREE-TUNED": score_rule(-g["dis_t"], td, pt),
        }
        res["thresholds"][f"FORCED-COST_{s}"] = t_cost
        res["thresholds"][f"CHOW_{s}"] = (b, a)
        res["thresholds"][f"CHOW-TUNED_{s}"] = (bt, at)
        res["thresholds"][f"MSP-TUNED_{s}"] = tm
        res["thresholds"][f"DISAGREE-TUNED_{s}"] = td
        for r, d in rules.items():
            res["decisions"][f"{r}_{s}"] = decision_metrics(yt, d)
        # matched-coverage comparison: MSP ranking at the coverage CHOW achieved
        k = int((rules["CHOW"] != -1).sum())
        d_m = np.full(len(pt), -1, dtype=int)
        if k > 0:
            top = np.argsort(-msp_t, kind="mergesort")[:k]
            d_m[top] = (pt[top] >= 0.5).astype(int)
        res["decisions"][f"MSP@CHOW-COVERAGE_{s}"] = decision_metrics(yt, d_m)

    # matched coverage for the entropy-adjusted score
    k = int((d_by_rule["ORIGINAL-SDF"] != -1).sum())
    d_m = np.full(len(pt), -1, dtype=int)
    if k > 0:
        top = np.argsort(-msp_t, kind="mergesort")[:k]
        d_m[top] = (pt[top] >= 0.5).astype(int)
    res["decisions"]["MSP@ORIGINAL-COVERAGE"] = decision_metrics(yt, d_m)

    # risk-coverage (prediction at 0.5)
    correct = ((pt >= 0.5).astype(int) == yt).astype(float)
    scores = {"MSP": msp_t, "ENTROPY": -binary_entropy(pt), "DISAGREEMENT": -g["dis_t"]}
    res["rc"] = {}
    for name, sc in scores.items():
        a_, e_ = aurc_eaurc(sc, correct)
        res["rc"][name] = {"AURC": a_, "E-AURC": e_, "risk_at": risk_at_coverage(sc, correct)}
    res["rc"]["MSP_vs_ENTROPY_identical_ranking"] = bool(
        np.array_equal(np.argsort(-msp_t, kind="mergesort"),
                       np.argsort(binary_entropy(pt), kind="mergesort"))
        or abs(res["rc"]["MSP"]["AURC"] - res["rc"]["ENTROPY"]["AURC"]) < 1e-12)

    # calibration of the ensemble (before / after ensemble-level temperature scaling)
    res["calibration"] = {"ece_val_ens_avg": ece(pv_o, yv), "ece_test_ens_avg": ece(pt_o, yt),
                          "ece_val_ens": ece(pv, yv), "ece_test_ens": ece(pt, yt),
                          "T_ens": g["T_ens"], "temperatures": g["temperatures"], "weights": g["weights"]}

    # label-free shift indicators (no target labels used to compute them)
    conf_v, conf_t = msp_v, msp_t
    acc_v = float(((pv >= 0.5).astype(int) == yv).mean())
    ts = np.sort(conf_v)
    # ATC: threshold so that the share of validation points above it equals validation accuracy
    t_atc = ts[int(np.floor((1 - acc_v) * len(ts)))] if acc_v < 1 else ts[0]
    b2, a2 = chow_thresholds(SCENARIOS["S2"])
    res["label_free"] = {
        "val_accuracy": acc_v,
        "mean_conf_val": float(conf_v.mean()), "mean_conf_test": float(conf_t.mean()),
        "DoC_predicted_test_accuracy": float(acc_v - (conf_v.mean() - conf_t.mean())),
        "ATC_predicted_test_accuracy": float((conf_t > t_atc).mean()),
        "actual_test_accuracy": float(correct.mean()),
        "share_pred_deceptive_val": float((pv >= 0.5).mean()),
        "share_pred_deceptive_test": float((pt >= 0.5).mean()),
        "chow_S2_abstain_rate_val": float(((pv > b2) & (pv < a2)).mean()),
        "chow_S2_abstain_rate_test": float(((pt > b2) & (pt < a2)).mean()),
    }

    # cost-ratio sensitivity: CHOW vs FORCED-COST
    sens = []
    for cm in SENS_CMISS:
        for cr in SENS_CREJECT:
            c = {"C_miss": cm, "C_false": 1.0, "C_reject": cr}
            b, a = chow_thresholds(c)
            d_ch = three_way(pt, b, a)
            d_fc = (pt >= c["C_false"] / (c["C_false"] + cm)).astype(int)
            k_ch, k_fc = expected_cost(yt, d_ch, c), expected_cost(yt, d_fc, c)
            sens.append({"C_miss": cm, "C_reject": cr, "coverage": float((d_ch != -1).mean()),
                         "cost_chow": k_ch, "cost_forced": k_fc,
                         "reduction_pct": 100 * (k_fc - k_ch) / k_fc if k_fc > 0 else 0.0})
    res["sensitivity_cost"] = sens

    # minimum-coverage sensitivity (validation-tuned variants)
    mc = []
    for cmin in MIN_COVERAGES:
        b, a = tune_dual_f1_mincov(pv, yv, cmin)
        m1 = decision_metrics(yt, three_way(pt, b, a))
        lo, hi = original_sdf_thresholds(pv_o, yv, min_coverage=cmin)
        m2 = decision_metrics(yt, original_sdf_apply(pt_o, lo, hi))
        mc.append({"min_coverage": cmin, "DUAL-F1": m1, "ORIGINAL-SDF": m2})
    res["sensitivity_mincov"] = mc
    return res


def subgroup_analysis(g, meta_df):
    """Hotel: metrics per writer x polarity subgroup (labels used for scoring only)."""
    df = meta_df.set_index("row_id").loc[g["test_row"]].reset_index()
    pt, yt = g["pt"], g["yt"]
    b, a = chow_thresholds(SCENARIOS["S2"])
    d = three_way(pt, b, a)
    out = []
    for (w, pol), idx in df.groupby(["writer", "polarity"]).groups.items():
        ii = np.asarray(list(idx))
        acc_mask = d[ii] != -1
        out.append({
            "writer": w, "polarity": pol, "n": int(len(ii)),
            "label": int(yt[ii].mean() > 0.5),
            "error_at_0.5": float(((pt[ii] >= 0.5).astype(int) != yt[ii]).mean()),
            "mean_p_deceptive": float(pt[ii].mean()),
            "mean_confidence": float(np.maximum(pt[ii], 1 - pt[ii]).mean()),
            "chow_S2_abstain_rate": float((~acc_mask).mean()),
            "chow_S2_error_accepted": float((d[ii][acc_mask] != yt[ii][acc_mask]).mean())
            if acc_mask.any() else np.nan,
        })
    return out


# ================================================================ main
def mean_std(values):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))],
                   dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="roberta-base")
    ap.add_argument("--hotel_meta", default=None, help="CSV with row_id, writer, polarity for the Hotel file")
    ap.add_argument("--keep_degenerate", action="store_true",
                    help="keep runs that predict a single class on validation (excluded by default)")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    groups, excluded = load_runs(args.logits_dir, args.model_name, exclude_degenerate=not args.keep_degenerate)
    if not groups:
        raise SystemExit("No .npz files found for this model.")
    print(f"Loaded {sum(len(v) for v in groups.values())} runs in {len(groups)} split/repetition groups")
    print(f"Excluded as degenerate (single-class predictions on validation): {excluded if excluded else 'none'}")
    with open(os.path.join(args.out_dir, "excluded_runs.json"), "w") as f:
        json.dump({"rule": "predicts one class for every source-validation review", "excluded": excluded}, f, indent=1)
    for key in sorted(groups, key=lambda k: (TARGET_ORDER.index(k[1]) if k[1] in TARGET_ORDER else 9, k[2])):
        n = len(groups[key])
        flag = "" if n == 3 else f"   <-- expected 3 models, found {n}"
        print(f"  train={'+'.join(key[0]):<18} test={key[1]:<10} rep={key[2]}  models={n}{flag}")

    meta_df = pd.read_csv(args.hotel_meta) if args.hotel_meta else None
    per_group, singles_rows = {}, []
    for key, runs in groups.items():
        g = ensemble_group(runs)
        r = analyse_group(g)
        if meta_df is not None and key[1] == "hotel":
            r["hotel_subgroups"] = subgroup_analysis(g, meta_df)
        per_group[key] = r
        for s in g["singles"]:
            singles_rows.append({"train": "+".join(key[0]), "test": key[1], "rep": key[2], **s})

    # ---------- aggregate over repetitions
    by_split = defaultdict(list)
    for key, r in per_group.items():
        by_split[(key[0], key[1])].append(r)

    main_rows, rc_rows, cal_rows, lf_rows, sens_rows, mc_rows, sub_rows = [], [], [], [], [], [], []
    for (src, tgt), reps in sorted(by_split.items(),
                                   key=lambda kv: TARGET_ORDER.index(kv[0][1]) if kv[0][1] in TARGET_ORDER else 9):
        split = f"{'+'.join(src)}>{tgt}"
        for rule in reps[0]["decisions"]:
            row = {"split": split, "test": tgt, "rule": rule, "n_reps": len(reps)}
            for metric in reps[0]["decisions"][rule]:
                m, s, _ = mean_std([r["decisions"][rule][metric] for r in reps])
                row[f"{metric}_mean"], row[f"{metric}_sd"] = m, s
            main_rows.append(row)
        for name in ["MSP", "ENTROPY", "DISAGREEMENT"]:
            row = {"split": split, "test": tgt, "score": name}
            for metric in ["AURC", "E-AURC"]:
                row[f"{metric}_mean"], row[f"{metric}_sd"], _ = mean_std([r["rc"][name][metric] for r in reps])
            for cov in COVERAGE_POINTS:
                row[f"risk@{cov:.1f}"] = mean_std([r["rc"][name]["risk_at"][f"{cov:.1f}"] for r in reps])[0]
            rc_rows.append(row)
        cal_rows.append({"split": split, "test": tgt,
                         "msp_entropy_identical_all_reps": all(r["rc"]["MSP_vs_ENTROPY_identical_ranking"] for r in reps),
                         **{f"{k}_mean": mean_std([r["calibration"][k] for r in reps])[0]
                            for k in ["ece_val_ens_avg", "ece_test_ens_avg", "ece_val_ens", "ece_test_ens", "T_ens"]},
                         **{f"{k}_sd": mean_std([r["calibration"][k] for r in reps])[1]
                            for k in ["ece_val_ens_avg", "ece_test_ens_avg", "ece_val_ens", "ece_test_ens", "T_ens"]}})
        lf = {"split": split, "test": tgt}
        for k in reps[0]["label_free"]:
            lf[f"{k}_mean"], lf[f"{k}_sd"], _ = mean_std([r["label_free"][k] for r in reps])
        lf_rows.append(lf)
        for i, cell in enumerate(reps[0]["sensitivity_cost"]):
            row = {"split": split, "test": tgt, "C_miss": cell["C_miss"], "C_reject": cell["C_reject"]}
            for k in ["coverage", "cost_chow", "cost_forced", "reduction_pct"]:
                row[f"{k}_mean"], row[f"{k}_sd"], _ = mean_std([r["sensitivity_cost"][i][k] for r in reps])
            sens_rows.append(row)
        for i, cell in enumerate(reps[0]["sensitivity_mincov"]):
            for variant in ["DUAL-F1", "ORIGINAL-SDF"]:
                row = {"split": split, "test": tgt, "min_coverage": cell["min_coverage"], "variant": variant}
                for k in ["coverage", "f1_dec", "macro_f1", "cost_S1", "cost_S2", "cost_S3"]:
                    row[f"{k}_mean"], row[f"{k}_sd"], _ = mean_std(
                        [r["sensitivity_mincov"][i][variant][k] for r in reps])
                mc_rows.append(row)
        if "hotel_subgroups" in reps[0]:
            for j, sg in enumerate(reps[0]["hotel_subgroups"]):
                row = {"writer": sg["writer"], "polarity": sg["polarity"], "n": sg["n"], "label": sg["label"]}
                for k in ["error_at_0.5", "mean_p_deceptive", "mean_confidence", "chow_S2_abstain_rate",
                          "chow_S2_error_accepted"]:
                    row[f"{k}_mean"], row[f"{k}_sd"], _ = mean_std([r["hotel_subgroups"][j][k] for r in reps])
                sub_rows.append(row)

    tables = {"table_main.csv": main_rows, "table_risk_coverage.csv": rc_rows,
              "table_calibration.csv": cal_rows, "table_label_free.csv": lf_rows,
              "table_sensitivity_cost.csv": sens_rows, "table_sensitivity_mincov.csv": mc_rows,
              "table_single_models.csv": singles_rows, "table_hotel_subgroups.csv": sub_rows}
    for name, rows in tables.items():
        if rows:
            # Full precision (2026-10-18): the tables are rounded once, by make_tables.py. Writing the CSVs
            # with 4 decimals caused double rounding at the third decimal (e.g. 0.326486 -> 0.3265 -> 0.327).
            pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, name), index=False)
    with open(os.path.join(args.out_dir, "results_per_repetition.json"), "w") as f:
        json.dump({f"{'+'.join(k[0])}>{k[1]}|rep{k[2]}": v for k, v in per_group.items()}, f, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # ---------- short console summary
    df = pd.DataFrame(main_rows)
    show = ["FORCED-F1", "FORCED-COST_S2", "CHOW_S2", "CHOW-TUNED_S2", "MSP-TUNED_S2", "DISAGREE-TUNED_S2",
            "ORIGINAL-SDF", "MSP@ORIGINAL-COVERAGE"]
    print("\nTest-domain results (mean over repetitions)")
    for tgt in [t for t in TARGET_ORDER if t in set(df["test"])]:
        print(f"\n  target = {tgt}")
        sub = df[(df["test"] == tgt) & (df["rule"].isin(show))].set_index("rule").loc[
            [r for r in show if r in set(df["rule"])]]
        for rule, r in sub.iterrows():
            print(f"    {rule:<24} cov={r['coverage_mean']:.3f}  F1dec={r['f1_dec_mean']:.3f}  "
                  f"recall={r['recall_dec_mean']:.3f}  risk={r['risk_accepted_mean']:.3f}  "
                  f"cost_S2={r['cost_S2_mean']:.3f}±{r['cost_S2_sd']:.3f}")
    print(f"\nTables written to {args.out_dir}")


if __name__ == "__main__":
    main()
