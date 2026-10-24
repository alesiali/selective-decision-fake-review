#!/usr/bin/env python3
"""
make_figures.py - Figures 1-5 of the article (version of 2026-10-18)

  Figure 1  The decision layer (schematic; no data).
  Figure 2  The evaluation protocol: the leave-one-domain-out splits of the main corpus with their sizes, and the
            data used at each step (every count is read from the metadata of the logit files and checked).
  Figure 3  Calibrated ensemble probability that a review is deceptive, on the source-validation set
            and on the target domain (RoBERTa ensemble, pooled over the five repetitions), with the
            thresholds of cost scenario S2.
  Figure 4  Risk-coverage curves of the RoBERTa ensemble: maximum softmax probability (MSP) versus
            ensemble disagreement (mean over the five repetitions).
  Figure 5  Cost-ratio grid: reduction in expected cost (%) of the cost-derived rule relative to the
            cost-optimal forced decision, for the three models and the three target domains.

Precision. Every plotted number is computed with the functions that produced the tables
(analyze_logits.py: load_runs, ensemble_group, aurc_eaurc, rc_curve, chow_thresholds) or is read from
their output files. Before anything is drawn, the script rebuilds the ensembles from the logits and checks them
against the analysis output, and it checks every count in Figure 2 and every summary value shown in Figures 3-5
(shares, AURC, errors at 50/80/100% coverage, every cost-grid cell) against the result files used for the
tables. If a check fails, the script stops with an error. All checks are printed, and every plotted value is
written to figure_data.json. The numbers printed inside Figures 3 and 4 are read from the result files, so they always
equal the tables. Values recomputed from the logits must match those files: counts exactly, AURC within 1e-4
(MSP) or 2e-3 (ensemble disagreement), because fitted temperatures can differ in their last digits on another
machine or library version (see TOL_AURC).

Requirements: Python 3.9+, numpy, pandas, scipy, scikit-learn, matplotlib. CPU only (about 15 s).

Usage (from the repository root):
  python code/make_figures.py --analyze_py code/analyze_logits.py --roberta_logits logits/roberta \
                              --results_root results --out_dir figures
  --roberta_logits  folder with the 45 RoBERTa .npz files written by train_save_logits.py
  --results_root    folder with the analysis outputs: roberta/, bert/ and tfidf_logreg/ (analyze_logits.py)

Outputs (figure number in the file name; vector PDF and PNG at 450 dpi, 6.5 in wide):
  Figure 1.pdf/.png ... Figure 5.pdf/.png, figure_data.json
"""
from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
import glob
import importlib.util
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

# ------------------------------------------------------------------ style
# Colours from the validated reference palette (categorical slots 1-2 and the blue sequential ramp);
# the two-colour set passes the colour-vision checks on a white surface. Text is at least 8 pt
# (capital height about 2 mm, as PeerJ suggests).
SURFACE, INK, INK2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # genuine / MSP = blue; deceptive / disagreement = orange
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # steps 100-700
WIDTH = 6.5  # inches: text width of a US Letter page with 2.5 cm margins (PeerJ vector-figure size)
DPI = 450    # 6.5 in x 450 dpi = 2925 px (PeerJ: ideal 3000 px, at least 300 dpi)

TARGETS = ["doctor", "restaurant", "hotel"]
SOURCES = {"doctor": "hotel + restaurant", "restaurant": "hotel + doctor", "hotel": "doctor + restaurant"}
MODELS = [("roberta", "roberta-base", "RoBERTa ensemble"), ("bert", "bert-base-uncased", "BERT ensemble"),
          ("tfidf_logreg", "tfidf-logreg", "TF-IDF logistic regression")]
S2 = {"C_miss": 8.0, "C_false": 1.0, "C_reject": 0.5}
SCENARIO_CELLS = {(3.0, 0.5): "S1", (8.0, 0.5): "S2", (1.5, 0.5): "S3"}  # (C_miss, C_reject), Table 1

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True, "mathtext.fontset": "dejavusans"})

CHECKS = []


def check(cond, msg):
    """Record a check; stop the script if it fails."""
    CHECKS.append(("PASS" if cond else "FAIL") + "  " + msg)
    print(CHECKS[-1])
    if not cond:
        raise SystemExit(f"Check failed: {msg}")


def load_module(path):
    spec = importlib.util.spec_from_file_location("analyze_logits", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {name}.pdf and {name}.png")


def grid_lines(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def hu(x, d=3):
    """Round half up and format with d decimals (as make_tables.py): the value is first rounded to 10 decimals,
    so exact ties are not decided by floating-point noise. Negative values get a typographic minus."""
    v = Decimal(str(round(float(x), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{v:f}".replace("-", "\u2212")


def _lum(rgb):
    c = np.asarray(rgb, float)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(c @ [0.2126, 0.7152, 0.0722])


def text_on(fill):
    """White or ink, whichever has the higher WCAG contrast against the fill colour."""
    lf = _lum(to_rgb(fill))
    c_white = 1.05 / (lf + 0.05)
    c_ink = (lf + 0.05) / (_lum(to_rgb(INK)) + 0.05)
    return SURFACE if c_white > c_ink else INK


# ============================================================ data: RoBERTa ensembles from the logits
# Tolerances for AURC values recomputed from the logits. Temperature scaling is fitted by bounded scalar
# minimisation, which stops within 1e-5, so on another machine or library version the temperatures can differ in
# their last digits. Counts (reviews predicted deceptive) do not move; AURC can move slightly, mostly for ensemble
# disagreement, whose near-ties reorder. Measured on the 15 ensembles with every temperature perturbed by a
# relative 1e-4: AURC changed by at most 2.4e-5 (MSP) and 7.4e-4 (disagreement), and no count changed.
TOL_AURC = {"MSP": 1e-4, "DISAGREEMENT": 2e-3}


def roberta_ensembles(A, logits_dir, results_root):
    """Rebuild every RoBERTa ensemble exactly as the analysis did and check it against its output."""
    groups, excluded = A.load_runs(logits_dir, "roberta-base")  # degenerate runs excluded, as in the tables
    R = json.load(open(os.path.join(results_root, "roberta", "results_per_repetition.json")))
    ex_file = json.load(open(os.path.join(results_root, "roberta", "excluded_runs.json")))["excluded"]
    check(sorted(excluded) == sorted(ex_file), f"excluded runs match the analysis: {excluded}")
    check(len(groups) == 15, f"15 split x repetition groups found ({len(groups)})")
    ens = {}
    for (train, test, rep), runs in sorted(groups.items()):
        g = A.ensemble_group(runs)
        key = f"{'+'.join(train)}>{test}|rep{rep}"
        lf = R[key]["label_free"]
        sv, st = float((g["pv"] >= 0.5).mean()), float((g["pt"] >= 0.5).mean())
        same_counts = (abs(sv - lf["share_pred_deceptive_val"]) * len(g["pv"]) < 0.5 and
                       abs(st - lf["share_pred_deceptive_test"]) * len(g["pt"]) < 0.5)
        check(same_counts, f"{key}: share predicted deceptive (val {sv:.4f}, target {st:.4f}) equals the analysis "
                           f"output (same number of reviews)")
        msp = np.maximum(g["pt"], 1 - g["pt"])
        correct = ((g["pt"] >= 0.5).astype(int) == g["yt"]).astype(float)
        a_msp = A.aurc_eaurc(msp, correct)[0]
        a_dis = A.aurc_eaurc(-g["dis_t"], correct)[0]
        d_msp = abs(a_msp - R[key]["rc"]["MSP"]["AURC"])
        d_dis = abs(a_dis - R[key]["rc"]["DISAGREEMENT"]["AURC"])
        check(d_msp < TOL_AURC["MSP"] and d_dis < TOL_AURC["DISAGREEMENT"],
              f"{key}: AURC (MSP {a_msp:.4f}, disagreement {a_dis:.4f}) equals the analysis output "
              f"(differences {d_msp:.1e} and {d_dis:.1e})")
        ens[(test, rep)] = {"key": key, "members": len(runs), "yv": g["yv"], "pv": g["pv"], "yt": g["yt"],
                            "pt": g["pt"], "msp": msp, "dis": -g["dis_t"], "correct": correct}
    return ens


# ============================================================ Figure 1: schematic
def figure1(out_dir):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    fig.subplots_adjust(left=0.004, right=0.996, bottom=0.004, top=0.996)  # the diagram uses the full width
    ax.set_xlim(0, 100)
    ax.set_ylim(-6, 56)
    ax.axis("off")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def fits(t, x0b, x1b, y0b=None):
        """Stop if a text does not fit horizontally (and above y0b) inside its box."""
        bb = t.get_window_extent(rend)
        (x0, y0), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        if x0 < x0b + 0.6 or x1 > x1b - 0.6 or (y0b is not None and y0 < y0b + 0.4):
            raise SystemExit(f"Figure 1: text {t.get_text()!r} does not fit in its box")

    def box(x, y, w, h, title, body="", body_size=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=NEUTRAL, ec=AXIS,
                                    lw=0.6))
        t1 = ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=8, fontweight="bold")
        fits(t1, x, x + w, y)
        if body:
            t2 = ax.text(x + w / 2, y + h - 5.3, body, ha="center", va="top", fontsize=body_size, color=INK2,
                         linespacing=1.25)
            fits(t2, x, x + w, y)

    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK2,
                                     shrinkA=0, shrinkB=0))

    xs, w, h = [0.5, 35.0, 69.5], 30.0, 15.0
    y1, y2 = 36.5, 15.5                                   # bottoms of rows 1 and 2
    # row 1: fitted once, on the labelled source domains
    ax.text(0.5, 55.5, "Fitted on the labeled source domains", fontsize=8, color=MUTED, va="top")
    box(xs[0], y1, w, h, "Source domains", "two of the three\ndomains, with labels")
    box(xs[1], y1, w, h, "Base classifier", "RoBERTa or BERT\nensemble (3 seeds), or\nTF-IDF logistic regression")
    box(xs[2], y1, w, h, "Calibration", "temperature scaling\nfitted on the source-\nvalidation set")
    arrow((xs[0] + w, y1 + h / 2), (xs[1], y1 + h / 2))
    arrow((xs[1] + w, y1 + h / 2), (xs[2], y1 + h / 2))
    # row 2: applied to every review of the held-out target domain
    ax.text(0.5, 34.3, "Applied to each target-domain review", fontsize=8, color=MUTED, va="top")
    box(xs[0], y2, w, h, "Target review x", "held-out domain;\nits label is not used")
    box(xs[1], y2, w, h, "Probability p(x)", "calibrated probability\nthat x is deceptive")
    box(xs[2], y2, w, h, "Decision rule (Eq. 2)",
        "$\\beta = C_\\mathrm{reject}/C_\\mathrm{miss}$\n$\\alpha = 1 - C_\\mathrm{reject}/C_\\mathrm{false}$", body_size=11)
    arrow((xs[0] + w, y2 + h / 2), (xs[1], y2 + h / 2))
    arrow((xs[1] + w, y2 + h / 2), (xs[2], y2 + h / 2))
    arrow((xs[1] + w / 2, y1), (xs[1] + w / 2, y2 + h))           # the trained classifier scores x
    arrow((xs[2] + w / 2, y1), (xs[1] + w - 4.0, y2 + h))         # and its output is calibrated
    # row 3: the three actions on the probability scale (schematic)
    ax.add_patch(FancyBboxPatch((0.5, -5.6), 99, 17.2, boxstyle="round,pad=0,rounding_size=1.2", fc=SURFACE,
                                ec=AXIS, lw=0.6, zorder=0))
    arrow((xs[2] + w / 2, y2), (xs[2] + w / 2, 11.6))
    segs = [(2.5, 34.0, "Label genuine", "$p \\leq \\beta$", BLUE),
            (34.0, 66.0, "Defer to a human", "$\\beta < p < \\alpha$", MUTED),
            (66.0, 97.5, "Label deceptive", "$p \\geq \\alpha$", ORANGE)]
    for x0, x1, title, cond, col in segs:
        ax.add_patch(Rectangle((x0 + 0.4, 1.0), x1 - x0 - 0.8, 8.5, fc=SURFACE, ec=col, lw=1.4))
        t1 = ax.text((x0 + x1) / 2, 7.4, title, ha="center", va="center", fontsize=8, fontweight="bold")
        t2 = ax.text((x0 + x1) / 2, 3.4, cond, ha="center", va="center", fontsize=8, color=INK2)
        fits(t1, x0, x1)
        fits(t2, x0, x1)
    for xv, lab in [(2.5, "0"), (34.0, "β"), (66.0, "α"), (97.5, "1")]:
        ax.text(xv, 0.2, lab, ha="center", va="top", fontsize=8, color=INK2)
    ax.text(50, -2.6, "calibrated probability p (schematic, not to scale)", ha="center", va="top", fontsize=8,
            color=MUTED)
    save(fig, out_dir, "Figure 1")


# ============================================================ Figure 2: evaluation protocol
TABLE2 = {"hotel": (1875, 796, 1079), "restaurant": (400, 200, 200), "doctor": (555, 200, 355)}  # reviews, genuine, deceptive


def protocol_counts(logits_dir, data):
    """Read the split sizes from the metadata that train_save_logits.py stored in every RoBERTa logit file, and
    check that the 45 runs follow the protocol of the "Protocol" section (Table 2 sizes, 15% stratified validation split,
    one validation split per repetition, seeds)."""
    runs = []
    for f in sorted(glob.glob(os.path.join(logits_dir, "*.npz"))):
        with np.load(f, allow_pickle=False) as z:
            m = json.loads(str(z["meta"]))
            runs.append({"target": m["test_domain"], "sources": sorted(m["train_domains"]), "rep": m["repetition"],
                         "split_seed": m["split_seed"], "model_seed": m["model_seed"],
                         "n": (m["n_train"], m["n_val"], m["n_test"]), "val_strata": m["val_strata"],
                         "data_files": m["data_files"], "val_ids": tuple(sorted(z["val_ids"].tolist())),
                         "test_ids": tuple(z["test_ids"].tolist())})
    check(len(runs) == 45, f"Fig 2: 45 RoBERTa logit files ({len(runs)})")
    files = runs[0]["data_files"]
    check(all(r["data_files"] == files for r in runs), "Fig 2: every run used the same three data files (same MD5)")
    size = {d: (v["n"], v["genuine"], v["deceptive"]) for d, v in files.items()}
    check(size == TABLE2, f"Fig 2: reviews, genuine and deceptive per domain = Table 2 {size}")
    data["figure2"] = {"domains": {d: dict(zip(("reviews", "genuine", "deceptive"), v)) for d, v in size.items()},
                       "splits": {}}
    rows = []
    for t in TARGETS:
        rs = [r for r in runs if r["target"] == t]
        src = sorted(d for d in size if d != t)
        check(len(rs) == 15 and all(r["sources"] == src for r in rs), f"Fig 2 {t}: 15 runs trained on {' + '.join(src)}")
        ns = {r["n"] for r in rs}
        check(len(ns) == 1, f"Fig 2 {t}: the same training, validation and target sizes in every run {sorted(ns)}")
        n_train, n_val, n_test = ns.pop()
        check(n_test == size[t][0] and all(len(r["test_ids"]) == n_test and {i.split(":")[0] for i in r["test_ids"]} == {t}
                                           for r in rs), f"Fig 2 {t}: all {n_test} target reviews scored, none from another domain")
        n_src = sum(size[s][0] for s in src)
        check(n_train + n_val == n_src, f"Fig 2 {t}: training {n_train} + validation {n_val} = source reviews {n_src}")
        check(all(len(r["val_ids"]) == n_val and {i.split(":")[0] for i in r["val_ids"]} <= set(src) for r in rs),
              f"Fig 2 {t}: the {n_val} validation reviews come only from the source domains")
        vs = rs[0]["val_strata"]
        check(all(r["val_strata"] == vs for r in rs) and
              all(abs(vs[f"{s}|{c}"] - 0.15 * size[s][1 + c]) <= 1 for s in src for c in (0, 1)),
              f"Fig 2 {t}: validation = 15% of every domain x class stratum, within one review {vs}")
        reps = sorted({r["rep"] for r in rs})
        val_sets = {k: {r["val_ids"] for r in rs if r["rep"] == k} for k in reps}
        check(reps == [0, 1, 2, 3, 4] and all(len(v) == 1 for v in val_sets.values()) and
              len({next(iter(v)) for v in val_sets.values()}) == 5,
              f"Fig 2 {t}: one validation split per repetition (shared by its three seeds), a different one in each repetition")
        seeds_ok = all({r["split_seed"] for r in rs if r["rep"] == k} == {42 + k} and
                       sorted(r["model_seed"] for r in rs if r["rep"] == k) ==
                       ([42, 123, 2024] if k == 0 else [42 + k, 142 + k, 242 + k]) for k in reps)
        check(seeds_ok, f"Fig 2 {t}: split seeds 42-46 and model seeds as in the Protocol section")
        shown = SOURCES[t].split(" + ")  # the order used in the text and in Figure 3
        check(sorted(shown) == src, f"Fig 2 {t}: source names as in Figure 3 ({SOURCES[t]})")
        rows.append((t, shown, n_train, n_val, n_test))
        data["figure2"]["splits"][t] = {"sources": src, "n_training": n_train, "n_validation": n_val,
                                        "n_target": n_test, "validation_strata": vs}
    return rows, size


def figure2(logits_dir, out_dir, data):
    rows, size = protocol_counts(logits_dir, data)
    H = 4.65  # inches; the layout below needs 70.7 units of 0.065 in (checked at the end)
    fig, ax = plt.subplots(figsize=(WIDTH, H))
    fig.subplots_adjust(left=0.004, right=0.996, bottom=0.004, top=0.996)
    Y = H / (WIDTH / 100)  # the same physical unit on both axes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, Y)
    ax.axis("off")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def fits(t, x0b, x1b, y0b=None, y1b=None):
        """Stop if a text does not fit inside its box."""
        bb = t.get_window_extent(rend)
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        if x0 < x0b + 0.6 or x1 > x1b - 0.6 or (y0b is not None and y0 < y0b + 0.3) or (y1b is not None and y1 > y1b - 0.3):
            raise SystemExit(f"Figure 2: text {t.get_text()!r} does not fit in its box")

    VAL_FILL = SEQ_BLUE[0]
    SRC = [(0.5, 15.6), (17.9, 33.0)]
    TRAIN, VAL, TGT = (38.5, 55.5), (56.5, 74.0), (77.5, 99.5)
    SEP = 75.75

    def box(x, y_c, h, title, body, fc, ec=AXIS, lw=0.6):
        x0, x1 = x
        ax.add_patch(FancyBboxPatch((x0, y_c - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc,
                                    ec=ec, lw=lw, zorder=2))
        if body:
            t1 = ax.text((x0 + x1) / 2, y_c + 1.0, title, ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)
            t2 = ax.text((x0 + x1) / 2, y_c - 1.25, body, ha="center", va="center", fontsize=8, color=INK2, zorder=3)
            fits(t2, x0, x1, y_c - h / 2, y_c + h / 2)
        else:
            t1 = ax.text((x0 + x1) / 2, y_c, title, ha="center", va="center", fontsize=8, zorder=3)
        fits(t1, x0, x1, y_c - h / 2, y_c + h / 2)

    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK2, shrinkA=0,
                                     shrinkB=0, zorder=2))

    # ---------------- A: the three leave-one-domain-out splits
    yA = Y - 1.2
    ax.text(0.5, yA, "A", fontsize=9, fontweight="bold", va="top")
    ax.text(3.2, yA, "Leave-one-domain-out splits (main corpus)", fontsize=8, va="top")
    y_head = yA - 5.1
    for (x0, x1), lab in [((SRC[0][0], SRC[1][1]), "Source domains (labeled)"),
                          ((TRAIN[0], VAL[1]), "Source reviews, split per repetition"),
                          (TGT, "Target domain (held out)")]:
        t = ax.text((x0 + x1) / 2, y_head, lab, ha="center", va="center", fontsize=8, color=MUTED)
        fits(t, x0 - 0.5 if x0 > 1 else x0, x1 + 0.5 if x1 < 99 else x1 + 0.5)
    h_box, step = 5.4, 7.0
    for i, (t, src, n_train, n_val, n_test) in enumerate(rows):
        y_c = y_head - 6.4 - i * step
        for (x, s) in zip(SRC, src):
            box(x, y_c, h_box, s, f"{size[s][0]:,} reviews", NEUTRAL)
        ax.text((SRC[0][1] + SRC[1][0]) / 2, y_c, "+", ha="center", va="center", fontsize=9, color=INK2)
        arrow((SRC[1][1] + 0.5, y_c), (TRAIN[0] - 0.5, y_c))
        box(TRAIN, y_c, h_box, "training", f"{n_train:,}", NEUTRAL)
        box(VAL, y_c, h_box, "validation", f"{n_val:,} (15%)", VAL_FILL)
        box(TGT, y_c, h_box, t, f"{n_test:,} reviews", SURFACE, ec=INK2, lw=0.9)
    y_last = y_head - 6.4 - (len(rows) - 1) * step - h_box / 2
    note = ax.text(0.5, y_last - 2.2, "Five repetitions per split, each with a new validation split; transformers: three "
                   "seeds per repetition", fontsize=8, color=MUTED, va="top")
    fits(note, -0.2, 100)  # left-aligned at x = 0.5

    # ---------------- B: which data each step uses
    yB = y_last - 7.4
    ax.text(0.5, yB, "B", fontsize=9, fontweight="bold", va="top")
    ax.text(3.2, yB, "Data used at each step", fontsize=8, va="top")
    y_colhead = yB - 5.0
    box(TRAIN, y_colhead, 3.6, "Source training", "", NEUTRAL)
    box(VAL, y_colhead, 3.6, "Source validation", "", VAL_FILL)
    box(TGT, y_colhead, 3.6, "Target", "", SURFACE, ec=INK2, lw=0.9)
    LAB, UNL = "labels", "no labels"
    steps = [("Train the base classifier", {"train": LAB}),
             ("Calibration and ensemble weights", {"val": LAB}),
             ("Tuned comparison rules", {"val": LAB}),
             ("Cost-derived thresholds (Eq. 2)", None),
             ("Decide on each target review", {"tgt": UNL}),
             ("Score the decisions (evaluation only)", {"tgt": LAB})]
    centres = {"train": sum(TRAIN) / 2, "val": sum(VAL) / 2, "tgt": sum(TGT) / 2}
    row_h = 3.7
    y0_rows = y_colhead - 2.0 - 0.6
    sep_gaps = []
    for k, (label, use) in enumerate(steps):
        y_c = y0_rows - row_h / 2 - k * row_h
        ax.axhline(y_c - row_h / 2, xmin=0.005, xmax=0.995, color=GRID, lw=0.5, zorder=0)
        t = ax.text(0.5, y_c, label, ha="left", va="center", fontsize=8)
        fits(t, -0.2, TRAIN[0] - 0.6)  # left-aligned at x = 0.5; must end before the first column
        if use is None:
            t = ax.text((TRAIN[0] + TGT[1]) / 2, y_c, "no data: β and α follow from the three costs (Table 1)",
                        ha="center", va="center", fontsize=8, color=INK2)
            fits(t, TRAIN[0], TGT[1])
            sep_gaps.append((y_c - row_h / 2, y_c + row_h / 2))
            continue
        for col, kind in use.items():
            s = 1.9
            ax.add_patch(Rectangle((centres[col] - s / 2, y_c - s / 2), s, s, fc=INK2 if kind == LAB else SURFACE,
                                   ec=INK2, lw=0.9, zorder=3))
        data["figure2"].setdefault("steps", {})[label] = use
    ax.axhline(y0_rows, xmin=0.005, xmax=0.995, color=GRID, lw=0.5, zorder=0)
    y_bottom = y0_rows - len(steps) * row_h
    # dashed line between the source side and the held-out target; broken between the panels (where the note runs
    # across) and where a row spans both sides
    sep_gaps.append((y_colhead + 1.8 + 0.6, y_last - 0.6))  # (lower, upper)
    check(all(lo < hi for lo, hi in sep_gaps), "Fig 2: gaps of the dashed line are ordered (lower < upper)")
    y_top = y_head + 1.6
    segs, y_hi = [], y_top
    for lo, hi in sorted(sep_gaps, reverse=True):
        segs.append((hi, y_hi))
        y_hi = lo
    segs.append((y_bottom, y_hi))
    for lo, hi in segs:
        ax.plot([SEP, SEP], [lo, hi], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    # legend
    y_leg = y_bottom - 2.6
    s = 1.9
    ax.add_patch(Rectangle((TRAIN[0], y_leg - s / 2), s, s, fc=INK2, ec=INK2, lw=0.9))
    t1 = ax.text(TRAIN[0] + s + 0.9, y_leg, "reviews with their labels", ha="left", va="center", fontsize=8)
    x2 = TRAIN[0] + 30.0
    ax.add_patch(Rectangle((x2, y_leg - s / 2), s, s, fc=SURFACE, ec=INK2, lw=0.9))
    t2 = ax.text(x2 + s + 0.9, y_leg, "reviews without their labels", ha="left", va="center", fontsize=8)
    fits(t1, TRAIN[0], x2)
    fits(t2, x2, 100)
    check(y_leg - 1.5 > 0, f"Fig 2: the drawing fits in the figure (lowest element at {y_leg - 1.5:.1f} > 0)")
    save(fig, out_dir, "Figure 2")


# ============================================================ Figure 3: probability distributions
def figure3(ens, results_root, out_dir, data):
    lf = pd.read_csv(os.path.join(results_root, "roberta", "table_label_free.csv"))
    beta, alpha = 0.5 / 8.0, 0.5  # S2 thresholds (Table 1)
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.9), sharex=True, sharey=True)
    data["figure3"] = {"bins": bins.tolist(), "pooled_over": "5 repetitions", "panels": {}}
    ymax = 0
    for j, t in enumerate(TARGETS):
        reps = sorted(r for (tt, r) in ens if tt == t)
        check(reps == [0, 1, 2, 3, 4], f"{t}: five repetitions")
        for i, (part, label) in enumerate([("v", "Source validation"), ("t", "Target domain")]):
            p = np.concatenate([ens[(t, r)][f"p{part}"] for r in reps])
            y = np.concatenate([ens[(t, r)][f"y{part}"] for r in reps])
            n = len(p)
            gen, _ = np.histogram(p[y == 0], bins=bins)
            dec, _ = np.histogram(p[y == 1], bins=bins)
            gen_pct, dec_pct = 100 * gen / n, 100 * dec / n
            ymax = max(ymax, float((gen_pct + dec_pct).max()))
            share_pred = float(np.mean([(ens[(t, r)][f"p{part}"] >= 0.5).mean() for r in reps]))
            share_true = float(y.mean())
            col = "share_pred_deceptive_val_mean" if part == "v" else "share_pred_deceptive_test_mean"
            tab = float(lf.loc[lf.test == t, col].iloc[0])
            check(abs(share_pred - tab) < 5e-5, f"Fig 3 {t} {label}: share predicted deceptive {share_pred:.4f} "
                                               f"= Table H ({tab:.4f})")
            check(abs(share_pred - float((p >= 0.5).mean())) < 1e-12, f"Fig 3 {t} {label}: pooled = mean of repetitions")
            ax = axes[i, j]
            ax.axvspan(beta, alpha, color=NEUTRAL, zorder=0, lw=0)
            for xv in (beta, alpha):
                ax.axvline(xv, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
            w = bins[1] - bins[0]
            ax.bar(bins[:-1], gen_pct, width=w, align="edge", color=BLUE, edgecolor=SURFACE, lw=0.5, zorder=2)
            ax.bar(bins[:-1], dec_pct, width=w, bottom=gen_pct, align="edge", color=ORANGE, edgecolor=SURFACE,
                   lw=0.5, zorder=2)
            ax.text(0.03, 0.98, f"predicted deceptive: {hu(100 * tab, 0)}%\ntrue share: {hu(100 * share_true, 0)}%",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK, linespacing=1.3,
                    bbox=dict(fc=SURFACE, ec="none", pad=1.2), zorder=4)
            if i == 0 and j == 0:
                for xv, nm in ((beta, "β"), (alpha, "α")):
                    ax.text(xv + 0.015, 30, nm, ha="left", va="center", fontsize=8, color=INK2)
            grid_lines(ax)
            if i == 0:
                ax.set_title(f"Target: {t}\nsources: {SOURCES[t]}", fontsize=8, color=INK, pad=4, linespacing=1.3)
            if j == 0:
                ax.set_ylabel(f"{label}\nreviews (%)")
            if i == 1:
                ax.set_xlabel("Calibrated probability p")
            data["figure3"]["panels"][f"{t}|{label}"] = {
                "n_pooled": int(n), "genuine_pct": gen_pct.round(4).tolist(), "deceptive_pct": dec_pct.round(4).tolist(),
                "share_predicted_deceptive": share_pred, "true_deceptive_share": share_true}
    ylim = max(50.0, float(np.ceil((ymax + 11) / 10) * 10))  # headroom for the annotation above the bars
    check(ymax <= ylim - 9, f"Fig 3: tallest bar {ymax:.1f}% leaves room for the annotations (y limit {ylim:.0f}%)")
    for ax in axes.flat:
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_ylim(0, ylim)
    handles = [Patch(color=BLUE, label="Genuine (true class)"), Patch(color=ORANGE, label="Deceptive (true class)"),
               Patch(fc=NEUTRAL, ec=INK2, ls=(0, (3, 2)), lw=0.7, label="Deferred under S2 (β = 0.0625 < p < α = 0.5)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.035), handlelength=1.2,
               columnspacing=1.6)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, out_dir, "Figure 3")


# ============================================================ Figure 4: risk-coverage
def figure4(A, ens, results_root, out_dir, data):
    rc = pd.read_csv(os.path.join(results_root, "roberta", "table_risk_coverage.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.35), sharey=True)
    data["figure4"] = {}
    for j, t in enumerate(TARGETS):
        ax = axes[j]
        reps = sorted(r for (tt, r) in ens if tt == t)
        data["figure4"][t] = {}
        aurc_mean = {}
        for name, score, col in [("MSP", "msp", BLUE), ("DISAGREEMENT", "dis", ORANGE)]:
            curves, aurcs = [], []
            for r in reps:
                e = ens[(t, r)]
                cov, risk = A.rc_curve(e[score], e["correct"])
                curves.append(risk)
                aurcs.append(A.aurc_eaurc(e[score], e["correct"])[0])
            n = len(curves[0])
            check(all(len(c) == n for c in curves), f"Fig 4 {t} {name}: same target size in every repetition ({n})")
            mean_risk = np.mean(curves, axis=0)
            row = rc[(rc.test == t) & (rc.score == name)].iloc[0]
            check(abs(np.mean(aurcs) - row.AURC_mean) < TOL_AURC[name], f"Fig 4 {t} {name}: mean AURC {np.mean(aurcs):.4f} = Table 5 "
                                                                f"({row.AURC_mean:.4f})")
            for c in (0.5, 0.8, 1.0):
                k = max(int(np.ceil(c * n)) - 1, 0)
                # allow one review to differ at the cut in one repetition (near-ties; see TOL_AURC)
                check(abs(mean_risk[k] - row[f"risk@{c:.1f}"]) < 5e-5 + 1 / (len(reps) * (k + 1)),
                      f"Fig 4 {t} {name}: error at {c:.0%} coverage {mean_risk[k]:.4f} = Table 5 ({row[f'risk@{c:.1f}']:.4f})")
            aurc_mean[name] = float(row.AURC_mean)  # printed value = Table 5 (the recomputed mean is checked above)
            ax.plot(cov, mean_risk, color=col, lw=1.4, solid_joinstyle="round", solid_capstyle="round")
            data["figure4"][t][name] = {"coverage": np.round(cov, 6).tolist(), "mean_risk": np.round(mean_risk, 6).tolist(),
                                        "AURC_mean": float(np.mean(aurcs))}
        ax.plot([1.0], [mean_risk[-1]], "o", ms=4.5, color=INK2, mec=SURFACE, mew=1.0, zorder=5, clip_on=False)
        ax.text(0.06, 0.97, f"AURC: MSP {hu(aurc_mean['MSP'])}\ndisagreement {hu(aurc_mean['DISAGREEMENT'])}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK, linespacing=1.3)
        ax.set_title(f"Target: {t}", fontsize=8, pad=4)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 0.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        grid_lines(ax, "both")
    axes[0].set_ylabel("Error rate among\ndecided reviews")
    handles = [Line2D([], [], color=BLUE, lw=1.4, label="MSP (= entropy)"),
               Line2D([], [], color=ORANGE, lw=1.4, label="Ensemble disagreement"),
               Line2D([], [], color=INK2, marker="o", ls="none", ms=4.5, mec=SURFACE, label="All reviews decided")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), handlelength=1.8,
               columnspacing=1.8)
    axes[1].set_xlabel("Coverage (share of reviews decided, most confident first)")
    fig.tight_layout(w_pad=0.8)
    save(fig, out_dir, "Figure 4")


# ============================================================ Figure 5: cost-ratio grid
def figure5(results_root, out_dir, data):
    vmax = 80.0
    cmap, norm = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE), Normalize(0, vmax)
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 7.0), sharex=True, sharey=True)
    data["figure5"] = {}
    summary = {}
    for i, (folder, model_name, label) in enumerate(MODELS):
        sens = pd.read_csv(os.path.join(results_root, folder, "table_sensitivity_cost.csv"))
        R = json.load(open(os.path.join(results_root, folder, "results_per_repetition.json")))
        for j, t in enumerate(TARGETS):
            s = sens[sens.test == t].copy()
            # reduction in MEAN expected cost over the five repetitions (the measure of Tables C and F and of the text)
            s["reduction_of_mean_pct"] = 100 * (s.cost_forced_mean - s.cost_chow_mean) / s.cost_forced_mean
            pv = s.pivot(index="C_miss", columns="C_reject", values="reduction_of_mean_pct").sort_index(ascending=False)
            check(pv.shape == (7, 4), f"Fig 5 {label} {t}: 7 x 4 grid")
            # recompute the means from the per-repetition results and compare with the table
            reps = [v["sensitivity_cost"] for k, v in R.items() if k.split(">")[1].split("|")[0] == t]
            check(len(reps) == 5, f"Fig 5 {label} {t}: five repetitions")
            bad = []
            for cm in pv.index:
                for cr in pv.columns:
                    cells = [c for rr in reps for c in rr if c["C_miss"] == cm and c["C_reject"] == cr]
                    k_ch = np.mean([c["cost_chow"] for c in cells])
                    k_fc = np.mean([c["cost_forced"] for c in cells])
                    if len(cells) != 5 or abs(100 * (k_fc - k_ch) / k_fc - pv.loc[cm, cr]) >= 1e-9:
                        bad.append((cm, cr))
            check(not bad, f"Fig 5 {label} {t}: all 28 cells equal the reduction in mean cost recomputed from the repetitions"
                  + (f" (differ: {bad})" if bad else ""))
            check(float(pv.values.max()) <= vmax, f"Fig 5 {label} {t}: maximum {pv.values.max():.1f}% within the colour scale")
            ax = axes[i, j]
            vals = pv.values
            ax.imshow(np.clip(vals, 0, vmax), cmap=cmap, norm=norm, aspect="auto")
            n_empty = 0
            for a, cm in enumerate(pv.index):
                for b, cr in enumerate(pv.columns):
                    beta, alpha = cr / cm, 1.0 - cr / 1.0
                    v = vals[a, b]
                    if beta >= alpha:  # empty deferral region: the rule is the forced decision
                        n_empty += 1
                        if abs(v) > 1e-12:
                            raise SystemExit(f"Fig 5 {label} {t} C_miss={cm} C_reject={cr}: empty region but reduction {v}")
                        ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fc=NEUTRAL, ec=SURFACE, lw=1.0))
                        ax.text(b, a, "=", ha="center", va="center", fontsize=8, color=INK2)
                        continue
                    txt = hu(v, 1)
                    ax.text(b, a, txt, ha="center", va="center", fontsize=8, color=text_on(cmap(norm(np.clip(v, 0, vmax)))))
                    if (cm, cr) in SCENARIO_CELLS:
                        ax.add_patch(Rectangle((b - 0.46, a - 0.46), 0.92, 0.92, fill=False, ec=INK, lw=0.9))
            check(n_empty == 5, f"Fig 5 {label} {t}: the 5 cells without a deferral region have reduction exactly 0")
            # white gaps between cells
            for a in range(1, 7):
                ax.axhline(a - 0.5, color=SURFACE, lw=1.0)
            for b in range(1, 4):
                ax.axvline(b - 0.5, color=SURFACE, lw=1.0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
            ax.set_xticks(range(4))
            ax.set_xticklabels([f"{c:g}" for c in pv.columns])
            ax.set_yticks(range(7))
            ax.set_yticklabels([f"{c:g}" for c in pv.index])
            if i == 0:
                ax.set_title(f"Target: {t}", fontsize=8, pad=4)
            if j == 0:
                ax.set_ylabel("$C_\\mathrm{miss}$", fontsize=11, labelpad=2)
                ax.text(-0.31, 0.5, label, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=8,
                        fontweight="bold")
            if i == 2:
                ax.set_xlabel("$C_\\mathrm{reject}$", fontsize=11)
            data["figure5"][f"{label}|{t}"] = {"C_miss": [float(c) for c in pv.index], "C_reject": [float(c) for c in pv.columns],
                                               "reduction_in_mean_cost_pct": np.round(vals, 6).tolist()}
            summary[(label, t)] = pv
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.0, w_pad=0.6)
    cax = fig.add_axes([0.25, -0.005, 0.5, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=[0, 20, 40, 60, 80])
    cb.ax.set_xticklabels(["0 or less", "20", "40", "60", "80"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    cb.set_label("Reduction in mean expected cost relative to the cost-optimal forced decision (%)", fontsize=8)
    save(fig, out_dir, "Figure 5")
    # values quoted in the article text ("Sensitivity to Costs and Coverage Constraints", RoBERTa) and the abstract (all models)
    print("\nValues for the text (from the plotted grid):")
    for (label, t), pv in summary.items():
        print(f"  {label:<27} {t:<10} min {pv.values.min():6.2f}%  "
              f"C_reject=0.1: {pv[0.1].min():.1f}-{pv[0.1].max():.1f}%  C_reject=0.5: max {pv[0.5].max():.1f}%")
    worst = min((pv.values.min(), label, t) for (label, t), pv in summary.items())
    print(f"  largest increase over all models and targets: {-worst[0]:.3f}% ({worst[1]}, {worst[2]})")
    data["figure5_largest_increase_pct"] = {"value": float(-worst[0]), "model": worst[1], "target": worst[2]}


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze_py", default="analyze_logits.py")
    ap.add_argument("--roberta_logits", required=True)
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out_dir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    A = load_module(a.analyze_py)
    data = {"script": "make_figures.py (2026-10-18, version 2)"}
    print("Figure 1 (schematic)")
    figure1(a.out_dir)
    print("\nFigure 2 (protocol; counts read from the logit files)")
    figure2(a.roberta_logits, a.out_dir, data)
    print("\nRebuilding the RoBERTa ensembles from the logits and checking them against the analysis")
    ens = roberta_ensembles(A, a.roberta_logits, a.results_root)
    print("\nFigure 3")
    figure3(ens, a.results_root, a.out_dir, data)
    print("\nFigure 4")
    figure4(A, ens, a.results_root, a.out_dir, data)
    print("\nFigure 5")
    figure5(a.results_root, a.out_dir, data)
    data["checks"] = CHECKS
    with open(os.path.join(a.out_dir, "figure_data.json"), "w") as f:
        json.dump(data, f, indent=1)
    n_fail = sum(c.startswith("FAIL") for c in CHECKS)
    print(f"\n{len(CHECKS)} checks, {n_fail} failed. Plotted values: {os.path.join(a.out_dir, 'figure_data.json')}")


if __name__ == "__main__":
    main()
