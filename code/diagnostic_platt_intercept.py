"""Post-hoc diagnostic: cost-derived rule after Platt (slope + intercept) calibration on source validation,
compared with the pre-specified temperature scaling and with the validation-tuned rules.

Usage: python diagnostic_platt_intercept.py LOGITS_DIR MODEL_NAME RESULTS_DIR/results_per_repetition.json
Called by make_supp_tables_s11_s13.py for Supplementary Table S12; analyze_logits.py must be in the same folder."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # analyze_logits.py lives next to this file
import analyze_logits as A
from scipy.optimize import minimize

def platt_fit(z, y):
    def nll(w):
        q = 1 / (1 + np.exp(-(w[0] * z + w[1]))); q = np.clip(q, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return minimize(nll, x0=[1.0, 0.0], method='L-BFGS-B').x

logit = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
logits_dir, model_name, results_json = sys.argv[1], sys.argv[2], sys.argv[3]
groups, excluded = A.load_runs(logits_dir, model_name)
res = json.load(open(results_json))
rows = []
for key, runs in groups.items():
    g = A.ensemble_group(runs)
    zv, zt = logit(g['pv_orig']), logit(g['pt_orig'])
    a, b = platt_fit(zv, g['yv'])
    pv_p, pt_p = 1 / (1 + np.exp(-(a * zv + b))), 1 / (1 + np.exp(-(a * zt + b)))
    src, tgt, rep = key
    rkey = f"{'+'.join(src)}>{tgt}|rep{rep}"
    r = res[rkey]
    row = {'target': tgt, 'rep': rep, 'a': a, 'b': b,
           'ece_t_ts': A.ece(g['pt'], g['yt']), 'ece_t_platt': A.ece(pt_p, g['yt']),
           'share_t_platt': float((pt_p >= .5).mean()), 'share_v_platt': float((pv_p >= .5).mean())}
    for s, c in A.SCENARIOS.items():
        b_, a_ = A.chow_thresholds(c)
        row[f'chow_ts_{s}'] = r['decisions'][f'CHOW_{s}'][f'cost_{s}']
        row[f'chow_platt_{s}'] = A.expected_cost(g['yt'], A.three_way(pt_p, b_, a_), c)
        row[f'forced_platt_{s}'] = A.expected_cost(g['yt'], (pt_p >= c['C_false'] / (c['C_false'] + c['C_miss'])).astype(int), c)
        row[f'forced_ts_{s}'] = r['decisions'][f'FORCED-COST_{s}'][f'cost_{s}']
        row[f'tuned_{s}'] = r['decisions'][f'CHOW-TUNED_{s}'][f'cost_{s}']
        row[f'msp_{s}'] = r['decisions'][f'MSP-TUNED_{s}'][f'cost_{s}']
        row[f'f1forced_{s}'] = r['decisions']['FORCED-F1'][f'cost_{s}']
    rows.append(row)
import pandas as pd
df = pd.DataFrame(rows)
order = [t for t in ['doctor', 'restaurant', 'hotel', 'books', 'clothing', 'electronics', 'home', 'kindle', 'movies', 'pets', 'sports', 'tools', 'toys'] if t in set(df.target)]
m = df.groupby('target').mean(numeric_only=True).loc[order]
pd.set_option('display.width', 250)
print(f'== {model_name} | {logits_dir.split("/")[-1]} | groups {len(df)}')
print(m[['a', 'b', 'ece_t_ts', 'ece_t_platt', 'share_v_platt', 'share_t_platt']].round(3).to_string())
for s in ['S1', 'S2', 'S3']:
    cols = [f'chow_ts_{s}', f'chow_platt_{s}', f'forced_ts_{s}', f'forced_platt_{s}', f'tuned_{s}', f'msp_{s}', f'f1forced_{s}']
    print(f'-- {s} mean expected cost'); print(m[cols].round(3).to_string())
    best_other = m[[f'tuned_{s}', f'msp_{s}', f'f1forced_{s}']].min(axis=1)
    print(f'   CHOW-Platt <= best validation-tuned rule in {(m[f"chow_platt_{s}"] <= best_other + 1e-9).sum()}/{len(m)} targets; '
          f'CHOW-TS <= best tuned in {(m[f"chow_ts_{s}"] <= best_other + 1e-9).sum()}/{len(m)}; '
          f'CHOW-Platt < CHOW-TS in {(m[f"chow_platt_{s}"] < m[f"chow_ts_{s}"]).sum()}/{len(m)}')
