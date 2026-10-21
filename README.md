# Selective Decision-Making for Cross-Domain Fake Review Detection under Uncertainty and Cost Constraints: code, saved logits and processed data (v2.0)

This repository accompanies the article *Selective Decision-Making for Cross-Domain Fake Review Detection under Uncertainty and Cost Constraints* (under review). Version 2.0 contains the code, the saved model outputs (logits) of every run, a processed version of the second dataset, and the analysis outputs. Every decision-layer result of the article (all tables, figures and supplementary tables) can be reproduced from the saved logits on a CPU in about 48 minutes, without the review texts of the main corpus; only retraining the models, including the TF-IDF and SVM baselines of Supplementary Table S15, needs these texts.

## Overview

A fake review classifier trained on some domains is often applied to a new domain for which no labels exist. The article studies a decision layer for this setting. A base classifier is trained on the labeled source domains, and its probabilities are calibrated by temperature scaling on a source-validation set. For a review of the held-out target domain with calibrated probability *p* of being deceptive, a cost-derived three-way rule labels the review deceptive if *p* ≥ α = 1 − C_reject/C_false, labels it genuine if *p* ≤ β = C_reject/C_miss, and defers it to a human moderator otherwise. Here C_miss is the cost of missing a deceptive review, C_false the cost of flagging a genuine review, and C_reject the cost of a deferral. The thresholds follow from the three costs alone, so the rule uses no target-domain labels. The rule is applied to the temperature-scaled probabilities of RoBERTa and BERT ensembles (three fine-tuning seeds each) and of TF-IDF logistic regression. It is evaluated under leave-one-domain-out transfer on the hotel, restaurant and doctor reviews of the corpus of Li et al. (2014) and on ten Amazon product categories of the dataset of Salminen et al. (2022), and it is compared with forced decisions (no deferral), with deferral rules whose thresholds are tuned on source-validation data, and with an entropy-adjusted score.

## Repository contents

```
README.md                  this file
LICENSE                    MIT license
CITATION.cff               citation metadata
.zenodo.json               metadata that Zenodo uses when it archives a GitHub release
requirements.txt           Python packages for reproducing the results from the saved logits (CPU)
requirements-gpu.txt       Python packages for retraining the transformers (GPU)
.gitignore                 files that should not be committed (extracted logits, local copies of the main corpus)
MANIFEST.md5               MD5 checksum of every file of the v2.0 package
code/
  train_save_logits.py           fine-tunes RoBERTa or BERT under leave-one-domain-out transfer; saves the logits of every run (GPU)
  train_classical.py             TF-IDF logistic regression on the same splits, same output format (CPU)
  prepare_salminen.py            prepares the second dataset: de-duplication, cleaning, truncation control, balanced sample
  train_salminen.py              leave-one-domain-out runs on the second dataset (RoBERTa on a GPU, TF-IDF on a CPU)
  published_protocol.py          base classifiers under the protocol of earlier studies (Supplementary Table S15)
  analyze_logits.py              decision-layer analysis of the saved logits: calibration, ensembles, decision rules, costs (CPU)
  make_tables.py                 result tables of the main corpus
  make_tables_salminen.py        result tables of the second dataset
  diagnostic_platt_intercept.py  post-hoc Platt-scaling diagnostic (called by make_supp_tables_s11_s13.py)
  make_supp_tables_s11_s13.py    Supplementary Tables S11-S13
  make_figures.py                Figures 1-5, with 155 consistency checks against the analysis outputs
data/
  hotel_meta_v2.csv              author type and polarity of every hotel review (row_id, writer, polarity); no review text
  salminen_processed.csv.gz      processed second dataset (gzip-compressed CSV, 40,363 reviews)
logits/                          saved logits, one zip file per group of runs (one .npz file per run)
  roberta.zip                    main corpus, RoBERTa, 45 runs (3 target domains x 5 repetitions x 3 seeds)
  bert.zip                       main corpus, BERT, 45 runs
  tfidf_logreg.zip               main corpus, TF-IDF logistic regression, 15 runs
  salminen_roberta.zip           second dataset, RoBERTa, 30 runs (10 categories x repetitions 0-2)
  salminen_tfidf_sample.zip      second dataset, TF-IDF trained on the 400-review sample per category, 50 runs
  salminen_tfidf_all.zip         second dataset, TF-IDF trained on all source reviews, 50 runs
  published_roberta.zip          published protocol, RoBERTa, 15 runs
results/
  results.zip                    all analysis outputs (CSV and JSON files) written by the commands below
tables/                          result tables (Markdown) and every number in them (JSON)
figures/                         Figures 1-5 (PDF and PNG) and figure_data.json (every plotted value and check)
```

Each `.npz` file holds the validation and target logits, labels and row numbers of one run, and a `meta` JSON string with the split, seeds, hyper-parameters, library versions, device and the MD5 checksums of the data files.

## Data

### Main corpus (third-party data, not redistributed)

The hotel, restaurant and doctor reviews come from the cross-domain deceptive opinion spam corpus of Li et al. (2014), which we obtained from its authors on request. The corpus is third-party data and is not redistributed in this repository; researchers can request it from the authors of Li et al. (2014). The hotel reviews written by crowd workers (Turkers) and customers are also publicly available in the op_spam corpus of Ott et al. (2011; 2013): https://myleott.com/op-spam.html. The review texts belong to their authors and remain under the terms of the original corpora.

The experiments used the de-duplicated files with the `clean_review` column (the model input) described in Section 4.1 of the article. Copies of these files can be checked against the following MD5 checksums, which are also recorded in the metadata of every logit file:

| File | Domain | Reviews | Genuine | Deceptive | MD5 |
|---|---|---|---|---|---|
| `LIWC_Hotel_readability_cleaned_dedup.csv` | hotel | 1,875 | 796 | 1,079 | `da94cf942361bb65c7cb571f666f700c` |
| `LIWC_Restaurant_readability_cleaned_dedup.csv` | restaurant | 400 | 200 | 200 | `b1ce128c69595084541cd906a632c2c6` |
| `LIWC_doctor_readability_cleaned_dedup.csv` | doctor | 555 | 200 | 355 | `9fdb1f4bce23487fe68ffd7bb45857d5` |

The metadata of the transformer runs (RoBERTa, BERT and the published-protocol runs) records the hotel and restaurant files as `LIWC.Hotel_readability_cleaned_dedup.csv` and `LIWC.Restaurant_readability_cleaned_dedup.csv` (with a dot after `LIWC`); the MD5 checksums are the same. `train_save_logits.py`, `train_classical.py` and `published_protocol.py` accept both spellings; `prepare_salminen.py` expects the spelling of the table.

The review texts are not needed for any decision-layer result: every table, figure and supplementary table is reproduced from the saved logits and `data/hotel_meta_v2.csv` alone, except the TF-IDF and SVM rows of Supplementary Table S15, whose models are fitted on the texts. The three files are needed only to retrain the models (including those two baselines) and for the cleaning check of `prepare_salminen.py` (option `--main_dir`). For these steps, place the three files in the repository root; `.gitignore` lists them so that they are not committed by accident.

`data/hotel_meta_v2.csv` gives, for each row of the hotel file (`row_id` 0-1874), the author type (`writer`: `expert` = written by hotel employees, `turker`, `customer`) and the polarity (`positive`, `negative`). It contains no review text. The labels were recovered from the block order of the hotel file (Section 4.1).

### Second dataset (Salminen et al. 2022)

The fake product reviews of Salminen et al. (2022) are available from the authors' repository, https://osf.io/tyue9/, under the Creative Commons Attribution 4.0 International license (CC BY 4.0). `data/salminen_processed.csv.gz` was produced from their file `fake_reviews_dataset.csv` (MD5 `177a386fab45e3f85d140728b2387967`) by `code/prepare_salminen.py`, as described in Section 4.5 of the article: exact de-duplication, the text cleaning of the main corpus, removal of the last token of every review (truncation control), removal of near-duplicates across categories, and a balanced training sample of 200 fake and 200 original reviews per category (column `in_sample`). It is shared under the same license (CC BY 4.0). The decompressed file `salminen_processed.csv` has the MD5 checksum `e750693b5621ec3a349fce0f6e9b5d99`. The model input is the column `model_text`, the product category is `domain`, and `y` = 1 marks a computer-generated (fake) review. The file is needed only for retraining; to decompress it on any system:

```
python -c "import gzip; open('data/salminen_processed.csv', 'wb').write(gzip.open('data/salminen_processed.csv.gz').read())"
```

## Requirements

- **Reproduction from the saved logits (CPU):** Python 3 with numpy, pandas, scipy, scikit-learn and matplotlib. `requirements.txt` pins the exact versions with which the commands below were tested (Python 3.13.16): `python -m pip install -r requirements.txt`. A second test with older versions (numpy 2.1.3, pandas 2.2.3, scipy 1.14.1, scikit-learn 1.6.1, matplotlib 3.9.2) gave byte-identical analysis outputs, tables and `figures/figure_data.json` and passed all figure checks; only the image files (PNG, PDF), drawn by a different matplotlib version, differed. Other versions were not tested.
- **Retraining (GPU):** `requirements-gpu.txt` lists the versions recorded in the logit files of the transformer runs (Google Colab, Python 3.13.15, PyTorch 2.11.0, Transformers 5.18.0, scikit-learn 1.6.1, numpy 2.1.3, pandas 2.2.3; one NVIDIA T4 GPU), and NLTK 3.10.3 for `prepare_salminen.py`. The TF-IDF models need no GPU; their saved logits were produced with the versions of `requirements.txt`.

## Reproduce the results from the saved logits (CPU, minutes)

Run the following commands from the repository root, in this order. They extract the logits, run the analysis, and regenerate the tables and figures. They need neither a GPU nor the review texts. The times were measured on a Linux machine with two CPU cores (total about 48 minutes).

```bash
# 1. Extract the saved logits (creates logits/roberta/, logits/bert/, ...)
python -m zipfile -e logits/roberta.zip logits/
python -m zipfile -e logits/bert.zip logits/
python -m zipfile -e logits/tfidf_logreg.zip logits/
python -m zipfile -e logits/salminen_roberta.zip logits/
python -m zipfile -e logits/salminen_tfidf_sample.zip logits/
python -m zipfile -e logits/salminen_tfidf_all.zip logits/
python -m zipfile -e logits/published_roberta.zip logits/

# 2. Main corpus: decision-layer analysis (about 9 min)
python code/analyze_logits.py --logits_dir logits/roberta --out_dir results/roberta --model_name roberta-base --hotel_meta data/hotel_meta_v2.csv
python code/analyze_logits.py --logits_dir logits/bert --out_dir results/bert --model_name bert-base-uncased --hotel_meta data/hotel_meta_v2.csv
python code/analyze_logits.py --logits_dir logits/tfidf_logreg --out_dir results/tfidf_logreg --model_name tfidf-logreg --hotel_meta data/hotel_meta_v2.csv

# 3. Main corpus: tables (about 1 min)
python code/make_tables.py --results results/roberta --logits logits/roberta --model roberta-base --results2 results/tfidf_logreg --logits2 logits/tfidf_logreg --model2 tfidf-logreg --out tables --name main_corpus_roberta_tfidf
python code/make_tables.py --results results/bert --logits logits/bert --model bert-base-uncased --out tables --name main_corpus_bert

# 4. Second dataset: analysis and tables (about 38 min, most of it for the TF-IDF model trained on all source reviews)
python code/analyze_logits.py --logits_dir logits/salminen_roberta --out_dir results/salminen_roberta --model_name roberta-base
python code/analyze_logits.py --logits_dir logits/salminen_tfidf_sample --out_dir results/salminen_tfidf_sample --model_name tfidf-logreg
python code/analyze_logits.py --logits_dir logits/salminen_tfidf_all --out_dir results/salminen_tfidf_all --model_name tfidf-logreg-alltrain
python code/make_tables_salminen.py --results results/salminen_roberta --label "RoBERTa (3 repetitions)" --results2 results/salminen_tfidf_sample --label2 "TF-IDF + LR, same splits (repetitions 0-2)" --reps 0 1 2 --same_targets --out tables --name second_dataset_S11_roberta_vs_tfidf
python code/make_tables_salminen.py --results results/salminen_tfidf_sample --label "Supplementary Table S11, part 3: TF-IDF + LR trained on the sample, five repetitions" --results2 results/salminen_tfidf_all --label2 "Supplementary Table S10: TF-IDF + LR trained on all source reviews (robustness), five repetitions" --out tables --name second_dataset_S10_S11part3
python code/make_supp_tables_s11_s13.py --logits_root logits --results_root results --s11_md tables/second_dataset_S11_roberta_vs_tfidf.md --out tables/second_dataset_S11_S13.md

# 5. Published protocol, RoBERTa ensemble (Supplementary Table S15) (a few seconds)
python code/published_protocol.py analyze --logits_dir logits/published_roberta --model_name roberta-base --variant all --out results/published

# 6. Figures 1-5 (about 10 s); the last line printed must start with "155 checks, 0 failed."
python code/make_figures.py --analyze_py code/analyze_logits.py --roberta_logits logits/roberta --results_root results --out_dir figures
```

What to expect:

- `analyze_logits.py` lists the groups of runs it found. For TF-IDF and for RoBERTa on the second dataset, which have one model per repetition, it prints "expected 3 models, found 1"; this is expected. On the main corpus, one RoBERTa run is excluded as degenerate (it predicts one class for every source-validation review; Section 4.2).
- `results/<group>/` receives the analysis outputs; they are identical to the content of `results/results.zip` (extract it elsewhere to compare, for example `python -m zipfile -e results/results.zip results_shipped/`). The file `results/published/classical_published_protocol.json` in the zip is not regenerated by these commands, because it needs the review texts (see "Retrain the models").
- The Markdown and JSON files in `tables/` are overwritten with identical copies (in our test of the package). `tables/published_protocol_S15.md` is a static table (see its notes): its RoBERTa row is regenerated by step 5 (`results/published/roberta-base_published_all.json`), and its TF-IDF and SVM rows need the main-corpus files.
- `make_figures.py` rebuilds the RoBERTa ensembles from the logits and checks every count and plotted value against the analysis outputs before drawing; it stops at the first failed check. In our test with the versions of `requirements.txt`, the PNG files were identical to the shipped ones and the PDF files differed only in their creation date. On another machine or with other library versions, the fitted temperatures can differ in their last digits, so the script compares recomputed values with small tolerances (see `TOL_AURC` in `make_figures.py`).

**Windows** (not tested): the commands use forward slashes and Python's own zip module, so they should also run in Command Prompt or PowerShell. Enable Python's UTF-8 mode first, because the table scripts write characters such as ±, → and ≥: `set PYTHONUTF8=1` (Command Prompt) or `$env:PYTHONUTF8=1` (PowerShell). Use `py` instead of `python` if `python` is not found.

**Google Colab** (not tested): clone or upload the repository, change into its folder (`%cd selective-decision-fake-review`), and run each command with `!` in front. Colab's preinstalled versions of numpy, pandas, scipy, scikit-learn and matplotlib may differ from `requirements.txt`; install the pinned versions (`!pip install -r requirements.txt`) if you want to compare files byte by byte.

## Article item → file

The tables are in `tables/`; each Markdown file has one section per table. `main_corpus_roberta_tfidf.md` has two parts, "Model: roberta-base" and "Model: tfidf-logreg", each with sections A-L. The numbers in each Markdown file are also in the JSON file of the same name (prefix `numbers_`).

| Article item | Content | File and section |
|---|---|---|
| Table 1 | Cost scenarios and thresholds | computed from the costs: `SCENARIOS` and `chow_thresholds` in `code/analyze_logits.py` (no generated table) |
| Table 2 | De-duplicated data | metadata of the logit files (`data_files`); counts checked and listed by `make_figures.py` in `figures/figure_data.json` (`figure2`, `domains`) |
| Table 3 | RoBERTa ensemble, forced decision at p = 0.5 | `main_corpus_roberta_tfidf.md`, roberta-base, section A |
| Table 4 | Decision rules under S2 (RoBERTa) | `main_corpus_roberta_tfidf.md`, roberta-base, section B-S2 |
| Table 5 | Risk-coverage summary | `main_corpus_roberta_tfidf.md`, roberta-base, section E (MSP and DISAGREEMENT; ENTROPY = MSP) |
| Table 6 | Hotel subgroups | `main_corpus_roberta_tfidf.md`, roberta-base, section K |
| Table 7 | Label-free indicators | `main_corpus_roberta_tfidf.md`, roberta-base, section H |
| Table 8 | Second dataset | TF-IDF block: `second_dataset_S10_S11part3.md`, "Supplementary Table S11, part 3", sections A and C; RoBERTa block: `second_dataset_S11_S13.md`, "RoBERTa (3 repetitions)", sections A and C |
| Figure 1 | Decision layer (schematic) | `figures/Figure 1.pdf`, `.png` |
| Figure 2 | Evaluation protocol | `figures/Figure 2.pdf`, `.png` (counts in `figure_data.json`, `figure2`) |
| Figure 3 | Calibrated probabilities (RoBERTa) | `figures/Figure 3.pdf`, `.png` (values in `figure_data.json`, `figure3`) |
| Figure 4 | Risk-coverage curves (RoBERTa) | `figures/Figure 4.pdf`, `.png` (values in `figure_data.json`, `figure4`) |
| Figure 5 | Cost-ratio grid, three models | `figures/Figure 5.pdf`, `.png` (values in `figure_data.json`, `figure5`) |
| Supplementary Table S1 | Decision rules under S1 (RoBERTa) | `main_corpus_roberta_tfidf.md`, roberta-base, section B-S1 |
| Supplementary Table S2 | Decision rules under S3 (RoBERTa) | `main_corpus_roberta_tfidf.md`, roberta-base, section B-S3 |
| Supplementary Table S3 | Cost-ratio grid (RoBERTa) | `main_corpus_roberta_tfidf.md`, roberta-base, section F (relative changes for S1-S3 in section C) |
| Supplementary Table S4 | Coverage floors | `main_corpus_roberta_tfidf.md`, roberta-base, section G |
| Supplementary Tables S5-S6 | TF-IDF logistic regression | `main_corpus_roberta_tfidf.md`, tfidf-logreg, sections A-L |
| Supplementary Table S7 | Mechanism of the entropy-adjusted score | `main_corpus_roberta_tfidf.md`, roberta-base, section J (tfidf-logreg: section J of its part) |
| Supplementary Table S8 | Single-model variability and calibration | `main_corpus_roberta_tfidf.md`, roberta-base, section I (ensemble calibration in section A) |
| Supplementary Table S9 | Ablation of the ensemble-level temperature scaling | `main_corpus_roberta_tfidf.md`, roberta-base, section L |
| Supplementary Table S10 | TF-IDF trained on all source reviews (second dataset) | `second_dataset_S10_S11part3.md`, "Supplementary Table S10" |
| Supplementary Table S11 | Expected cost of every rule on the second dataset | parts 1-2 (RoBERTa and TF-IDF on repetitions 0-2): `second_dataset_S11_S13.md`, "Supplementary Table S11"; part 3 (TF-IDF, five repetitions): `second_dataset_S10_S11part3.md`, "Supplementary Table S11, part 3" |
| Supplementary Table S12 | Post-hoc Platt diagnostic | `second_dataset_S11_S13.md`, "Supplementary Table S12" |
| Supplementary Table S13 | Excess cost over the cheapest rule | `second_dataset_S11_S13.md`, "Supplementary Table S13" |
| Supplementary Table S14 | BERT ensemble | `main_corpus_bert.md` (sections A-L) |
| Supplementary Table S15 | Comparison with published results under their protocol | `published_protocol_S15.md`; full numbers in `results/published/` (in `results/results.zip`) |

`tables/second_dataset_S11_roberta_vs_tfidf.md` is an intermediate file (parts 1-2 of Supplementary Table S11) that `make_supp_tables_s11_s13.py` reads. Section D of the main-corpus tables (same coverage, different selection) supports the coverage-matched comparison of Section 3.5; it has no table number.

## Retrain the models (GPU, optional)

The decision-layer results above are reproduced exactly from the saved logits. Retraining produces new logits: GPU fine-tuning is not bit-wise reproducible across hardware and library versions, which is why the logits of every run are released. New runs should therefore agree with the article up to the run-to-run variation reported there, not to the last digit. The TF-IDF models are deterministic and run on a CPU.

All commands are run from the repository root, with `requirements-gpu.txt` installed for the transformers and with the three main-corpus files in the root folder (except for the second dataset). New logits are written to `logits_new/`; the analysis commands above accept them with `--logits_dir`. Each training script skips runs whose output file already exists, so an interrupted run can be resumed with the same command. `train_save_logits.py`, `train_salminen.py` and `published_protocol.py train` accept `--dry_run`, which checks the data and the splits without training.

```bash
# Main corpus: RoBERTa and BERT ensembles (45 runs each; 2.3-5.3 min per run on one T4 GPU)
python code/train_save_logits.py --data_dir . --out_dir logits_new/roberta --model_name roberta-base
python code/train_save_logits.py --data_dir . --out_dir logits_new/bert --model_name bert-base-uncased
# Main corpus: TF-IDF logistic regression (15 runs, CPU)
python code/train_classical.py --data_dir . --out_dir logits_new/tfidf_logreg

# Second dataset (decompress data/salminen_processed.csv.gz first, see "Data")
python code/train_salminen.py --data data/salminen_processed.csv --out_dir logits_new/salminen_roberta --model_name roberta-base --reps 0 1 2
python code/train_salminen.py --data data/salminen_processed.csv --out_dir logits_new/salminen_tfidf_sample --model_name tfidf-logreg
python code/train_salminen.py --data data/salminen_processed.csv --out_dir logits_new/salminen_tfidf_all --model_name tfidf-logreg --train_pool all

# Published protocol (Supplementary Table S15): RoBERTa (GPU) and TF-IDF / SVM (CPU)
python code/published_protocol.py train --data_dir . --meta data/hotel_meta_v2.csv --out_dir logits_new/published_roberta --model_name roberta-base --variant all
python code/published_protocol.py classical --data_dir . --meta data/hotel_meta_v2.csv --out results/published

# Second dataset from the original file of Salminen et al. (2022) (fake_reviews_dataset.csv from https://osf.io/tyue9/).
# Needs NLTK and its WordNet data (python -m nltk.downloader wordnet). --main_dir . checks the cleaning function
# against the clean_review column of the main corpus and can be left out if the main-corpus files are not available.
python code/prepare_salminen.py --raw fake_reviews_dataset.csv --out_dir data_salminen_new --main_dir .
```

In the article, RoBERTa on the second dataset uses repetitions 0-2 (`--reps 0 1 2`), with one model per repetition; TF-IDF uses all five repetitions. `make_figures.py` checks the logits against the analysis outputs, so run `analyze_logits.py` on new logits before drawing figures from them.

## Keys in the results files

`analyze_logits.py` writes, for each group of runs, `results_per_repetition.json` (one entry per split and repetition, keyed `<sources>><target>|rep<r>`, for example `hotel+restaurant>doctor|rep0`) and summary CSV files with the mean (`_mean`) and standard deviation (`_sd`) over repetitions (`table_main.csv`, `table_calibration.csv`, `table_label_free.csv`, `table_risk_coverage.csv`, `table_sensitivity_cost.csv`, `table_sensitivity_mincov.csv`, `table_single_models.csv`, `table_hotel_subgroups.csv`), and `excluded_runs.json` (runs that predict one class for every source-validation review; one RoBERTa run of the main corpus). The decision rules have these keys; `_S1`, `_S2` and `_S3` give the cost scenario:

| Key | Rule |
|---|---|
| `FORCED-0.5` | forced decision at p = 0.5 |
| `FORCED-F1` | forced, F1-tuned: one threshold that maximizes validation macro-F1 |
| `FORCED-COST_S*` | forced, cost-optimal: threshold C_false / (C_false + C_miss) |
| `CHOW_S*` | cost-derived three-way rule (the rule of the article) |
| `CHOW-TUNED_S*` | three-way rule with two thresholds tuned on validation to minimize expected cost |
| `MSP-TUNED_S*` | confidence rule: defer if max(p, 1 - p) is below a threshold tuned on validation (MSP = entropy) |
| `DISAGREE-TUNED_S*` | ensemble disagreement rule: defer if the standard deviation of the members' probabilities exceeds a tuned threshold (ensembles only) |
| `ORIGINAL-SDF` | entropy-adjusted score p * (1 - H(p)) with two thresholds tuned on validation (comparison rule, Section 3.5) |
| `FORCED-COST-NOENSCAL_S*`, `CHOW-NOENSCAL_S*` | the same rules without the ensemble-level temperature scaling (Supplementary Table S9) |
| `MSP@CHOW-COVERAGE_S*`, `MSP@ORIGINAL-COVERAGE` | MSP ranking at the coverage of the cost-derived rule or of the entropy-adjusted score |
| `DUAL-F1` | (coverage-floor analysis) two thresholds on p maximizing accepted macro-F1 subject to a coverage floor |

For each rule, `decisions` gives `coverage`, `acc_accepted`, `risk_accepted` (error on accepted reviews), `f1_dec`, `recall_dec`, `precision_dec` (deceptive class), `macro_f1` and the expected costs `cost_S1`, `cost_S2`, `cost_S3`. Other entries: `thresholds`; `rc` (risk-coverage: `AURC`, `E-AURC` and `risk_at` for the scores `MSP`, `ENTROPY`, `DISAGREEMENT`); `calibration` (ECE before and after the ensemble-level scaling, temperatures, ensemble weights); `label_free` (label-free indicators); `sensitivity_cost` (28-setting cost grid); `sensitivity_mincov` (coverage floors); `hotel_subgroups` (hotel target only). In the JSON files written by `make_tables_salminen.py`, the prefixes `f1forced`, `forced`, `chow`, `tuned`, `msp` and `reviewed` denote the forced F1-tuned, forced cost-optimal, cost-derived, tuned three-way, tuned MSP and entropy-adjusted rules. The internal names `ORIGINAL-SDF`, `reviewed` and `original_sdf_*` (function names) refer to the entropy-adjusted score; they are kept so that the files remain comparable with earlier outputs. The published-protocol files (`results/published/`) give accuracy, balanced accuracy, precision, recall, F1, macro-F1 and the share of reviews predicted deceptive for the restaurant and doctor test sets.

## License

The code and the saved logits are released under the MIT license (see `LICENSE`). The main-corpus review texts are not part of this repository; they belong to their authors and remain under the terms of the original corpora (Ott et al. 2011; 2013; Li et al. 2014). `data/salminen_processed.csv.gz` is derived from the dataset of Salminen et al. (2022) and is shared under CC BY 4.0, with the changes described above. `data/hotel_meta_v2.csv` contains no review text.

## Citation

Please cite the archive of this repository (see `CITATION.cff`; Zenodo assigns a DOI to each release) and the article:

> Ali OAJ, Abu Bakar A, Kaur W. Selective decision-making for cross-domain fake review detection under uncertainty and cost constraints. Under review.

Please also cite the data sources:

> Ott M, Choi Y, Cardie C, Hancock JT. 2011. Finding deceptive opinion spam by any stretch of the imagination. In: *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, 309-319.
>
> Ott M, Cardie C, Hancock JT. 2013. Negative deceptive opinion spam. In: *Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 497-501.
>
> Li J, Ott M, Cardie C, Hovy E. 2014. Towards a general rule for identifying deceptive opinion spam. In: *Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 1566-1576. https://doi.org/10.3115/v1/P14-1147
>
> Salminen J, Kandpal C, Kamel AM, Jung S-G, Jansen BJ. 2022. Creating and detecting fake reviews of online products. *Journal of Retailing and Consumer Services* 64:102771. https://doi.org/10.1016/j.jretconser.2021.102771

## Version

Version 2.0 is the version described in the article and replaces versions 1.0 and 1.1.

## Contact

Questions about the code or the data: please open an issue at https://github.com/alesiali/selective-decision-fake-review/issues or write to the corresponding author of the article.
