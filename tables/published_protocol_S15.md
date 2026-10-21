# Supplementary Table S15. Base classifiers under the published cross-domain protocol

This table is static (it is not written by a script); the commands that regenerate its numbers are given at the end.

Protocol of Li et al. (2014, Section 5.2, Table 4) and Ren & Zhang (2016, Section 4.4, Table 4): the classifiers are trained on hotel reviews, truthful (customer) versus deceptive (Turker) only, and tested on all restaurant reviews (200 genuine, 200 deceptive) and all doctor reviews (200 genuine, 355 deceptive) of the main corpus. Employee-written hotel reviews are excluded (`data/hotel_meta_v2.csv`). Variant "all": both hotel polarities (800 Turker and 796 customer reviews; primary analysis). Variant "positive": positive hotel reviews only (400 and 400). As in the main experiments, 15% of the training pool (stratified by label and polarity; split seeds 42-46) is held out as source validation for temperature scaling and ensemble weights; the RoBERTa ensemble has three models per repetition, with the seeds and hyper-parameters of the main runs. Predictions are made at p = 0.5. Mean ± SD over five repetitions where applicable; the SVM rows are single runs trained on the whole pool. Values are rounded once, half up (for example, 0.7795 = 1559/2000 is reported as 0.780).

| Model | Variant | Restaurant accuracy | Restaurant: predicted deceptive (true 0.500) | Doctor accuracy | Doctor balanced accuracy | Doctor: predicted deceptive (true 0.640) |
|---|---|---|---|---|---|---|
| RoBERTa ensemble | all | 0.770 ± 0.010 | 0.702 | 0.716 ± 0.014 | 0.615 ± 0.021 | 0.895 |
| TF-IDF + LR | all | 0.780 ± 0.008 | 0.571 | 0.660 ± 0.010 | 0.635 ± 0.013 | 0.629 |
| TF-IDF + LR | positive | 0.796 ± 0.015 | 0.520 | 0.658 ± 0.011 | 0.567 ± 0.015 | 0.846 |
| Linear SVM, unigram counts, raw text, whole pool | all | 0.708 | 0.683 | 0.625 | 0.611 | 0.582 |
| Linear SVM, unigram counts, raw text, whole pool | positive | 0.760 | 0.490 | 0.638 | 0.603 | 0.652 |

Notes

- Li et al. (2014) tested on doctor reviews with 200 Turker and 200 customer reviews; the doctor file used here has 355 deceptive and 200 genuine reviews, so the balanced accuracy (mean of the two class recalls) is the closer comparator to their accuracy.
- The linear SVM on unigram counts of the raw review text is a rough check against the "SVM unigram" row of Li et al. (2014), not a replication: their feature weighting, tokenization and hotel subset are not reported.
- The published accuracies under this protocol are compared in Section 5.1 of the article.

Regenerating the numbers (from the repository root)

- RoBERTa row, from the saved logits (no review texts needed):
  `python code/published_protocol.py analyze --logits_dir logits/published_roberta --model_name roberta-base --variant all --out results/published`
  writes `results/published/roberta-base_published_all.json`.
- TF-IDF and SVM rows; these need the three main-corpus files in the repository root (see README, "Data"):
  `python code/published_protocol.py classical --data_dir . --meta data/hotel_meta_v2.csv --out results/published`
  writes `results/published/classical_published_protocol.json`.

Both JSON files are included in `results/results.zip`; they also give precision, recall, F1 and macro-F1 of the deceptive class for each test domain.
