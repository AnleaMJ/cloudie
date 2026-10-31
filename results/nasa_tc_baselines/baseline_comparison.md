# Baseline comparison

Software check only. Do not quote these numbers as experimental results. Operational baselines (ADT, SATCON, Dvorak) are not joined. The locked test split was not used for fitting, selection, or SMOTE. This is not a forecast or a warning.

Validation scores. The test split was not used for fitting, selection, or SMOTE.
Persistence is the previous same-storm wind, not a feature.
Learned models are sklearn pipelines. SMOTE and class weights stay inside the training fold.
The official competition side, when present, stays in `source_split`.
`quote_as_result` is no. This is not a forecast and not an operational-readiness claim.

## Run

- command: `python -m src.baselines --config config/config.yaml --manifest data/nasa_tc/manifest.csv --out results/nasa_tc_baselines --workers 4 --feature-cache data/nasa_tc/feature_table.csv --manifest-command 'python -m src.nasa_tc --root data/torchgeo_tc --out data/nasa_tc/manifest.csv --reader csv'`
- manifest_command: `python -m src.nasa_tc --root data/torchgeo_tc --out data/nasa_tc/manifest.csv --reader csv`
- manifest: `data/nasa_tc/manifest.csv`
- split_strategy: `storm_wise`
- split_seed: `42`
- baseline_seed: `42`
- feature_selection_seed: `42`
- test_fraction: `0.2`
- val_fraction: `0.2`
- workers: `4`
- rows: `114632`
- storms: `638`
- rows_after_features: `114632`
- storms_after_features: `638`
- modeling_split: `test:rows=23434,storms=128; train:rows=73035,storms=408; val:rows=18163,storms=102`
- dropped_cross_split_links: `0`
- use_smote: `True`
- quote_as_result: `no`
- official_split_used_for_training: `no`
- source_split: `test:rows=44377,storms=371; train:rows=70255,storms=494`
- storms_on_both_source_sides: `227`

| task | model | n_train_rows | n_val_rows | n_train_storms | n_val_storms | mae_kt | rmse_kt | bias_kt | macro_f1 | fit_notes | quote_as_result | synthetic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | mean | 73035 | 18163 | 408 | 102 | 19.7336 | 26.3213 | -6.9502 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | persistence | 72627 | 18061 | 408 | 102 | 0.4762 | 1.5293 | -0.0114 |  | previous_same_storm_wind;not_a_feature;no_pipeline;inner_cv:previous_same_storm_wind | no | no |
| regression | ridge | 73035 | 18163 | 408 | 102 | 12.9375 | 17.7091 | -3.0352 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | lr | 73035 | 18163 | 408 | 102 | 12.9370 | 17.7081 | -3.0347 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | knn | 73035 | 18163 | 408 | 102 | 13.4103 | 18.7815 | -3.0638 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | rf | 73035 | 18163 | 408 | 102 | 11.9229 | 16.6626 | -2.6013 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | xgboost | 73035 | 18163 | 408 | 102 | 11.3326 | 15.7998 | -2.4761 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | svr | 73035 | 18163 | 408 | 102 | 11.7257 | 16.7581 | -4.2285 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| classification | mean | 72479 | 18055 | 408 | 102 |  |  |  | 0.0908 | majority_train_class;inner_cv:majority_train_class/majority_train_class/majority_train_class | no | no |
| classification | persistence | 72071 | 17952 | 408 | 102 |  |  |  | 0.9761 | previous_same_storm_wind;not_a_feature;no_pipeline;inner_cv:previous_same_storm_wind | no | no |
| classification | logreg | 72479 | 18055 | 408 | 102 |  |  |  | 0.4751 | smote;inner_cv:smote/smote/smote | no | no |
| classification | knn | 72479 | 18055 | 408 | 102 |  |  |  | 0.4120 | smote;inner_cv:smote/smote/smote | no | no |
| classification | rf | 72479 | 18055 | 408 | 102 |  |  |  | 0.4549 | smote;inner_cv:smote/smote/smote | no | no |
| classification | xgboost | 72479 | 18055 | 408 | 102 |  |  |  | 0.4870 | smote;inner_cv:smote/smote/smote | no | no |
| classification | svc | 72479 | 18055 | 408 | 102 |  |  |  | 0.5181 | smote;inner_cv:smote/smote/smote | no | no |
