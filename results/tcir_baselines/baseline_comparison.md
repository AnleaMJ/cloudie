# Baseline comparison

Software check only. Do not quote these numbers as experimental results. Operational baselines (ADT, SATCON, Dvorak) are not joined. The locked test split was not used for fitting, selection, or SMOTE. This is not a forecast or a warning.

Validation scores. The test split was not used for fitting, selection, or SMOTE.
Persistence is a reference baseline. It uses the previous frame's best-track label, not the imagery. Image-only models cannot be compared against it.
Learned models are sklearn pipelines. SMOTE and class weights stay inside the training fold.
The official competition side, when present, stays in `source_split`.
`quote_as_result` is no. This is not a forecast and not an operational-readiness claim.

## Run

- command: `python -m src.baselines --config config/config.yaml --manifest data/tcir/manifest.csv --out results/tcir_baselines --workers 4 --feature-cache data/tcir/feature_table.csv --manifest-command 'python -m src.tcir --h5 data/tcir_src/TCIR-ATLN_EPAC_WPAC.h5 data/tcir_src/TCIR-CPAC_IO_SH.h5 --out data/tcir/manifest.csv'`
- manifest_command: `python -m src.tcir --h5 data/tcir_src/TCIR-ATLN_EPAC_WPAC.h5 data/tcir_src/TCIR-CPAC_IO_SH.h5 --out data/tcir/manifest.csv`
- manifest: `data/tcir/manifest.csv`
- split_strategy: `storm_wise`
- split_seed: `42`
- baseline_seed: `42`
- feature_selection_seed: `42`
- test_fraction: `0.2`
- val_fraction: `0.2`
- workers: `4`
- rows: `69867`
- storms: `1285`
- rows_after_features: `69867`
- storms_after_features: `1285`
- modeling_split: `test:rows=15810,storms=257; train:rows=42138,storms=822; val:rows=11919,storms=206`
- dropped_cross_split_links: `0`
- use_smote: `True`
- quote_as_result: `no`
- official_split_used_for_training: `no`

| task | model | role | n_train_rows | n_val_rows | n_train_storms | n_val_storms | mae_kt | rmse_kt | bias_kt | macro_f1 | fit_notes | quote_as_result | synthetic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | mean | image_features | 42138 | 11919 | 822 | 206 | 23.1716 | 31.4942 | -11.1009 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | persistence | reference_baseline | 41316 | 11713 | 822 | 206 | 2.2081 | 3.4582 | -0.0979 |  | reference_baseline;previous_best_track_label;not_imagery;previous_same_storm_wind;not_a_feature;no_pipeline;inner_cv:previous_same_storm_wind | no | no |
| regression | ridge | image_features | 42138 | 11919 | 822 | 206 | 15.9387 | 21.3593 | -4.8037 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | lr | image_features | 42138 | 11919 | 822 | 206 | 15.9381 | 21.3583 | -4.8025 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | knn | image_features | 42138 | 11919 | 822 | 206 | 15.9762 | 22.0833 | -4.9699 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | rf | image_features | 42138 | 11919 | 822 | 206 | 15.3341 | 20.9887 | -5.3414 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | xgboost | image_features | 42138 | 11919 | 822 | 206 | 14.4033 | 19.5066 | -4.7112 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| regression | svr | image_features | 42138 | 11919 | 822 | 206 | 14.8926 | 21.0172 | -7.4850 |  | no_smote;inner_cv:no_smote/no_smote/no_smote | no | no |
| classification | mean | image_features | 41185 | 11750 | 822 | 206 |  |  |  | 0.0889 | majority_train_class;inner_cv:majority_train_class/majority_train_class/majority_train_class | no | no |
| classification | persistence | reference_baseline | 40348 | 11542 | 822 | 206 |  |  |  | 0.8991 | reference_baseline;previous_best_track_label;not_imagery;previous_same_storm_wind;not_a_feature;no_pipeline;inner_cv:previous_same_storm_wind | no | no |
| classification | logreg | image_features | 41185 | 11750 | 822 | 206 |  |  |  | 0.4514 | smote;inner_cv:smote/smote/smote | no | no |
| classification | knn | image_features | 41185 | 11750 | 822 | 206 |  |  |  | 0.3881 | smote;inner_cv:smote/smote/smote | no | no |
| classification | rf | image_features | 41185 | 11750 | 822 | 206 |  |  |  | 0.4208 | smote;inner_cv:smote/smote/smote | no | no |
| classification | xgboost | image_features | 41185 | 11750 | 822 | 206 |  |  |  | 0.4540 | smote;inner_cv:smote/smote/smote | no | no |
| classification | svc | image_features | 41185 | 11750 | 822 | 206 |  |  |  | 0.4791 | smote;inner_cv:smote/smote/smote | no | no |
