# Baseline comparison

Software check only. Do not quote these numbers as experimental results. Operational baselines (ADT, SATCON, Dvorak) are not joined. The locked test split was not used for fitting, selection, or SMOTE. This is not a forecast or a warning.

Validation scores. The test split was not used for fitting, selection, or SMOTE.
Persistence is a reference baseline. It uses the previous frame's best-track label, not the imagery. Image-only models cannot be compared against it.
Learned models are sklearn pipelines. SMOTE and class weights stay inside the training fold.
The official competition side, when present, stays in `source_split`.
`quote_as_result` is no. This is not a forecast and not an operational-readiness claim.
Per-class validation RMSE, MAE, and bias are in `per_class_metrics.csv`.
Those intervals resample storm IDs. The test split is not in that table.

## Run

- command: `python -m src.baselines --config config/config_incyde.yaml --manifest data/incyde/manifest.csv --out results/incyde_baselines --workers 4 --feature-cache data/incyde/feature_table.csv --manifest-command 'python -m incyde.manifest --root /tmp/incyde/unpacked --out data/incyde/manifest.csv --zip /tmp/incyde/Cropped_cyclone_data.zip --skip-image-check'`
- manifest_command: `python -m incyde.manifest --root /tmp/incyde/unpacked --out data/incyde/manifest.csv --zip /tmp/incyde/Cropped_cyclone_data.zip --skip-image-check`
- manifest: `data/incyde/manifest.csv`
- git_commit: `94a1a180fb1cbcf87481c72cfc8d37ac39998898`
- split_strategy: `recorded`
- split_seed: `not_used`
- baseline_seed: `42`
- feature_selection_seed: `42`
- test_fraction: `not_used`
- val_fraction: `not_used`
- cv_pool: `train_val`
- cv_group_column: `cv_group`
- cv_collapse_rare: `False`
- cv_assign: `round_robin`
- workers: `4`
- rows: `21240`
- storms: `91`
- rows_after_features: `21240`
- storms_after_features: `91`
- modeling_split: `test:rows=4415,storms=17; train:rows=12757,storms=60; val:rows=4068,storms=14`
- dropped_cross_split_links: `0`
- use_smote: `True`
- bootstrap_resamples: `1000`
- per_storm_breakdown: `True`
- per_class_eval_split: `val`
- per_class_resample: `storm_id`
- quote_as_result: `no`
- official_split_used_for_training: `no`
- source_split: `test:rows=2503,storms=11; train:rows=16338,storms=71; val:rows=2399,storms=9`
- storms_on_both_source_sides: `0`
- wind_averaging: `3-min`
- wind_averaging_s: `180`
- label_scope: `INSAT-only`
- comparability: `INSAT-3D/3DR only. Not comparable with NASA or TCIR tables. Not like-for-like with Agrawal et al. 2024. All metrics use raw 3-minute IMD labels. wind_averaging is 3-min and wind_averaging_s is 180 on every row. There is no 1-minute conversion.`
- metrics_use: `All metrics use raw 3-minute IMD labels. wind_averaging is 3-min and wind_averaging_s is 180 on every row. There is no 1-minute conversion.`
- on_fix_definition: `on_fix is joined from incyde/audit/frame_flags.csv on filename. It is true only when that file says true: the UTC timestamp equals an IMD best-track time_resolved for that storm. near_land and dist_to_coast_km come from the same file. unknown stays unknown and is not defaulted to false or zero.`
- augmented_dropped: `82055`
- default_excluded_originals: `73`
- flagged_not_default_originals: `90`
- drop_flagged: `false`
- on_fix_rows: `1854`
- on_fix_false_rows: `19386`
- on_fix_unknown_rows: `0`
- near_land_unknown_rows: `201`
- dist_to_coast_unknown_rows: `201`
- frame_flags_source: `/agent/repos/cloudie/incyde/audit/frame_flags.csv`
- originals_before_exclusions: `21313`
- rows_written: `21240`
- pixel_units: `8bit_uncalibrated`
- year_holdout_2019: `not_built`
- zenodo_doi: `10.5281/zenodo.8015544`
- zip_md5: `ed5dac254c689e5a327dee7580324347`

| task | model | role | n_train_rows | n_val_rows | n_train_storms | n_val_storms | mae_kt | rmse_kt | bias_kt | macro_f1 | fit_notes | quote_as_result | synthetic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regression | mean | image_features | 12757 | 4068 | 60 | 14 | 14.2646 | 18.3398 | 2.6290 |  | no_smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:no_smote/no_smote/no_smote/no_smote/no_smote | no | no |
| regression | persistence | reference_baseline | 12697 | 4054 | 60 | 14 | 0.1801 | 0.9995 | 0.0074 |  | reference_baseline;previous_best_track_label;not_imagery;previous_same_storm_wind;not_a_feature;no_pipeline;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:previous_same_storm_wind | no | no |
| regression | ridge | image_features | 12757 | 4068 | 60 | 14 | 14.6592 | 19.1601 | 2.4649 |  | no_smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:no_smote/no_smote/no_smote/no_smote/no_smote | no | no |
| regression | lr | image_features | 12757 | 4068 | 60 | 14 | 14.6606 | 19.1638 | 2.4637 |  | no_smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:no_smote/no_smote/no_smote/no_smote/no_smote | no | no |
| regression | knn | image_features | 12757 | 4068 | 60 | 14 | 14.9004 | 20.9545 | 1.4703 |  | no_smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:no_smote/no_smote/no_smote/no_smote/no_smote | no | no |
| regression | rf | image_features | 12757 | 4068 | 60 | 14 | 13.0810 | 18.1955 | 0.6441 |  | no_smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:no_smote/no_smote/no_smote/no_smote/no_smote | no | no |
| regression | xgboost | image_features | 12757 | 4068 | 60 | 14 | 13.6517 | 19.0568 | 0.4630 |  | no_smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:no_smote/no_smote/no_smote/no_smote/no_smote | no | no |
| regression | svr | image_features | 12757 | 4068 | 60 | 14 | 12.2395 | 18.8555 | -4.2398 |  | no_smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:no_smote/no_smote/no_smote/no_smote/no_smote | no | no |
| classification | mean | image_features | 12736 | 4068 | 60 | 14 |  |  |  | 0.1493 | majority_train_class;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:majority_train_class/majority_train_class/majority_train_class/majority_train_class/majority_train_class | no | no |
| classification | persistence | reference_baseline | 12676 | 4054 | 60 | 14 |  |  |  | 0.9863 | reference_baseline;previous_best_track_label;not_imagery;previous_same_storm_wind;not_a_feature;no_pipeline;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:previous_same_storm_wind | no | no |
| classification | logreg | image_features | 12736 | 4068 | 60 | 14 |  |  |  | 0.2108 | smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:smote/smote/smote/smote/smote | no | no |
| classification | knn | image_features | 12736 | 4068 | 60 | 14 |  |  |  | 0.2161 | smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:smote/smote/smote/smote/smote | no | no |
| classification | rf | image_features | 12736 | 4068 | 60 | 14 |  |  |  | 0.2060 | smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:smote/smote/smote/smote/smote | no | no |
| classification | xgboost | image_features | 12736 | 4068 | 60 | 14 |  |  |  | 0.2150 | smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:smote/smote/smote/smote/smote | no | no |
| classification | svc | image_features | 12736 | 4068 | 60 | 14 |  |  |  | 0.2193 | smote;cv_pool:train_val;cv_group:cv_group;cv_assign:round_robin;inner_cv:smote/smote/smote/smote/smote | no | no |
