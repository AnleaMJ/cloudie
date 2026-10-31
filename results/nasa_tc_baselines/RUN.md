# Baseline run

Software check only. Do not quote these numbers as experimental results. Operational baselines (ADT, SATCON, Dvorak) are not joined. The locked test split was not used for fitting, selection, or SMOTE. This is not a forecast or a warning.

Official competition labels stay in `source_split`. The modeling split is storm-wise.
The test split was not fit. `quote_as_result` is no.

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
