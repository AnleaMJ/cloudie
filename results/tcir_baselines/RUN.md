# Baseline run

Software check only. Do not quote these numbers as experimental results. Operational baselines (ADT, SATCON, Dvorak) are not joined. The locked test split was not used for fitting, selection, or SMOTE. This is not a forecast or a warning.

Official competition labels stay in `source_split`. The modeling split is storm-wise.
The test split was not fit. `quote_as_result` is no.
Persistence is a reference baseline. It uses the previous frame's best-track label, not the imagery. Image-only models cannot be compared against it.

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

IR1 is channel 0, kept at native 201×201. Values are kelvin brightness temperature. Feature names stay `pixel_*` because `features.calibrated` is false. Southern Hemisphere frames were not flipped. Validation scores only; the test split was not fit.
