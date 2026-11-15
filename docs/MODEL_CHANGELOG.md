# Model changelog

Every change made to train or improve the Cloudie intensity model, newest last.
Times are IST (UTC+5:30). "Measured effect" is filled in only from real runs; "pending" means not measured yet.
All metrics are validation / storm-grouped CV only. Nothing here is an operational claim.

| # | Date & time (IST) | Change | Reason | Measured effect |
|---|---|---|---|---|
| 1 | 2026-11-15 23:38 IST | Opened branch `bean/incyde-cpu-model-v0.1` from main `8eff560` to train the first real Cloudie model on the shared box CPU (no GPU, no paid cloud, nothing run on Anlea's machines). Targets: max sustained wind (kt, regression) and intensity category (classification). | Anlea asked for a trained model, shipped weights and real eval. | pending |
| 2 | 2026-11-15 23:38 IST | Data decisions carried in: INCYDE is the fine-tune/eval set; drop only `default_exclude=true` rows of `incyde/audit/exclusions.csv` (438 S42 bad-date rows); keep `label_ne_imd_interp` and `outside_imd_track_span` rows; storm-ID splits from `storm_split.csv` with `cooccurrence_group` kept together; storm-grouped CV; `year_holdout_2019` off; augmentation = INCYDE `rot_180` plus circle-masked rotations, no flips, augmented copies stay in their original's fold. | Agreed team decisions (12 Nov) and the INCYDE audit (#31). | n/a (data policy) |
| 3 | 2026-11-15 23:38 IST | Label standard: IMD 3-minute winds. INCYDE labels unchanged. TCIR/NASA 1-minute labels converted to `wind_kt_3min = 0.93 x raw` as a single config value marked derived: "0.93, derived from WMO/TD-1555 Tables 1.1 and 1.2, at-sea exposure" (Harper, Kepert & Ginger 2010), not a WMO constant, pending Researchy's verification. Raw label and `wind_averaging_s` kept on every row; near-landfall rows (within ~20 km of land) flagged rather than silently converted. | Projects Manager's plan; target should match what IMD users compare against. | pending |
| 4 | 2026-11-15 23:38 IST | Plan: SimpleCNN (from main) pretrained on TCIR + NASA on box CPU, checkpoint every epoch, run manifest; then fine-tune on INCYDE via `--init-from` and compare against INCYDE from scratch. Report pretrain metrics with and without the 0.93 conversion. ResNet-18 deferred to Liora. Data prep kept in a separate script so it does not collide with Clara's INCYDE loader PR. | Projects Manager's plan; avoid duplicating Clara's work. | pending |
