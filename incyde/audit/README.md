# INCYDE cropped-set audit

Audit of the cropped INCYDE cyclone images (CPU only, 12 November 2026).
This folder holds the scripts, the proposed storm-wise split, and a small
exclusions list. It does **not** hold the images, the zip, the pickles, or
the full ~103k-row `manifest.csv`.

## Source data

- Zenodo record: [8015544](https://zenodo.org/records/8015544)
- File: `Cropped_cyclone_data.zip`
- MD5: `ed5dac254c689e5a327dee7580324347` (matches the Zenodo value)
- Contents: 103,368 JPEGs plus `Cropped_cyclone_labels.csv`
  (columns: Sequence, Name, MSWS kts, Category, filename, Set)
- Filename pattern: `DDMONYYYY_HHMM_<obj>[_aug].jpg`

The IMD best-track workbook used for the label audit is
`Best_Tracks_Data_1982-2026.xlsx` from
[RSMC New Delhi](https://rsmcnewdelhi.imd.gov.in), parsed with
`imdtrack` 0.3.0 (`time_resolved` fixes day/month swaps).

## Key findings

- Labels are IMD **3-minute** sustained wind (knots). Filename times are
  **IST**: after subtracting 5 hours 30 minutes, a linear interpolation of
  the IMD best track (rounded to 5 kt) matches **99.68%** of in-span
  original frames. Treating the filename time as UTC matches only 55%.
- About **91%** of original-frame labels are interpolated: 19,364 of
  21,217 in-span originals are not at an exact IMD observation time.
- Only **2 SuCS** storms (KYAAR, AMPHAN) and **11 storms with ESCS-class
  frames** (the two SuCS storms plus 9 whose peak is ESCS). This is too
  few storms to treat a top-end improvement as proven.
- The default INCYDE train/val/test split has **789 co-scene timestamps**
  (1,578 original frames whose time also appears in another official set)
  and **0 SuCS in test** (KYAAR is in Validation, AMPHAN in Train).
- The proposed split is **train 60 / val 14 / test 17** storms, with
  **0 crossings**: no storm, source frame, or timestamp appears in more
  than one proposed split. Concurrent storms (shared timestamp) are kept
  in the same split. Augmented copies follow their source storm; they are
  used only when that storm is in train. Evaluation should use originals.
- Perceptual hash (pHash) finds **no near-duplicates**. The closest
  val/test-to-train pair is Hamming distance 6, and those two frames are
  from different storms three years apart (KYAAR vs an unnamed 2016
  storm), so they are not a copy.

## Files in this folder

| File | What it is |
| --- | --- |
| `storm_split.csv` | One row per INCYDE `Sequence` (91 storms). Columns: `Sequence`, `storm_id` (`INCYDE-Sxx_IMD-yyyy-nnn`), `name`, `basin`, `peak_imd_class`, `proposed_split` (`train`/`val`/`test`), `cooccurrence_group`. |
| `exclusions.csv` | `filename`, `qc_flag`, `default_exclude` for every flagged frame (163 originals and their 770 augmented copies). Only `wrong_date_imd_daymonth_swap` (73 Sequence-42 originals plus 365 augs) has `default_exclude=true`. `label_ne_imd_interp` (67 originals) and `outside_imd_track_span` (23 originals) are flag-only (`default_exclude=false`), so the loader keeps them unless asked to drop them. The 789 co-scene frames of INCYDE's default split are not listed here; the storm split handles them. |
| `proposed_split_groups.csv` | Co-occurrence groups and the peak-class assignment used to draw the split (seed `20261112`). |
| `build_manifest.py` | Rebuilds `manifest.csv` plus the three CSVs above. |
| `match.py` | `SEQUENCE_TO_IMD` mapping, including Sequence 46 → `2017-010`. |
| `checks.py` | Split-leakage and QC sanity checks on a built manifest. |
| `tables.py` | Per-class / per-split counts. |

Unnamed depressions have an empty `name`. `peak_imd_class` uses the IMD
scale on each storm's maximum labelled wind: D ≤27, DD ≤33, CS ≤47,
SCS ≤63, VSCS ≤89, ESCS ≤119, SuCS ≥120 kt.

## Regenerating the manifest

The full manifest is about 30 MB and is not in this repository. To
rebuild it:

1. Download `Cropped_cyclone_data.zip` from Zenodo 8015544 and confirm
   `md5sum` is `ed5dac254c689e5a327dee7580324347`.
2. Unzip it. You also need the parsed frame table `incyde2.pkl`
   (labels CSV + filename fields + pHash) and the per-original
   interpolation table `R.pkl` from a prior audit run, plus the IMD
   best-track workbook.
3. Install audit-only dependencies (not required for CI):
   `pandas`, `numpy`, `networkx`, `imdtrack`.
4. Run:

```bash
python incyde/audit/build_manifest.py   --root /path/to/incyde-data   --labels-pickle /path/to/incyde2.pkl   --imd-interp-pickle /path/to/R.pkl   --imd-best-track /path/to/imd_best_tracks_1982-2026.xlsx   --output-dir incyde/audit
```

`build_manifest.py` writes `<root>/manifest.csv` and refreshes
`storm_split.csv`, `exclusions.csv`, and `proposed_split_groups.csv`.

Then:

```bash
python incyde/audit/checks.py --manifest /path/to/manifest.csv
python incyde/audit/tables.py --manifest /path/to/manifest.csv
```

These scripts are not part of CI and do not need the dataset to be
present for the test suite to pass.
