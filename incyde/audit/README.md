# INCYDE cropped-set audit

Audit of the cropped INCYDE cyclone images (CPU only, 12 November 2026).
This folder holds the scripts, the proposed storm-wise split, a small
exclusions list, and per-frame `on_fix` / near-land flags. It does **not**
hold the images, the zip, the pickles, or the full ~103k-row
`manifest.csv`.

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
- About **91%** of original-frame labels are interpolated: **19,363 of
  21,217** in-span originals are not at an exact IMD observation time
  (`on_fix=false`). The earlier #31 note said 19,364; the exact count
  from the 0-second coincidence rule in `frame_flags.csv` is 19,363
  (1,854 on-fix originals). Widening the tolerance to 5 minutes does
  not change any flag.
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
| `frame_flags.csv` | One row per **original** INCYDE filename (21,313 rows, same basename key as `exclusions.csv`). Augmented copies inherit the source frame's flags (strip `_(flip_h|flip_v|rot_90|rot_180|rot_270)`). Columns: `filename`, `on_fix`, `lat`, `lon`, `dist_to_coast_km`, `near_land`. |
| `build_frame_flags.py` | Rebuilds `frame_flags.csv` from the manifest, the IMD workbook, and Natural Earth 10m land. |

Unnamed depressions have an empty `name`. `peak_imd_class` uses the IMD
scale on each storm's maximum labelled wind: D ≤27, DD ≤33, CS ≤47,
SCS ≤63, VSCS ≤89, ESCS ≤119, SuCS ≥120 kt.


## Per-frame `on_fix` and near-land flags

`frame_flags.csv` is keyed by `filename`, using the same basename keys as
`exclusions.csv` (for example `08OCT2013_0830_1.jpg`). The committed file
has one row per original frame (21,313). Augmented copies are not listed;
they inherit `on_fix`, `lat`, `lon`, `dist_to_coast_km`, and `near_land`
from their source filename by stripping `_(flip_h|flip_v|rot_90|rot_180|rot_270)`.
Rebuild with `--all-frames` if you need a row for every JPEG.

### `on_fix`

`on_fix` is true when the frame's UTC timestamp **exactly equals** an
IMD best-track `time_resolved` value for that storm. The rule and
tolerance:

- Filename clocks are IST, as established in the #31 audit. The
  manifest already stores UTC as IST minus 5 hours 30 minutes.
- Fix times come from `Best_Tracks_Data_1982-2026.xlsx` (RSMC New
  Delhi), parsed with `imdtrack` 0.3.0, using `time_resolved` (the
  same field as the #31 interpolation, which corrects day/month swaps).
- **Tolerance is 0 seconds.** Both clocks are minute-aligned. A
  1-minute or 5-minute window flags the same 1,854 in-span originals.
- A frame outside that storm's fix span is `on_fix=false`. We do not
  guess a match against another storm.

Counts in the committed file: **1,854 of 21,313 originals** are
`on_fix=true` (1,854 of 21,217 in-span originals; all 96
outside-span originals are false). The remaining 19,459 originals are
`on_fix=false`.

### Storm-centre position, `dist_to_coast_km`, and `near_land`

The storm centre at the frame time is taken from the coincident IMD
fix, or linearly interpolated in time between the two neighbouring
**usable** fixes. A fix is usable only if latitude and longitude are
both finite and `pos_suspect` is false. We never interpolate across
an unusable fix, never extrapolate past the first or last fix, and
never invent a position.

`dist_to_coast_km` is the distance in kilometres from that point to
Natural Earth 10m land, in a regional azimuthal-equidistant projection
centred on 15°N 80°E. It is **0** when the centre is on or inside a
land polygon. `near_land` is:

- `true` if `dist_to_coast_km` ≤ 20, or the centre is over land
- `false` if the centre is more than 20 km offshore
- `unknown` if there is no usable position (`lat`, `lon`, and
  `dist_to_coast_km` are then left blank)

Coastline source: **Natural Earth 10m land, version 5.1.1**, public
domain, downloaded on 16 November 2026 from
[naciscdn.org](https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip)
(`ne_10m_land.VERSION.txt` in that zip is `5.1.1`, dated 9 May 2022).
At 1:10 million scale the shoreline can be off by on the order of a
kilometre, which is well below the 20 km threshold for most of this
basin, but small islands and complex deltas are less certain.

Counts in the committed file (21,313 originals): `near_land` true
5,547 / false 15,492 / unknown 274.

The 274 originals with `near_land=unknown` (and their copies) are
left unknown on purpose:

- Sequence 43 / IMD `2017-007`: 143 originals after 19 Oct 2017
  18:00 UTC, because 11 of 15 IMD fixes have latitude but no
  longitude.
- Sequence 42 / IMD `2017-006`: 73 `wrong_date_imd_daymonth_swap`
  originals whose filename times fall in mid-September, outside the
  October track span.
- Sequence 56 / IMD `2018-010`: 35 originals between the 21 Sep 2018
  18:00 and 22 Sep 00:00 UTC fixes, because the 21:00 UTC fix is
  marked `pos_suspect`.
- 23 originals already flagged `outside_imd_track_span` (12 from
  Sequence 90 / `2021-009` before the first fix, 11 from Sequence 12
  / `2014-005` after the last fix).

### Regenerating `frame_flags.csv`

Audit-only dependencies, in addition to those for the manifest:
`shapely`, `pyproj`, `pyshp`, `imdtrack`. This is **not** part of CI.

```bash
python incyde/audit/build_frame_flags.py \
  --manifest /path/to/manifest.csv \
  --imd-best-track /path/to/imd_best_tracks_1982-2026.xlsx \
  --land-shapefile /path/to/ne_10m_land.shp \
  --output incyde/audit/frame_flags.csv
```

`--originals-only` is the default and matches the committed file. Pass
`--all-frames` to emit all 103,368 JPEGs, including augmented copies.

The 20 km threshold and 0-second `on_fix` tolerance are the defaults.
`lat` and `lon` are included so a loader can audit the join; they are
not a substitute for the IMD workbook.

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
