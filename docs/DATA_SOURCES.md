# Data sources

Do not commit imagery or best-track dumps. CI must not download them.

- NASA Tropical Storm Wind Speed, TorchGeo `TropicalCyclone`.
  Atlantic and East Pacific, 2000–2019. Chips are 366×366 (Gurung et al. 2023).
  Labels are HURDAT2 1-minute sustained winds (`data.wind_averaging: 1min`
  in `config/config.yaml`). Download uses azcopy. Do not download in CI.
  Cite https://doi.org/10.1109/JSTARS.2020.3011907.
  The Hugging Face mirror `torchgeo/tropical_cyclone` lists CC-BY-4.0 and
  about 2.4 GB. Do not depend on the Hub viewer in CI. A local copy can
  be turned into a manifest later; this phase does not ship a downloader.
  It is a mirror only. The main loader is TorchGeo
  `TropicalCyclone`: `python -m src.nasa_tc --root data/torchgeo_tc`
  builds `data/nasa_tc/manifest.csv` from TorchGeo's CSVs and
  `<root>/<split>/<Image ID>.jpg`. Download once with
  `TropicalCyclone(root, split, download=True)` (needs `azcopy`, public
  blob, no credentials). Images stay at native size, recorded as
  `native_height`/`native_width`/`orig_size`: 103,590 frames are 366×366
  and 11,042 are 1093×1093. The 1093 frames cover the same footprint at
  about 3× pixel density, so `src.nasa_tc.to_working_grid` area-averages
  them to the 366 grid (`working_resample=area_average`). `npf_000` and
  `npf_001` are colour-enhanced, not grey IR. They are dropped and listed
  in `nasa_tc_exclusions.csv` next to the manifest. The manifest has no
  `split` column; `src.splits.assign_storms` adds it. Pixels are an 8-bit render, not Kelvin:
  call them pixel stats. Labels are HURDAT2 1-minute sustained winds, so
  `wind_averaging=1min`. The competition train/test sides share storms,
  so they are kept only as `source_split`; use storm-wise splits on the
  union. Tests use tiny fake tables and never download.
  There is no distance-to-land column. `storm_id` is a 3-letter id and
  the only time is `relative_time_s`, so this loader does not join
  IBTrACS. `near_land` stays unknown. See `nasa_tc/audit/README.md`.
- TCIR: https://github.com/BoyoChen/TCIR — MIT repository, 201×201, four
  channels. Confirm satellite redistribution terms separately. Southern
  Hemisphere storms are not silently flipped. Phase 2 features use the
  first channel only.
  `python -m src.tcir --h5 TCIR-ATLN_EPAC_WPAC.h5 TCIR-CPAC_IO_SH.h5`
  reads only IR1 (channel index 0 in the README order IR1, WV, VIS, PMW),
  keeps 201×201, and writes `ir1/<sample_id>.npy`, a manifest, and
  `tcir_exclusions.csv`. The `info` table is a pandas 0.15 fixed frame
  that current pandas cannot read, so the loader decodes it with h5py.
  Labels: JTWC for WPAC, HURDAT2 for ATLN and EPAC (documented, 1-minute).
  TCIR does not document CPAC, IO or SH; an IBTrACS v04r01 match shows they
  equal the US agency 1-minute winds (JTWC for IO/SH, HURDAT2 for CPAC), so
  those rows say `label_source_documented=false` and `*_inferred`.
  `hemisphere` is recorded per row. `--flip-southern` is an opt-in
  north-south mirror and sets `flipped`; the default is no flip. TCIR has
  no official split in these files, so there is no `source_split`. The
  2017 file `TCIR-ALL_2017.h5` is a separate year and is not loaded here.
  `src.tcir` joins `tcir/audit/distance_to_land.csv` on `sample_id`
  (`distance_to_land_km`, `near_land` at 20 km). Unmatched rows and blank
  distances stay unknown. `labels.open_water_km` is the threshold the
  training label uses. The audit CSV is not a coastline download.
  ATLN and EPAC storms from 2003–2016 overlap NASA Tropical Storm storms;
  map both to one storm key before splitting a combined manifest.
- INCYDE, the cropped INSAT-3D North Indian Ocean cyclone set, Zenodo
  record 8015544, file `Cropped_cyclone_data.zip`, MD5
  `ed5dac254c689e5a327dee7580324347`. `python -m src.incyde --root <extract> --out data/incyde/manifest.csv`
  reads local JPEGs and `Cropped_cyclone_labels.csv`. It does not download
  the zip. Filename clocks are IST. `Sequence` is the storm key; the
  filename `obj` is not. The split and the exclusion list committed under
  `incyde/audit/` are the source of truth. Labels are IMD 3-minute winds.
  About 91% of in-span original labels are interpolated, and only two
  storms reach SuCS. `src.incyde` joins `incyde/audit/frame_flags.csv`
  on the source filename and carries `on_fix`, `distance_to_land_km`,
  and `near_land`. Augmented copies inherit the source row. Unknown
  stays unknown. The 3-minute label is not rescaled. Tests use tiny
  fixtures and never download.
- IBTrACS v4: https://doi.org/10.25921/82ty-9e16. Labels only.
- INSAT-3D / 3DR: MOSDAC, registration required, not redistributed here.
  The intensity page reads the public Infrared-1 JPEG on
  https://mausam.imd.gov.in/responsive/satellite.php when someone clicks
  it. That file is not committed, and tests do not download it.
  Cropped INCYDE IR is Zenodo 8015544 (`Cropped_cyclone_data.zip`,
  MD5 `ed5dac254c689e5a327dee7580324347`). `python -m incyde.manifest` builds
  a local manifest from an already downloaded unzip and does not fetch
  it. CI must not download it. Labels stay raw IMD 3-minute winds
  (`wind_averaging=3-min`, `wind_averaging_s=180`) with no conversion.
  The modeling split is `incyde/audit/storm_split.csv`. Original frames
  only. `exclusions.csv` drops a row only when `default_exclude` is true.
- TC PRIMED: https://doi.org/10.25921/dmy1-0595. Optional later source.

`wind_averaging` must be a column on every manifest, for example `1-min`,
`3-min`, or `10-min`. Mixed periods in one manifest are rejected. The
17/34/48/64/91 kt names are this project's grouping, not official IMD classes,
and they are not a warning product. IMD itself uses 3-minute winds, so those
bins are not the IMD 3-minute scale.

`center_source` may be `dataset_precentered` or a fix available at image
time. Do not crop with a best-track center revised after the image time.
Phase 2 radial and eye features assume the crop is already centered.

Smoke manifests from `src/make_smoke_data.py` use `wind_averaging=synthetic`
and `is_synthetic=true`. They are not a dataset.
