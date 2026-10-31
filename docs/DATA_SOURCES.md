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
  ATLN and EPAC storms from 2003–2016 overlap NASA Tropical Storm storms;
  map both to one storm key before splitting a combined manifest.
- IBTrACS v4: https://doi.org/10.25921/82ty-9e16. Labels only.
- INSAT-3D / 3DR: MOSDAC, registration required, not redistributed here.
  The intensity page reads the public Infrared-1 JPEG on
  https://mausam.imd.gov.in/responsive/satellite.php when someone clicks
  it. That file is not committed, and tests do not download it.
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
