# Data sources

Do not commit imagery or best-track dumps. CI must not download them.

- NASA Tropical Storm Wind Speed, TorchGeo `TropicalCyclone`.
  Atlantic and East Pacific, 2000–2019. Chips are 366×366 (Gurung et al. 2023).
  Labels are HURDAT2 1-minute sustained winds (`data.wind_averaging: 1min`
  in `config/config.yaml`). Download uses azcopy. Do not download in CI.
  Cite https://doi.org/10.1109/JSTARS.2020.3011907.
  The Hugging Face mirror `torchgeo/tropical_cyclone` lists CC-BY-4.0 and
  about 2.4 GB. It is a mirror only. The main loader is TorchGeo
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
